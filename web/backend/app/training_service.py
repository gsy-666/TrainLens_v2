"""Web wrapper around the desktop training_center services (guided mode).

Reuses the Qt-free building blocks: JobManager (singleton, mutex for one
active job), UltralyticsAdapter, HistoryStore, MetricStore, PreflightWorker.
Events from the JobManager are appended to a ring buffer that the API
polls incrementally.
"""

import datetime
import logging
import math
import re
import shutil
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

from anylabeling.services.training_center.adapters.ultralytics_adapter import (
    UltralyticsAdapter,
)
from anylabeling.services.training_center.event_protocol import (
    TrainingEventType,
    create_console_output_event,
    create_stopped_event,
)
from anylabeling.services.training_center.history import get_history_store
from anylabeling.services.training_center.job_manager import get_job_manager
from anylabeling.services.training_center.metrics.models import MetricSample
from anylabeling.services.training_center.metrics.store import MetricStore
from anylabeling.services.training_center.models import (
    TrainingJob,
    TrainingMode,
    TrainingStatus,
    normalize_execution_mode,
)
from anylabeling.services.training_center.preflight.models import (
    GuidedPreflightContext,
)

_log = logging.getLogger(__name__)


def _resolve_output_dir(project: str, name: str) -> Path:
    """Mimic ultralytics' auto-incrementing run directory."""
    base = Path(project) / name
    if not base.exists():
        return base
    i = 2
    while (Path(project) / f"{name}{i}").exists():
        i += 1
    return Path(project) / f"{name}{i}"


class WebTrainingService:
    def __init__(self):
        from .web_runner import register_web_runner
        from .web_ssh_runner import register_web_ssh_runner

        register_web_runner()  # QProcess-based LocalRunner needs a Qt loop
        register_web_ssh_runner()  # QThread-based SSHRemoteRunner too
        self.jm = get_job_manager()
        self.events: deque = deque(maxlen=5000)
        self.seq = 0
        self.lock = threading.RLock()
        self.metric_store = MetricStore()
        self.adapter: Optional[UltralyticsAdapter] = None
        self.output_dir: Optional[Path] = None
        self.active_job_id: Optional[str] = None
        # recent (epoch, timestamp) marks used for ETA estimation
        self._epoch_marks: deque = deque(maxlen=6)
        self._eta_seconds: Optional[float] = None
        self.jm.subscribe_events(self._on_event)

    # ---- events ---------------------------------------------------------------
    _ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07")

    @classmethod
    def _strip_ansi(cls, obj):
        """Remove ANSI escape sequences from log-ish payload text."""
        if isinstance(obj, str):
            return cls._ANSI_RE.sub("", obj)
        if isinstance(obj, dict):
            return {k: cls._strip_ansi(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [cls._strip_ansi(v) for v in obj]
        return obj

    def _on_event(self, event):
        with self.lock:
            self.seq += 1
            try:
                payload = event.to_dict()
            except Exception:
                payload = {
                    "event_type": str(event.event_type),
                    "job_id": event.job_id,
                    "payload": getattr(event, "payload", {}),
                }
            self.events.append({"seq": self.seq, **self._strip_ansi(payload)})
            if event.event_type == TrainingEventType.EPOCH_METRICS:
                self._update_eta(event)
                self._feed_remote_metrics(event)
            elif event.event_type in (
                TrainingEventType.COMPLETED,
                TrainingEventType.FAILED,
                TrainingEventType.STOPPED,
            ):
                # terminal state: drop stale ETA so status() stops
                # reporting the last estimate forever
                self._epoch_marks.clear()
                self._eta_seconds = None
                self._cleanup_remote_staged_dataset(getattr(event, "job_id", ""))

    def _cleanup_remote_staged_dataset(self, job_id: str) -> None:
        """Delete the locally generated dataset of a finished remote job.

        Remote jobs upload the entire dataset to the server, so the local
        copy under the trainer datasets root is pure staging. Only directories
        under that generated root are removed — user data elsewhere is never
        touched. Caller must hold self.lock.
        """
        try:
            data = ""
            mode = ""
            job = self.jm.get_current_job()
            if job is not None and getattr(job, "job_id", None) == job_id:
                mode = getattr(job, "execution_mode", "") or ""
                data = getattr(job, "data", "") or ""
            if not data:
                for record in self.history(200):
                    if str(record.get("job_id")) == job_id:
                        mode = record.get("execution_mode", "") or ""
                        data = record.get("data", "") or ""
                        break
            if (
                normalize_execution_mode(mode or "local") != "remote_ssh"
                or not data
            ):
                return

            from anylabeling.services.auto_training.ultralytics.config import (
                get_dataset_path,
            )

            root = Path(get_dataset_path()).resolve()
            p = Path(data)
            ds_dir = (
                p.parent if p.suffix.lower() in (".yaml", ".yml") else p
            ).resolve()
            if root != ds_dir and root in ds_dir.parents and ds_dir.is_dir():
                shutil.rmtree(ds_dir)
                self.seq += 1
                self.events.append(
                    {
                        "seq": self.seq,
                        "event_type": "console_output",
                        "job_id": job_id,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "payload": {
                            "message": f"已清理本机暂存的训练集目录: {ds_dir}"
                        },
                    }
                )
        except Exception:
            _log.exception("Failed to clean remote-staged dataset")

    def _feed_remote_metrics(self, event) -> None:
        """Feed EPOCH_METRICS payloads into the MetricStore for remote jobs.

        Remote runs have no local results.csv while training (artifacts are
        downloaded only at completion), so live curves come from the event
        stream. Local jobs are intentionally skipped — their CSV tailer
        already feeds the store and double-feeding would duplicate points.
        Caller must hold self.lock.
        """
        try:
            job = self.jm.get_current_job()
            if job is None:
                return
            mode = normalize_execution_mode(getattr(job, "execution_mode", None))
            if mode != "remote_ssh":
                return
            payload = getattr(event, "payload", None) or {}
            metrics = payload.get("metrics") or {}
            if not isinstance(metrics, dict):
                return
            values = {
                str(k): float(v)
                for k, v in metrics.items()
                if isinstance(v, (int, float)) and math.isfinite(v)
            }
            if not values:
                return
            epoch = payload.get("epoch")
            total = payload.get("total_epochs")
            self.metric_store.add_sample(
                MetricSample(
                    job_id=event.job_id,
                    epoch=float(epoch) if epoch is not None else None,
                    total_epochs=int(total) if total else None,
                    values=values,
                    timestamp=float(getattr(event, "timestamp", None) or time.time()),
                )
            )
        except Exception:  # noqa - metrics feeding must never break events
            _log.debug("remote metrics feed failed", exc_info=True)

    def _update_eta(self, event) -> None:
        """ETA = mean duration of recent epochs x remaining epochs.

        Computed inline (instead of ETAEstimator) because only
        epoch-completion events are available here; the estimator's
        start/complete record pair does not fit that stream.
        Caller must hold self.lock.
        """
        payload = getattr(event, "payload", None) or {}
        try:
            epoch = int(payload.get("epoch"))
            total = int(payload.get("total_epochs"))
            ts = float(getattr(event, "timestamp", None) or time.time())
        except (TypeError, ValueError):
            return
        marks = self._epoch_marks
        if marks and epoch <= marks[-1][0]:
            return  # duplicate or out-of-order event
        marks.append((epoch, ts))
        remaining = total - epoch
        if remaining <= 0:
            self._eta_seconds = 0.0
            return
        per_epoch = [
            (marks[i][1] - marks[i - 1][1]) / (marks[i][0] - marks[i - 1][0])
            for i in range(1, len(marks))
            if marks[i][0] > marks[i - 1][0] and marks[i][1] > marks[i - 1][1]
        ]
        if not per_epoch:
            self._eta_seconds = None
            return
        self._eta_seconds = round(sum(per_epoch) / len(per_epoch) * remaining, 1)

    def events_since(self, seq: int) -> Dict[str, Any]:
        with self.lock:
            items = [e for e in self.events if e["seq"] > seq]
            return {"latest": self.seq, "events": items}

    # ---- guided training --------------------------------------------------------
    @staticmethod
    def _resolve_remote_profile(params: Dict[str, Any]):
        """Validate remote execution params and return the RemoteProfile."""
        from anylabeling.services.training_center.remote.models import AuthMethod
        from anylabeling.services.training_center.remote.storage import (
            get_profile_store,
        )

        profile_id = params.get("remote_profile_id") or ""
        if not profile_id:
            raise RuntimeError("未选择远程服务器档案")
        profile = get_profile_store().get(profile_id)
        if profile is None:
            raise RuntimeError(f"远程服务器档案不存在: {profile_id}")
        if not profile.host:
            raise RuntimeError("远程服务器档案未配置主机地址")
        if not profile.remote_workspace:
            raise RuntimeError("远程服务器档案未配置远程工作目录(remote_workspace)")
        if not profile.remote_python:
            raise RuntimeError("远程服务器档案未配置远程 Python 路径(remote_python)")
        if profile.auth_method == AuthMethod.PASSWORD and not (
            params.get("remote_password") or ""
        ):
            raise RuntimeError("该档案使用密码认证，启动训练时需填写密码（仅本次会话，不会保存）")
        return profile

    def start_guided(self, params: Dict[str, Any]) -> Dict[str, Any]:
        current = self.jm.get_current_job()
        if current and current.status.is_active():
            raise RuntimeError(f"已有任务在运行: {current.job_id}")

        # Reset progress estimation up front so a *failed* start never
        # leaves stale ETA/marks behind for status() to report.
        with self.lock:
            self._epoch_marks.clear()
            self._eta_seconds = None

        execution_mode = normalize_execution_mode(params.get("execution_mode"))
        remote_profile = None
        if execution_mode == "remote_ssh":
            remote_profile = self._resolve_remote_profile(params)

        project = params["project"]
        name = params["name"]
        output_dir = _resolve_output_dir(project, name)

        # ultralytics classification training only accepts the dataset
        # *directory* (check_cls_dataset rejects a file path). The dataset
        # preparation flow hands us <dataset_dir>/data.yaml, so rewrite it
        # to the parent directory (mirrors the desktop guided widget).
        task = str(params.get("task") or "detect").strip().lower()
        data = params["data"]
        classify_data_fix: Optional[tuple] = None
        if task == "classify":
            data_path = Path(str(data)).expanduser()
            if data_path.is_file() and data_path.suffix.lower() in (".yaml", ".yml"):
                data = str(data_path.parent)
                classify_data_fix = (str(data_path), data)

        train_args: Dict[str, Any] = {
            "data": data,
            "model": params["model"],
            "project": project,
            "name": name,
            "device": params.get("device", "cpu"),
        }
        # advanced params pass through to YOLO.train()
        for key in (
            "epochs",
            "batch",
            "imgsz",
            "patience",
            "lr0",
            "lrf",
            "momentum",
            "weight_decay",
            "warmup_epochs",
            "workers",
            "optimizer",
            "seed",
            "save_period",
            "val",
            "plots",
        ):
            if key in params and params[key] is not None:
                train_args[key] = params[key]

        job_id = f"guided_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        job = TrainingJob(
            job_id=job_id,
            mode=TrainingMode.GUIDED_ULTRALYTICS,
            status=TrainingStatus.IDLE,
            created_at=datetime.datetime.now(),
            started_at=None,
            ended_at=None,
            workspace=project,
            output_directory=output_dir,
            display_name=f"Guided: {name}",
            framework="ultralytics",
            command=[],
            metadata={},
            error_message=None,
            task=params.get("task", "detect"),
            model=params["model"],
            data=data,
            project=project,
            name=name,
        )
        job.execution_mode = execution_mode
        if remote_profile is not None:
            job.remote_profile_id = remote_profile.profile_id
            job.remote_host = remote_profile.host
            job.remote_port = remote_profile.port
            job.remote_username = remote_profile.username
            job.remote_workspace = remote_profile.remote_workspace
            job.remote_python = remote_profile.remote_python
            job._session_password = params.get("remote_password") or ""

        self.adapter = UltralyticsAdapter()
        if execution_mode == "remote_ssh":
            # runner.start() connects + uploads the whole dataset
            # synchronously — far too long for an HTTP request. Reserve
            # synchronously (fast, surfaces slot/adapter errors), then run
            # the start phase on a daemon thread; failures surface as
            # FAILED events/history like on the desktop.
            ok, msg = self.jm.reserve_job(job, self.adapter)
            if not ok:
                raise RuntimeError(msg)
            threading.Thread(
                target=self._start_reserved_remote,
                args=(job.job_id, train_args),
                daemon=True,
                name=f"remote-start-{job.job_id}",
            ).start()
        else:
            ok, msg = self.jm.request_start(job, self.adapter, train_args)
            if not ok:
                raise RuntimeError(msg)

        self.output_dir = output_dir
        self.active_job_id = job_id
        if classify_data_fix is not None:
            self._on_event(
                create_console_output_event(
                    job_id=job_id,
                    timestamp=time.time(),
                    message=(
                        "分类任务的数据集参数已自动调整为目录：\n"
                        f"  原: {classify_data_fix[0]}\n"
                        f"  新: {classify_data_fix[1]}\n"
                        "(ultralytics 分类训练只接受数据集目录，不接受 data.yaml 文件路径)"
                    ),
                    source="training_service",
                )
            )
        self.metric_store.start_run(job_id, str(output_dir))
        return self.status()

    def _start_reserved_remote(self, job_id: str, train_args: Dict[str, Any]):
        """Run the long remote start phase (connect/upload/launch).

        start_reserved_job already transitions the job to FAILED and
        finalizes history on errors; this wrapper additionally guards
        against unexpected escapes (e.g. the runner type check raising)
        so the job never stays stuck in PREPARING forever.
        """
        try:
            self.jm.start_reserved_job(job_id, train_args)
        except Exception as e:  # noqa
            _log.exception("remote start phase crashed for job %s", job_id)
            try:
                # no-op unless the job is still PREPARING (e.g. already
                # STOPPED via stop() is left untouched)
                self.jm.fail_reserved_job(job_id, str(e))
            except Exception:  # noqa
                _log.debug("fail_reserved_job failed for %s", job_id, exc_info=True)

    def stop(self) -> Dict[str, Any]:
        job = self.jm.get_current_job()
        remote_preparing = (
            job is not None
            and job.status == TrainingStatus.PREPARING
            and normalize_execution_mode(getattr(job, "execution_mode", None))
            == "remote_ssh"
        )
        ok = self.jm.request_stop()
        if ok and remote_preparing:
            # request_stop() only flips a PREPARING job to STOPPED — the
            # daemon thread in _start_reserved_remote may still be
            # connecting/uploading inside runner.start(), keeping
            # runner._active_job_id occupied (the next remote job would
            # then fail with "Runner already has an active job"). Cancel
            # the runner so it aborts promptly and releases the slot.
            job_id = job.job_id
            runner = None
            try:
                from anylabeling.services.training_center.runners.factory import (
                    RunnerFactory,
                )

                runner = RunnerFactory.get_instance().get("remote_ssh")
            except Exception:  # noqa
                runner = None
            if runner is not None:
                try:
                    runner.cancel(job_id)
                except Exception:  # noqa - stop must never raise
                    _log.warning(
                        "remote runner cancel failed for %s", job_id, exc_info=True
                    )
            # request_stop() emits no event for PREPARING jobs — surface
            # the terminal transition in the web event feed.
            self._on_event(
                create_stopped_event(
                    job_id=job_id, timestamp=time.time(), source="training_service"
                )
            )
        return {"stopped": ok}

    def status(self) -> Dict[str, Any]:
        job = self.jm.get_current_job()
        active = bool(job and job.status.is_active())
        return {
            "job": job.to_dict() if job else None,
            "running": active,
            "output_dir": str(self.output_dir) if self.output_dir else None,
            # never report a stale ETA once the job left an active state
            "eta_seconds": self._eta_seconds if active else None,
        }

    # ---- metrics ------------------------------------------------------------------
    def metrics(self) -> Dict[str, Any]:
        job = self.jm.get_current_job()
        job_id = job.job_id if job else self.active_job_id
        if not job_id:
            return {"series": []}
        run = self.metric_store.poll_csv()
        if run is None and self.output_dir:
            run = self.metric_store.load_from_output_dir(
                job_id, str(self.output_dir)
            )
        if run is None:
            # Remote jobs: no local CSV during training — use the samples
            # fed from EPOCH_METRICS events (also covers pre-epoch-0 local).
            run = self.metric_store.get_run(job_id)
        if run is None:
            return {"series": []}
        series = run.to_series()
        return {
            "series": [
                {"name": s.name, "group": s.group, "points": s.points}
                for s in series
            ]
        }

    # ---- history -------------------------------------------------------------------
    def history(self, limit: int = 50) -> List[Dict[str, Any]]:
        store = get_history_store()
        records = store.list_jobs(limit=limit)
        result = []
        for r in records:
            d = r.to_dict() if hasattr(r, "to_dict") else dict(r.__dict__)
            result.append(d)
        return result

    # ---- preflight ------------------------------------------------------------------
    @staticmethod
    def _resolve_cached_model(model: str) -> Optional[str]:
        """Mirror resolve_training_model_path's lookup without downloading.

        Returns the cached weights path for a bare ``xxx.pt`` filename, or
        None when the model is not a bare name / not cached yet."""
        if not isinstance(model, str) or not model:
            return None
        model_path = Path(model)
        if (
            model.startswith(("http://", "https://"))
            or model_path.is_absolute()
            or model_path.parent != Path(".")
            or model_path.suffix.lower() != ".pt"
        ):
            return None
        from anylabeling.services.auto_training.ultralytics.trainer import (
            get_training_weights_dir,
        )

        cached = Path(get_training_weights_dir()) / model_path.name
        return str(cached) if cached.exists() else None

    def run_preflight(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Blocking preflight check (run in a worker thread)."""
        mode = normalize_execution_mode(params.get("execution_mode"))
        if mode == "remote_ssh":
            # The local machine's GPU/model/dataset checks are meaningless
            # for a remote run — the server diagnostics are authoritative.
            return {
                "can_start": True,
                "issues": [
                    {
                        "severity": "pass",
                        "title": "远程训练模式",
                        "message": "远程任务以服务器环境诊断结果为准，已跳过本机预检查。",
                        "suggestion": "启动前可在「执行位置」卡片点击「检测服务器」确认远端 GPU 与环境。",
                    }
                ],
            }

        from anylabeling.services.training_center.preflight.worker import (
            PreflightWorker,
        )

        model = params["model"]
        bare_auto_download = self._resolve_cached_model(model) is None and (
            isinstance(model, str)
            and model
            and not model.startswith(("http://", "https://"))
            and not Path(model).is_absolute()
            and Path(model).parent == Path(".")
            and Path(model).suffix.lower() == ".pt"
        )
        ctx = GuidedPreflightContext(
            task_type=params.get("task", "detect"),
            model_path=self._resolve_cached_model(model) or model,
            dataset_yaml=params["data"],
            epochs=params.get("epochs", 100),
            batch=params.get("batch", 16),
            imgsz=params.get("imgsz", 640),
            device=params.get("device", "cpu"),
            output_dir=str(Path(params["project"]) / params["name"]),
        )
        worker = PreflightWorker()
        holder: Dict[str, Any] = {}
        worker.finished.connect(lambda result: holder.update(result=result))
        worker.error.connect(lambda msg: holder.update(error=msg))
        worker.run_guided(ctx, False)

        if "error" in holder:
            raise RuntimeError(holder["error"])
        result = holder.get("result")
        if result is None:
            raise RuntimeError("Preflight returned no result")

        from anylabeling.services.training_center.preflight.models import (
            PreflightSeverity,
        )

        # A bare "xxx.pt" name is auto-downloaded by the trainer on start, so
        # a missing local file is a warning, not a blocking error.
        if bare_auto_download:
            for issue in result.issues:
                if (
                    issue.code == "MODEL_NOT_FOUND"
                    and issue.severity == PreflightSeverity.ERROR
                ):
                    issue.severity = PreflightSeverity.WARNING
                    issue.message = (
                        f"模型 {model} 尚未下载，训练启动时将自动下载。"
                    )
                    issue.suggestion = "首次下载需要联网；也可以提前下载到训练权重目录。"

        # GuidedStartRequest accepts batch=-1 (ultralytics auto-batch), but
        # the shared positive-int check flags it as BATCH_INVALID — downgrade
        # to pass so auto-batch never blocks can_start.
        if params.get("batch") == -1:
            for issue in result.issues:
                if (
                    issue.code == "BATCH_INVALID"
                    and issue.severity == PreflightSeverity.ERROR
                ):
                    issue.severity = PreflightSeverity.PASS
                    issue.title = "batch 自动模式"
                    issue.message = (
                        "batch=-1：训练时由 ultralytics 根据可用显存自动选择"
                        "批次大小（auto-batch）。"
                    )
                    issue.suggestion = None

        return {
            "can_start": result.can_start,
            "issues": [
                {
                    "severity": str(getattr(i.severity, "value", i.severity)),
                    "title": getattr(i, "title", ""),
                    "message": getattr(i, "message", ""),
                    "suggestion": getattr(i, "suggestion", ""),
                }
                for i in result.issues
            ],
        }


_service: Optional[WebTrainingService] = None


def get_training_service() -> WebTrainingService:
    global _service
    if _service is None:
        _service = WebTrainingService()
    return _service
