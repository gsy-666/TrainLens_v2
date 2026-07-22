"""Web wrapper around the desktop training_center services (guided mode).

Reuses the Qt-free building blocks: JobManager (singleton, mutex for one
active job), UltralyticsAdapter, HistoryStore, MetricStore, PreflightWorker.
Events from the JobManager are appended to a ring buffer that the API
polls incrementally.
"""

import datetime
import sys
import threading
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

from anylabeling.services.training_center.adapters.ultralytics_adapter import (
    UltralyticsAdapter,
)
from anylabeling.services.training_center.event_protocol import TrainingEventType
from anylabeling.services.training_center.history import get_history_store
from anylabeling.services.training_center.job_manager import get_job_manager
from anylabeling.services.training_center.metrics.store import MetricStore
from anylabeling.services.training_center.models import (
    TrainingJob,
    TrainingMode,
    TrainingStatus,
)
from anylabeling.services.training_center.preflight.models import (
    GuidedPreflightContext,
)


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

        register_web_runner()  # QProcess-based LocalRunner needs a Qt loop
        self.jm = get_job_manager()
        self.events: deque = deque(maxlen=5000)
        self.seq = 0
        self.lock = threading.RLock()
        self.metric_store = MetricStore()
        self.adapter: Optional[UltralyticsAdapter] = None
        self.output_dir: Optional[Path] = None
        self.active_job_id: Optional[str] = None
        self.jm.subscribe_events(self._on_event)

    # ---- events ---------------------------------------------------------------
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
            self.events.append({"seq": self.seq, **payload})

    def events_since(self, seq: int) -> Dict[str, Any]:
        with self.lock:
            items = [e for e in self.events if e["seq"] > seq]
            return {"latest": self.seq, "events": items}

    # ---- guided training --------------------------------------------------------
    def start_guided(self, params: Dict[str, Any]) -> Dict[str, Any]:
        current = self.jm.get_current_job()
        if current and current.status.is_active():
            raise RuntimeError(f"已有任务在运行: {current.job_id}")

        project = params["project"]
        name = params["name"]
        output_dir = _resolve_output_dir(project, name)

        train_args: Dict[str, Any] = {
            "data": params["data"],
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
            data=params["data"],
            project=project,
            name=name,
        )

        self.adapter = UltralyticsAdapter()
        ok, msg = self.jm.request_start(job, self.adapter, train_args)
        if not ok:
            raise RuntimeError(msg)

        self.output_dir = output_dir
        self.active_job_id = job_id
        self.metric_store.start_run(job_id, str(output_dir))
        return self.status()

    def stop(self) -> Dict[str, Any]:
        ok = self.jm.request_stop()
        return {"stopped": ok}

    def status(self) -> Dict[str, Any]:
        job = self.jm.get_current_job()
        return {
            "job": job.to_dict() if job else None,
            "running": bool(job and job.status.is_active()),
            "output_dir": str(self.output_dir) if self.output_dir else None,
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
    def run_preflight(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Blocking preflight check (run in a worker thread)."""
        from anylabeling.services.training_center.preflight.worker import (
            PreflightWorker,
        )

        ctx = GuidedPreflightContext(
            task_type=params.get("task", "detect"),
            model_path=params["model"],
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
