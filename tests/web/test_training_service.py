"""Unit tests for WebTrainingService internals that the router tests don't
reach: ETA lifecycle, classify data rewriting, remote stop orchestration
and the reserved-job failure fallback.

The service is built via __new__ (no JobManager singleton side effects);
the job manager is a fake.
"""

from __future__ import annotations

import importlib
import threading
import time
from collections import deque
from types import SimpleNamespace

import pytest

from anylabeling.services.training_center.metrics.store import MetricStore
from anylabeling.services.training_center.models import TrainingStatus


def _build_service(monkeypatch, fake_jm):
    ts = importlib.import_module("web.backend.app.training_service")
    monkeypatch.setattr(ts, "UltralyticsAdapter", lambda: SimpleNamespace())
    svc = ts.WebTrainingService.__new__(ts.WebTrainingService)
    svc.jm = fake_jm
    svc.events = deque(maxlen=100)
    svc.seq = 0
    svc.lock = threading.RLock()
    svc.metric_store = MetricStore()
    svc.adapter = None
    svc.output_dir = None
    svc.active_job_id = None
    svc._epoch_marks = deque(maxlen=6)
    svc._eta_seconds = None
    return svc


def _epoch_event(job_id: str, epoch: int, total: int = 5, ts: float | None = None):
    from anylabeling.services.training_center.event_protocol import (
        TrainingEvent,
        TrainingEventType,
    )

    return TrainingEvent(
        schema_version=1,
        job_id=job_id,
        event_type=TrainingEventType.EPOCH_METRICS,
        timestamp=ts if ts is not None else time.time(),
        payload={"epoch": epoch, "total_epochs": total, "metrics": {}},
        source="test",
    )


def _terminal_event(job_id: str, event_type):
    from anylabeling.services.training_center.event_protocol import TrainingEvent

    return TrainingEvent(
        schema_version=1,
        job_id=job_id,
        event_type=event_type,
        timestamp=time.time(),
        payload={},
        source="test",
    )


def _guided_params(tmp_path, **over):
    p = {
        "task": "detect",
        "model": "yolov8n.pt",
        "data": "data.yaml",
        "project": str(tmp_path / "runs"),
        "name": "train",
        "device": "cpu",
    }
    p.update(over)
    return p


# ---- ETA lifecycle (m9) --------------------------------------------------------


def test_terminal_event_clears_eta(monkeypatch):
    from anylabeling.services.training_center.event_protocol import (
        TrainingEventType,
    )

    svc = _build_service(monkeypatch, SimpleNamespace(get_current_job=lambda: None))
    base = time.time()
    # explicit increasing timestamps: time.time() granularity on Windows
    # (~15ms) can make back-to-back events look simultaneous
    svc._on_event(_epoch_event("j", 1, ts=base))
    svc._on_event(_epoch_event("j", 2, ts=base + 2.0))
    assert svc._eta_seconds is not None  # computed from two epoch marks

    svc._on_event(_terminal_event("j", TrainingEventType.COMPLETED))
    assert svc._eta_seconds is None
    assert len(svc._epoch_marks) == 0
    assert svc.status()["eta_seconds"] is None


def test_status_hides_eta_when_job_not_active(monkeypatch):
    svc = _build_service(monkeypatch, SimpleNamespace(get_current_job=lambda: None))
    svc._eta_seconds = 42.0  # stale value from a previous run
    assert svc.status()["eta_seconds"] is None


def test_failed_start_clears_eta_marks(monkeypatch, tmp_path):
    class _JM:
        def get_current_job(self):
            return None

        def subscribe_events(self, cb):
            pass

        def request_start(self, job, adapter, config):
            return False, "boom"

    svc = _build_service(monkeypatch, _JM())
    svc._eta_seconds = 42.0
    svc._epoch_marks.append((1, time.time()))
    with pytest.raises(RuntimeError, match="boom"):
        svc.start_guided(_guided_params(tmp_path))
    assert svc._eta_seconds is None
    assert len(svc._epoch_marks) == 0


# ---- classify data rewrite (B1 service half) ------------------------------------


def test_start_guided_classify_rewrites_data_yaml_to_directory(monkeypatch, tmp_path):
    ds = tmp_path / "cls_dataset"
    ds.mkdir()
    data_yaml = ds / "data.yaml"
    data_yaml.write_text("names: {0: cat}\n", encoding="utf-8")

    captured: dict = {}

    class _JM:
        def get_current_job(self):
            return None

        def subscribe_events(self, cb):
            pass

        def request_start(self, job, adapter, config):
            captured.update(config)
            captured["job_data"] = job.data
            return True, "ok"

    svc = _build_service(monkeypatch, _JM())
    svc.start_guided(
        _guided_params(
            tmp_path, task="classify", model="yolov8n-cls.pt", data=str(data_yaml)
        )
    )
    assert captured["data"] == str(ds)  # train args use the directory
    assert captured["job_data"] == str(ds)  # history record too
    hints = [
        e
        for e in svc.events_since(0)["events"]
        if e["event_type"] == "console_output" and "分类任务" in str(e["payload"])
    ]
    assert hints


def test_start_guided_detect_keeps_data_yaml(monkeypatch, tmp_path):
    captured: dict = {}

    class _JM:
        def get_current_job(self):
            return None

        def subscribe_events(self, cb):
            pass

        def request_start(self, job, adapter, config):
            captured.update(config)
            return True, "ok"

    svc = _build_service(monkeypatch, _JM())
    svc.start_guided(_guided_params(tmp_path, data=str(tmp_path / "ds" / "data.yaml")))
    assert captured["data"] == str(tmp_path / "ds" / "data.yaml")


# ---- remote PREPARING stop orchestration (M4 service half) -----------------------


def test_stop_remote_preparing_cancels_runner_and_emits_stopped(monkeypatch):
    from anylabeling.services.training_center.runners.factory import RunnerFactory

    job = SimpleNamespace(
        job_id="j1", status=TrainingStatus.PREPARING, execution_mode="remote_ssh"
    )
    svc = _build_service(
        monkeypatch,
        SimpleNamespace(get_current_job=lambda: job, request_stop=lambda: True),
    )
    cancelled = []
    fake_runner = SimpleNamespace(
        cancel=lambda job_id: cancelled.append(job_id) or True
    )
    factory = RunnerFactory.get_instance()
    monkeypatch.setattr(
        factory, "_runners", {**factory._runners, "remote_ssh": fake_runner}
    )

    out = svc.stop()
    assert out == {"stopped": True}
    assert cancelled == ["j1"]
    stopped = [
        e
        for e in svc.events_since(0)["events"]
        if e["event_type"] == "stopped" and e["job_id"] == "j1"
    ]
    assert stopped


def test_stop_local_preparing_does_not_touch_remote_runner(monkeypatch):
    from anylabeling.services.training_center.runners.factory import RunnerFactory

    job = SimpleNamespace(
        job_id="j2", status=TrainingStatus.PREPARING, execution_mode="local"
    )
    svc = _build_service(
        monkeypatch,
        SimpleNamespace(get_current_job=lambda: job, request_stop=lambda: True),
    )
    cancelled = []
    fake_runner = SimpleNamespace(
        cancel=lambda job_id: cancelled.append(job_id) or True
    )
    factory = RunnerFactory.get_instance()
    monkeypatch.setattr(
        factory, "_runners", {**factory._runners, "remote_ssh": fake_runner}
    )

    out = svc.stop()
    assert out == {"stopped": True}
    assert cancelled == []
    assert svc.events_since(0)["events"] == []


# ---- reserved-job crash fallback (m13) --------------------------------------------


def test_start_reserved_remote_crash_fails_reserved_job(monkeypatch):
    failed = []

    class _JM:
        def get_current_job(self):
            return None

        def subscribe_events(self, cb):
            pass

        def start_reserved_job(self, job_id, config):
            raise RuntimeError("runner type mismatch")

        def fail_reserved_job(self, job_id, error):
            failed.append((job_id, error))

    svc = _build_service(monkeypatch, _JM())
    svc._start_reserved_remote("j1", {"data": "x"})  # must not raise
    assert failed == [("j1", "runner type mismatch")]


def test_events_strip_ansi_codes(monkeypatch):
    """Console payloads must reach the web log panel without ANSI escapes."""
    svc = _build_service(monkeypatch, SimpleNamespace(get_current_job=lambda: None))
    from anylabeling.services.training_center.event_protocol import (
        TrainingEventType,
    )

    event = SimpleNamespace(
        event_type=TrainingEventType.CONSOLE_OUTPUT,
        job_id="j1",
        to_dict=lambda: {
            "event_type": "console_output",
            "job_id": "j1",
            "payload": {"message": "\x1b[1m/root/train path\x1b[0m tail"},
        },
    )
    svc._on_event(event)
    items = svc.events_since(0)["events"]
    assert items[-1]["payload"]["message"] == "/root/train path tail"


# ---- remote staged-dataset cleanup -------------------------------------------------


def _completed_event(job_id: str):
    from anylabeling.services.training_center.event_protocol import (
        TrainingEventType,
    )

    return _terminal_event(job_id, TrainingEventType.COMPLETED)


def test_remote_terminal_cleans_staged_dataset(monkeypatch, tmp_path):
    from anylabeling.services.auto_training.ultralytics import config as ul_cfg

    root = tmp_path / "datasets"
    ds = root / "detect" / "auto_dataset_x"
    ds.mkdir(parents=True)
    (ds / "data.yaml").write_text("names: [a]", encoding="utf-8")
    monkeypatch.setattr(ul_cfg, "get_dataset_path", lambda: str(root))

    job = SimpleNamespace(
        job_id="j1", execution_mode="remote_ssh", data=str(ds / "data.yaml")
    )
    svc = _build_service(monkeypatch, SimpleNamespace(get_current_job=lambda: job))
    svc._on_event(_completed_event("j1"))
    assert not ds.exists()
    assert any(
        "已清理" in str(e.get("payload", {}).get("message", "")) for e in svc.events
    )


def test_remote_terminal_never_deletes_outside_root(monkeypatch, tmp_path):
    from anylabeling.services.auto_training.ultralytics import config as ul_cfg

    monkeypatch.setattr(
        ul_cfg, "get_dataset_path", lambda: str(tmp_path / "datasets")
    )
    outside = tmp_path / "user_data" / "real"
    outside.mkdir(parents=True)
    (outside / "data.yaml").write_text("names: [a]", encoding="utf-8")
    job = SimpleNamespace(
        job_id="j1", execution_mode="remote_ssh", data=str(outside / "data.yaml")
    )
    svc = _build_service(monkeypatch, SimpleNamespace(get_current_job=lambda: job))
    svc._on_event(_completed_event("j1"))
    assert outside.exists()


def test_local_terminal_keeps_dataset(monkeypatch, tmp_path):
    from anylabeling.services.auto_training.ultralytics import config as ul_cfg

    root = tmp_path / "datasets"
    ds = root / "detect" / "auto_dataset_x"
    ds.mkdir(parents=True)
    (ds / "data.yaml").write_text("names: [a]", encoding="utf-8")
    monkeypatch.setattr(ul_cfg, "get_dataset_path", lambda: str(root))

    job = SimpleNamespace(
        job_id="j1", execution_mode="local", data=str(ds / "data.yaml")
    )
    svc = _build_service(monkeypatch, SimpleNamespace(get_current_job=lambda: job))
    svc._on_event(_completed_event("j1"))
    assert ds.exists()
