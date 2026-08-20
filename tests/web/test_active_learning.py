"""Tests for the active-learning hard-example scan endpoints:

    POST /api/active_learning/scan
    GET  /api/active_learning/scan/status
    POST /api/active_learning/scan/stop
    GET  /api/active_learning/scores
    POST /api/active_learning/clear
"""

from __future__ import annotations

import importlib
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _rect(label: str, score=None) -> dict:
    shape = {
        "label": label,
        "shape_type": "rectangle",
        "points": [[0, 0], [10, 0], [10, 10], [0, 10]],
        "group_id": None,
        "flags": {},
    }
    if score is not None:
        shape["score"] = score
    return shape


def _write_labelme_json(path: Path, shapes: list) -> None:
    data = {
        "version": "3.0.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": path.with_suffix(".jpg").name,
        "imageData": None,
        "imageHeight": 100,
        "imageWidth": 100,
    }
    path.write_text(json.dumps(data), encoding="utf-8")


class FakeService:
    """predict() returns per-image shapes from `behavior` (or raises)."""

    def __init__(self, behavior, loaded=True, gate: threading.Event = None):
        self.behavior = behavior
        self.loaded = loaded
        self.gate = gate
        self.calls = []

    def loaded_info(self):
        if not self.loaded:
            return None
        return {"display_name": "Fake YOLO", "type": "yolov8"}

    def predict(self, image_path, text_prompt=None, conf=None, iou=None):
        name = Path(image_path).name
        self.calls.append(name)
        if self.gate is not None:
            self.gate.wait(timeout=10)
        outcome = self.behavior[name]
        if isinstance(outcome, Exception):
            raise outcome
        return {"shapes": outcome, "replace": True, "description": ""}


@pytest.fixture()
def env(monkeypatch, tmp_path):
    al = importlib.import_module("web.backend.app.routers.active_learning")
    al = importlib.reload(al)
    deps = importlib.import_module("web.backend.app.deps")

    # a.jpg is labeled, b.jpg has no label file
    (tmp_path / "a.jpg").write_bytes(b"\xff\xd8\xff")  # bytes never read
    (tmp_path / "b.jpg").write_bytes(b"\xff\xd8\xff")
    _write_labelme_json(tmp_path / "a.json", [_rect("cat")])

    deps.session.set_dir(tmp_path, ["a.jpg", "b.jpg"])
    app = FastAPI()
    app.include_router(al.router, prefix="/api")
    client = TestClient(app)

    fake = FakeService(
        {
            "a.jpg": [_rect("cat", 0.9), _rect("dog", 0.4)],
            "b.jpg": [],
        }
    )
    monkeypatch.setattr(al, "get_model_service", lambda: fake)

    yield SimpleNamespace(
        client=client, module=al, svc=fake, tmp_path=tmp_path, deps=deps
    )

    # never leave a scan thread running between tests
    al._STATE.stop_event.set()
    if al._STATE.thread is not None:
        al._STATE.thread.join(timeout=10)
    deps.session.set_dir(None, [])


def _wait_done(client, timeout=10.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = client.get("/api/active_learning/scan/status").json()
        if st["done"] and not st["running"]:
            return st
        time.sleep(0.01)
    raise AssertionError("scan did not finish in time")


# ---- happy path -------------------------------------------------------------


def test_scan_unlabeled_scope_only_scores_unlabeled(env):
    r = env.client.post("/api/active_learning/scan", json={})
    assert r.status_code == 200
    assert r.json() == {"started": True, "total": 1}

    st = _wait_done(env.client)
    assert st["total"] == 1
    assert st["processed"] == 1
    assert st["error"] is None
    assert st["error_count"] == 0

    # only the unlabeled image was scanned; no detections -> 0.0
    assert env.svc.calls == ["b.jpg"]
    scores = env.client.get("/api/active_learning/scores").json()
    assert scores["scores"] == {"b.jpg": {"score": 0.0, "count": 0}}
    assert scores["updated_at"] is not None


def test_scan_all_scope_takes_max_confidence(env):
    r = env.client.post("/api/active_learning/scan", json={"scope": "all"})
    assert r.status_code == 200
    assert r.json() == {"started": True, "total": 2}

    st = _wait_done(env.client)
    assert st["processed"] == 2

    scores = env.client.get("/api/active_learning/scores").json()["scores"]
    assert scores["a.jpg"] == {"score": 0.9, "count": 2}  # max(0.9, 0.4)
    assert scores["b.jpg"] == {"score": 0.0, "count": 0}


def test_shape_without_score_counts_as_fully_confident(env):
    env.svc.behavior["b.jpg"] = [_rect("cat")]  # no "score" key
    env.svc.behavior["a.jpg"] = [_rect("cat", 0.3)]
    r = env.client.post("/api/active_learning/scan", json={"scope": "all"})
    assert r.status_code == 200
    _wait_done(env.client)
    scores = env.client.get("/api/active_learning/scores").json()["scores"]
    assert scores["b.jpg"] == {"score": 1.0, "count": 1}
    assert scores["a.jpg"] == {"score": 0.3, "count": 1}


def test_scan_conf_is_forwarded(env):
    env.svc.gate = None

    class ConfCapture(FakeService):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.conf_seen = None

        def predict(self, image_path, text_prompt=None, conf=None, iou=None):
            self.conf_seen = conf
            return super().predict(image_path, text_prompt, conf, iou)

    env.svc.__class__ = ConfCapture
    r = env.client.post("/api/active_learning/scan", json={"conf": 0.55})
    assert r.status_code == 200
    _wait_done(env.client)
    assert env.svc.conf_seen == 0.55


# ---- per-image failure ------------------------------------------------------


def test_predict_failure_marks_null_and_continues(env):
    env.svc.behavior["b.jpg"] = RuntimeError("boom")
    r = env.client.post("/api/active_learning/scan", json={})
    assert r.status_code == 200
    st = _wait_done(env.client)
    assert st["error_count"] == 1
    assert st["error"] is None
    scores = env.client.get("/api/active_learning/scores").json()["scores"]
    assert scores["b.jpg"] == {"score": None, "count": 0}


def test_model_unloaded_mid_scan_aborts_with_error(env):
    class UnloadingService(FakeService):
        def predict(self, image_path, text_prompt=None, conf=None, iou=None):
            self.loaded = False
            raise RuntimeError("No model loaded")

    env.svc.__class__ = UnloadingService
    r = env.client.post("/api/active_learning/scan", json={"scope": "all"})
    assert r.status_code == 200
    st = _wait_done(env.client)
    assert st["error"] == "Model unloaded during scan"
    assert st["processed"] == 0


# ---- concurrency / stop -------------------------------------------------------


def test_second_scan_while_running_returns_409(env, monkeypatch):
    gate = threading.Event()
    env.svc.gate = gate
    r = env.client.post("/api/active_learning/scan", json={"scope": "all"})
    assert r.status_code == 200
    try:
        r2 = env.client.post("/api/active_learning/scan", json={"scope": "all"})
        assert r2.status_code == 409
    finally:
        gate.set()
    _wait_done(env.client)


def test_stop_finishes_current_image_then_halts(env):
    gate = threading.Event()
    env.svc.gate = gate
    r = env.client.post("/api/active_learning/scan", json={"scope": "all"})
    assert r.status_code == 200

    # let the thread block inside predict on the first image
    deadline = time.time() + 10
    while not env.svc.calls and time.time() < deadline:
        time.sleep(0.01)
    assert env.svc.calls == ["a.jpg"]

    r = env.client.post("/api/active_learning/scan/stop")
    assert r.status_code == 200
    assert r.json() == {"stopping": True}

    gate.set()  # allow the in-flight image to finish
    st = _wait_done(env.client)
    assert st["processed"] == 1  # a.jpg kept, b.jpg never started
    assert st["total"] == 2
    scores = env.client.get("/api/active_learning/scores").json()["scores"]
    assert set(scores) == {"a.jpg"}

    # stop with nothing running is a no-op
    r = env.client.post("/api/active_learning/scan/stop")
    assert r.json() == {"stopping": False}


# ---- clear / validation -------------------------------------------------------


def test_clear_resets_scores(env):
    r = env.client.post("/api/active_learning/scan", json={})
    assert r.status_code == 200
    _wait_done(env.client)

    r = env.client.post("/api/active_learning/clear")
    assert r.status_code == 200
    assert r.json() == {"cleared": True}
    scores = env.client.get("/api/active_learning/scores").json()
    assert scores["scores"] == {}
    assert scores["updated_at"] is None


def test_initial_status_is_idle(env):
    st = env.client.get("/api/active_learning/scan/status").json()
    assert st == {
        "running": False,
        "processed": 0,
        "total": 0,
        "done": False,
        "error": None,
        "error_count": 0,
    }


def test_scan_without_dir_returns_400(env):
    env.deps.session.set_dir(None, [])
    r = env.client.post("/api/active_learning/scan", json={})
    assert r.status_code == 400


def test_scan_without_model_returns_400(env):
    env.svc.loaded = False
    r = env.client.post("/api/active_learning/scan", json={})
    assert r.status_code == 400


def test_scan_unknown_scope_returns_400(env):
    r = env.client.post("/api/active_learning/scan", json={"scope": "labeled"})
    assert r.status_code == 400
