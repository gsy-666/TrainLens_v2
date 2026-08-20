"""Tests for POST /api/training/quickstart.

create_yolo_dataset / device detection / the training service are mocked;
these tests cover the router's data-parameter contract:
- Classify training must receive the dataset *directory* (ultralytics
  check_cls_dataset rejects a data.yaml file path);
- every other task receives <dataset_dir>/data.yaml;
- a (None, message) tuple from create_yolo_dataset is a 400, not a crash.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture()
def client_env(monkeypatch, tmp_path):
    quickstart = importlib.import_module("web.backend.app.routers.quickstart")
    importlib.reload(quickstart)
    deps = importlib.import_module("web.backend.app.deps")

    for name in ("a.jpg", "b.jpg"):
        (tmp_path / name).write_bytes(b"\xff\xd8\xff")  # bytes never read
    deps.session.set_dir(tmp_path, ["a.jpg", "b.jpg"])

    app = FastAPI()
    app.include_router(quickstart.router, prefix="/api")
    client = TestClient(app)

    dataset_dir = tmp_path / "yolo_dataset"
    dataset_dir.mkdir()
    monkeypatch.setattr(
        quickstart, "create_yolo_dataset", lambda *a, **k: str(dataset_dir)
    )
    monkeypatch.setattr(quickstart, "_ensure_val_split", lambda d: "")
    monkeypatch.setattr(quickstart, "_read_info", lambda d: "")
    monkeypatch.setattr(
        quickstart, "_detect_device", lambda: {"recommended": "cpu"}
    )

    captured: dict = {}

    def fake_start(params):
        captured.update(params)
        return {
            "job": {"job_id": "j1"},
            "running": True,
            "output_dir": None,
            "eta_seconds": None,
        }

    monkeypatch.setattr(
        quickstart,
        "get_training_service",
        lambda: SimpleNamespace(start_guided=fake_start),
    )

    yield SimpleNamespace(
        client=client,
        captured=captured,
        dataset_dir=dataset_dir,
        quickstart=quickstart,
        monkeypatch=monkeypatch,
    )
    deps.session.set_dir(None, [])


def test_quickstart_classify_passes_dataset_directory(client_env):
    r = client_env.client.post(
        "/api/training/quickstart", json={"task_type": "Classify"}
    )
    assert r.status_code == 200
    assert r.json()["task_type"] == "Classify"
    assert client_env.captured["task"] == "classify"
    # the dataset directory itself, not a data.yaml file path
    assert client_env.captured["data"] == str(client_env.dataset_dir)
    assert client_env.captured["model"] == "yolov8n-cls.pt"


def test_quickstart_detect_passes_data_yaml(client_env):
    r = client_env.client.post(
        "/api/training/quickstart", json={"task_type": "Detect"}
    )
    assert r.status_code == 200
    assert client_env.captured["data"] == str(
        client_env.dataset_dir / "data.yaml"
    )


def test_quickstart_pose_tuple_returns_400(client_env):
    client_env.monkeypatch.setattr(
        client_env.quickstart,
        "create_yolo_dataset",
        lambda *a, **k: (
            None,
            "Pose configuration file is required for pose detection tasks",
        ),
    )
    r = client_env.client.post(
        "/api/training/quickstart", json={"task_type": "Pose"}
    )
    assert r.status_code == 400
    assert "Pose configuration" in r.json()["detail"]


def test_quickstart_dataset_exception_returns_500(client_env):
    client_env.monkeypatch.setattr(
        client_env.quickstart,
        "create_yolo_dataset",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    r = client_env.client.post(
        "/api/training/quickstart", json={"task_type": "Detect"}
    )
    assert r.status_code == 500
    assert "boom" in r.json()["detail"]


def test_quickstart_no_dir_returns_400(tmp_path):
    quickstart = importlib.import_module("web.backend.app.routers.quickstart")
    importlib.reload(quickstart)
    deps = importlib.import_module("web.backend.app.deps")
    deps.session.set_dir(None, [])
    app = FastAPI()
    app.include_router(quickstart.router, prefix="/api")
    r = TestClient(app).post("/api/training/quickstart", json={})
    assert r.status_code == 400
