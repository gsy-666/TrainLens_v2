"""Tests for POST /api/training/preflight + /training/guided/start
request validation: numeric field constraints (422) and task×model
matching (400). The training service itself is mocked out — these tests
cover the router-level guardrails only."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _payload(**over) -> dict:
    p = {
        "task": "detect",
        "model": "yolov8n.pt",
        "data": "data.yaml",
        "project": "runs",
    }
    p.update(over)
    return p


@pytest.fixture()
def client(monkeypatch):
    training = importlib.import_module("web.backend.app.routers.training")
    importlib.reload(training)
    app = FastAPI()
    app.include_router(training.router, prefix="/api")
    monkeypatch.setattr(
        training,
        "get_training_service",
        lambda: SimpleNamespace(
            run_preflight=lambda params: {"can_start": True, "issues": []},
            start_guided=lambda params: {
                "job": None,
                "running": True,
                "output_dir": None,
                "eta_seconds": None,
            },
        ),
    )
    return TestClient(app)


# ---- numeric field validation (pydantic → 422) ------------------------------


def test_preflight_epochs_zero_returns_422(client):
    r = client.post("/api/training/preflight", json=_payload(epochs=0))
    assert r.status_code == 422


def test_guided_start_epochs_negative_returns_422(client):
    r = client.post("/api/training/guided/start", json=_payload(epochs=-5))
    assert r.status_code == 422


def test_preflight_batch_zero_returns_422(client):
    r = client.post("/api/training/preflight", json=_payload(batch=0))
    assert r.status_code == 422


def test_preflight_batch_minus_one_is_auto_and_accepted(client):
    r = client.post("/api/training/preflight", json=_payload(batch=-1))
    assert r.status_code == 200


@pytest.mark.parametrize("imgsz", [16, 8192])
def test_preflight_imgsz_out_of_range_returns_422(client, imgsz):
    r = client.post("/api/training/preflight", json=_payload(imgsz=imgsz))
    assert r.status_code == 422


def test_preflight_negative_patience_returns_422(client):
    r = client.post("/api/training/preflight", json=_payload(patience=-1))
    assert r.status_code == 422


def test_preflight_negative_lr0_returns_422(client):
    r = client.post("/api/training/preflight", json=_payload(lr0=-0.01))
    assert r.status_code == 422


# ---- task×model matching (400) -----------------------------------------------


def test_guided_start_task_model_mismatch_returns_400(client):
    r = client.post(
        "/api/training/guided/start",
        json=_payload(task="detect", model="yolov8n-seg.pt"),
    )
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "yolov8n-seg.pt" in detail
    assert "segment" in detail


def test_preflight_task_model_mismatch_returns_400(client):
    r = client.post(
        "/api/training/preflight",
        json=_payload(task="detect", model="yolov8n-seg.pt"),
    )
    assert r.status_code == 400
    assert "segment" in r.json()["detail"]


@pytest.mark.parametrize(
    "task,model",
    [
        ("segment", "yolov8n.pt"),  # bare yolo name => detect weights
        ("detect", "yolo11n-cls.pt"),
        ("classify", "yolov8n-pose.pt"),
        ("obb", "yolov8n-pose.pt"),
    ],
)
def test_preflight_other_mismatches_return_400(client, task, model):
    r = client.post("/api/training/preflight", json=_payload(task=task, model=model))
    assert r.status_code == 400


def test_preflight_matching_suffix_accepted(client):
    r = client.post(
        "/api/training/preflight",
        json=_payload(task="segment", model="yolov8n-seg.pt"),
    )
    assert r.status_code == 200
    assert r.json()["can_start"] is True


def test_preflight_custom_pt_path_undecidable_allowed(client):
    # custom weights without a known suffix cannot be classified → pass
    r = client.post(
        "/api/training/preflight",
        json=_payload(task="segment", model="D:/runs/exp/weights/best.pt"),
    )
    assert r.status_code == 200


# ---- happy path ---------------------------------------------------------------


def test_preflight_bare_model_ok(client):
    r = client.post("/api/training/preflight", json=_payload())
    assert r.status_code == 200
    payload = r.json()
    assert payload["can_start"] is True
    assert isinstance(payload["issues"], list)


def test_guided_start_ok(client):
    r = client.post("/api/training/guided/start", json=_payload())
    assert r.status_code == 200
    assert r.json()["running"] is True


# ---- task/model matching edge cases (m14) ------------------------------------------


def test_preflight_task_case_insensitive_match_accepted(client):
    r = client.post(
        "/api/training/preflight",
        json=_payload(task="Segment", model="yolov8n-seg.pt"),
    )
    assert r.status_code == 200


def test_guided_start_uppercase_task_mismatch_returns_400(client):
    r = client.post(
        "/api/training/guided/start",
        json=_payload(task="Detect", model="yolov8n-seg.pt"),
    )
    assert r.status_code == 400
    assert "segment" in r.json()["detail"]


def test_preflight_task_with_whitespace_mismatch_returns_400(client):
    r = client.post(
        "/api/training/preflight",
        json=_payload(task="  detect ", model="yolov8n-cls.pt"),
    )
    assert r.status_code == 400
    assert "classify" in r.json()["detail"]


def test_preflight_model_url_with_query_mismatch_returns_400(client):
    r = client.post(
        "/api/training/preflight",
        json=_payload(
            task="detect",
            model="https://example.com/weights/yolov8n-seg.pt?download=1",
        ),
    )
    assert r.status_code == 400
    assert "segment" in r.json()["detail"]


def test_preflight_model_url_with_fragment_match_accepted(client):
    r = client.post(
        "/api/training/preflight",
        json=_payload(
            task="segment", model="https://example.com/yolov8n-seg.pt#release"
        ),
    )
    assert r.status_code == 200
