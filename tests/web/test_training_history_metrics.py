"""Tests for GET /api/training/history/{job_id}/metrics (multi-run compare)."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

RESULTS_CSV = (
    "epoch,time,train/box_loss,val/box_loss,metrics/mAP50(B),lr/pg0\n"
    "1,0.5,1.5,1.4,0.30,0.001\n"
    "2,1.0,1.2,1.1,0.50,0.0009\n"
    "3,1.5,0.9,0.8,0.62,0.0008\n"
)


def _build_client(monkeypatch, tmp_path: Path, *, with_csv: bool = True) -> TestClient:
    training = importlib.import_module("web.backend.app.routers.training")
    importlib.reload(training)

    app = FastAPI()
    app.include_router(training.router, prefix="/api")
    client = TestClient(app)

    output_dir = tmp_path / "runs" / "exp"
    output_dir.mkdir(parents=True)
    if with_csv:
        (output_dir / "results.csv").write_text(RESULTS_CSV, encoding="utf-8")
        # weights live in a subdirectory; must not confuse the parser
        weights = output_dir / "weights"
        weights.mkdir()
        (weights / "best.pt").write_bytes(b"best")

    jobs = [
        {
            "job_id": "job-1",
            "output_directory": str(output_dir),
            "output_dir": str(output_dir),
            "workspace": str(output_dir.parent),
            "status": "completed",
            "started_at": "2025-04-01T00:00:00Z",
        }
    ]
    monkeypatch.setattr(
        training,
        "get_training_service",
        lambda: SimpleNamespace(history=lambda limit=50: jobs),
    )
    return client


@pytest.fixture()
def client_with_csv(monkeypatch, tmp_path):
    return _build_client(monkeypatch, tmp_path, with_csv=True)


@pytest.fixture()
def client_no_csv(monkeypatch, tmp_path):
    return _build_client(monkeypatch, tmp_path, with_csv=False)


def test_metrics_unknown_job_returns_404(client_with_csv):
    r = client_with_csv.get("/api/training/history/missing/metrics")
    assert r.status_code == 404


def test_metrics_missing_output_dir_returns_404(monkeypatch, tmp_path):
    training = importlib.import_module("web.backend.app.routers.training")
    importlib.reload(training)
    app = FastAPI()
    app.include_router(training.router, prefix="/api")
    client = TestClient(app)

    missing = tmp_path / "missing"
    jobs = [{"job_id": "job-1", "output_directory": str(missing), "status": "completed"}]
    monkeypatch.setattr(
        training,
        "get_training_service",
        lambda: SimpleNamespace(history=lambda limit=50: jobs),
    )
    r = client.get("/api/training/history/job-1/metrics")
    assert r.status_code == 404


def test_metrics_no_results_csv_returns_empty_series(client_no_csv):
    r = client_no_csv.get("/api/training/history/job-1/metrics")
    assert r.status_code == 200
    assert r.json() == {"series": []}


def test_metrics_parses_results_csv_into_grouped_series(client_with_csv):
    r = client_with_csv.get("/api/training/history/job-1/metrics")
    assert r.status_code == 200
    payload = r.json()
    series = {s["name"]: s for s in payload["series"]}

    # shape matches /training/metrics: name + group + points
    assert set(series) == {
        "train/box_loss",
        "val/box_loss",
        "metrics/mAP50(B)",
        "lr/pg0",
    }
    assert series["train/box_loss"]["group"] == "loss"
    assert series["val/box_loss"]["group"] == "loss"
    assert series["metrics/mAP50(B)"]["group"] == "quality"
    assert series["lr/pg0"]["group"] == "learning_rate"

    # epoch/time columns are skipped, points are [epoch, value] pairs
    assert series["train/box_loss"]["points"] == [[1, 1.5], [2, 1.2], [3, 0.9]]
    assert series["metrics/mAP50(B)"]["points"] == [[1, 0.30], [2, 0.50], [3, 0.62]]
    assert series["lr/pg0"]["points"] == [[1, 0.001], [2, 0.0009], [3, 0.0008]]
