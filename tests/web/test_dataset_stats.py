"""Tests for GET /api/dataset/stats: real label statistics for the
dataset-inspection panel in the training center."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _rect(label: str) -> dict:
    return {
        "label": label,
        "shape_type": "rectangle",
        "points": [[0, 0], [10, 10]],
        "group_id": None,
        "flags": {},
    }


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


@pytest.fixture()
def client_with_dataset(tmp_path):
    dataset = importlib.import_module("web.backend.app.routers.dataset")
    deps = importlib.import_module("web.backend.app.deps")

    for name in ("a.jpg", "b.jpg"):
        (tmp_path / name).write_bytes(b"\xff\xd8\xff")  # bytes never read
    _write_labelme_json(tmp_path / "a.json", [_rect("cat"), _rect("cat")])
    _write_labelme_json(tmp_path / "b.json", [_rect("dog")])

    deps.session.set_dir(tmp_path, ["a.jpg", "b.jpg"])
    app = FastAPI()
    app.include_router(dataset.router, prefix="/api")
    yield TestClient(app)
    deps.session.set_dir(None, [])


def test_dataset_stats_returns_real_counts(client_with_dataset):
    r = client_with_dataset.get("/api/dataset/stats")
    assert r.status_code == 200
    payload = r.json()
    assert payload["total_images"] == 2
    assert payload["class_counts"] == {"cat": 2, "dog": 1}
    assert payload["per_task_valid"]["Detect"] == 2
    assert payload["recommended_task"] == "Detect"
    assert payload["label_infos"]["cat"]["rectangle"] == 2
    warning_codes = {w["code"] for w in payload["warnings"]}
    assert "few_images" in warning_codes  # only 2 valid images
    assert "few_instances" in warning_codes  # every class has < 5 instances


def test_dataset_stats_no_dir_returns_400(tmp_path):
    dataset = importlib.import_module("web.backend.app.routers.dataset")
    deps = importlib.import_module("web.backend.app.deps")
    deps.session.set_dir(None, [])
    app = FastAPI()
    app.include_router(dataset.router, prefix="/api")
    client = TestClient(app)
    r = client.get("/api/dataset/stats")
    assert r.status_code == 400


def test_dataset_stats_malformed_json_is_skipped(tmp_path):
    """A broken label file must not 500 the endpoint — it is skipped and
    surfaced as a warning instead."""
    dataset = importlib.import_module("web.backend.app.routers.dataset")
    deps = importlib.import_module("web.backend.app.deps")

    for name in ("a.jpg", "b.jpg", "c.jpg"):
        (tmp_path / name).write_bytes(b"\xff\xd8\xff")
    _write_labelme_json(tmp_path / "a.json", [_rect("cat")])
    (tmp_path / "b.json").write_text("{not valid json", encoding="utf-8")
    (tmp_path / "c.json").write_text("[1, 2, 3]", encoding="utf-8")  # not a dict

    deps.session.set_dir(tmp_path, ["a.jpg", "b.jpg", "c.jpg"])
    app = FastAPI()
    app.include_router(dataset.router, prefix="/api")
    try:
        r = TestClient(app).get("/api/dataset/stats")
    finally:
        deps.session.set_dir(None, [])

    assert r.status_code == 200
    payload = r.json()
    assert payload["total_images"] == 3
    assert payload["class_counts"] == {"cat": 1}
    assert payload["per_task_valid"]["Detect"] == 1
    codes = {w["code"] for w in payload["warnings"]}
    assert "malformed_labels" in codes
