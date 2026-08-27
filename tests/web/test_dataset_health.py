"""Tests for the dataset health check endpoints:

    POST /api/dataset/health/scan
    GET  /api/dataset/health/scan/status
    POST /api/dataset/health/scan/stop
    GET  /api/dataset/health/report
"""

from __future__ import annotations

import importlib
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

SIZE = 200  # all synthetic images are SIZE x SIZE


def _noise(seed: int, mean: float = 127.0, sigma: float = 40.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = rng.normal(mean, sigma, (SIZE, SIZE, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def _write_image(path: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    path.write_bytes(buf.tobytes())


def _rect(points) -> dict:
    return {
        "label": "cat",
        "shape_type": "rectangle",
        "points": points,
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
        "imageHeight": SIZE,
        "imageWidth": SIZE,
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def _build_dataset(tmp_path: Path) -> list:
    """Create the synthetic dataset; returns the image name list."""
    dup = _noise(1)
    _write_image(tmp_path / "dup_a.jpg", dup)
    _write_image(tmp_path / "dup_b.jpg", dup)  # identical -> duplicate group
    _write_image(tmp_path / "blur.jpg", cv2.GaussianBlur(_noise(2), (25, 25), 0))
    _write_image(tmp_path / "dark.jpg", _noise(3, mean=30.0, sigma=3.0))
    _write_image(tmp_path / "bright.jpg", _noise(4, mean=230.0, sigma=3.0))
    _write_image(tmp_path / "normal.jpg", _noise(5))

    # label anomalies
    _write_image(tmp_path / "oob.jpg", _noise(6))
    _write_labelme_json(
        tmp_path / "oob.json",
        [_rect([[10, 10], [250, 10], [250, 50], [10, 50]])],  # right edge +50px
    )
    _write_image(tmp_path / "big.jpg", _noise(7))
    _write_labelme_json(
        tmp_path / "big.json",
        [_rect([[0, 0], [SIZE, 0], [SIZE, SIZE], [0, SIZE]])],  # 100% of image
    )
    _write_image(tmp_path / "degen.jpg", _noise(8))
    _write_labelme_json(
        tmp_path / "degen.json",
        [
            {
                "label": "poly",
                "shape_type": "polygon",
                "points": [[1, 1], [2, 2]],
                "group_id": None,
                "flags": {},
            }
        ],
    )
    return [
        "dup_a.jpg",
        "dup_b.jpg",
        "blur.jpg",
        "dark.jpg",
        "bright.jpg",
        "normal.jpg",
        "oob.jpg",
        "big.jpg",
        "degen.jpg",
    ]


@pytest.fixture()
def env(monkeypatch, tmp_path):
    health = importlib.import_module("web.backend.app.routers.health")
    health = importlib.reload(health)
    deps = importlib.import_module("web.backend.app.deps")

    images = _build_dataset(tmp_path)
    deps.session.set_dir(tmp_path, images)
    app = FastAPI()
    app.include_router(health.router, prefix="/api")
    client = TestClient(app)

    yield SimpleNamespace(
        client=client,
        module=health,
        tmp_path=tmp_path,
        deps=deps,
        images=images,
    )

    # never leave a scan thread running between tests
    health._STATE.stop_event.set()
    if health._STATE.thread is not None:
        health._STATE.thread.join(timeout=10)
    deps.session.set_dir(None, [])


def _wait_done(client, timeout=30.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = client.get("/api/dataset/health/scan/status").json()
        if st["done"] and not st["running"]:
            return st
        time.sleep(0.01)
    raise AssertionError("scan did not finish in time")


# ---- happy path -------------------------------------------------------------


def test_full_scan_report_hits_all_groups(env):
    r = env.client.post("/api/dataset/health/scan", json={})
    assert r.status_code == 200
    assert r.json() == {"started": True, "total": len(env.images)}

    st = _wait_done(env.client)
    assert st["processed"] == len(env.images)
    assert st["error"] is None
    assert st["error_count"] == 0

    report = env.client.get("/api/dataset/health/report").json()
    assert report["updated_at"] is not None

    summary = report["summary"]
    assert summary["total_images"] == len(env.images)

    # duplicates: the two identical images form one group
    assert ["dup_a.jpg", "dup_b.jpg"] in report["duplicate_groups"]
    assert summary["duplicate_groups"] == len(report["duplicate_groups"])
    assert summary["duplicate_images"] == sum(
        len(g) for g in report["duplicate_groups"]
    )

    # blurry / dark / bright
    blurry_names = {e["name"] for e in report["blurry"]}
    assert "blur.jpg" in blurry_names
    assert "normal.jpg" not in blurry_names
    blur_entry = next(e for e in report["blurry"] if e["name"] == "blur.jpg")
    assert blur_entry["score"] < 100
    # report is sorted ascending by score
    scores = [e["score"] for e in report["blurry"]]
    assert scores == sorted(scores)

    dark_names = {e["name"] for e in report["dark"]}
    assert "dark.jpg" in dark_names
    assert "normal.jpg" not in dark_names
    dark_entry = next(e for e in report["dark"] if e["name"] == "dark.jpg")
    assert dark_entry["brightness"] < 60

    bright_names = {e["name"] for e in report["bright"]}
    assert "bright.jpg" in bright_names
    bright_entry = next(e for e in report["bright"] if e["name"] == "bright.jpg")
    assert bright_entry["brightness"] > 195

    assert summary["blurry"] == len(report["blurry"])
    assert summary["dark"] == len(report["dark"])
    assert summary["bright"] == len(report["bright"])

    # shape issues
    issues_by_name = {e["name"]: e["issues"] for e in report["shape_issues"]}
    assert summary["shape_issue_images"] == len(report["shape_issues"])

    oob_kinds = {i["issue"] for i in issues_by_name["oob.jpg"]}
    assert "out_of_bounds" in oob_kinds
    oob_issue = next(
        i for i in issues_by_name["oob.jpg"] if i["issue"] == "out_of_bounds"
    )
    assert oob_issue["label"] == "cat"
    assert oob_issue["detail"]  # non-empty Chinese detail

    big_kinds = {i["issue"] for i in issues_by_name["big.jpg"]}
    assert "oversized" in big_kinds

    degen_kinds = {i["issue"] for i in issues_by_name["degen.jpg"]}
    assert "degenerate" in degen_kinds

    # a clean image with no label file produces no issues
    assert "normal.jpg" not in issues_by_name


def test_report_empty_before_any_scan(env):
    assert env.client.get("/api/dataset/health/report").json() == {
        "updated_at": None
    }


def test_initial_status_is_idle(env):
    st = env.client.get("/api/dataset/health/scan/status").json()
    assert st == {
        "running": False,
        "processed": 0,
        "total": 0,
        "done": False,
        "error": None,
        "error_count": 0,
    }


# ---- per-image failure --------------------------------------------------------


def test_undecodable_image_increments_error_count(env):
    (env.tmp_path / "bad.jpg").write_bytes(b"not an image")
    env.deps.session.set_dir(env.tmp_path, env.images + ["bad.jpg"])

    r = env.client.post("/api/dataset/health/scan", json={})
    assert r.status_code == 200
    st = _wait_done(env.client)
    assert st["error_count"] == 1
    assert st["processed"] == len(env.images) + 1
    assert st["error"] is None

    report = env.client.get("/api/dataset/health/report").json()
    assert report["summary"]["total_images"] == len(env.images)


# ---- concurrency / stop -------------------------------------------------------


def _gate_analyze(env, monkeypatch, gate: threading.Event):
    """Block inside _analyze_image until `gate` is set."""
    real = env.module._analyze_image
    calls = []

    def gated(image_dir, name):
        calls.append(name)
        gate.wait(timeout=10)
        return real(image_dir, name)

    monkeypatch.setattr(env.module, "_analyze_image", gated)
    return calls


def test_second_scan_while_running_returns_409(env, monkeypatch):
    gate = threading.Event()
    _gate_analyze(env, monkeypatch, gate)
    r = env.client.post("/api/dataset/health/scan", json={})
    assert r.status_code == 200
    try:
        r2 = env.client.post("/api/dataset/health/scan", json={})
        assert r2.status_code == 409
    finally:
        gate.set()
    _wait_done(env.client)


def test_stop_halts_scan(env, monkeypatch):
    gate = threading.Event()
    calls = _gate_analyze(env, monkeypatch, gate)
    r = env.client.post("/api/dataset/health/scan", json={})
    assert r.status_code == 200

    # wait until the thread is blocked inside the first image
    deadline = time.time() + 10
    while not calls and time.time() < deadline:
        time.sleep(0.01)
    assert calls

    r = env.client.post("/api/dataset/health/scan/stop")
    assert r.status_code == 200
    assert r.json() == {"stopping": True}

    gate.set()  # let the in-flight image finish
    st = _wait_done(env.client)
    assert st["processed"] == 1
    assert st["total"] == len(env.images)

    # stop with nothing running is a no-op
    r = env.client.post("/api/dataset/health/scan/stop")
    assert r.json() == {"stopping": False}


# ---- validation -----------------------------------------------------------------


def test_scan_without_dir_returns_400(env):
    env.deps.session.set_dir(None, [])
    r = env.client.post("/api/dataset/health/scan", json={})
    assert r.status_code == 400
