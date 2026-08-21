"""Tests for the model library scan endpoints (/api/models/local-files,
/models/cache, /models/scan-dir)."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _cfg(**kw):
    base = {
        "config_file": ":/x.yaml",
        "name": "x",
        "display_name": "X",
        "type": "yolov8",
        "is_custom_model": False,
    }
    base.update(kw)
    return base


@pytest.fixture()
def env(monkeypatch, tmp_path):
    models = importlib.import_module("web.backend.app.routers.models")
    importlib.reload(models)

    app = FastAPI()
    app.include_router(models.router, prefix="/api")
    client = TestClient(app)

    root = tmp_path / "models"
    monkeypatch.setenv("XANYLABELING_MODELS_DIR", str(root))

    # downloaded built-in model: root/yolov8n-r1/yolov8n.onnx
    cached = root / "yolov8n-r1"
    cached.mkdir(parents=True)
    (cached / "yolov8n.onnx").write_bytes(b"12345")

    # custom model with an existing local file
    custom_onnx = tmp_path / "mine.onnx"
    custom_onnx.write_bytes(b"abc")

    configs = [
        _cfg(
            config_file=":/yolov8n.yaml",
            name="yolov8n-r1",
            display_name="YOLOv8n",
            model_path="https://example.com/yolov8n.onnx",
        ),
        _cfg(
            config_file=":/yolov8s.yaml",
            name="yolov8s-r1",
            display_name="YOLOv8s",
            model_path="https://example.com/yolov8s.onnx",
        ),
        _cfg(
            config_file=str(tmp_path / "custom.yaml"),
            name="_custom_mine",
            display_name="Mine",
            model_path=str(custom_onnx),
            is_custom_model=True,
        ),
        _cfg(
            config_file=str(tmp_path / "gone.yaml"),
            name="_custom_gone",
            display_name="Gone",
            model_path=str(tmp_path / "gone.onnx"),
            is_custom_model=True,
        ),
    ]
    manager = SimpleNamespace(model_configs=configs)
    monkeypatch.setattr(
        models, "get_model_service", lambda: SimpleNamespace(manager=manager)
    )
    return SimpleNamespace(client=client, root=root, tmp_path=tmp_path)


def test_local_files_report(env):
    r = env.client.get("/api/models/local-files")
    assert r.status_code == 200
    data = r.json()
    by_name = {i["name"]: i for i in data["items"]}
    assert by_name["yolov8n-r1"]["downloaded"] is True
    assert by_name["yolov8n-r1"]["size_bytes"] == 5
    assert by_name["yolov8s-r1"]["downloaded"] is False
    assert by_name["_custom_mine"]["downloaded"] is True
    assert by_name["_custom_gone"]["downloaded"] is False
    assert data["total_bytes"] == 5 + 3
    assert data["root"] == str(env.root)


def test_delete_cache_happy_path(env):
    target = env.root / "yolov8n-r1" / "yolov8n.onnx"
    assert target.is_file()
    r = env.client.delete("/api/models/cache", params={"config_file": ":/yolov8n.yaml"})
    assert r.status_code == 200
    assert r.json() == {"deleted": True, "freed_bytes": 5}
    assert not target.exists()
    # empty per-model dir removed as well
    assert not (env.root / "yolov8n-r1").exists()
    # second call: nothing to delete
    r2 = env.client.delete("/api/models/cache", params={"config_file": ":/yolov8n.yaml"})
    assert r2.json() == {"deleted": False, "freed_bytes": 0}


def test_delete_cache_rejects_custom(env):
    r = env.client.delete(
        "/api/models/cache", params={"config_file": str(env.tmp_path / "custom.yaml")}
    )
    assert r.status_code == 400


def test_delete_cache_unknown_model(env):
    r = env.client.delete("/api/models/cache", params={"config_file": ":/nope.yaml"})
    assert r.status_code == 404


def test_delete_cache_path_escape_blocked(env, monkeypatch):
    models = importlib.import_module("web.backend.app.routers.models")
    evil = _cfg(name="..\\..\\evil", model_path="https://example.com/x.onnx")
    manager = SimpleNamespace(model_configs=[evil])
    monkeypatch.setattr(
        models, "get_model_service", lambda: SimpleNamespace(manager=manager)
    )
    r = env.client.delete("/api/models/cache", params={"config_file": ":/x.yaml"})
    assert r.status_code in (400, 404)


def test_scan_dir(env, tmp_path):
    d = tmp_path / "scan"
    d.mkdir()
    (d / "a.onnx").write_bytes(b"1")
    sub = d / "sub"
    sub.mkdir()
    (sub / "b.onnx").write_bytes(b"22")
    (d / "c.txt").write_text("no", encoding="utf-8")

    r = env.client.get("/api/models/scan-dir", params={"path": str(d)})
    assert r.status_code == 200
    files = r.json()["files"]
    assert sorted(f["name"] for f in files) == ["a.onnx", "b.onnx"]
    assert all(f["size_bytes"] > 0 for f in files)


def test_scan_dir_not_a_dir(env):
    r = env.client.get("/api/models/scan-dir", params={"path": "D:/no/such/dir"})
    assert r.status_code == 400
