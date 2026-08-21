"""Tests for POST /api/models/register-local (local .onnx registration)."""

from __future__ import annotations

import copy
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

TEMPLATE = {
    "type": "yolov8",
    "name": "yolov8n-r20230520",
    "display_name": "YOLOv8n Ultralytics",
    "provider": "Ultralytics",
    "config_file": ":/yolov8n.yaml",
    "model_path": "https://example.com/yolov8n.onnx",
    "input_width": 640,
    "input_height": 640,
    "score_threshold": 0.5,
    "nms_threshold": 0.45,
    "confidence_threshold": 0.45,
    "classes": ["cat", "dog"],
    "is_custom_model": False,
}


@pytest.fixture()
def env(monkeypatch, tmp_path):
    models = importlib.import_module("web.backend.app.routers.models")
    importlib.reload(models)
    model_service = importlib.import_module("web.backend.app.model_service")

    app = FastAPI()
    app.include_router(models.router, prefix="/api")
    client = TestClient(app)

    reloads: list[bool] = []
    manager = SimpleNamespace(
        MAX_NUM_CUSTOM_MODELS=10,
        model_configs=[copy.deepcopy(TEMPLATE)],
        load_model_configs=lambda: reloads.append(True),
    )
    monkeypatch.setattr(
        models, "get_model_service", lambda: SimpleNamespace(manager=manager)
    )

    store: dict = {}
    monkeypatch.setattr(model_service, "get_config", lambda: copy.deepcopy(store))
    monkeypatch.setattr(models, "get_config", lambda: copy.deepcopy(store))

    def _save(cfg):
        store.clear()
        store.update(copy.deepcopy(cfg))

    monkeypatch.setattr(model_service, "save_config", _save)
    monkeypatch.setattr(models, "save_config", _save)

    # redirect the yaml config dir into tmp_path
    monkeypatch.setenv("XANYLABELING_MODELS_DIR", str(tmp_path / "models"))

    onnx = tmp_path / "my_model.onnx"
    onnx.write_bytes(b"fake-onnx")

    return SimpleNamespace(
        client=client,
        store=store,
        reloads=reloads,
        onnx=onnx,
        tmp_path=tmp_path,
    )


def _register(env, **extra):
    payload = {
        "template_config_file": ":/yolov8n.yaml",
        "local_path": str(env.onnx),
        **extra,
    }
    return env.client.post("/api/models/register-local", json=payload)


def test_register_local_happy_path(env):
    r = _register(env)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["registered"] is True
    assert data["model_type"] == "yolov8"
    assert "本地 my_model.onnx" in data["display_name"]

    config_file = Path(data["config_file"])
    assert config_file.is_file()
    assert config_file.parent.parent == env.tmp_path / "models"
    cfg = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    assert cfg["type"] == "yolov8"
    assert cfg["model_path"] == str(env.onnx.resolve())
    assert cfg["classes"] == ["cat", "dog"]
    assert cfg["input_width"] == 640
    assert cfg["name"].startswith("local_yolov8_my_model_")

    custom = env.store.get("custom_models") or []
    assert len(custom) == 1
    assert custom[0]["config_file"] == str(config_file)
    assert env.reloads == [True]


def test_register_local_idempotent(env):
    r1 = _register(env)
    r2 = _register(env)
    assert r1.json()["config_file"] == r2.json()["config_file"]
    assert len(env.store.get("custom_models") or []) == 1


def test_register_local_custom_display_name(env):
    r = _register(env, display_name="我的模型")
    assert r.status_code == 200
    assert r.json()["display_name"] == "我的模型"


def test_register_local_missing_file(env):
    r = _register(env, local_path=str(env.tmp_path / "nope.onnx"))
    assert r.status_code == 404


def test_register_local_rejects_non_onnx(env, tmp_path):
    pt = tmp_path / "model.pt"
    pt.write_bytes(b"pt")
    r = _register(env, local_path=str(pt))
    assert r.status_code == 400
    assert "onnx" in r.json()["detail"]


def test_register_local_unknown_template(env):
    r = _register(env, template_config_file=":/nonexistent.yaml")
    assert r.status_code == 404


def test_template_url_model_path_not_copied(env):
    """The template's http model_path must never leak into the local config."""
    r = _register(env)
    cfg = yaml.safe_load(Path(r.json()["config_file"]).read_text(encoding="utf-8"))
    assert not cfg["model_path"].startswith(("http://", "https://"))


def test_delete_custom_model(env):
    r = _register(env)
    config_file = r.json()["config_file"]
    rd = env.client.delete(
        "/api/models/custom", params={"config_file": config_file}
    )
    assert rd.status_code == 200
    assert rd.json() == {"deleted": True, "deleted_file": False}
    assert env.store.get("custom_models") == []
    # yaml kept without delete_file=true
    assert Path(config_file).is_file()
    # gone from the catalog
    rd2 = env.client.delete(
        "/api/models/custom", params={"config_file": config_file}
    )
    assert rd2.status_code == 404


def test_delete_custom_model_with_file(env):
    r = _register(env)
    config_file = r.json()["config_file"]
    rd = env.client.delete(
        "/api/models/custom",
        params={"config_file": config_file, "delete_file": True},
    )
    assert rd.status_code == 200
    assert rd.json()["deleted_file"] is True
    assert not Path(config_file).exists()


def test_delete_custom_model_never_deletes_outside_local(env, tmp_path):
    """delete_file must not touch yaml files outside the managed _local dir."""
    outside = tmp_path / "elsewhere" / "custom.yaml"
    outside.parent.mkdir(parents=True)
    outside.write_text("type: yolov8\n", encoding="utf-8")
    env.store["custom_models"] = [
        {"config_file": str(outside), "name": "x", "last_used": 1.0}
    ]
    rd = env.client.delete(
        "/api/models/custom",
        params={"config_file": str(outside), "delete_file": True},
    )
    assert rd.status_code == 200
    assert rd.json()["deleted_file"] is False
    assert outside.is_file()
