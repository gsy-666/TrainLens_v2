"""Tests for POST /api/playground/predict: one-off inference on an
uploaded image with the currently loaded model, without touching the
session/dataset state.
"""

from __future__ import annotations

import importlib
import os
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64

SHAPES = [
    {
        "label": "cat",
        "score": 0.9,
        "points": [[0, 0], [10, 0], [10, 10], [0, 10]],
        "group_id": None,
        "shape_type": "rectangle",
        "flags": {},
    }
]


class FakeService:
    def __init__(self, loaded=True):
        self.loaded = loaded
        self.calls = []

    def loaded_info(self):
        if not self.loaded:
            return None
        return {"display_name": "Fake YOLO", "type": "yolov8"}

    def predict(self, image_path, text_prompt=None, conf=None, iou=None):
        self.calls.append(
            {"path": image_path, "text_prompt": text_prompt, "conf": conf, "iou": iou}
        )
        return {"shapes": SHAPES, "replace": True, "description": ""}


@pytest.fixture()
def env(monkeypatch):
    playground = importlib.import_module("web.backend.app.routers.playground")
    importlib.reload(playground)

    app = FastAPI()
    app.include_router(playground.router, prefix="/api")
    client = TestClient(app)

    svc = FakeService()
    monkeypatch.setattr(playground, "get_model_service", lambda: svc)
    return SimpleNamespace(client=client, module=playground, svc=svc)


def _post(client, **kwargs):
    args = {
        "files": {"file": ("cat.png", PNG_BYTES, "image/png")},
    }
    args.update(kwargs)
    return client.post("/api/playground/predict", **args)


def test_playground_predict_success(env):
    r = _post(env.client, data={"conf": "0.3", "iou": "0.5"})
    assert r.status_code == 200
    payload = r.json()
    assert payload["shapes"] == SHAPES
    assert payload["model"] == {"display_name": "Fake YOLO", "type": "yolov8"}

    # conf/iou form fields are forwarded to the service
    call = env.svc.calls[0]
    assert call["conf"] == 0.3
    assert call["iou"] == 0.5
    # the temp file passed to predict is cleaned up afterwards
    assert not os.path.exists(call["path"])


def test_playground_predict_without_conf(env):
    r = _post(env.client)
    assert r.status_code == 200
    call = env.svc.calls[0]
    assert call["conf"] is None
    assert call["iou"] is None


def test_playground_no_model_loaded_returns_400(env):
    env.svc.loaded = False
    r = _post(env.client)
    assert r.status_code == 400
    assert "加载模型" in r.json()["detail"]


def test_playground_non_image_returns_400(env):
    r = env.client.post(
        "/api/playground/predict",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert r.status_code == 400
    assert env.svc.calls == []


def test_playground_oversized_image_returns_400(env, monkeypatch):
    monkeypatch.setattr(env.module, "MAX_UPLOAD_BYTES", 8)
    r = _post(env.client)
    assert r.status_code == 400
    assert env.svc.calls == []


def test_playground_does_not_touch_session(env):
    deps = importlib.import_module("web.backend.app.deps")
    deps.session.set_dir(None, [])
    r = _post(env.client)
    assert r.status_code == 200
    assert deps.session.get_dir() is None
    assert deps.session.get_images() == []
