"""Tests for POST /api/training/history/{job_id}/artifacts/register-model.

The training service, model service and user-config persistence are all
mocked out — these tests cover the router-level registration logic:
YAML generation, custom_models bookkeeping and idempotency.
"""

from __future__ import annotations

import copy
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _make_run_dir(tmp_path: Path, *, names, imgsz: int = 640) -> Path:
    output_dir = tmp_path / "runs" / "exp"
    weights = output_dir / "weights"
    weights.mkdir(parents=True)
    (weights / "best.onnx").write_bytes(b"onnx")
    (weights / "best.pt").write_bytes(b"pt")
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({"names": names}), encoding="utf-8")
    (output_dir / "args.yaml").write_text(
        yaml.safe_dump({"data": str(data_yaml), "imgsz": imgsz}),
        encoding="utf-8",
    )
    return output_dir


@pytest.fixture()
def env(monkeypatch, tmp_path):
    training = importlib.import_module("web.backend.app.routers.training")
    importlib.reload(training)

    app = FastAPI()
    app.include_router(training.router, prefix="/api")
    client = TestClient(app)

    output_dir = _make_run_dir(tmp_path, names=["cat", "dog"])
    record = {
        "job_id": "job-1",
        "task": "detect",
        "output_directory": str(output_dir),
        "status": "completed",
    }
    monkeypatch.setattr(
        training,
        "get_training_service",
        lambda: SimpleNamespace(history=lambda limit=50: [record]),
    )

    reloads: list[bool] = []
    manager = SimpleNamespace(
        MAX_NUM_CUSTOM_MODELS=10,
        load_model_configs=lambda: reloads.append(True),
    )
    monkeypatch.setattr(
        training, "get_model_service", lambda: SimpleNamespace(manager=manager)
    )

    store: dict = {}
    monkeypatch.setattr(training, "get_config", lambda: copy.deepcopy(store))

    def _save(cfg):
        store.clear()
        store.update(copy.deepcopy(cfg))

    monkeypatch.setattr(training, "save_config", _save)

    return SimpleNamespace(
        client=client,
        training=training,
        output_dir=output_dir,
        record=record,
        store=store,
        reloads=reloads,
        monkeypatch=monkeypatch,
    )


def _register(env, path: str, **extra):
    return env.client.post(
        "/api/training/history/job-1/artifacts/register-model",
        json={"path": path, **extra},
    )


def _read_generated_yaml(env, payload) -> dict:
    config_file = Path(payload["config_file"])
    assert config_file.is_file()
    return yaml.safe_load(config_file.read_text(encoding="utf-8"))


# ---- happy path ---------------------------------------------------------------


def test_register_onnx_detect_success(env):
    r = _register(env, "weights/best.onnx")
    assert r.status_code == 200
    payload = r.json()
    assert payload["registered"] is True
    assert payload["model_type"] == "yolov8"

    cfg = _read_generated_yaml(env, payload)
    assert cfg["type"] == "yolov8"
    assert cfg["name"].startswith("trainlens_detect_")
    assert cfg["display_name"] == payload["display_name"]
    # bare filename — resolved relative to the YAML's directory at load time
    assert cfg["model_path"] == "best.onnx"
    assert cfg["classes"] == ["cat", "dog"]
    assert cfg["input_width"] == 640
    assert cfg["input_height"] == 640
    assert cfg["score_threshold"] == 0.25
    assert cfg["nms_threshold"] == 0.45
    assert cfg["confidence_threshold"] == 0.25

    # YAML sits next to the onnx so model_path resolves
    assert Path(payload["config_file"]).parent == (
        env.output_dir / "weights"
    ).resolve()

    # custom_models appended (with config_file) and manager reloaded
    custom = env.store["custom_models"]
    assert len(custom) == 1
    assert custom[0]["config_file"] == payload["config_file"]
    assert env.reloads == [True]


def test_register_is_idempotent_and_overwrites_yaml(env):
    r1 = _register(env, "weights/best.onnx", display_name="我的模型")
    assert r1.status_code == 200
    first = r1.json()
    assert _read_generated_yaml(env, first)["display_name"] == "我的模型"

    r2 = _register(env, "weights/best.onnx")
    assert r2.status_code == 200
    second = r2.json()

    assert second["config_file"] == first["config_file"]
    # YAML content overwritten (custom display_name gone)
    assert _read_generated_yaml(env, second)["display_name"] == second["display_name"]
    # config deduplicated by config_file, manager reloaded again
    assert len(env.store["custom_models"]) == 1
    assert env.reloads == [True, True]


def test_register_pt_exports_onnx_first(env, monkeypatch):
    (env.output_dir / "weights" / "best.onnx").unlink()

    def fake_export(job_id, rel_path, fmt, holder):
        assert fmt == "onnx"
        assert rel_path == "weights/best.pt"
        out = env.output_dir / "weights" / "best.onnx"
        out.write_bytes(b"onnx-from-pt")
        holder["output"] = str(out)

    monkeypatch.setattr(env.training, "_run_model_export", fake_export)

    r = _register(env, "weights/best.pt")
    assert r.status_code == 200
    cfg = _read_generated_yaml(env, r.json())
    assert cfg["model_path"] == "best.onnx"


def test_register_segment_task_writes_seg_config(env):
    env.record["task"] = "segment"
    r = _register(env, "weights/best.onnx")
    assert r.status_code == 200
    payload = r.json()
    assert payload["model_type"] == "yolov8_seg"
    cfg = _read_generated_yaml(env, payload)
    assert cfg["type"] == "yolov8_seg"
    assert cfg["classes"] == ["cat", "dog"]
    assert cfg["conf_threshold"] == 0.25
    assert cfg["iou_threshold"] == 0.45
    assert "input_width" not in cfg


# ---- task inference ------------------------------------------------------------


def test_register_unsupported_task_returns_400(env):
    env.record["task"] = "obb"
    r = _register(env, "weights/best.onnx")
    assert r.status_code == 400
    assert "obb" in r.json()["detail"]


def test_infer_segment_from_filename_when_record_lacks_task(env):
    env.record.pop("task")
    seg = env.output_dir / "weights" / "best-seg.onnx"
    seg.write_bytes(b"seg-onnx")
    r = _register(env, "weights/best-seg.onnx")
    assert r.status_code == 200
    assert r.json()["model_type"] == "yolov8_seg"


def test_uninferable_task_returns_400(env):
    env.record.pop("task")
    r = _register(env, "weights/best.onnx")
    assert r.status_code == 400


# ---- classes / args.yaml parsing -------------------------------------------------


def test_names_dict_sorted_by_key(env, tmp_path):
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump({"names": {1: "dog", 0: "cat"}}), encoding="utf-8"
    )
    r = _register(env, "weights/best.onnx")
    assert r.status_code == 200
    assert _read_generated_yaml(env, r.json())["classes"] == ["cat", "dog"]


def test_missing_args_yaml_returns_400(env):
    (env.output_dir / "args.yaml").unlink()
    r = _register(env, "weights/best.onnx")
    assert r.status_code == 400
    assert "args.yaml" in r.json()["detail"]


# ---- path validation --------------------------------------------------------------


def test_missing_artifact_returns_404(env):
    r = _register(env, "weights/missing.onnx")
    assert r.status_code == 404


def test_non_model_suffix_returns_400(env):
    r = _register(env, "args.yaml")
    assert r.status_code == 400


def test_path_traversal_returns_400(env):
    r = _register(env, "../../../outside.onnx")
    assert r.status_code == 400


def test_unknown_job_returns_404(env):
    r = env.client.post(
        "/api/training/history/nope/artifacts/register-model",
        json={"path": "weights/best.onnx"},
    )
    assert r.status_code == 404


# ---- model naming (uuid segment + artifact stem) -------------------------------


def test_register_name_contains_uuid_segment_and_stem(env):
    env.record["job_id"] = "guided_20260814_122455_ab12cd"
    r = env.client.post(
        "/api/training/history/guided_20260814_122455_ab12cd/artifacts/register-model",
        json={"path": "weights/best.onnx"},
    )
    assert r.status_code == 200
    cfg = _read_generated_yaml(env, r.json())
    assert cfg["name"] == "trainlens_detect_ab12cd_best"
    assert "20260814_122455" in cfg["display_name"]
    assert cfg["display_name"].endswith("best")


def test_register_best_and_last_same_job_do_not_overwrite(env):
    (env.output_dir / "weights" / "last.onnx").write_bytes(b"onnx-last")
    r1 = _register(env, "weights/best.onnx")
    r2 = _register(env, "weights/last.onnx")
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json()["config_file"] != r2.json()["config_file"]
    names = {c["name"] for c in env.store["custom_models"]}
    assert names == {"trainlens_detect_job-1_best", "trainlens_detect_job-1_last"}
    assert len(env.store["custom_models"]) == 2


# ---- names dict validation (sparse / non-integer keys) ---------------------------


def test_register_sparse_names_dict_returns_400(env, tmp_path):
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump({"names": {0: "cat", 2: "dog"}}), encoding="utf-8"
    )
    r = _register(env, "weights/best.onnx")
    assert r.status_code == 400
    assert "连续" in r.json()["detail"]


def test_register_non_integer_names_key_returns_400(env, tmp_path):
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump({"names": {"cat": "cat"}}), encoding="utf-8"
    )
    r = _register(env, "weights/best.onnx")
    assert r.status_code == 400
    assert "整数" in r.json()["detail"]


# ---- remote job fallback: record["data"] ------------------------------------------


def test_register_remote_job_falls_back_to_record_data(env, tmp_path):
    # a remote run downloads args.yaml whose data points at a *remote* path
    (env.output_dir / "args.yaml").write_text(
        yaml.safe_dump(
            {"data": "/data/trainlens/jobs/job-1/config/data.yaml", "imgsz": 640}
        ),
        encoding="utf-8",
    )
    # the local data.yaml used at submission time is still on disk
    env.record["data"] = str(tmp_path / "data.yaml")
    env.record["execution_mode"] = "remote_ssh"
    r = _register(env, "weights/best.onnx")
    assert r.status_code == 200
    assert _read_generated_yaml(env, r.json())["classes"] == ["cat", "dog"]


def test_register_missing_args_yaml_uses_record_data(env, tmp_path):
    (env.output_dir / "args.yaml").unlink()
    env.record["data"] = str(tmp_path / "data.yaml")
    r = _register(env, "weights/best.onnx")
    assert r.status_code == 200
    assert _read_generated_yaml(env, r.json())["classes"] == ["cat", "dog"]


def test_register_remote_job_without_record_data_returns_400(env):
    (env.output_dir / "args.yaml").write_text(
        yaml.safe_dump({"data": "/data/trainlens/jobs/job-1/config/data.yaml"}),
        encoding="utf-8",
    )
    r = _register(env, "weights/best.onnx")
    assert r.status_code == 400
    assert "args.yaml" in r.json()["detail"]
