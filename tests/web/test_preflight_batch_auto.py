"""Preflight with batch=-1 (ultralytics auto-batch) must not be blocked by
the shared positive-int check (BATCH_INVALID). Runs the real PreflightWorker
against a temporary, valid YOLO dataset — no mocks except the service shell.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import yaml


def _make_yolo_dataset(root: Path) -> Path:
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        (root / "images" / split / "a.jpg").write_bytes(b"\xff\xd8\xff\xd9")
        (root / "labels" / split / "a.txt").write_text(
            "0 0.5 0.5 0.5 0.5\n", encoding="utf-8"
        )
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "cat"},
                "nc": 1,
            }
        ),
        encoding="utf-8",
    )
    return data_yaml


def test_preflight_batch_minus_one_auto_batch_not_blocked(tmp_path):
    ts = importlib.import_module("web.backend.app.training_service")
    # run_preflight only touches staticmethods + params — no __init__ needed
    svc = ts.WebTrainingService.__new__(ts.WebTrainingService)

    data_yaml = _make_yolo_dataset(tmp_path / "ds")
    out = svc.run_preflight(
        {
            "task": "detect",
            "model": "yolov8n.pt",  # bare name: auto-download warning, not error
            "data": str(data_yaml),
            "project": str(tmp_path / "runs"),
            "name": "train",
            "device": "cpu",
            "epochs": 100,
            "batch": -1,
            "imgsz": 640,
        }
    )

    # the auto-batch downgrade is a pass with an explanation
    batch_issues = [
        i for i in out["issues"] if "batch" in (i["title"] + i["message"]).lower()
    ]
    assert batch_issues, "expected a batch-related issue entry"
    assert all(i["severity"] == "pass" for i in batch_issues)
    assert any("auto" in i["message"].lower() for i in batch_issues)

    errors = [i for i in out["issues"] if i["severity"] == "error"]
    assert errors == [], f"unexpected preflight errors: {errors}"
    assert out["can_start"] is True


def test_preflight_batch_invalid_positive_check_still_works(tmp_path):
    """A genuinely invalid batch (e.g. -5 rejected by pydantic upstream) is
    not downgraded — only -1 is. Uses batch=-5 directly at service level."""
    ts = importlib.import_module("web.backend.app.training_service")
    svc = ts.WebTrainingService.__new__(ts.WebTrainingService)

    data_yaml = _make_yolo_dataset(tmp_path / "ds")
    out = svc.run_preflight(
        {
            "task": "detect",
            "model": "yolov8n.pt",
            "data": str(data_yaml),
            "project": str(tmp_path / "runs"),
            "name": "train",
            "device": "cpu",
            "epochs": 100,
            "batch": -5,  # invalid: only -1 means auto-batch
            "imgsz": 640,
        }
    )
    batch_errors = [
        i
        for i in out["issues"]
        if i["severity"] == "error" and "batch" in i["message"].lower()
    ]
    assert batch_errors
    assert out["can_start"] is False
