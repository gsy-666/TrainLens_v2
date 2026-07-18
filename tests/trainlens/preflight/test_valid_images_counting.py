"""Tests verifying Valid Images counting and Data Check / Preflight labeling."""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from PyQt6.QtWidgets import QApplication

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from anylabeling.services.auto_training.ultralytics.utils import get_task_valid_images
from anylabeling.services.training_center.preflight.models import (
    PreflightResult, PreflightSeverity, PreflightIssue,
)


# ── Valid Images counting ───────────────────────────────────────────

def _make_label_json(path: str, shapes: list):
    """Write a minimal X-AnyLabeling JSON label file."""
    data = {
        "version": "1.0",
        "shapes": [
            {
                "label": s["label"],
                "shape_type": s.get("shape_type", "rectangle"),
                "points": s.get("points", [[0, 0], [10, 10]]),
            }
            for s in shapes
        ],
        "imagePath": os.path.basename(path).replace(".json", ".jpg"),
    }
    Path(path).write_text(json.dumps(data), encoding="utf-8")


class TestValidImagesCounting:
    """get_task_valid_images must count distinct images, not instances."""

    def test_five_images_twenty_three_boxes(self, tmp_path):
        """5 images, 23 total boxes → count = 5."""
        images = []
        for i in range(5):
            img = tmp_path / f"img{i:04d}.jpg"
            img.write_text("fake")
            images.append(str(img))
            _make_label_json(
                str(tmp_path / f"img{i:04d}.json"),
                [{"label": "obj"}] * (5 if i < 3 else 4),  # 5+5+5+4+4=23
            )
        count = get_task_valid_images(images, "Detect", str(tmp_path))
        assert count == 5, f"Expected 5 images, got {count}"

    def test_twenty_three_images_one_box_each(self, tmp_path):
        """23 images, 1 box each → count = 23."""
        images = []
        for i in range(23):
            img = tmp_path / f"img{i:04d}.jpg"
            img.write_text("fake")
            images.append(str(img))
            _make_label_json(
                str(tmp_path / f"img{i:04d}.json"),
                [{"label": "obj"}],
            )
        count = get_task_valid_images(images, "Detect", str(tmp_path))
        assert count == 23

    def test_empty_label_image_not_counted(self, tmp_path):
        """Images with empty shapes list → not counted."""
        images = []
        for i in range(3):
            img = tmp_path / f"img{i:04d}.jpg"
            img.write_text("fake")
            images.append(str(img))
        # img0 has valid label
        _make_label_json(str(tmp_path / "img0000.json"), [{"label": "obj"}])
        # img1 has empty shapes
        _make_label_json(str(tmp_path / "img0001.json"), [])
        # img2 has no label file at all
        count = get_task_valid_images(images, "Detect", str(tmp_path))
        assert count == 1  # Only img0

    def test_invalid_label_image_not_counted(self, tmp_path):
        """Images with only invalid/no shape_type → not counted."""
        images = []
        for i in range(3):
            img = tmp_path / f"img{i:04d}.jpg"
            img.write_text("fake")
            images.append(str(img))
        # img0: valid rectangle
        _make_label_json(str(tmp_path / "img0000.json"), [{"label": "obj", "shape_type": "rectangle"}])
        # img1: only "point" shape (not in Detect valid shapes)
        Path(tmp_path / "img0001.json").write_text(json.dumps({
            "shapes": [{"label": "x", "shape_type": "point", "points": [[1, 1]]}],
        }), encoding="utf-8")
        # img2: corrupted JSON
        Path(tmp_path / "img0002.json").write_text("not json {{{", encoding="utf-8")
        count = get_task_valid_images(images, "Detect", str(tmp_path))
        assert count == 1  # Only img0

    def test_same_image_multiple_boxes_still_one(self, tmp_path):
        """One image with 10 valid boxes → count = 1."""
        img = tmp_path / "img0000.jpg"
        img.write_text("fake")
        _make_label_json(
            str(tmp_path / "img0000.json"),
            [{"label": f"obj{i}"} for i in range(10)],
        )
        count = get_task_valid_images([str(img)], "Detect", str(tmp_path))
        assert count == 1


# ── Summary labeling ────────────────────────────────────────────────

class TestSummaryLabeling:
    def test_data_check_says_data_check(self):
        r = PreflightResult(mode="data_check")
        r.add(PreflightIssue(code="P1", severity=PreflightSeverity.PASS, title="p", message="x"))
        assert "Data Check" in r.summary()
        assert "Preflight" not in r.summary()

    def test_preflight_says_preflight(self):
        r = PreflightResult(mode="guided")
        r.add(PreflightIssue(code="P1", severity=PreflightSeverity.PASS, title="p", message="x"))
        assert "Preflight" in r.summary()
        assert "Data Check" not in r.summary()

    def test_custom_mode_says_preflight(self):
        r = PreflightResult(mode="custom")
        r.add(PreflightIssue(code="P1", severity=PreflightSeverity.PASS, title="p", message="x"))
        assert "Preflight" in r.summary()


# ── Threshold consistency ───────────────────────────────────────────

class TestThresholdConsistency:
    def test_min_labeled_threshold_used_in_ui_and_gating(self):
        """MIN_LABELED_IMAGES_THRESHOLD must be the single source of truth."""
        from anylabeling.services.auto_training.ultralytics.config import MIN_LABELED_IMAGES_THRESHOLD
        from anylabeling.services.auto_training.ultralytics.validators import validate_task_requirements
        assert MIN_LABELED_IMAGES_THRESHOLD == 20
        # With images but no labels, should mention threshold
        ok, msg = validate_task_requirements("Detect", ["/fake/img.jpg"], None)
        assert not ok
        assert str(MIN_LABELED_IMAGES_THRESHOLD) in msg
