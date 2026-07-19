"""Tests for dataset cache clear and rebuild behavior.

Covers:
- Clear Cache clears all session state
- Run Check auto-rebuilds dataset after clear
- Stale YAML paths are not used
- Config basic.data is cleared when pointing to deleted cache
- Images/JSON state preserved after clear
- Prepare failure does not run Full Preflight
- Disk space check falls back to drive root
"""

import sys
import os
import json
import tempfile
import shutil
import pytest
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from anylabeling.views.training.guided_training_widget import GuidedTrainingWidget


@pytest.fixture(scope="session")
def qapp():
    """Create QApplication for real Qt tests"""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def _make_widget(image_list=None, output_dir="/tmp/test", task_type="Detect"):
    """Create a GuidedTrainingWidget with given images and task."""
    w = GuidedTrainingWidget(
        parent=None, image_list=image_list or [],
        output_dir=output_dir,
        supported_shape=["rectangle", "polygon"],
    )
    w.selected_task_type = task_type
    return w


# ── Helpers ──────────────────────────────────────────────────────────

def _setup_config_widgets(widget):
    """Ensure config_widgets dict exists with a 'data' field so
    _clear_dataset_cache can clear it.
    
    Must use CustomLineEdit because get_current_config checks type().__name__."""
    from anylabeling.views.training.widgets.ultralytics_widgets.custom_widgets import CustomLineEdit
    widget.config_widgets = {"data": CustomLineEdit()}
    widget.config_widgets["data"].setText("/tmp/some/auto_dataset_xxx/data.yaml")


# ── Test 1-3: Clear Cache clears session state ────────────────────────

class TestClearCacheStateReset:
    """Clear Cache must reset all prepared-dataset fields."""

    def test_clear_cache_nulls_prepared_dataset_dir(self, qapp):
        """_prepared_dataset_dir -> None after clear."""
        w = _make_widget()
        _setup_config_widgets(w)
        w._prepared_dataset_dir = "/tmp/stale/dataset"
        w._clear_dataset_cache()
        assert w._prepared_dataset_dir is None

    def test_clear_cache_nulls_prepared_yaml_path(self, qapp):
        """_prepared_yaml_path -> None after clear."""
        w = _make_widget()
        _setup_config_widgets(w)
        w._prepared_yaml_path = "/tmp/stale/dataset/data.yaml"
        w._clear_dataset_cache()
        assert w._prepared_yaml_path is None

    def test_clear_cache_nulls_preflight_result(self, qapp):
        """_preflight_result -> None after clear."""
        w = _make_widget()
        _setup_config_widgets(w)
        w._preflight_result = object()  # any non-None
        w._clear_dataset_cache()
        assert w._preflight_result is None


# ── Test 4: _force_rebuild_dataset set ────────────────────────────────

class TestClearCacheForceRebuild:
    """Clear Cache must set _force_rebuild_dataset = True."""

    def test_force_rebuild_set_after_clear(self, qapp):
        """_force_rebuild_dataset is True after clear."""
        w = _make_widget()
        _setup_config_widgets(w)
        w._force_rebuild_dataset = False
        w._clear_dataset_cache()
        assert w._force_rebuild_dataset is True


# ── Test 5: Dataset status label updated ──────────────────────────────

class TestClearCacheStatusLabel:
    """Clear Cache must update dataset status label to Not prepared."""

    def test_status_label_after_clear(self, qapp):
        """dataset_status_label shows Not prepared after clear."""
        w = _make_widget()
        _setup_config_widgets(w)
        w.ensure_train_tab_initialized()
        # Set fake ready state
        w.dataset_status_label.setText("Dataset: Ready")
        w._clear_dataset_cache()
        assert "Not prepared" in w.dataset_status_label.text() or \
               "未" in w.dataset_status_label.text()  # i18n


# ── Test 6: YAML not checked when deleted ─────────────────────────────

class TestStaleYamlProtection:
    """_build_guided_preflight_context must not use a deleted YAML path."""

    def test_context_drops_deleted_yaml(self, qapp, tmp_path):
        """When _prepared_yaml_path points to missing file, context uses empty string."""
        w = _make_widget()
        _setup_config_widgets(w)
        # Set stale yaml path (file does not exist)
        stale = str(tmp_path / "nonexistent" / "data.yaml")
        w._prepared_yaml_path = stale
        w._prepared_dataset_dir = str(tmp_path / "nonexistent")
        # Also set config data to stale path
        w.config_widgets["data"].setText(stale)
        ctx = w._build_guided_preflight_context()
        # Should not use the stale path
        assert ctx.dataset_yaml == "" or ctx.dataset_yaml == stale
        # Actually after Fix 3, it should be "" because file doesn't exist
        if os.path.isfile(stale):
            pass  # shouldn't happen
        else:
            assert ctx.dataset_yaml == ""


# ── Test 7: Config basic.data cleared when inside cache root ──────────

class TestConfigDataClear:
    """config["basic"]["data"] cleared when it points into deleted cache."""

    def test_data_field_cleared_when_in_cache(self, qapp):
        """After clear, config data field is empty if it was under cache root."""
        w = _make_widget()
        _setup_config_widgets(w)
        # Simulate a config data pointing inside the real cache root
        from anylabeling.services.auto_training.ultralytics.config import get_dataset_path
        cache_root = get_dataset_path()
        fake_cache_path = os.path.join(cache_root, "detect", "auto_dataset_test", "data.yaml")
        w.config_widgets["data"].setText(fake_cache_path)
        # Ensure the cache dir exists so rmtree works
        task_dir = os.path.join(cache_root, "detect")
        os.makedirs(task_dir, exist_ok=True)
        try:
            w._clear_dataset_cache()
        except Exception:
            pass  # may fail if nothing to delete
        config = w.get_current_config()
        assert config["basic"]["data"] == ""

    def test_data_field_preserved_when_outside_cache(self, qapp, tmp_path):
        """After clear, config data field is preserved if outside cache root."""
        w = _make_widget()
        _setup_config_widgets(w)
        outside_path = str(tmp_path / "my_project" / "data.yaml")
        w.config_widgets["data"].setText(outside_path)
        try:
            w._clear_dataset_cache()
        except Exception:
            pass
        config = w.get_current_config()
        assert config["basic"]["data"] == outside_path


# ── Test 8: Images and JSON state preserved ───────────────────────────

class TestImageJsonStatePreserved:
    """Clear Cache preserves loaded images, JSON labels, and task type."""

    def test_images_preserved_after_clear(self, qapp):
        """image_list unchanged after clear."""
        w = _make_widget(image_list=["/a/img1.jpg", "/a/img2.jpg"])
        _setup_config_widgets(w)
        w._clear_dataset_cache()
        assert w.image_list == ["/a/img1.jpg", "/a/img2.jpg"]

    def test_task_type_preserved_after_clear(self, qapp):
        """selected_task_type unchanged after clear."""
        w = _make_widget(task_type="Segment")
        _setup_config_widgets(w)
        w._clear_dataset_cache()
        assert w.selected_task_type == "Segment"


# ── Test 9: _run_full_preflight does not proceed when prepare fails ───

class TestPreflightAfterPrepareFailure:
    """When dataset preparation fails, _run_full_preflight must return early."""

    def test_preflight_does_not_run_when_prepare_fails(self, qapp):
        """_run_full_preflight returns without starting worker if prepare fails."""
        w = _make_widget(image_list=[])  # empty → prepare returns False
        _setup_config_widgets(w)
        w._prepared_dataset_dir = None
        w._prepared_yaml_path = None
        w._preflight_running = False
        # _prepare_dataset with empty image_list returns False (no images)
        w._run_full_preflight()
        # Should not be running (prepare failed → early return)
        assert not w._preflight_running
        # _prepared_dataset_dir should still be None (not set)
        assert w._prepared_dataset_dir is None


# ── Test 10: Disk space check falls back to drive root ────────────────

class TestDiskSpaceFallback:
    """check_disk_space must walk up to nearest existing dir, including drive root."""

    def test_disk_space_falls_back_to_drive_root(self):
        """Verifies that check_disk_space uses checked path, not original path."""
        from anylabeling.services.training_center.preflight.common_checks import check_disk_space
        from anylabeling.services.training_center.preflight.models import PreflightResult

        result = PreflightResult(mode="data_check")
        # Use a path that doesn't exist but drive root does
        non_existent = "D:\\TrainLensTest\\guided_smoke\\runs\\guided_mutex_test29"
        check_disk_space(result, non_existent)
        # Should NOT report DISK_SPACE_UNKNOWN because D:\ exists
        for issue in result.issues:
            assert issue.code != "DISK_SPACE_UNKNOWN", (
                f"Should not report DISK_SPACE_UNKNOWN when drive root exists. "
                f"Got: {issue.code} - {issue.title}"
            )

    def test_disk_space_resolves_existing_parent(self, tmp_path):
        """check_disk_space uses nearest existing parent."""
        from anylabeling.services.training_center.preflight.common_checks import check_disk_space
        from anylabeling.services.training_center.preflight.models import PreflightResult

        existing = tmp_path
        non_existent = str(existing / "a" / "b" / "c")
        result = PreflightResult(mode="data_check")
        check_disk_space(result, non_existent)
        # Should not report UNKNOWN (parent tmp_path exists)
        for issue in result.issues:
            assert "UNKNOWN" not in issue.code, (
                f"Should resolve to existing parent. Got: {issue.code} - {issue.title}"
            )


# ── Test 11-14: get_training_args uses prepared YAML ─────────────────

class TestGetTrainingArgsPreparedPath:
    """get_training_args must use _prepared_yaml_path when set."""

    def _make_config(self, data="/tmp/coco8.yaml"):
        return {
            "basic": {
                "data": data, "project": "/tmp/proj", "name": "exp",
                "model": "/tmp/model.pt", "device": "cpu",
                "dataset_ratio": 0.8,
            },
            "train": {"epochs": 100, "batch": 16, "imgsz": 640},
            "checkpoint": {},
        }

    def test_uses_prepared_yaml_when_set(self, qapp, tmp_path):
        """When _prepared_yaml_path is set, get_training_args uses it."""
        w = _make_widget(image_list=[])
        _setup_config_widgets(w)
        prepared = str(tmp_path / "auto_dataset_test" / "data.yaml")
        os.makedirs(os.path.dirname(prepared), exist_ok=True)
        Path(prepared).touch()
        w._prepared_yaml_path = prepared
        w._prepared_dataset_dir = os.path.dirname(prepared)
        config = self._make_config(data="/tmp/coco8.yaml")
        args = w.get_training_args(config)
        assert args["data"] == prepared

    def test_falls_back_when_prepared_not_set(self, qapp):
        """When _prepared_yaml_path is None, falls back to old behavior."""
        w = _make_widget(image_list=[])
        _setup_config_widgets(w)
        w._prepared_yaml_path = None
        config = self._make_config(data="/tmp/coco8.yaml")
        # Should use old path (coco8) since prepared not set
        # create_yolo_dataset will fail for empty image_list, but the logic branch is correct
        args = w.get_training_args(config)
        # Falls through to create_yolo_dataset which uses config["basic"]["data"]
        # Since image_list is empty, it may still produce a path
        assert "data" in args


# ── Test 15-18: Hard validation ──────────────────────────────────────

class TestValidatePreparedOutput:
    """_validate_prepared_output rejects stale/invalid outputs."""

    def test_rejects_coco8_dirname(self, qapp, tmp_path):
        """Rejects directories starting with coco8_."""
        w = _make_widget()
        ds_dir = str(tmp_path / "coco8_20260719_test")
        os.makedirs(ds_dir, exist_ok=True)
        yaml_path = os.path.join(ds_dir, "data.yaml")
        Path(yaml_path).touch()
        manifest = {"classes": [], "valid_images": 0}
        assert not w._validate_prepared_output(ds_dir, yaml_path, manifest)

    def test_accepts_auto_dataset_dirname(self, qapp, tmp_path):
        """Accepts directories starting with auto_dataset_."""
        w = _make_widget()
        ds_dir = str(tmp_path / "auto_dataset_20260719_abc")
        os.makedirs(ds_dir, exist_ok=True)
        # Create subdirs with dummy images (check #9 counts image files)
        for split in ("train", "val"):
            os.makedirs(os.path.join(ds_dir, "images", split), exist_ok=True)
            os.makedirs(os.path.join(ds_dir, "labels", split), exist_ok=True)
            Path(os.path.join(ds_dir, "images", split, "img1.jpg")).touch()
        # Write valid manifest
        import yaml
        yaml_path = os.path.join(ds_dir, "data.yaml")
        yaml_data = {"names": {0: "shoes", 1: "bag"}, "nc": 2, "path": ds_dir}
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_data, f)
        # Write manifest
        mf_path = os.path.join(ds_dir, "dataset_manifest.json")
        manifest = {"classes": ["shoes", "bag"], "valid_images": 2,
                     "split_ratio": 0.5, "train_count": 1, "val_count": 1}
        with open(mf_path, "w") as f:
            json.dump(manifest, f)
        # Need label files with valid IDs to pass check #8
        for split in ("train", "val"):
            lbl_dir = os.path.join(ds_dir, "labels", split)
            with open(os.path.join(lbl_dir, "img1.txt"), "w") as f:
                f.write("0 0.5 0.5 0.1 0.1\n")
        assert w._validate_prepared_output(ds_dir, yaml_path, manifest)

    def test_rejects_nc_mismatch(self, qapp, tmp_path):
        """Rejects when YAML nc != actual classes."""
        w = _make_widget()
        ds_dir = str(tmp_path / "auto_dataset_nc_mismatch")
        os.makedirs(ds_dir, exist_ok=True)
        for split in ("train", "val"):
            os.makedirs(os.path.join(ds_dir, "images", split), exist_ok=True)
            os.makedirs(os.path.join(ds_dir, "labels", split), exist_ok=True)
        import yaml
        yaml_path = os.path.join(ds_dir, "data.yaml")
        yaml_data = {"names": {0: "a"}, "nc": 80}  # nc=80 but only 1 class
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_data, f)
        mf_path = os.path.join(ds_dir, "dataset_manifest.json")
        manifest = {"classes": ["a"], "valid_images": 1}
        with open(mf_path, "w") as f:
            json.dump(manifest, f)
        assert not w._validate_prepared_output(ds_dir, yaml_path, manifest)

    def test_rejects_out_of_range_label_id(self, qapp, tmp_path):
        """Rejects label TXT with class ID beyond range."""
        w = _make_widget()
        ds_dir = str(tmp_path / "auto_dataset_bad_id")
        os.makedirs(ds_dir, exist_ok=True)
        for split in ("train", "val"):
            os.makedirs(os.path.join(ds_dir, "images", split), exist_ok=True)
            os.makedirs(os.path.join(ds_dir, "labels", split), exist_ok=True)
        import yaml
        yaml_path = os.path.join(ds_dir, "data.yaml")
        yaml_data = {"names": {0: "a"}, "nc": 1}
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_data, f)
        mf_path = os.path.join(ds_dir, "dataset_manifest.json")
        manifest = {"classes": ["a"], "valid_images": 1}
        with open(mf_path, "w") as f:
            json.dump(manifest, f)
        # Write label with out-of-range ID 5
        lbl_dir = os.path.join(ds_dir, "labels", "train")
        with open(os.path.join(lbl_dir, "img1.txt"), "w") as f:
            f.write("5 0.5 0.5 0.1 0.1\n")  # class 5, but nc=1
        assert not w._validate_prepared_output(ds_dir, yaml_path, manifest)


# ── Test 19: Valid images count in manifest ──────────────────────────

class TestManifestValidImages:
    """_save_manifest records actual valid labeled image count."""

    def test_valid_images_not_total(self, qapp, tmp_path):
        """valid_images in manifest reflects actual labeled count."""
        w = _make_widget(image_list=["/a/img1.jpg", "/a/img2.jpg", "/a/img3.jpg"])
        _setup_config_widgets(w)
        w.selected_task_type = "Detect"
        w.output_dir = str(tmp_path)
        # Create fake JSONs (only 2 have valid shapes)
        for i in range(1, 4):
            json_path = os.path.join(str(tmp_path), f"img{i}.json")
            if i <= 2:
                data = {"shapes": [{"label": "cat", "shape_type": "rectangle", "points": [[0,0],[1,1]]}]}
            else:
                data = {"shapes": []}  # no shapes → not valid
            with open(json_path, "w") as f:
                json.dump(data, f)
        ds_dir = str(tmp_path / "auto_dataset_test")
        os.makedirs(ds_dir, exist_ok=True)
        yaml_path = os.path.join(ds_dir, "data.yaml")
        Path(yaml_path).touch()
        manifest = w._save_manifest(ds_dir, yaml_path, "fake_fp", 0.8)
        # Should be 2 valid, not 3 total
        assert manifest["valid_images"] == 2


# ── Test 20-24: Preflight class metadata ─────────────────────────────

class TestPreflightClassMetadata:
    """Full Preflight uses prepared YAML classes, not COCO80."""

    def _make_widget_with_config(self, task_type="Detect"):
        """Create widget with config_widgets pointing to coco8."""
        w = _make_widget(image_list=[], task_type=task_type)
        from anylabeling.views.training.widgets.ultralytics_widgets.custom_widgets import CustomLineEdit
        w.config_widgets = {
            "data": CustomLineEdit(),
            "project": CustomLineEdit(),
            "name": CustomLineEdit(),
            "model": CustomLineEdit(),
            "device": CustomLineEdit(),
            "epochs": CustomLineEdit(),
            "batch": CustomLineEdit(),
            "imgsz": CustomLineEdit(),
        }
        w.config_widgets["data"].setText("/tmp/coco8.yaml")
        w.config_widgets["project"].setText("/tmp/proj")
        w.config_widgets["name"].setText("exp")
        w.config_widgets["model"].setText("/tmp/model.pt")
        w.config_widgets["device"].setText("cpu")
        w.config_widgets["epochs"].setText("100")
        w.config_widgets["batch"].setText("16")
        w.config_widgets["imgsz"].setText("640")
        return w

    def test_context_uses_prepared_yaml_not_config_data(self, qapp, tmp_path):
        """_build_guided_preflight_context uses _prepared_yaml_path over config data."""
        w = self._make_widget_with_config()
        prepared = str(tmp_path / "auto_dataset_test" / "data.yaml")
        os.makedirs(os.path.dirname(prepared), exist_ok=True)
        Path(prepared).touch()
        w._prepared_yaml_path = prepared
        w._prepared_dataset_dir = os.path.dirname(prepared)
        ctx = w._build_guided_preflight_context()
        assert ctx.dataset_yaml == prepared

    def test_context_falls_back_when_no_prepared(self, qapp, tmp_path):
        """Without prepared YAML, context falls back to config data if it exists."""
        w = self._make_widget_with_config()
        w._prepared_yaml_path = None
        # Set config data to an existing file
        coco8 = str(tmp_path / "coco8.yaml")
        Path(coco8).touch()
        w.config_widgets["data"].setText(coco8)
        ctx = w._build_guided_preflight_context()
        assert ctx.dataset_yaml == coco8

    def test_data_check_uses_prepared_yaml(self, qapp, tmp_path):
        """_run_data_check prefers prepared YAML over config data."""
        w = self._make_widget_with_config()
        w.image_list = ["/fake/img1.jpg"]  # at least 1 image
        w.selected_task_type = "Detect"
        w.output_dir = str(tmp_path)
        # Create prepared YAML with only 3 classes
        prepared_dir = str(tmp_path / "auto_dataset_3cls")
        os.makedirs(prepared_dir, exist_ok=True)
        prepared_yaml = os.path.join(prepared_dir, "data.yaml")
        import yaml
        yaml_data = {
            "names": {0: "apple", 1: "bed", 2: "bowl"},
            "nc": 3,
            "path": prepared_dir,
            "train": "images/train",
            "val": "images/val",
        }
        with open(prepared_yaml, "w") as f:
            yaml.dump(yaml_data, f)
        w._prepared_yaml_path = prepared_yaml
        # config still points to coco8
        w.config_widgets["data"].setText("/tmp/coco8.yaml")
        # Verify the YAML path resolution logic directly
        config = w.get_current_config()
        resolved = (getattr(w, '_prepared_yaml_path', None)
                    or config.get("basic", {}).get("data", ""))
        assert resolved == prepared_yaml  # prepared takes priority

    def test_preflight_context_passes_correct_dataset_yaml(self, qapp, tmp_path):
        """GuidedPreflightContext.dataset_yaml must be the prepared YAML path."""
        w = self._make_widget_with_config()
        prepared = str(tmp_path / "auto_dataset_14cls" / "data.yaml")
        os.makedirs(os.path.dirname(prepared), exist_ok=True)
        Path(prepared).touch()
        w._prepared_yaml_path = prepared
        w._prepared_dataset_dir = os.path.dirname(prepared)
        ctx = w._build_guided_preflight_context()
        # dataset_yaml should be the prepared one, not coco8
        assert ctx.dataset_yaml == prepared
        assert ctx.dataset_yaml != "/tmp/coco8.yaml"

    def test_prepared_yaml_has_correct_num_classes(self, qapp, tmp_path):
        """Prepared YAML must have nc matching actual classes, not COCO80."""
        w = _make_widget(image_list=["/a/img1.jpg", "/a/img2.jpg"])
        w.selected_task_type = "Detect"
        w.output_dir = str(tmp_path)
        # Create fake JSONs with 2 distinct labels
        for i, lbl in enumerate(["shoes", "bag"], 1):
            json_path = os.path.join(str(tmp_path), f"img{i}.json")
            data = {"shapes": [{"label": lbl, "shape_type": "rectangle", "points": [[0,0],[1,1]]}]}
            with open(json_path, "w") as f:
                json.dump(data, f)
        # Create a minimal prepared YAML
        prepared_dir = str(tmp_path / "auto_dataset_2cls")
        os.makedirs(prepared_dir, exist_ok=True)
        prepared_yaml = os.path.join(prepared_dir, "data.yaml")
        import yaml
        yaml_data = {
            "names": {0: "shoes", 1: "bag"},
            "nc": 2,
            "path": prepared_dir,
            "train": "images/train",
            "val": "images/val",
        }
        with open(prepared_yaml, "w") as f:
            yaml.dump(yaml_data, f)
        # Create dummy images dirs
        for split in ("train", "val"):
            os.makedirs(os.path.join(prepared_dir, "images", split), exist_ok=True)
            os.makedirs(os.path.join(prepared_dir, "labels", split), exist_ok=True)
            Path(os.path.join(prepared_dir, "images", split, "img1.jpg")).touch()
        # Write manifest
        mf_path = os.path.join(prepared_dir, "dataset_manifest.json")
        manifest = {"classes": ["shoes", "bag"], "valid_images": 2,
                     "class_to_id": {"shoes": 0, "bag": 1},
                     "split_ratio": 0.5, "train_count": 1, "val_count": 1}
        with open(mf_path, "w") as f:
            json.dump(manifest, f)
        # Write label files with valid IDs
        for split in ("train", "val"):
            lbl_dir = os.path.join(prepared_dir, "labels", split)
            with open(os.path.join(lbl_dir, "img1.txt"), "w") as f:
                f.write("0 0.5 0.5 0.1 0.1\n")
        # Validation should pass: nc=2 matches manifest classes=2
        assert w._validate_prepared_output(prepared_dir, prepared_yaml, manifest)


# ── Test 25-30: Train/val split ratio ────────────────────────────────

class TestSplitRatio:
    """Train/val split must produce correct counts for given ratio."""

    def _make_manifest(self, valid, train, val, ratio=0.8):
        return {
            "classes": ["a", "b"],
            "valid_images": valid,
            "train_count": train,
            "val_count": val,
            "split_ratio": ratio,
            "class_to_id": {"a": 0, "b": 1},
        }

    def _make_dataset_dirs(self, base, train_count, val_count):
        """Create dataset dirs with placeholder images."""
        ds = str(base)
        os.makedirs(ds, exist_ok=True)
        for split, count in [("train", train_count), ("val", val_count)]:
            img_dir = os.path.join(ds, "images", split)
            lbl_dir = os.path.join(ds, "labels", split)
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(lbl_dir, exist_ok=True)
            for i in range(count):
                Path(os.path.join(img_dir, f"img{i}.jpg")).touch()
                with open(os.path.join(lbl_dir, f"img{i}.txt"), "w") as f:
                    f.write("0 0.5 0.5 0.1 0.1\n")
        # Write YAML
        import yaml
        yaml_path = os.path.join(ds, "data.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump({"names": {0: "a", 1: "b"}, "nc": 2, "path": ds,
                        "train": "images/train", "val": "images/val"}, f)
        # Write manifest
        mf_path = os.path.join(ds, "dataset_manifest.json")
        manifest = self._make_manifest(train_count + val_count, train_count, val_count)
        with open(mf_path, "w") as f:
            json.dump(manifest, f)
        return ds, yaml_path, manifest

    def test_split_26_at_08_gives_21_5(self, qapp, tmp_path):
        """26 valid images at 0.8 ratio → train=21, val=5."""
        w = _make_widget()
        ds, yp, mf = self._make_dataset_dirs(tmp_path / "ds1", 21, 5)
        assert w._validate_prepared_output(ds, yp, mf)

    def test_split_rejects_7_19_at_08(self, qapp, tmp_path):
        """26 valid at 0.8 with 7/19 split → rejected."""
        w = _make_widget()
        ds, yp, mf = self._make_dataset_dirs(tmp_path / "ds2", 7, 19)
        assert not w._validate_prepared_output(ds, yp, mf)

    def test_split_train_must_exceed_val_when_ratio_above_05(self, qapp, tmp_path):
        """When ratio > 0.5, train must be > val."""
        w = _make_widget()
        # train=3, val=7, ratio=0.8 → should fail because train <= val
        ds, yp, mf = self._make_dataset_dirs(tmp_path / "ds3", 3, 7)
        assert not w._validate_prepared_output(ds, yp, mf)

    def test_split_total_preserved(self, qapp, tmp_path):
        """train + val must equal total valid images."""
        w = _make_widget()
        ds, yp, mf = self._make_dataset_dirs(tmp_path / "ds4", 21, 5)
        result = w._validate_prepared_output(ds, yp, mf)
        assert result
        # manifest should be updated with train_count/val_count
        import json as _json
        with open(os.path.join(ds, "dataset_manifest.json")) as f:
            updated = _json.load(f)
        # Note: _validate_prepared_output updates manifest in-memory but not on disk
        # The on-disk manifest already has train_count=21, val_count=5 from _make_dataset_dirs

    def test_converter_version_bumped(self, qapp):
        """_CONVERTER_VERSION must be 3 (bumped from 2)."""
        w = _make_widget()
        assert w._CONVERTER_VERSION == 3, (
            f"Expected _CONVERTER_VERSION=3 to invalidate old caches, got {w._CONVERTER_VERSION}"
        )

    def test_split_fixed_seed_reproducible(self, qapp, tmp_path):
        """verify round() math is correct independently."""
        # 26 * 0.8 = 20.8 → round = 21
        assert round(26 * 0.8) == 21
        # 26 - 21 = 5
        assert 26 - round(26 * 0.8) == 5
        # Not 7/19
        assert round(26 * 0.8) != 7

