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
