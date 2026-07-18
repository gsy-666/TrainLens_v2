"""Tests for preflight common checks (output dir, disk, active job)."""

import os
import tempfile

from anylabeling.services.training_center.preflight import (
    PreflightResult, PreflightSeverity, PreflightIssue,
)
from anylabeling.services.training_center.preflight.common_checks import (
    check_active_job, check_output_directory, check_disk_space,
    check_config_not_empty, check_positive_int,
)


class TestActiveJob:
    def test_no_active_job_passes(self):
        r = PreflightResult(mode="test")
        check_active_job(r, False)
        assert r.error_count == 0

    def test_active_job_errors(self):
        r = PreflightResult(mode="test")
        check_active_job(r, True)
        assert r.error_count > 0
        assert r.errors()[0].code == "ACTIVE_JOB"


class TestOutputDirectory:
    def test_writable_dir_passes(self, tmp_path):
        r = PreflightResult(mode="test")
        check_output_directory(r, str(tmp_path))
        assert r.error_count == 0, f"Expected no errors, got: {[i.code for i in r.errors()]}"

    def test_empty_dir_warns(self):
        r = PreflightResult(mode="test")
        check_output_directory(r, "")
        assert any(i.code == "OUTPUT_DIR_EMPTY" for i in r.issues)

    def test_nonexistent_creatable(self, tmp_path):
        r = PreflightResult(mode="test")
        check_output_directory(r, str(tmp_path / "new_subdir"))
        assert r.error_count == 0

    def test_invalid_chars_error(self, tmp_path):
        r = PreflightResult(mode="test")
        check_output_directory(r, str(tmp_path / "bad:name"))
        if os.name == 'nt':
            assert r.error_count > 0


class TestDiskSpace:
    def test_disk_space_on_valid_dir(self, tmp_path):
        r = PreflightResult(mode="test")
        check_disk_space(r, str(tmp_path))
        # Should always have some disk space result
        assert len(r.issues) > 0


class TestConfigNotEmpty:
    def test_empty_value_error(self):
        r = PreflightResult(mode="test")
        check_config_not_empty(r, "model", "")
        assert any(i.code == "MODEL_EMPTY" for i in r.issues)

    def test_filled_value_passes(self):
        r = PreflightResult(mode="test")
        check_config_not_empty(r, "model", "yolov8n.pt")
        assert any(i.code == "MODEL_OK" for i in r.issues)


class TestPositiveInt:
    def test_zero_errors(self):
        r = PreflightResult(mode="test")
        check_positive_int(r, "epochs", 0)
        assert any(i.code == "EPOCHS_INVALID" for i in r.issues)

    def test_positive_passes(self):
        r = PreflightResult(mode="test")
        check_positive_int(r, "epochs", 100)
        assert any(i.code == "EPOCHS_OK" for i in r.issues)

    def test_negative_errors(self):
        r = PreflightResult(mode="test")
        check_positive_int(r, "batch", -1)
        assert any(i.code == "BATCH_INVALID" for i in r.issues)
