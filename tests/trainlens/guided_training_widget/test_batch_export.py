"""Tests for batch multi-format model export."""

import os
import sys
import json
import tempfile
import uuid
from unittest.mock import MagicMock, patch, PropertyMock, call

import pytest
from PyQt6.QtWidgets import QApplication, QMessageBox

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from anylabeling.services.auto_training.ultralytics.export_formats import (
    EXPORT_FORMATS,
    FORMAT_BY_CODE,
    FormatInfo,
    FormatCapability,
    FormatStatus,
    get_format_status,
    get_missing_pip_packages,
    get_large_missing_packages,
    has_any_large_missing,
    DEPRECATED_FORMATS,
    DEPRECATED_REDIRECT,
)
from anylabeling.services.auto_training.ultralytics.batch_exporter import (
    BatchExportRequest,
    ExportJob,
    BatchExportManager,
    get_batch_export_manager,
)
from anylabeling.services.auto_training.ultralytics.exporter import (
    get_export_validator,
    validate_torchscript_export_environment,
    validate_litert_export_environment,
)
from anylabeling.services.auto_training.ultralytics.package_installer import PackageInstaller


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


# ── Format Definitions Tests ─────────────────────────────────────────────

class TestFormatDefinitions:
    """Verify format definitions are correct."""

    def test_all_formats_have_unique_codes(self):
        codes = [f.format_code for f in EXPORT_FORMATS]
        assert len(codes) == len(set(codes)), "Duplicate format codes found"

    def test_litert_is_present(self):
        assert "litert" in FORMAT_BY_CODE
        info = FORMAT_BY_CODE["litert"]
        assert info.display_name == "LiteRT"
        assert info.category == "Mobile & Edge"

    def test_tflite_is_deprecated(self):
        assert "tflite" in DEPRECATED_FORMATS
        assert DEPRECATED_REDIRECT.get("tflite") == "litert"

    def test_tfjs_is_deprecated(self):
        assert "tfjs" in DEPRECATED_FORMATS
        assert DEPRECATED_REDIRECT.get("tfjs") == "litert"

    def test_torchscript_has_format_info(self):
        assert "torchscript" in FORMAT_BY_CODE
        info = FORMAT_BY_CODE["torchscript"]
        assert info.format_code == "torchscript"
        assert info.required_packages == []
        assert info.capability.supports_half is True

    def test_onnx_output_is_file(self):
        info = FORMAT_BY_CODE["onnx"]
        assert not info.is_directory
        assert info.output_path_template == "best.onnx"

    def test_saved_model_output_is_directory(self):
        info = FORMAT_BY_CODE["saved_model"]
        assert info.is_directory
        assert info.output_path_template == "best_saved_model/"

    def test_engine_requires_gpu(self):
        info = FORMAT_BY_CODE["engine"]
        assert info.capability.requires_gpu

    def test_coreml_platforms(self):
        info = FORMAT_BY_CODE["coreml"]
        assert "darwin" in info.capability.supported_platforms

    def test_rknn_linux_only(self):
        info = FORMAT_BY_CODE["rknn"]
        assert info.capability.supported_platforms == ["linux"]


class TestFormatStatus:
    """Verify environment status detection."""

    def test_torchscript_status_is_not_unsupported_platform(self):
        info = FORMAT_BY_CODE["torchscript"]
        status = get_format_status(info)
        # TorchScript should never be UNSUPPORTED_PLATFORM
        assert status != FormatStatus.UNSUPPORTED_PLATFORM

    def test_onnx_missing_deps_returns_correct_status(self):
        info = FORMAT_BY_CODE["onnx"]
        status = get_format_status(info)
        # May be READY or MISSING_DEPENDENCY depending on environment
        assert status in (FormatStatus.READY, FormatStatus.MISSING_DEPENDENCY)

    def test_coreml_status_on_windows(self):
        """CoreML should show UNSUPPORTED_PLATFORM on Windows."""
        import platform
        info = FORMAT_BY_CODE["coreml"]
        status = get_format_status(info)
        if platform.system() != "Darwin":
            assert status == FormatStatus.UNSUPPORTED_PLATFORM

    def test_engine_requires_gpu_status(self):
        info = FORMAT_BY_CODE["engine"]
        status = get_format_status(info)
        # Check that it detects GPU requirement
        try:
            import torch
            if not torch.cuda.is_available():
                assert status == FormatStatus.UNSUPPORTED_DEVICE
        except ImportError:
            assert status == FormatStatus.UNSUPPORTED_DEVICE


class TestDeprecatedFormats:
    """Deprecated format handling."""

    def test_tflite_maps_to_litert_validator(self):
        validator = get_export_validator("tflite")
        # Should return the litert validator, not tensorflow validator
        assert validator is validate_litert_export_environment

    def test_tfjs_maps_to_litert_validator(self):
        validator = get_export_validator("tfjs")
        assert validator is validate_litert_export_environment

    def test_litert_validator_returns_empty(self):
        missing = validate_litert_export_environment()
        assert missing == []


class TestTorchScriptValidator:
    """TorchScript no longer uses lambda: []."""

    def test_torchscript_validator_is_real_function(self):
        validator = get_export_validator("torchscript")
        assert validator is validate_torchscript_export_environment
        # Not a lambda
        assert validator.__name__ == "validate_torchscript_export_environment"

    def test_torchscript_validator_runs(self):
        missing = validate_torchscript_export_environment()
        # Should check for torch
        try:
            import torch
            assert missing == []
        except ImportError:
            assert "torch" in missing


# ── BatchExportManager Tests ─────────────────────────────────────────────

class TestBatchExportRequest:
    """BatchExportRequest data structure."""

    def test_request_creation(self):
        req = BatchExportRequest(
            source_model="/path/to/best.pt",
            formats=["onnx", "torchscript", "mnn"],
            output_root="/path/to/exports",
            imgsz=640,
            half=False,
        )
        assert req.source_model == "/path/to/best.pt"
        assert len(req.formats) == 3
        assert "tflite" not in req.formats


class TestBatchExportManager:
    """BatchExportManager serial execution and job management."""

    def test_create_jobs_deduplicates(self, qapp):
        manager = BatchExportManager()
        req = BatchExportRequest(
            source_model="/fake/best.pt",
            formats=["onnx", "onnx", "torchscript"],
            output_root="/fake/exports",
        )
        jobs = manager.create_jobs(req)
        assert len(jobs) == 2
        codes = [j.format_code for j in jobs]
        assert codes.count("onnx") == 1

    def test_create_jobs_redirects_deprecated(self, qapp):
        manager = BatchExportManager()
        req = BatchExportRequest(
            source_model="/fake/best.pt",
            formats=["tflite", "tfjs"],
            output_root="/fake/exports",
        )
        jobs = manager.create_jobs(req)
        # Both should be redirected to litert, deduplicated to 1
        assert len(jobs) == 1
        assert jobs[0].format_code == "litert"

    def test_create_jobs_rejects_unknown(self, qapp):
        manager = BatchExportManager()
        req = BatchExportRequest(
            source_model="/fake/best.pt",
            formats=["onnx", "nonexistent_format"],
            output_root="/fake/exports",
        )
        jobs = manager.create_jobs(req)
        # Unknown format should be skipped
        assert len(jobs) == 1
        assert jobs[0].format_code == "onnx"

    def test_cannot_start_two_batches(self, qapp):
        manager = BatchExportManager()
        req = BatchExportRequest(
            source_model="/fake/best.pt",
            formats=["onnx"],
            output_root="/fake/exports",
        )
        assert manager.start_batch(req) is True
        assert manager.is_running is True
        assert manager.start_batch(req) is False

    def test_cancel_batch(self, qapp):
        manager = BatchExportManager()
        req = BatchExportRequest(
            source_model="/fake/best.pt",
            formats=["onnx", "torchscript"],
            output_root="/fake/exports",
        )
        # Don't actually start execution (would need real model)
        # Just verify the cancel mechanism
        manager._jobs = [
            ExportJob(job_id="1", format_code="onnx", status="pending"),
            ExportJob(job_id="2", format_code="torchscript", status="pending"),
        ]
        manager._is_running = True
        manager.cancel_batch()
        for job in manager._jobs:
            assert job.status == "cancelled"


class TestExportJob:
    """ExportJob data structure."""

    def test_job_initial_state(self):
        job = ExportJob(job_id="test-1", format_code="onnx")
        assert job.job_id == "test-1"
        assert job.format_code == "onnx"
        assert job.status == "pending"
        assert job.output_path is None
        assert job.error_message is None
        assert job.progress_pct == 0

    def test_job_status_transitions(self):
        job = ExportJob(job_id="test-2", format_code="torchscript")
        job.status = "running"
        assert job.status == "running"
        job.status = "completed"
        job.output_path = "/path/to/output.torchscript"
        assert job.status == "completed"
        assert job.output_path == "/path/to/output.torchscript"


class TestFormatCapability:
    """ExportCapability parameter compatibility."""

    def test_onnx_supports_all(self):
        info = FORMAT_BY_CODE["onnx"]
        assert info.capability.supports_half
        assert info.capability.supports_int8
        assert info.capability.supports_dynamic
        assert info.capability.supports_simplify

    def test_torchscript_partial_support(self):
        info = FORMAT_BY_CODE["torchscript"]
        assert info.capability.supports_half
        assert not info.capability.supports_int8
        assert not info.capability.supports_dynamic
        assert not info.capability.supports_simplify

    def test_litert_capability(self):
        info = FORMAT_BY_CODE["litert"]
        assert info.capability.supports_int8
        assert info.capability.requires_dataset  # INT8 needs calibration data

    def test_edgetpu_capability(self):
        info = FORMAT_BY_CODE["edgetpu"]
        assert info.capability.requires_dataset
        assert info.capability.supports_int8


class TestOutputPaths:
    """Output path correctness for files and directories."""

    def test_file_output_paths(self):
        """All file formats have .extension output."""
        file_formats = ["onnx", "torchscript", "engine", "litert", "edgetpu"]
        for fmt in file_formats:
            info = FORMAT_BY_CODE[fmt]
            assert not info.is_directory, f"{fmt} should be a file"
            assert info.extension, f"{fmt} should have an extension"
            assert not info.output_path_template.endswith("/"), (
                f"{fmt} file should not end with /"
            )

    def test_directory_output_paths(self):
        """All directory formats end with /."""
        dir_formats = [
            "openvino", "saved_model", "paddle", "mnn", "ncnn", "imx", "rknn"
        ]
        for fmt in dir_formats:
            info = FORMAT_BY_CODE[fmt]
            assert info.is_directory, f"{fmt} should be a directory"
            assert info.output_path_template.endswith("/"), (
                f"{fmt} dir should end with /"
            )


class TestLargeDependencyDetection:
    """Large dependency detection."""

    def test_large_dependency_list(self):
        large = get_large_missing_packages(FORMAT_BY_CODE["saved_model"])
        assert any("tensorflow" in d for d in large)

    def test_has_any_large_missing(self):
        assert has_any_large_missing([FORMAT_BY_CODE["saved_model"]]) is True
        assert has_any_large_missing([FORMAT_BY_CODE["onnx"]]) is False


class TestPackageInstaller:
    """PackageInstaller QProcess-based non-blocking install."""

    def test_installer_not_running_initially(self, qapp):
        installer = PackageInstaller()
        assert not installer.is_running

    def test_cancel_when_not_running(self, qapp):
        installer = PackageInstaller()
        installer.cancel()  # Should not raise
        assert not installer.is_running


# ── Manifest Tests ───────────────────────────────────────────────────────

class TestManifestGeneration:
    """export_manifest.json generation."""

    def test_manifest_structure(self, qapp, tmp_path):
        """Manifest contains expected fields."""
        req = BatchExportRequest(
            source_model="/fake/best.pt",
            formats=["onnx", "torchscript"],
            output_root=str(tmp_path),
        )
        manager = BatchExportManager()
        manager._request = req
        manager._jobs = [
            ExportJob(job_id="1", format_code="onnx", status="completed",
                       output_path=str(tmp_path / "onnx" / "best.onnx")),
            ExportJob(job_id="2", format_code="torchscript", status="failed",
                       error_message="Test error"),
        ]
        # Create fake output directories for manifest
        (tmp_path / "onnx").mkdir(parents=True, exist_ok=True)
        (tmp_path / "onnx" / "best.onnx").write_text("mock")

        manifest_path = manager.generate_manifest(str(tmp_path))
        assert os.path.exists(manifest_path)

        with open(manifest_path) as f:
            data = json.load(f)

        assert data["source_model"] == "/fake/best.pt"
        assert "onnx" in data["requested_formats"]
        assert "torchscript" in data["requested_formats"]
        assert len(data["exports"]) == 2

        onnx_entry = [e for e in data["exports"] if e["format"] == "onnx"][0]
        assert onnx_entry["status"] == "completed"
        assert "output" in onnx_entry

        ts_entry = [e for e in data["exports"] if e["format"] == "torchscript"][0]
        assert ts_entry["status"] == "failed"
        assert "error" in ts_entry
