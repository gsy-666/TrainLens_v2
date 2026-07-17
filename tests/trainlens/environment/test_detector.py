"""Tests for environment detector functions."""

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from anylabeling.services.training_center.environment.detector import (
    find_project_python,
    find_system_python_for_venv,
    inspect_python_environment,
    _validate_python,
    _supports_venv,
    _is_packaged_python,
    _VENV_CANDIDATES_WIN,
    _VENV_CANDIDATES_UNIX,
)
from anylabeling.services.training_center.environment.models import (
    EnvironmentInfo,
    EnvironmentStatus,
)


class TestFindProjectPython:
    """Tests for find_project_python."""

    def test_finds_dot_venv_windows(self, tmp_path):
        r"""Finds .venv\Scripts\python.exe on Windows."""
        if platform.system() != "Windows":
            pytest.skip("Windows-only test")
        venv = tmp_path / ".venv" / "Scripts"
        venv.mkdir(parents=True)
        py = venv / "python.exe"
        py.write_text("fake")

        def fake_validate(p):
            return p == py or str(p).endswith("python.exe")

        with patch("anylabeling.services.training_center.environment.detector._validate_python", side_effect=fake_validate):
            found = find_project_python(tmp_path)
            assert found == py

    def test_finds_venv_windows(self, tmp_path):
        r"""Finds venv\Scripts\python.exe on Windows (only venv created, no .venv)."""
        if platform.system() != "Windows":
            pytest.skip("Windows-only test")
        venv = tmp_path / "venv" / "Scripts"
        venv.mkdir(parents=True)
        py = venv / "python.exe"
        py.write_text("fake")

        # Only validate the venv entry, not .venv
        def fake_validate(p):
            return p == py

        with patch("anylabeling.services.training_center.environment.detector._validate_python", side_effect=fake_validate):
            found = find_project_python(tmp_path)
            assert found == py

    def test_priority_dot_venv_over_venv(self, tmp_path):
        """.venv is preferred over venv."""
        if platform.system() != "Windows":
            pytest.skip("Windows-only test")
        v1 = tmp_path / ".venv" / "Scripts"
        v1.mkdir(parents=True)
        py1 = v1 / "python.exe"
        py1.write_text("fake1")
        v2 = tmp_path / "venv" / "Scripts"
        v2.mkdir(parents=True)
        (v2 / "python.exe").write_text("fake2")

        def fake_validate(p):
            return True  # both valid

        with patch("anylabeling.services.training_center.environment.detector._validate_python", side_effect=fake_validate):
            found = find_project_python(tmp_path)
            assert found == py1

    def test_invalid_candidate_skipped(self, tmp_path):
        """Non-executable candidate is skipped."""
        venv = tmp_path / ".venv" / "Scripts"
        venv.mkdir(parents=True)
        py = venv / "python.exe"
        py.write_text("fake")

        with patch("anylabeling.services.training_center.environment.detector._validate_python", return_value=False):
            found = find_project_python(tmp_path)
            assert found is None

    def test_no_venv_returns_none(self, tmp_path):
        """Empty directory returns None."""
        found = find_project_python(tmp_path)
        assert found is None

    def test_nonexistent_dir_returns_none(self):
        """Non-existent directory returns None."""
        found = find_project_python(Path("/nonexistent/path/xyz"))
        assert found is None


class TestSystemPython:
    """Tests for find_system_python_for_venv."""

    def test_current_python_is_valid(self):
        """Current sys.executable is a valid system Python."""
        found = find_system_python_for_venv()
        assert found is not None

    def test_trainlens_exe_filtered(self):
        """TrainLens.exe is NOT considered a system Python."""
        with patch("anylabeling.services.training_center.environment.detector._is_packaged_python", return_value=True):
            with patch("anylabeling.services.training_center.environment.detector.sys") as mock_sys:
                mock_sys.executable = "C:/app/TrainLens.exe"
                found = find_system_python_for_venv()
                # Should not return TrainLens.exe — will try py launcher or PATH
                assert found is None or "TrainLens" not in str(found)


class TestInspectEnvironment:
    """Tests for inspect_python_environment."""

    def test_valid_python_detection(self):
        """Current Python is detected correctly."""
        info = inspect_python_environment(Path(sys.executable))
        assert info.executable_valid
        assert info.status in (EnvironmentStatus.READY, EnvironmentStatus.WARNING)
        assert info.python_version

    def test_json_parse_with_valid_output(self):
        """JSON from detection script parses correctly."""
        sample = json.dumps({
            "version": "3.11.9",
            "executable_valid": True,
            "pip_available": True,
            "torch_installed": False,
            "torch_version": "",
            "cuda_available": False,
            "cuda_version": "",
            "gpu_name": "",
            "ultralytics_installed": False,
            "ultralytics_version": "",
            "prefix": "/some/venv",
            "base_prefix": "/usr",
        })
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout=sample, stderr="")
            info = inspect_python_environment(Path(sys.executable))
            assert info.status == EnvironmentStatus.READY
            assert info.python_version == "3.11.9"
            assert info.pip_available

    def test_missing_pip_is_warning(self):
        """Missing pip → WARNING."""
        sample = json.dumps({
            "version": "3.11.9",
            "executable_valid": True,
            "pip_available": False,
            "torch_installed": False,
            "torch_version": "",
            "cuda_available": False,
            "cuda_version": "",
            "gpu_name": "",
            "ultralytics_installed": False,
            "ultralytics_version": "",
            "prefix": "/usr",
            "base_prefix": "/usr",
        })
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout=sample, stderr="")
            info = inspect_python_environment(Path(sys.executable))
            assert info.status == EnvironmentStatus.WARNING
            assert "pip" in (info.warning_message or "").lower()

    def test_cpu_only_torch_not_error(self):
        """CPU-only PyTorch is NOT ERROR."""
        sample = json.dumps({
            "version": "3.11.9",
            "executable_valid": True,
            "pip_available": True,
            "torch_installed": True,
            "torch_version": "2.0.0+cpu",
            "cuda_available": False,
            "cuda_version": "",
            "gpu_name": "",
            "ultralytics_installed": True,
            "ultralytics_version": "8.0.0",
            "prefix": "/usr",
            "base_prefix": "/usr",
        })
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout=sample, stderr="")
            info = inspect_python_environment(Path(sys.executable))
            assert info.status != EnvironmentStatus.ERROR

    def test_invalid_json_is_error(self):
        """Invalid JSON → ERROR."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="not json", stderr="")
            info = inspect_python_environment(Path(sys.executable))
            assert info.status == EnvironmentStatus.ERROR

    def test_detection_timeout_is_error(self):
        """Timeout → ERROR."""
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 30)):
            info = inspect_python_environment(Path(sys.executable))
            assert info.status == EnvironmentStatus.ERROR

    def test_nonexistent_python_is_error(self):
        """Non-existent Python → ERROR."""
        info = inspect_python_environment(Path("/nonexistent/python"))
        assert info.status == EnvironmentStatus.ERROR

    def test_venv_type_detected(self):
        """Prefix != base_prefix → venv."""
        sample = json.dumps({
            "version": "3.11.9",
            "executable_valid": True,
            "pip_available": True,
            "torch_installed": False, "torch_version": "",
            "cuda_available": False, "cuda_version": "", "gpu_name": "",
            "ultralytics_installed": False, "ultralytics_version": "",
            "prefix": "/project/.venv",
            "base_prefix": "/usr/local",
        })
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout=sample, stderr="")
            info = inspect_python_environment(Path(sys.executable))
            assert info.environment_type == "venv"


class TestHelpers:
    """Tests for internal helpers."""

    def test_validate_python(self):
        """Current Python validates."""
        assert _validate_python(Path(sys.executable))

    def test_validate_nonexistent(self):
        """Non-existent Python fails validation."""
        assert not _validate_python(Path("/nonexistent/python_xyz"))

    def test_supports_venv(self):
        """Current Python supports venv."""
        assert _supports_venv(Path(sys.executable))

    def test_packaged_python_detection(self):
        """TrainLens.exe-like names are detected as packaged."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert _is_packaged_python(Path("C:/app/TrainLens.exe"))
