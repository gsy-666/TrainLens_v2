"""Tests for environment operations (venv creation, pip, requirements)."""

import platform
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from anylabeling.services.training_center.environment.operations import (
    create_project_venv,
    ensure_pip,
    install_requirements,
    _pip_available,
)
from anylabeling.services.training_center.environment.detector import _validate_python


class TestCreateVenv:
    """Tests for create_project_venv."""

    def test_creates_with_args_list_shell_false(self, tmp_path):
        """venv creation uses argument list and shell=False."""
        sys_py = tmp_path / "sys_python"
        sys_py.write_text("fake")

        with patch("subprocess.run") as mock_run:
            with patch("anylabeling.services.training_center.environment.operations._validate_python", return_value=True):
                with patch("anylabeling.services.training_center.environment.operations._pip_available", return_value=True):
                    mock_run.return_value = Mock(returncode=0)
                    ok, path, msg = create_project_venv(tmp_path, sys_py)
                    assert ok
                    call_args = mock_run.call_args[0][0]
                    assert isinstance(call_args, list)
                    assert "-m" in call_args
                    assert "venv" in call_args
                    # shell must not be True
                    assert mock_run.call_args[1].get("shell", False) is False

    def test_existing_valid_venv_not_overwritten(self, tmp_path):
        """Valid existing .venv is returned without recreation."""
        sys_py = tmp_path / "sys_python"
        sys_py.write_text("fake")
        venv_py = tmp_path / ".venv" / "Scripts" / "python.exe"
        if platform.system() != "Windows":
            venv_py = tmp_path / ".venv" / "bin" / "python"
        venv_py.parent.mkdir(parents=True)
        venv_py.write_text("fake")

        with patch("anylabeling.services.training_center.environment.operations._validate_python", return_value=True):
            ok, path, msg = create_project_venv(tmp_path, sys_py)
            assert ok
            assert "existing" in msg.lower() or "valid" in msg.lower()

    def test_damaged_venv_not_deleted(self, tmp_path):
        """Damaged .venv returns error, does NOT auto-delete."""
        sys_py = tmp_path / "sys_python"
        sys_py.write_text("fake")
        venv_dir = tmp_path / ".venv"
        venv_dir.mkdir()

        with patch("anylabeling.services.training_center.environment.operations._validate_python", return_value=False):
            ok, path, msg = create_project_venv(tmp_path, sys_py)
            assert not ok
            assert venv_dir.exists()  # NOT deleted

    def test_nonexistent_project_returns_error(self, tmp_path):
        """Non-existent project dir returns error."""
        sys_py = tmp_path / "sys_python"
        sys_py.write_text("fake")
        ok, path, msg = create_project_venv(Path("/nonexistent/project"), sys_py)
        assert not ok


class TestEnsurePip:
    """Tests for ensure_pip."""

    def test_pip_already_available(self):
        """pip already available → no action."""
        with patch("anylabeling.services.training_center.environment.operations._pip_available", return_value=True):
            ok, msg = ensure_pip(Path("python"))
            assert ok
            assert "already" in msg.lower()

    def test_pip_missing_calls_ensurepip(self, tmp_path):
        """Missing pip triggers ensurepip."""
        py = tmp_path / "python"
        py.write_text("fake")
        with patch("anylabeling.services.training_center.environment.operations._pip_available", side_effect=[False, False]):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = Mock(returncode=0)
                ok, msg = ensure_pip(py)
                call_args = mock_run.call_args[0][0]
                assert "-m" in call_args
                assert "ensurepip" in call_args


class TestInstallRequirements:
    """Tests for install_requirements."""

    def test_pip_install_args_correct(self, tmp_path):
        """pip install uses correct arguments."""
        py = tmp_path / "python"
        py.write_text("fake")
        req = tmp_path / "requirements.txt"
        req.write_text("numpy\n")

        with patch("anylabeling.services.training_center.environment.operations._pip_available", return_value=True):
            with patch("subprocess.Popen") as mock_popen:
                mock_proc = Mock()
                mock_proc.returncode = 0
                mock_proc.communicate.return_value = ("ok\n", "")
                mock_popen.return_value = mock_proc
                ok, msg = install_requirements(py, req)
                call_args = mock_popen.call_args[0][0]
                assert "-m" in call_args
                assert "pip" in call_args
                assert "install" in call_args
                assert "-r" in call_args
                assert str(req) in call_args

    def test_requirements_not_found_rejected(self, tmp_path):
        """Missing requirements.txt → error."""
        py = tmp_path / "python"
        py.write_text("fake")
        ok, msg = install_requirements(py, tmp_path / "nonexistent.txt")
        assert not ok

    def test_no_pip_rejected(self, tmp_path):
        """No pip available → error."""
        py = tmp_path / "python"
        py.write_text("fake")
        req = tmp_path / "requirements.txt"
        req.write_text("numpy\n")
        with patch("anylabeling.services.training_center.environment.operations._pip_available", return_value=False):
            ok, msg = install_requirements(py, req)
            assert not ok

    def test_chinese_path_not_split(self, tmp_path):
        """Chinese/spaces in path are NOT split into separate args."""
        chinese_dir = tmp_path / "我的项目"
        chinese_dir.mkdir()
        py = chinese_dir / "python.exe"
        py.write_text("fake")
        req = chinese_dir / "requirements.txt"
        req.write_text("numpy\n")

        with patch("anylabeling.services.training_center.environment.operations._pip_available", return_value=True):
            with patch("subprocess.Popen") as mock_popen:
                mock_proc = Mock()
                mock_proc.returncode = 0
                mock_proc.communicate.return_value = ("ok\n", "")
                mock_popen.return_value = mock_proc
                install_requirements(py, req)
                call_args = mock_popen.call_args[0][0]
                # The path should appear as a single element
                req_str = str(req)
                assert req_str in call_args or any(req_str in a for a in call_args)
