"""Tests for resource_utils, frozen worker mode, and packaging infrastructure."""

import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from anylabeling.services.training_center.resource_utils import (
    resource_path,
    is_frozen,
)


class TestResourcePathSourceMode:
    """resource_path() in source mode (normal development)."""

    def test_resolves_training_worker_in_source_mode(self):
        """Worker script should be resolvable from source tree."""
        path = resource_path(
            "anylabeling/services/auto_training/ultralytics/training_worker.py"
        )
        assert path.exists(), f"Expected worker at {path}"
        assert path.name == "training_worker.py"

    def test_resolves_relative_path_from_project_root(self):
        """Any relative path should be resolved relative to project root."""
        path = resource_path("anylabeling/app_info.py")
        assert path.exists()
        assert "app_info.py" in str(path)

    def test_is_frozen_returns_false_in_source(self):
        """In normal unittest environment, is_frozen should be False."""
        assert is_frozen() is False


class TestResourcePathFrozenMode:
    """resource_path() in frozen (PyInstaller) mode."""

    @patch.object(sys, "frozen", True, create=True)
    @patch.object(sys, "_MEIPASS", "/fake/meipass", create=True)
    def test_resolves_under_meipass_when_frozen(self):
        """When frozen, paths should resolve under sys._MEIPASS."""
        path = resource_path("anylabeling/training_worker.py")
        expected = Path("/fake/meipass/anylabeling/training_worker.py")
        assert path == expected

    @patch.object(sys, "frozen", True, create=True)
    def test_is_frozen_returns_true(self):
        """is_frozen should return True when sys.frozen is set."""
        assert is_frozen() is True


class TestLocalRunnerFrozenMode:
    """LocalRunner behavior in frozen mode."""

    def test_resolve_worker_script_returns_sentinel_in_frozen(self):
        """_resolve_worker_script() should return ':frozen:' when frozen."""
        with patch.object(sys, "frozen", True, create=True):
            from anylabeling.services.training_center.runners.local import _resolve_worker_script
            result = _resolve_worker_script()
            assert result == ":frozen:"

    def test_resolve_worker_script_returns_path_in_source(self):
        """In source mode, returns actual file path."""
        from anylabeling.services.training_center.runners.local import _resolve_worker_script
        result = _resolve_worker_script()
        assert result != ":frozen:"
        assert "training_worker.py" in result


class TestSSHRemoteWorkerUpload:
    """SSHRemoteRunner worker upload path resolution."""

    def test_resolve_worker_for_upload_in_source(self):
        """In source mode, _resolve_worker_for_upload returns existing file."""
        from anylabeling.services.training_center.runners.ssh_remote import (
            _resolve_worker_for_upload,
        )
        path = _resolve_worker_for_upload()
        assert os.path.isfile(path), f"Worker not found at {path}"
        assert "training_worker.py" in path

    @patch.object(sys, "frozen", True, create=True)
    @patch.object(sys, "_MEIPASS", "/fake/meipass", create=True)
    def test_resolve_worker_for_upload_in_frozen(self):
        """In frozen mode, resolves under _MEIPASS."""
        from anylabeling.services.training_center.runners.ssh_remote import (
            _resolve_worker_for_upload,
        )
        path = _resolve_worker_for_upload()
        assert "/fake/meipass/" in path.replace("\\", "/")
        assert "training_worker.py" in path


class TestBuildInfo:
    """build_info module."""

    def test_get_user_data_dir_in_localappdata(self):
        """User data should be under %LOCALAPPDATA%\\TrainLens."""
        from anylabeling.services.training_center.build_info import get_user_data_dir
        path = get_user_data_dir()
        assert "TrainLens" in str(path)

    def test_get_log_dir_is_under_user_data(self):
        """Log directory should be under user data."""
        from anylabeling.services.training_center.build_info import (
            get_user_data_dir,
            get_log_dir,
        )
        log_dir = get_log_dir()
        user_dir = get_user_data_dir()
        assert str(log_dir).startswith(str(user_dir))

    def test_ensure_user_dirs_creates_directories(self):
        """ensure_user_dirs should create directories without error."""
        from anylabeling.services.training_center.build_info import ensure_user_dirs
        # Should not raise
        ensure_user_dirs()
        from anylabeling.services.training_center.build_info import get_log_dir
        assert get_log_dir().exists()

    def test_get_build_info_returns_expected_keys(self):
        """get_build_info should return all required keys."""
        from anylabeling.services.training_center.build_info import get_build_info
        info = get_build_info()
        expected_keys = [
            "trainlens_version", "app_version", "build_date",
            "git_commit", "frozen", "python_version", "python_executable",
            "platform", "user_data_dir",
        ]
        for key in expected_keys:
            assert key in info, f"Missing key: {key}"
        assert info["trainlens_version"] == "0.1.0-beta"
        assert isinstance(info["frozen"], bool)


class TestPasswordNotInLogs:
    """Security: passwords must not appear in logs or build artifacts."""

    def test_password_not_in_resource_path(self):
        """resource_path should not leak any password strings."""
        sample_passwords = ["hunter2", "admin123", "password"]
        path_str = str(resource_path("anylabeling/app.py"))
        for pw in sample_passwords:
            assert pw not in path_str.lower()

    def test_build_info_excludes_sensitive_keys(self):
        """Build info must NOT contain password/secret/private keys."""
        from anylabeling.services.training_center.build_info import get_build_info
        info = get_build_info()
        info_str = str(info).lower()
        for sensitive in ["password", "secret", "private", "token", "passwd"]:
            assert sensitive not in info_str, f"Leaked: {sensitive}"


class TestUserDataNotInInstallDir:
    """User data MUST be outside the install/dist directory."""

    def test_user_data_not_under_project_root(self):
        """get_user_data_dir should NOT be under the project root."""
        from anylabeling.services.training_center.build_info import get_user_data_dir
        ud = str(get_user_data_dir()).lower()
        project_root = str(Path(__file__).resolve().parent.parent.parent).lower()
        # User data should be in %LOCALAPPDATA%, not in the project
        assert not ud.startswith(project_root), (
            f"User data {ud} is under project root {project_root}"
        )
