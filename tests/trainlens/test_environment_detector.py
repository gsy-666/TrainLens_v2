"""Test environment detector"""

import sys
import pytest
from pathlib import Path

from anylabeling.services.run_monitor.environment_detector import EnvironmentDetector


def test_environment_detector_system_python():
    """Test system Python detection"""
    detector = EnvironmentDetector()

    # System Python should always be detected
    envs = detector.detect(Path.cwd())

    system_envs = [e for e in envs if e.env_type == "system"]
    assert len(system_envs) > 0
    assert system_envs[0].is_valid
    assert system_envs[0].version


def test_environment_detector_venv(tmp_path):
    """Test virtual environment detection"""
    detector = EnvironmentDetector()

    # Create fake venv structure
    if sys.platform == "win32":
        venv_python = tmp_path / ".venv" / "Scripts" / "python.exe"
    else:
        venv_python = tmp_path / ".venv" / "bin" / "python"

    venv_python.parent.mkdir(parents=True, exist_ok=True)

    # Create symlink to real Python
    try:
        venv_python.symlink_to(sys.executable)
    except OSError:
        # On Windows without admin rights, copy instead
        import shutil
        shutil.copy(sys.executable, venv_python)

    envs = detector.detect(tmp_path)

    venv_envs = [e for e in envs if e.env_type == "venv"]
    if venv_envs:
        assert venv_envs[0].env_path == tmp_path / ".venv"


def test_environment_detector_empty_workspace(tmp_path):
    """Test detection in empty workspace"""
    detector = EnvironmentDetector()

    envs = detector.detect(tmp_path)

    # Should at least find system Python
    assert len(envs) > 0
    system_envs = [e for e in envs if e.env_type == "system"]
    assert len(system_envs) > 0
