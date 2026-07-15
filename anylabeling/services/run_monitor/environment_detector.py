"""Environment detection for Run Monitor

Detects Python installations and virtual environments.
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from .models import PythonEnvironment


class EnvironmentDetector:
    """Detects Python environments in workspace"""

    def detect(self, workspace_path: Path) -> List[PythonEnvironment]:
        """
        Find all Python environments in workspace.

        Args:
            workspace_path: Root directory to search

        Returns:
            List of detected Python environments
        """
        environments = []

        # Check for common venv directory names
        venv_names = [".venv", "venv", "env"]
        for venv_name in venv_names:
            venv_path = workspace_path / venv_name
            if venv_path.exists() and venv_path.is_dir():
                env = self._check_venv(venv_path)
                if env:
                    environments.append(env)

        # Check for system Python
        system_python = self._get_system_python()
        if system_python:
            environments.append(system_python)

        return environments

    def _check_venv(self, venv_path: Path) -> Optional[PythonEnvironment]:
        """Check if directory is a valid virtual environment"""
        # Try common Python executable locations
        if sys.platform == "win32":
            python_paths = [
                venv_path / "Scripts" / "python.exe",
                venv_path / "Scripts" / "python3.exe",
            ]
        else:
            python_paths = [
                venv_path / "bin" / "python",
                venv_path / "bin" / "python3",
            ]

        for python_path in python_paths:
            if python_path.exists():
                version = self._get_python_version(python_path)
                if version:
                    return PythonEnvironment(
                        python_path=python_path,
                        version=version,
                        env_type="venv",
                        env_path=venv_path,
                        is_valid=True,
                    )

        return None

    def _get_system_python(self) -> Optional[PythonEnvironment]:
        """Get system Python installation"""
        # Use current Python executable
        python_path = Path(sys.executable)
        if not python_path.exists():
            return None

        version = self._get_python_version(python_path)
        if not version:
            return None

        return PythonEnvironment(
            python_path=python_path,
            version=version,
            env_type="system",
            env_path=None,
            is_valid=True,
        )

    def _get_python_version(self, python_path: Path) -> Optional[str]:
        """Get Python version by executing python --version"""
        try:
            result = subprocess.run(
                [str(python_path), "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                # Parse "Python 3.11.5" -> "3.11.5"
                version_str = result.stdout.strip()
                if version_str.startswith("Python "):
                    return version_str[7:]
            return None
        except Exception:
            return None
