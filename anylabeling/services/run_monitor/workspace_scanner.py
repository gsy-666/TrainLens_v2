"""Workspace scanner for Run Monitor

Scans workspace directories to detect training scripts and environments.
"""

from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from .environment_detector import EnvironmentDetector
from .models import Workspace
from .script_detector import ScriptDetector


class WorkspaceScanner:
    """Scans workspace for training scripts and Python environments"""

    # Directories to exclude from scanning
    EXCLUDE_DIRS = {
        ".git",
        "__pycache__",
        "node_modules",
        "build",
        "dist",
        ".venv",
        "venv",
        "env",
        ".eggs",
        "*.egg-info",
        ".tox",
        ".pytest_cache",
        ".mypy_cache",
        ".cache",
        "runs",
        "checkpoints",
        "logs",
    }

    # Maximum files to scan (safety limit)
    MAX_FILES_TO_SCAN = 10000

    def __init__(self):
        self.script_detector = ScriptDetector()
        self.environment_detector = EnvironmentDetector()
        self._cancelled = False

    def scan(
        self,
        workspace_path: Path,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Workspace:
        """
        Scan workspace directory for training scripts and environments.

        Args:
            workspace_path: Root directory to scan
            progress_callback: Optional callback(current, total, message)

        Returns:
            Workspace object with detected scripts and environments
        """
        if not workspace_path.exists() or not workspace_path.is_dir():
            raise ValueError(f"Invalid workspace path: {workspace_path}")

        self._cancelled = False
        workspace = Workspace(
            path=workspace_path,
            scan_timestamp=datetime.now(),
        )

        # Detect environments first (fast)
        if progress_callback:
            progress_callback(0, 2, "Detecting Python environments...")

        workspace.detected_environments = self.environment_detector.detect(
            workspace_path
        )

        if self._cancelled:
            return workspace

        # Scan for Python scripts
        if progress_callback:
            progress_callback(1, 2, "Scanning for training scripts...")

        python_files = self._find_python_files(workspace_path)
        total_files = len(python_files)

        for i, script_path in enumerate(python_files):
            if self._cancelled:
                break

            if progress_callback and i % 10 == 0:
                progress_callback(
                    i,
                    total_files,
                    f"Analyzing {script_path.name}...",
                )

            detected = self.script_detector.detect(script_path)
            if detected:
                workspace.detected_scripts.append(detected)

        if progress_callback:
            progress_callback(total_files, total_files, "Scan complete")

        return workspace

    def cancel(self):
        """Cancel ongoing scan"""
        self._cancelled = True

    def _find_python_files(self, workspace_path: Path) -> list[Path]:
        """Find all Python files in workspace, excluding common directories"""
        python_files = []
        files_scanned = 0

        try:
            for path in workspace_path.rglob("*.py"):
                if files_scanned >= self.MAX_FILES_TO_SCAN:
                    break

                # Check if file is in excluded directory
                if self._should_exclude(path, workspace_path):
                    continue

                python_files.append(path)
                files_scanned += 1

        except PermissionError:
            # Skip directories without permission
            pass

        return python_files

    def _should_exclude(self, path: Path, workspace_path: Path) -> bool:
        """Check if path should be excluded from scanning"""
        try:
            relative_path = path.relative_to(workspace_path)
            parts = relative_path.parts

            for part in parts:
                if part in self.EXCLUDE_DIRS:
                    return True
                # Check for pattern matches (e.g., *.egg-info)
                for exclude_pattern in self.EXCLUDE_DIRS:
                    if "*" in exclude_pattern:
                        pattern = exclude_pattern.replace("*", "")
                        if pattern in part:
                            return True
            return False
        except ValueError:
            return True
