"""TrainLens Environment Wizard — background worker (QObject for QThread)."""

from enum import Enum
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QObject, pyqtSignal

from .detector import find_project_python, find_system_python_for_venv, inspect_python_environment
from .models import EnvironmentInfo, EnvironmentStatus
from .operations import create_project_venv, ensure_pip, install_requirements


class EnvironmentTask(str, Enum):
    DETECT = "detect"
    CREATE_VENV = "create_venv"
    INSTALL_REQUIREMENTS = "install_requirements"


class EnvironmentWorker(QObject):
    """Background worker for environment operations.

    Runs on a QThread. Communicates via Qt signals only — never touches widgets.
    """

    # Result signals
    detection_done = pyqtSignal(object)  # EnvironmentInfo
    venv_created = pyqtSignal(bool, str, str)  # success, venv_python_path, message
    requirements_done = pyqtSignal(bool, str)  # success, message

    # Progress signals
    log_message = pyqtSignal(str)
    status_changed = pyqtSignal(str)  # EnvironmentStatus value

    # Lifecycle signal — emitted after run() completes (success or failure)
    finished = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._task: Optional[EnvironmentTask] = None
        self._project_dir: Optional[Path] = None
        self._python_path: Optional[Path] = None
        self._generation: int = 0  # incremented per task to prevent stale results

    # ── public API (called from main thread) ────────────────────────────

    def request_detect(self, python_path: str, project_dir: str = "", generation: int = 0):
        """Request environment detection."""
        self._task = EnvironmentTask.DETECT
        self._python_path = Path(python_path) if python_path else None
        self._project_dir = Path(project_dir) if project_dir else None
        self._generation = generation

    def request_create_venv(self, project_dir: str, generation: int = 0):
        """Request .venv creation."""
        self._task = EnvironmentTask.CREATE_VENV
        self._project_dir = Path(project_dir) if project_dir else None
        self._generation = generation

    def request_install_requirements(self, python_path: str, requirements_path: str, generation: int = 0):
        """Request requirements installation."""
        self._task = EnvironmentTask.INSTALL_REQUIREMENTS
        self._python_path = Path(python_path) if python_path else None
        self._project_dir = Path(requirements_path).parent if requirements_path else None
        self._requirements_path = Path(requirements_path) if requirements_path else None
        self._generation = generation

    # ── main entry (called from QThread) ────────────────────────────────

    def run(self):
        """Execute the requested task. Called after moveToThread + started signal."""
        try:
            if self._task == EnvironmentTask.DETECT:
                self._run_detect()
            elif self._task == EnvironmentTask.CREATE_VENV:
                self._run_create_venv()
            elif self._task == EnvironmentTask.INSTALL_REQUIREMENTS:
                self._run_install_requirements()
        finally:
            self.finished.emit()

    # ── task implementations ────────────────────────────────────────────

    def _run_detect(self):
        if not self._python_path:
            self.detection_done.emit(_make_error("No Python path provided"))
            return

        self.status_changed.emit(EnvironmentStatus.CHECKING.value)
        info = inspect_python_environment(self._python_path)
        self.detection_done.emit(info)

    def _run_create_venv(self):
        if not self._project_dir:
            self.venv_created.emit(False, "", "No project directory provided")
            return

        self.status_changed.emit(EnvironmentStatus.CREATING.value)

        system_python = find_system_python_for_venv()
        if not system_python:
            self.venv_created.emit(
                False, "",
                "No usable system Python found. Install Python 3.10–3.12 or select an existing interpreter.",
            )
            return

        def log(msg: str):
            self.log_message.emit(msg)

        ok, venv_path, msg = create_project_venv(self._project_dir, system_python, log)
        self.venv_created.emit(ok, str(venv_path) if ok else "", msg)

    def _run_install_requirements(self):
        req_path = getattr(self, '_requirements_path', None)
        if not req_path:
            self.requirements_done.emit(False, "No requirements.txt path provided")
            return
        if not self._python_path:
            self.requirements_done.emit(False, "No Python path provided")
            return
        if not req_path.exists():
            self.requirements_done.emit(False, f"requirements.txt not found: {req_path}")
            return

        self.status_changed.emit(EnvironmentStatus.INSTALLING.value)

        def log(msg: str):
            self.log_message.emit(msg)

        ok, msg = install_requirements(self._python_path, req_path, log)
        self.requirements_done.emit(ok, msg)


def _make_error(message: str) -> EnvironmentInfo:
    info = EnvironmentInfo()
    info.status = EnvironmentStatus.ERROR
    info.error_message = message
    return info
