"""TrainLens Preflight — background worker (QObject, runs on QThread)."""

import os
import time
from enum import Enum

from PyQt6.QtCore import QObject, pyqtSignal

from . import PreflightIssue, PreflightResult, PreflightSeverity
from .common_checks import (
    check_active_job, check_config_not_empty, check_disk_space,
    check_output_directory, check_positive_int,
)
from .guided_checks import (
    check_device, check_device_with_runtime, check_model, check_python_packages,
)
from .guided_yaml import (
    check_yaml_dataset_paths, check_yaml_structure, read_yaml_safe,
)
from .guided_detect_labels import check_detect_labels
from .custom_checks import (
    check_environment_busy, check_python_executable, check_script,
    check_working_directory, check_workspace,
)
from .models import CustomPreflightContext, GuidedPreflightContext


class PreflightTask(str, Enum):
    GUIDED = "guided"
    CUSTOM = "custom"
    CANCEL = "cancel"


class PreflightWorker(QObject):
    """Background worker for preflight checks. No widget access."""

    # Progress signals
    progress = pyqtSignal(str)          # status message
    issue_found = pyqtSignal(object)    # PreflightIssue (for live feedback)

    # Completion signals
    finished = pyqtSignal(object)       # PreflightResult
    error = pyqtSignal(str)             # fatal error message

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cancelled = False

    def cancel(self):
        """Request cancellation. Worker checks this between steps."""
        self._cancelled = True

    def run_guided(self, ctx: GuidedPreflightContext, is_active_job: bool):
        """Run all Guided preflight checks."""
        result = PreflightResult(
            mode="guided",
            started_at=time.time(),
            config_fingerprint=ctx.fingerprint(),
        )

        try:
            self._run_guided_checks(result, ctx, is_active_job)
        except Exception as e:
            result.add(PreflightIssue(
                code="PREFLIGHT_ERROR",
                severity=PreflightSeverity.ERROR,
                title="Preflight check failed",
                message=f"Internal error during preflight: {e}",
                details=str(e),
            ))

        if self._cancelled:
            result.cancelled = True
        result.finished_at = time.time()
        self.finished.emit(result)

    def run_custom(self, ctx: CustomPreflightContext, is_active_job: bool):
        """Run all Custom preflight checks."""
        result = PreflightResult(
            mode="custom",
            started_at=time.time(),
            config_fingerprint=ctx.fingerprint(),
        )

        try:
            self._run_custom_checks(result, ctx, is_active_job)
        except Exception as e:
            result.add(PreflightIssue(
                code="PREFLIGHT_ERROR",
                severity=PreflightSeverity.ERROR,
                title="Preflight check failed",
                message=f"Internal error during preflight: {e}",
                details=str(e),
            ))

        if self._cancelled:
            result.cancelled = True
        result.finished_at = time.time()
        self.finished.emit(result)

    # ── Guided checks ────────────────────────────────────────────────

    def _run_guided_checks(self, result: PreflightResult, ctx: GuidedPreflightContext, is_active_job: bool):
        # 1. Active job
        self.progress.emit("Checking for active training tasks...")
        check_active_job(result, is_active_job)
        if self._cancelled:
            return

        # 2. Python packages
        self.progress.emit("Checking Python packages...")
        check_python_packages(result)
        if self._cancelled:
            return

        # 3. Model
        self.progress.emit("Checking model file...")
        check_config_not_empty(result, "model", ctx.model_path)
        check_model(result, ctx.model_path, ctx.task_type)
        if self._cancelled:
            return

        # 4. Device — use runtime-aware check if external runtime available
        self.progress.emit("Checking device configuration...")
        runtime_python = getattr(ctx, "runtime_python", "") or ""
        if runtime_python and os.path.isfile(runtime_python):
            check_device_with_runtime(result, ctx)
        else:
            check_device(result, ctx.device)
        if self._cancelled:
            return

        # 5. Epochs, batch, imgsz
        self.progress.emit("Checking training parameters...")
        check_positive_int(result, "epochs", ctx.epochs)
        check_positive_int(result, "batch", ctx.batch)
        check_positive_int(result, "imgsz", ctx.imgsz)
        if self._cancelled:
            return

        # 6. Output directory
        self.progress.emit("Checking output directory...")
        check_output_directory(result, ctx.output_dir)
        check_disk_space(result, ctx.output_dir)
        if self._cancelled:
            return

        # 7. Dataset YAML
        self.progress.emit("Reading dataset YAML...")
        check_config_not_empty(result, "dataset_yaml", ctx.dataset_yaml)
        yaml_data, yaml_error = read_yaml_safe(ctx.dataset_yaml)
        if yaml_error:
            result.add(PreflightIssue(
                code="YAML_READ_ERROR", severity=PreflightSeverity.ERROR,
                title="Cannot read dataset YAML",
                message=yaml_error,
                path=ctx.dataset_yaml,
            ))
            return
        if self._cancelled:
            return

        check_yaml_structure(result, ctx.dataset_yaml, yaml_data)
        if self._cancelled:
            return

        check_yaml_dataset_paths(result, ctx.dataset_yaml, yaml_data)
        if self._cancelled:
            return

        # 8. Detect label validation
        self.progress.emit("Scanning training labels...")
        check_detect_labels(result, ctx.dataset_yaml, yaml_data, ctx.task_type)
        if self._cancelled:
            return

        self.progress.emit("Finalizing report...")

    # ── Custom checks ────────────────────────────────────────────────

    def _run_custom_checks(self, result: PreflightResult, ctx: CustomPreflightContext, is_active_job: bool):
        # 1. Active job
        self.progress.emit("Checking for active training tasks...")
        check_active_job(result, is_active_job)
        if self._cancelled:
            return

        # 2. Environment busy
        self.progress.emit("Checking environment status...")
        check_environment_busy(result, ctx.environment_busy)
        if self._cancelled:
            return

        # 3. Project directory
        self.progress.emit("Checking project directory...")
        check_workspace(result, ctx.project_dir)
        if self._cancelled:
            return

        # 4. Python
        self.progress.emit("Checking Python...")
        check_python_executable(result, ctx.python_path)
        if self._cancelled:
            return

        # 5. Script
        self.progress.emit("Checking training script...")
        check_script(result, ctx.script_path, ctx.project_dir)
        if self._cancelled:
            return

        # 6. Working directory
        if ctx.working_directory:
            self.progress.emit("Checking working directory...")
            check_working_directory(result, ctx.working_directory)
            if self._cancelled:
                return

        # 7. Output directory (optional)
        if ctx.output_dir:
            self.progress.emit("Checking output directory...")
            check_output_directory(result, ctx.output_dir)
            check_disk_space(result, ctx.output_dir)
            if self._cancelled:
                return

        self.progress.emit("Finalizing report...")
