"""Batch multi-format model export manager.

Serial execution of ExportJobs with per-format status tracking,
cancel support, and structured output directory management.
"""

import os
import sys
import json
import time
import shutil
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Callable, List, Optional

from PyQt6.QtCore import QObject, pyqtSignal


from .export_formats import (
    EXPORT_FORMATS,
    FORMAT_BY_CODE,
    FormatInfo,
    FormatStatus,
    get_format_status,
    get_missing_pip_packages,
    DEPRECATED_FORMATS,
    DEPRECATED_REDIRECT,
)
from .package_installer import PackageInstaller
from .utils import check_package_installed

# For backward compatibility
from .exporter import ExportLogRedirector


# ── Data Structures ──────────────────────────────────────────────────────

@dataclass
class BatchExportRequest:
    """Request to export one model to multiple formats."""

    source_model: str  # path to best.pt
    formats: List[str]  # list of format codes
    output_root: str  # parent directory for exports/
    imgsz: int = 640
    half: bool = False
    int8: bool = False
    dynamic: bool = False
    simplify: bool = True
    device: str = ""
    dataset: Optional[str] = None


@dataclass
class ExportJob:
    """Status of a single format export job."""

    job_id: str
    format_code: str
    status: str = "pending"  # pending|running|completed|failed|cancelled
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    progress_pct: int = 0
    started_at: Optional[float] = None
    finished_at: Optional[float] = None


# ── Format output path mapping ───────────────────────────────────────────

# Maps ultralytics format code → expected output suffix/name
# Used to locate the exported artifact after model.export() returns
FORMAT_OUTPUT_MAP = {
    format_code: info.output_path_template
    for format_code, info in FORMAT_BY_CODE.items()
}
# Ensure deprecated formats redirect correctly
FORMAT_OUTPUT_MAP["tflite"] = FORMAT_OUTPUT_MAP.get("litert", "best.litert")
FORMAT_OUTPUT_MAP["tfjs"] = FORMAT_OUTPUT_MAP.get("litert", "best.litert")


# ── BatchExportManager ───────────────────────────────────────────────────

class BatchExportManager(QObject):
    """Manages serial batch export execution."""

    # Signals for UI integration
    job_updated = pyqtSignal(ExportJob)  # emitted when any job changes
    all_completed = pyqtSignal(dict)  # summary dict
    export_log = pyqtSignal(str)  # real-time log messages
    package_install_query = pyqtSignal(list, list)  # (all_packages, large_packages) -> ask user

    def __init__(self, parent=None):
        super().__init__(parent)
        self._jobs: List[ExportJob] = []
        self._request: Optional[BatchExportRequest] = None
        self._is_running = False
        self._current_job_index = -1
        self._cancelled = False
        self._export_thread: Optional[threading.Thread] = None
        self._package_installer: Optional[PackageInstaller] = None
        self._pending_large_install: Optional[List[str]] = None
        self._install_accepted = False

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def jobs(self) -> List[ExportJob]:
        return list(self._jobs)

    def create_jobs(self, request: BatchExportRequest) -> List[ExportJob]:
        """Validate request and create export jobs."""
        jobs = []

        # Filter out deprecated formats with redirect warnings
        cleaned_formats = []
        for fmt in request.formats:
            if fmt in DEPRECATED_FORMATS:
                redirected = DEPRECATED_REDIRECT.get(fmt, "litert")
                self.export_log.emit(
                    f"WARNING: '{fmt}' is deprecated. Redirecting to '{redirected}'."
                )
                if redirected not in cleaned_formats:
                    cleaned_formats.append(redirected)
            else:
                cleaned_formats.append(fmt)

        # Deduplicate
        seen = set()
        unique_formats = []
        for fmt in cleaned_formats:
            if fmt not in seen and fmt in FORMAT_BY_CODE:
                unique_formats.append(fmt)
                seen.add(fmt)
            elif fmt not in FORMAT_BY_CODE:
                self.export_log.emit(f"WARNING: Unknown format '{fmt}' — skipping.")

        for fmt in unique_formats:
            job = ExportJob(
                job_id=str(uuid.uuid4())[:8],
                format_code=fmt,
            )
            jobs.append(job)

        return jobs

    def start_batch(self, request: BatchExportRequest) -> bool:
        """Start a batch export. Returns False if already running."""
        if self._is_running:
            return False

        self._request = request
        self._cancelled = False
        self._is_running = True
        self._current_job_index = -1

        self._jobs = self.create_jobs(request)

        # Emit all jobs as pending
        for job in self._jobs:
            self.job_updated.emit(job)

        # Check environment for all formats
        self._check_environments()
        return True

    def _check_environments(self):
        """Check each format's environment and emit status."""
        for job in self._jobs:
            info = FORMAT_BY_CODE.get(job.format_code)
            if info is None:
                job.status = "failed"
                job.error_message = f"Unknown format: {job.format_code}"
                self.job_updated.emit(job)
                continue

            status = get_format_status(info)
            if status == FormatStatus.UNSUPPORTED_PLATFORM:
                job.status = "skipped"
                job.error_message = (
                    f"Unsupported platform: {sys.platform}. "
                    f"Requires: {info.capability.supported_platforms}"
                )
                self.job_updated.emit(job)
            elif status == FormatStatus.UNSUPPORTED_DEVICE:
                job.status = "skipped"
                job.error_message = "Requires NVIDIA GPU with CUDA"
                self.job_updated.emit(job)
            # MISSING_DEPENDENCY handled before each job's execution

        # Count ready jobs
        pending_jobs = [j for j in self._jobs if j.status == "pending"]
        if pending_jobs:
            # Check if any need large package installs
            self._check_large_deps_and_proceed(pending_jobs)
        else:
            # No jobs to run — all skipped
            self._finish_batch()

    def _check_large_deps_and_proceed(self, pending_jobs: List[ExportJob]):
        """Check for large dependencies and prompt user before proceeding."""
        all_missing = []
        large_missing = []

        for job in pending_jobs:
            info = FORMAT_BY_CODE.get(job.format_code)
            if info:
                missing = get_missing_pip_packages(info)
                all_missing.extend(missing)
                from .export_formats import get_large_missing_packages
                large_missing.extend(get_large_missing_packages(info))

        # Deduplicate
        all_missing = list(dict.fromkeys(all_missing))
        large_missing = list(dict.fromkeys(large_missing))

        if large_missing:
            # Ask user for confirmation
            self._pending_large_install = all_missing
            self.package_install_query.emit(all_missing, large_missing)
            # Execution continues in _on_install_response
        elif all_missing:
            # Small packages only — auto-install (non-blocking)
            self._auto_install_and_proceed(all_missing)
        else:
            # All deps present — go!
            self._start_executing()

    def _auto_install_and_proceed(self, packages: List[str]):
        """Install small packages without user prompt."""
        if not packages:
            self._start_executing()
            return

        self.export_log.emit(f"Installing missing packages: {', '.join(packages)}")
        self._package_installer = PackageInstaller(self)
        self._package_installer.install_output.connect(self.export_log.emit)
        self._package_installer.install_finished.connect(
            lambda success, stdout, stderr: self._on_auto_install_done(success)
        )
        self._package_installer.install(packages)

    def _on_auto_install_done(self, success: bool):
        if success:
            self.export_log.emit("Package installation completed.")
            self._start_executing()
        else:
            self.export_log.emit("ERROR: Package installation failed. Some formats will fail.")
            # Still proceed — each job will handle its own deps
            self._start_executing()

    def on_install_confirmed(self, accepted: bool):
        """User responded to large-package install prompt."""
        self._install_accepted = accepted
        if accepted and self._pending_large_install:
            self._auto_install_and_proceed(self._pending_large_install)
        else:
            # User declined — skip formats needing large deps
            self._mark_large_dep_jobs_skipped()
            self._start_executing()
        self._pending_large_install = None

    def on_install_skip_format(self):
        """User chose 'Skip this format' — skip large-dep formats, proceed."""
        self._mark_large_dep_jobs_skipped()
        self._pending_large_install = None
        self._start_executing()

    def on_install_cancel_all(self):
        """User chose 'Cancel all' — abort entire batch."""
        self._pending_large_install = None
        self._cancelled = True
        for job in self._jobs:
            if job.status == "pending":
                job.status = "cancelled"
                job.error_message = "Cancelled by user"
                self.job_updated.emit(job)
        self._finish_batch()

    def _mark_large_dep_jobs_skipped(self):
        """Mark jobs that need large dependencies as skipped."""
        for job in self._jobs:
            if job.status == "pending":
                info = FORMAT_BY_CODE.get(job.format_code)
                if info:
                    from .export_formats import get_large_missing_packages
                    if get_large_missing_packages(info):
                        job.status = "skipped"
                        job.error_message = "Large dependency installation declined by user"
                        self.job_updated.emit(job)

    def _start_executing(self):
        """Begin serial execution of pending jobs."""
        if self._cancelled:
            self._finish_batch()
            return

        pending_jobs = [j for j in self._jobs if j.status == "pending"]
        if not pending_jobs:
            self._finish_batch()
            return

        self._export_thread = threading.Thread(target=self._run_serial, daemon=True)
        self._export_thread.start()

    def _run_serial(self):
        """Execute export jobs one at a time on a background thread."""
        for idx, job in enumerate(self._jobs):
            if self._cancelled:
                break
            if job.status in ("skipped", "failed", "cancelled"):
                continue

            self._current_job_index = idx
            self._execute_one_job(job)

        self._is_running = False
        self.all_completed.emit(self._build_summary())

    def _execute_one_job(self, job: ExportJob):
        """Execute a single export job."""
        job.status = "running"
        job.started_at = time.time()
        job.progress_pct = 0
        self.job_updated.emit(job)

        info = FORMAT_BY_CODE.get(job.format_code)
        if info is None:
            job.status = "failed"
            job.error_message = f"Unknown format: {job.format_code}"
            self.job_updated.emit(job)
            return

        # Check deps
        missing = get_missing_pip_packages(info)
        if missing:
            job.status = "failed"
            job.error_message = f"Missing packages: {', '.join(missing)}"
            self.job_updated.emit(job)
            return

        # Build output directory
        if self._request is None:
            job.status = "failed"
            job.error_message = "No export request"
            self.job_updated.emit(job)
            return

        output_dir = os.path.join(
            self._request.output_root,
            job.format_code,
        )
        os.makedirs(output_dir, exist_ok=True)

        self.export_log.emit(f"[{job.format_code}] Starting export...")
        job.progress_pct = 10
        self.job_updated.emit(job)

        # Run ultralytics export
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        log_redirector = ExportLogRedirector()
        log_redirector.log_signal.connect(self.export_log.emit)
        sys.stdout = log_redirector
        sys.stderr = log_redirector

        try:
            from ultralytics import YOLO

            model = YOLO(self._request.source_model)
            job.progress_pct = 30
            self.job_updated.emit(job)

            # Build export kwargs
            export_kwargs = {"format": job.format_code}

            # Only pass parameters the format supports
            if info.capability.supports_half and self._request.half:
                export_kwargs["half"] = True
            if info.capability.supports_int8 and self._request.int8:
                export_kwargs["int8"] = True
            if info.capability.supports_dynamic and self._request.dynamic:
                export_kwargs["dynamic"] = True
            if info.capability.supports_simplify and self._request.simplify:
                export_kwargs["simplify"] = True

            export_kwargs["imgsz"] = self._request.imgsz
            if self._request.device:
                export_kwargs["device"] = self._request.device

            # For int8 calibration, use the dataset if provided
            if self._request.int8 and self._request.dataset:
                export_kwargs["data"] = self._request.dataset

            job.progress_pct = 50
            self.job_updated.emit(job)

            # Run export — output goes to the model's directory
            result = model.export(**export_kwargs)
            exported_path = result if isinstance(result, str) else str(result)
            job.progress_pct = 80
            self.job_updated.emit(job)

            # Move exported artifact to our organized output directory
            if exported_path and os.path.exists(exported_path):
                expected_name = info.output_path_template
                dest_path = os.path.join(output_dir, expected_name)

                if os.path.isdir(exported_path):
                    # Directory format: copy to dest
                    if os.path.exists(dest_path):
                        shutil.rmtree(dest_path, ignore_errors=True)
                    shutil.copytree(exported_path, dest_path)
                else:
                    # File format: move to dest
                    os.makedirs(output_dir, exist_ok=True)
                    if os.path.exists(dest_path):
                        os.remove(dest_path)
                    shutil.move(exported_path, dest_path)

                # Verify destination exists
                if os.path.exists(dest_path):
                    job.output_path = dest_path
                    job.status = "completed"
                    job.progress_pct = 100
                    self.export_log.emit(
                        f"[{job.format_code}] Export completed → {dest_path}"
                    )
                else:
                    job.status = "failed"
                    job.error_message = (
                        f"Failed to move exported artifact to {dest_path}"
                    )
                    self.export_log.emit(
                        f"[{job.format_code}] ERROR: Could not move output to {dest_path}"
                    )
            else:
                # Ultralytics may have written to a default location
                # Try to find the output using our output path mapping
                expected_name = info.output_path_template
                weights_dir = os.path.dirname(self._request.source_model)
                possible_path = os.path.join(weights_dir, expected_name)

                if os.path.exists(possible_path):
                    dest_path = os.path.join(output_dir, expected_name)
                    if os.path.isdir(possible_path):
                        if os.path.exists(dest_path):
                            shutil.rmtree(dest_path, ignore_errors=True)
                        shutil.copytree(possible_path, dest_path)
                    else:
                        os.makedirs(output_dir, exist_ok=True)
                        if os.path.exists(dest_path):
                            os.remove(dest_path)
                        shutil.move(possible_path, dest_path)

                    if os.path.exists(dest_path):
                        job.output_path = dest_path
                        job.status = "completed"
                        job.progress_pct = 100
                        self.export_log.emit(
                            f"[{job.format_code}] Export completed (found at {possible_path}) → {dest_path}"
                        )
                    else:
                        job.status = "failed"
                        job.error_message = "Could not locate exported file"
                        self.export_log.emit(
                            f"[{job.format_code}] ERROR: Exported file not found"
                        )
                else:
                    # Last attempt: check for the extension-less path
                    model_name = os.path.splitext(
                        os.path.basename(self._request.source_model)
                    )[0]
                    # Try directory suffix pattern
                    alt_path = os.path.join(
                        weights_dir,
                        f"{model_name}_{job.format_code}_model",
                    )
                    if not os.path.exists(alt_path):
                        alt_path = os.path.join(
                            weights_dir,
                            f"{model_name}_model",
                        )

                    if os.path.exists(alt_path):
                        dest_path = os.path.join(output_dir, expected_name)
                        if os.path.isdir(alt_path):
                            if os.path.exists(dest_path):
                                shutil.rmtree(dest_path, ignore_errors=True)
                            shutil.copytree(alt_path, dest_path)
                            if os.path.exists(dest_path):
                                job.output_path = dest_path
                                job.status = "completed"
                                job.progress_pct = 100
                            else:
                                job.status = "failed"
                                job.error_message = "Could not copy output"
                        else:
                            os.makedirs(output_dir, exist_ok=True)
                            shutil.move(alt_path, dest_path)
                            job.output_path = dest_path
                            job.status = "completed"
                            job.progress_pct = 100
                    else:
                        job.status = "failed"
                        job.error_message = "Export completed but output file not found"
                        self.export_log.emit(
                            f"[{job.format_code}] ERROR: Output file not found at any expected location"
                        )

        except ImportError as e:
            job.status = "failed"
            job.error_message = f"Import error: {e}"
            self.export_log.emit(f"[{job.format_code}] ERROR: {e}")
        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            self.export_log.emit(f"[{job.format_code}] ERROR: {e}")
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            job.finished_at = time.time()
            self.job_updated.emit(job)

    def cancel_batch(self):
        """Cancel the remaining batch. Current job will still complete."""
        self._cancelled = True
        for job in self._jobs:
            if job.status in ("pending",):
                job.status = "cancelled"
                job.error_message = "Cancelled by user"
                self.job_updated.emit(job)
        if self._package_installer and self._package_installer.is_running:
            self._package_installer.cancel()

    def _build_summary(self) -> dict:
        """Build summary dictionary for all_completed signal."""
        completed = [j for j in self._jobs if j.status == "completed"]
        failed = [j for j in self._jobs if j.status == "failed"]
        skipped = [j for j in self._jobs if j.status == "skipped"]
        cancelled = [j for j in self._jobs if j.status == "cancelled"]
        return {
            "total": len(self._jobs),
            "completed": len(completed),
            "failed": len(failed),
            "skipped": len(skipped),
            "cancelled": len(cancelled),
            "jobs": [asdict(j) for j in self._jobs],
        }

    def _finish_batch(self):
        """Finish batch (all skipped or cancelled from start)."""
        self._is_running = False
        self.all_completed.emit(self._build_summary())

    def generate_manifest(self, output_root: str) -> str:
        """Generate export_manifest.json in the output root directory."""
        if self._request is None:
            return ""

        # Try to read class names from source model
        classes = []
        task = "unknown"
        try:
            from ultralytics import YOLO
            model = YOLO(self._request.source_model)
            if hasattr(model, "names") and model.names:
                classes = list(model.names.values()) if isinstance(model.names, dict) else list(model.names)
            if hasattr(model, "task") and model.task:
                task = model.task
        except Exception:
            pass

        manifest = {
            "source_model": self._request.source_model,
            "task": task,
            "created_at": datetime.now().isoformat(),
            "imgsz": self._request.imgsz,
            "classes": classes,
            "requested_formats": self._request.formats,
            "exports": [],
        }

        for job in self._jobs:
            entry = {
                "format": job.format_code,
                "status": job.status,
            }
            if job.output_path:
                # Store path relative to output_root
                entry["output"] = os.path.relpath(job.output_path, output_root)
            if job.error_message:
                entry["error"] = job.error_message
            manifest["exports"].append(entry)

        manifest_path = os.path.join(output_root, "export_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        return manifest_path


# ── Singleton ────────────────────────────────────────────────────────────

_batch_export_manager: Optional[BatchExportManager] = None


def get_batch_export_manager() -> BatchExportManager:
    """Get or create the singleton BatchExportManager."""
    global _batch_export_manager
    if _batch_export_manager is None:
        _batch_export_manager = BatchExportManager()
    return _batch_export_manager
