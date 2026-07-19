"""Training Center - Job History Store

Persistent JSONL-based storage for training job history.
Thread-safe with atomic writes and corruption resistance.
"""

import csv
import json
import os
import shutil
import threading
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime

from .models import TrainingMode, TrainingStatus


@dataclass
class JobHistoryRecord:
    """Complete training job history record"""
    job_id: str
    mode: str  # TrainingMode value
    status: str  # TrainingStatus value
    created_at: str  # ISO 8601
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    workspace: Optional[str] = None
    output_directory: Optional[str] = None
    display_name: Optional[str] = None
    framework: Optional[str] = None
    python_executable: Optional[str] = None
    command: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

    # Training config (for Guided display)
    task: Optional[str] = None
    model: Optional[str] = None
    data: Optional[str] = None
    project: Optional[str] = None

    # Runtime (GPU environment)
    runtime_id: Optional[str] = None
    runtime_python: Optional[str] = None
    requested_device: Optional[str] = None
    resolved_device: Optional[str] = None
    execution_mode: Optional[str] = None

    # Epochs & metrics
    requested_epochs: Optional[int] = None
    completed_epochs: Optional[int] = None
    best_epoch: Optional[int] = None
    best_metric_name: Optional[str] = None
    best_metric_value: Optional[float] = None

    # Legacy metrics (kept for backward compat)
    total_epochs: Optional[int] = None
    final_epoch: Optional[int] = None
    best_map50: Optional[float] = None
    best_map50_95: Optional[float] = None
    final_train_loss: Optional[float] = None
    final_val_loss: Optional[float] = None

    # Timing
    duration_seconds: Optional[float] = None
    avg_epoch_seconds: Optional[float] = None

    # Resources
    peak_memory_mb: Optional[float] = None
    peak_cpu_percent: Optional[float] = None
    peak_gpu_percent: Optional[float] = None

    # Artifacts
    checkpoint_path: Optional[str] = None
    results_path: Optional[str] = None
    export_path: Optional[str] = None
    best_weights_path: Optional[str] = None
    last_weights_path: Optional[str] = None
    results_csv_path: Optional[str] = None
    results_image_path: Optional[str] = None
    log_path: Optional[str] = None

    # Final state
    exit_code: Optional[int] = None
    final_status: Optional[str] = None
    dataset_yaml: Optional[str] = None
    model_name: Optional[str] = None
    project_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobHistoryRecord":
        """Create from dictionary"""
        return cls(**data)


class HistoryStore:
    """Thread-safe JSONL-based job history storage

    Format: One JSON object per line in index.jsonl
    Location: <appdata>/trainlens/history/index.jsonl
    """

    def __init__(self, history_dir: Optional[Path] = None):
        """Initialize history store

        Args:
            history_dir: Custom directory (for testing). Defaults to user config.
        """
        if history_dir is None:
            history_dir = self._get_default_history_dir()

        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(parents=True, exist_ok=True)

        self.index_file = self.history_dir / "index.jsonl"
        self._lock = threading.Lock()
        self._cache: Dict[str, JobHistoryRecord] = {}
        self._loaded = False

    def _get_default_history_dir(self) -> Path:
        """Get default history directory from user config"""
        import os
        import platform

        system = platform.system()
        if system == "Windows":
            base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        else:
            base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))

        return base / "trainlens" / "history"

    def _ensure_loaded(self):
        """Lazy load index into cache"""
        if self._loaded:
            return

        if not self.index_file.exists():
            self._loaded = True
            return

        try:
            with open(self.index_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                        record = JobHistoryRecord.from_dict(data)
                        self._cache[record.job_id] = record
                    except (json.JSONDecodeError, TypeError, KeyError) as e:
                        warnings.warn(f"Skipping corrupted line {line_num} in history: {e}")
        except Exception as e:
            warnings.warn(f"Failed to load history: {e}")

        self._loaded = True

    def _write_index(self):
        """Atomically rewrite entire index"""
        temp_file = self.index_file.with_suffix(".tmp")

        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                for record in self._cache.values():
                    json.dump(record.to_dict(), f, ensure_ascii=False)
                    f.write("\n")

            # Atomic replace
            shutil.move(str(temp_file), str(self.index_file))
        except Exception:
            if temp_file.exists():
                temp_file.unlink()
            raise

    def append_job(self, record: JobHistoryRecord):
        """Add new job to history

        Args:
            record: Job history record
        """
        with self._lock:
            self._ensure_loaded()
            self._cache[record.job_id] = record
            self._write_index()

    def update_job(self, job_id: str, **updates):
        """Update existing job record

        Args:
            job_id: Job ID to update
            **updates: Fields to update

        Raises:
            KeyError: If job not found
        """
        with self._lock:
            self._ensure_loaded()

            if job_id not in self._cache:
                raise KeyError(f"Job {job_id} not found in history")

            record = self._cache[job_id]

            # Update fields
            for key, value in updates.items():
                if hasattr(record, key):
                    setattr(record, key, value)

            self._write_index()

    def finalize_job(
        self,
        job_id: str,
        status: TrainingStatus,
        ended_at: Optional[datetime] = None,
        error_message: Optional[str] = None,
        **metrics
    ):
        """Mark job as terminal and record final metrics.

        Idempotent: preserves existing terminal state.
        Auto-parses results.csv for epochs and best metric.

        Args:
            job_id: Job ID to finalize
            status: Terminal status
            ended_at: End timestamp
            error_message: Error message if failed
            **metrics: Additional metrics to record
        """
        if not status.is_terminal():
            raise ValueError(f"finalize_job requires terminal status, got {status}")

        with self._lock:
            self._ensure_loaded()

            if job_id not in self._cache:
                raise KeyError(f"Job {job_id} not found in history")

            record = self._cache[job_id]

            # Idempotent: don't overwrite existing terminal state
            try:
                current_status = TrainingStatus(record.status)
            except ValueError:
                current_status = None
            if current_status and current_status.is_terminal():
                return

            # Update terminal fields
            record.status = status.value
            record.final_status = status.value
            if ended_at:
                record.ended_at = ended_at.isoformat()

            if error_message:
                record.error_message = error_message

            # Write back real output directory
            output_directory = metrics.pop('output_directory', None)
            if output_directory:
                record.output_directory = str(output_directory)

            # Calculate duration
            if record.started_at:
                try:
                    start = datetime.fromisoformat(record.started_at)
                    end = ended_at if ended_at else datetime.now()
                    record.duration_seconds = max(0, (end - start).total_seconds())
                except (ValueError, TypeError):
                    pass

            # ── Parse results.csv for epochs and best metric ──
            out_dir = record.output_directory
            if out_dir and os.path.isdir(out_dir):
                results_csv = os.path.join(out_dir, "results.csv")
                if os.path.isfile(results_csv):
                    try:
                        epochs_metrics = _parse_results_csv(results_csv)
                        if epochs_metrics:
                            record.completed_epochs = len(epochs_metrics)
                            record.results_csv_path = results_csv
                            # Find best epoch by mAP50(B)
                            best = _find_best_epoch(epochs_metrics)
                            if best:
                                record.best_epoch = best["epoch"]
                                record.best_metric_name = best["metric_name"]
                                record.best_metric_value = best["metric_value"]
                    except Exception:
                        pass

                # Check for weights
                best_pt = os.path.join(out_dir, "weights", "best.pt")
                if os.path.isfile(best_pt):
                    record.best_weights_path = best_pt
                last_pt = os.path.join(out_dir, "weights", "last.pt")
                if os.path.isfile(last_pt):
                    record.last_weights_path = last_pt

                # results.png
                results_png = os.path.join(out_dir, "results.png")
                if os.path.isfile(results_png):
                    record.results_image_path = results_png

                # Legacy: also set for backward compat
                if record.best_map50 is None and record.best_metric_name == "metrics/mAP50(B)":
                    record.best_map50 = record.best_metric_value
                if record.best_map50_95 is None and record.best_metric_name == "metrics/mAP50-95(B)":
                    record.best_map50_95 = record.best_metric_value
                if record.final_epoch is None:
                    record.final_epoch = record.completed_epochs
                if record.total_epochs is None:
                    record.total_epochs = record.requested_epochs

            # Update metrics
            for key, value in metrics.items():
                if hasattr(record, key):
                    setattr(record, key, value)

            self._write_index()

    def get_job(self, job_id: str) -> Optional[JobHistoryRecord]:
        """Retrieve job by ID

        Args:
            job_id: Job ID

        Returns:
            Job record or None if not found
        """
        with self._lock:
            self._ensure_loaded()
            return self._cache.get(job_id)

    def list_jobs(
        self,
        mode: Optional[TrainingMode] = None,
        status: Optional[TrainingStatus] = None,
        limit: Optional[int] = None,
    ) -> List[JobHistoryRecord]:
        """List jobs with optional filtering

        Args:
            mode: Filter by training mode
            status: Filter by status
            limit: Maximum number of results (most recent first)

        Returns:
            List of job records
        """
        with self._lock:
            self._ensure_loaded()

            results = list(self._cache.values())

            # Filter by mode
            if mode is not None:
                results = [r for r in results if r.mode == mode.value]

            # Filter by status
            if status is not None:
                results = [r for r in results if r.status == status.value]

            # Sort by created_at descending (most recent first)
            results.sort(key=lambda r: r.created_at, reverse=True)

            # Apply limit
            if limit is not None:
                results = results[:limit]

            return results

    def clear(self):
        """Clear all history (for testing)"""
        with self._lock:
            self._cache.clear()
            if self.index_file.exists():
                self.index_file.unlink()
            self._loaded = True

    def delete_job(self, job_id: str):
        """Delete a single job record from history.

        Args:
            job_id: Job ID to delete
        """
        with self._lock:
            self._ensure_loaded()
            if job_id in self._cache:
                del self._cache[job_id]
                self._write_index()


# Singleton instance
_store: Optional[HistoryStore] = None
_store_lock = threading.Lock()


def get_history_store(history_dir: Optional[Path] = None) -> HistoryStore:
    """Get global history store singleton

    Args:
        history_dir: Custom directory (for testing)

    Returns:
        HistoryStore instance
    """
    global _store

    with _store_lock:
        if _store is None or history_dir is not None:
            _store = HistoryStore(history_dir=history_dir)
        return _store


# ── Results CSV parsing ───────────────────────────────────────────

def _parse_results_csv(csv_path: str) -> list:
    """Parse ultralytics results.csv into a list of per-epoch metric dicts.

    Returns empty list on any error.
    """
    rows = []
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Strip whitespace from keys
                cleaned = {k.strip(): v.strip() for k, v in row.items()}
                rows.append(cleaned)
    except Exception:
        pass
    return rows


def _find_best_epoch(epochs: list) -> dict | None:
    """Find the epoch with the best mAP50(B) value.

    Returns dict with epoch, metric_name, metric_value or None.
    """
    if not epochs:
        return None

    # Priority metrics for Detect tasks
    candidates = [
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ]

    best = None
    for metric in candidates:
        for i, row in enumerate(epochs):
            val_str = row.get(metric, "").strip()
            if not val_str:
                continue
            try:
                val = float(val_str)
                epoch_num = i + 1  # 1-based epoch
                if best is None or val > best["metric_value"]:
                    best = {
                        "epoch": epoch_num,
                        "metric_name": metric,
                        "metric_value": round(val, 4),
                    }
            except (ValueError, TypeError):
                continue
        if best is not None:
            break

    return best


def format_duration(seconds: float | None) -> str:
    """Format duration for display.

    < 60s: "23s"
    < 1h:   "4m 12s"
    >= 1h:  "1h 08m"
    """
    if seconds is None or seconds <= 0:
        return ""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins:02d}m"
