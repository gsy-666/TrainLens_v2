"""Training Center - Job History Store

Persistent JSONL-based storage for training job history.
Thread-safe with atomic writes and corruption resistance.
"""

import json
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

    # Metrics summary
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
        """Mark job as terminal and record final metrics

        Idempotent: preserves existing terminal state.

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
            current_status = TrainingStatus(record.status)
            if current_status.is_terminal():
                return

            # Update terminal fields
            record.status = status.value
            if ended_at:
                record.ended_at = ended_at.isoformat()

            if error_message:
                record.error_message = error_message

            # Calculate duration
            if record.started_at and record.ended_at:
                start = datetime.fromisoformat(record.started_at)
                end = datetime.fromisoformat(record.ended_at)
                record.duration_seconds = (end - start).total_seconds()

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
