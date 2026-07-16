"""Training Center - Job Manager

Global singleton that coordinates all training jobs across different modes.
Enforces mutual exclusion and manages job lifecycle.
"""

import threading
import time
from typing import Optional, Tuple, Dict, Any, Callable, List
from pathlib import Path

from .models import TrainingJob, TrainingMode, TrainingStatus
from .adapters.base import TrainingAdapter
from .event_protocol import TrainingEvent, TrainingEventType


class JobManager:
    """Global singleton coordinator for all training jobs

    Responsibilities:
    - Enforce mutual exclusion (only one job active at a time)
    - Manage job lifecycle state transitions
    - Route events from adapters to UI subscribers
    - Provide idempotent terminal event handlers
    """

    _instance: Optional['JobManager'] = None
    _lock = threading.Lock()

    def __new__(cls):
        """Thread-safe singleton creation"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize singleton (only once)"""
        if self._initialized:
            return

        self._state_lock = threading.Lock()
        self._current_job: Optional[TrainingJob] = None
        self._current_adapter: Optional[TrainingAdapter] = None
        self._status_callbacks: List[Callable[[TrainingJob], None]] = []
        self._event_callbacks: List[Callable[[TrainingEvent], None]] = []
        self._initialized = True

    def request_start(
        self,
        job: TrainingJob,
        adapter: TrainingAdapter,
        config: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Request to start a new training job

        Args:
            job: TrainingJob with initial metadata
            adapter: Training adapter for this mode
            config: System-specific configuration

        Returns:
            (success, message)
        """
        with self._state_lock:
            if self._current_job is not None:
                if self._current_job.status.is_active():
                    return False, f"Training already in progress: {self._current_job.display_name}"

            can_start, reason = adapter.can_start()
            if not can_start:
                return False, reason

            job.status = TrainingStatus.PREPARING
            self._current_job = job
            self._current_adapter = adapter

            adapter.subscribe(self._on_adapter_event)
            self._notify_status_change(job)

        success, message = adapter.start(job, config)

        if success:
            with self._state_lock:
                if self._current_job and self._current_job.job_id == job.job_id:
                    self._current_job.status = TrainingStatus.RUNNING
                    self._notify_status_change(self._current_job)
        else:
            with self._state_lock:
                if self._current_job and self._current_job.job_id == job.job_id:
                    self._current_job.status = TrainingStatus.FAILED
                    self._current_job.error_message = message
                    self._notify_status_change(self._current_job)
                    self._cleanup_job()

        return success, message

    def request_stop(self) -> bool:
        """Request to stop the current training job

        Returns:
            True if stop signal sent successfully
        """
        with self._state_lock:
            if self._current_job is None:
                return False

            if not self._current_job.status.is_active():
                return False

            self._current_job.status = TrainingStatus.STOPPING
            adapter = self._current_adapter
            self._notify_status_change(self._current_job)

        if adapter:
            return adapter.stop()
        return False

    def complete_job(self, job_id: str, **kwargs):
        """Mark job as completed (idempotent)

        Args:
            job_id: Job ID to validate
            **kwargs: Additional metadata to store
        """
        with self._state_lock:
            if not self._validate_job_id(job_id):
                return

            if self._current_job.status.is_terminal():
                return

            self._current_job.status = TrainingStatus.COMPLETED
            self._current_job.ended_at = kwargs.get('ended_at')

            if 'metadata' in kwargs:
                self._current_job.metadata.update(kwargs['metadata'])

            self._notify_status_change(self._current_job)
            self._cleanup_job()

    def fail_job(self, job_id: str, error: str, **kwargs):
        """Mark job as failed (idempotent)

        Args:
            job_id: Job ID to validate
            error: Error message
            **kwargs: Additional metadata to store
        """
        with self._state_lock:
            if not self._validate_job_id(job_id):
                return

            if self._current_job.status.is_terminal():
                return

            self._current_job.status = TrainingStatus.FAILED
            self._current_job.error_message = error
            self._current_job.ended_at = kwargs.get('ended_at')

            if 'metadata' in kwargs:
                self._current_job.metadata.update(kwargs['metadata'])

            self._notify_status_change(self._current_job)
            self._cleanup_job()

    def stop_job(self, job_id: str, **kwargs):
        """Mark job as stopped (idempotent)

        Args:
            job_id: Job ID to validate
            **kwargs: Additional metadata to store
        """
        with self._state_lock:
            if not self._validate_job_id(job_id):
                return

            if self._current_job.status.is_terminal():
                return

            self._current_job.status = TrainingStatus.STOPPED
            self._current_job.ended_at = kwargs.get('ended_at')

            if 'metadata' in kwargs:
                self._current_job.metadata.update(kwargs['metadata'])

            self._notify_status_change(self._current_job)
            self._cleanup_job()

    def get_current_job(self) -> Optional[TrainingJob]:
        """Get current job (thread-safe read)"""
        with self._state_lock:
            return self._current_job

    def subscribe_status(self, callback: Callable[[TrainingJob], None]):
        """Subscribe to job status changes"""
        if callback not in self._status_callbacks:
            self._status_callbacks.append(callback)

    def unsubscribe_status(self, callback: Callable[[TrainingJob], None]):
        """Unsubscribe from job status changes"""
        if callback in self._status_callbacks:
            self._status_callbacks.remove(callback)

    def subscribe_events(self, callback: Callable[[TrainingEvent], None]):
        """Subscribe to training events"""
        if callback not in self._event_callbacks:
            self._event_callbacks.append(callback)

    def unsubscribe_events(self, callback: Callable[[TrainingEvent], None]):
        """Unsubscribe from training events"""
        if callback in self._event_callbacks:
            self._event_callbacks.remove(callback)

    def _validate_job_id(self, job_id: str) -> bool:
        """Validate job_id matches current job"""
        if self._current_job is None:
            return False
        if self._current_job.job_id != job_id:
            return False
        return True

    def _cleanup_job(self):
        """Cleanup after job reaches terminal state"""
        if self._current_adapter:
            self._current_adapter.unsubscribe(self._on_adapter_event)

        self._current_adapter = None
        self._current_job = None

    def _on_adapter_event(self, event: TrainingEvent):
        """Forward adapter events to subscribers"""
        if event.event_type == TrainingEventType.COMPLETED:
            self.complete_job(
                event.job_id,
                ended_at=event.timestamp,
                metadata=event.payload
            )
        elif event.event_type == TrainingEventType.FAILED:
            self.fail_job(
                event.job_id,
                error=event.payload.get('error', 'Unknown error'),
                ended_at=event.timestamp,
                metadata=event.payload
            )
        elif event.event_type == TrainingEventType.STOPPED:
            self.stop_job(
                event.job_id,
                ended_at=event.timestamp,
                metadata=event.payload
            )

        for callback in self._event_callbacks[:]:
            try:
                callback(event)
            except Exception:
                pass

    def _notify_status_change(self, job: TrainingJob):
        """Notify subscribers of status change"""
        for callback in self._status_callbacks[:]:
            try:
                callback(job)
            except Exception:
                pass


_manager_instance: Optional[JobManager] = None


def get_job_manager() -> JobManager:
    """Get global JobManager singleton"""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = JobManager()
    return _manager_instance
