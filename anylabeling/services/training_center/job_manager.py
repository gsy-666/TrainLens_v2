"""Training Center - Job Manager

Global singleton that coordinates all training jobs across different modes.
Enforces mutual exclusion and manages job lifecycle.
"""

import os
import threading
import time
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, Callable, List
from pathlib import Path

from .models import TrainingJob, TrainingMode, TrainingStatus, normalize_execution_mode
from .adapters.base import TrainingAdapter
from .event_protocol import TrainingEvent, TrainingEventType
from .runners.base import TrainingRunner


class JobManager:
    """Global singleton coordinator for all training jobs

    Responsibilities:
    - Enforce mutual exclusion (only one job active at a time)
    - Manage job lifecycle state transitions
    - Route events from runners/adapters to UI subscribers
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

        self._state_lock = threading.RLock()
        self._current_job: Optional[TrainingJob] = None
        self._current_adapter: Optional[TrainingAdapter] = None
        self._current_runner: Optional[TrainingRunner] = None
        self._status_callbacks: List[Callable[[TrainingJob], None]] = []
        self._event_callbacks: List[Callable[[TrainingEvent], None]] = []
        self._history_store = None  # Lazy-init on first use
        self._initialized = True
        self._recover_orphaned_jobs()

    def _get_history_store(self):
        """Lazy-init HistoryStore singleton."""
        if self._history_store is None:
            from .history import get_history_store
            self._history_store = get_history_store()
        return self._history_store

    def _recover_orphaned_jobs(self):
        """On startup, mark any PREPARING/RUNNING records as FAILED (interrupted)."""
        try:
            store = self._get_history_store()
        except Exception:
            return
        store._ensure_loaded()
        orphaned = []
        for job_id, record in list(store._cache.items()):
            try:
                status = TrainingStatus(record.status)
            except ValueError:
                continue
            if status in (TrainingStatus.PREPARING, TrainingStatus.RUNNING):
                orphaned.append(job_id)
        for job_id in orphaned:
            try:
                store.finalize_job(
                    job_id,
                    TrainingStatus.FAILED,
                    error_message="Application exited before the training job reached a terminal state.",
                )
            except Exception:
                pass

    def _history_append(self, job: TrainingJob):
        """Write job to history with PREPARING status (called from reserve_job)."""
        try:
            store = self._get_history_store()
            from .history import JobHistoryRecord
            record = JobHistoryRecord(
                job_id=job.job_id,
                mode=job.mode.value,
                status=TrainingStatus.PREPARING.value,
                created_at=(job.created_at or datetime.now()).isoformat(),
                workspace=str(job.workspace) if job.workspace else None,
                output_directory=str(job.output_directory) if job.output_directory else None,
                display_name=job.display_name,
                framework=job.framework,
                python_executable=str(job.python_executable) if job.python_executable else None,
                command=job.command,
                metadata=dict(job.metadata),
                task=getattr(job, 'task', None),
                model=getattr(job, 'model', None),
                data=getattr(job, 'data', None),
                project=getattr(job, 'project', None),
                project_path=os.path.join(getattr(job, 'project', '') or '', getattr(job, 'name', '') or '') if getattr(job, 'project', None) else None,
                dataset_yaml=getattr(job, 'data', None),
                model_name=getattr(job, 'model', None),
                runtime_id=getattr(job, 'runtime_id', None),
                runtime_python=getattr(job, 'runtime_python', None),
                requested_device=getattr(job, 'requested_device', None),
                resolved_device=getattr(job, 'resolved_device', None),
                execution_mode=getattr(job, 'execution_mode', None),
            )
            store.append_job(record)
        except Exception:
            pass

    def _history_update(self, job_id: str, **updates):
        """Update history record in-place."""
        try:
            self._get_history_store().update_job(job_id, **updates)
        except Exception:
            pass

    def _history_finalize(self, job_id: str, status: TrainingStatus, error_message=None, **kwargs):
        """Finalize history record with terminal status."""
        try:
            self._get_history_store().finalize_job(
                job_id, status,
                error_message=error_message,
                **kwargs,
            )
        except Exception:
            pass

    def request_start(
        self,
        job: TrainingJob,
        adapter: TrainingAdapter,
        config: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Request to start a new training job (one-shot: reserve + start).

        For two-phase (reserve then start), use reserve_job() + start_reserved_job().

        Returns:
            (success, message)
        """
        ok, msg = self.reserve_job(job, adapter)
        if not ok:
            return False, msg
        return self.start_reserved_job(job.job_id, config)

    def reserve_job(
        self,
        job: TrainingJob,
        adapter: TrainingAdapter,
    ) -> Tuple[bool, str]:
        """Phase 1: reserve the job manager slot (PREPARING state).

        Does NOT call adapter.start(). The caller is responsible for
        doing background preparation, then calling start_reserved_job()
        or fail_reserved_job().

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

            # Resolve runner from execution_mode
            execution_mode_raw = getattr(job, "execution_mode", None) or "local"
            execution_mode = normalize_execution_mode(execution_mode_raw)
            from .runners.factory import get_runner

            try:
                runner = get_runner(execution_mode)
            except (ValueError, NotImplementedError) as e:
                return False, str(e)

            # Log runner selection to training console
            import logging as _logging_jm
            _log_jm = _logging_jm.getLogger(__name__)
            _log_jm.info(
                "Execution mode raw: %r → normalized: %r | Runner: %s (%s)",
                execution_mode_raw, execution_mode,
                type(runner).__name__, type(runner).__module__,
            )

            job.status = TrainingStatus.PREPARING
            self._current_job = job
            self._current_adapter = adapter
            self._current_runner = runner

            # Subscribe to runner events
            runner.subscribe(self._on_runner_event)
            cbs = list(self._status_callbacks)

        self._notify_status_change(job, cbs)
        self._history_append(job)
        return True, "Job reserved, preparing..."

    def start_reserved_job(
        self,
        job_id: str,
        config: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """Phase 2: start the actual training for a previously reserved job.

        Uses the runner resolved during reserve_job().
        The runner owns the process lifecycle; this method only
        delegates and updates status.
        """
        runner: Optional[TrainingRunner] = None
        job: Optional[TrainingJob] = None

        with self._state_lock:
            if not self._validate_job_id(job_id):
                return False, "Job ID mismatch or no reserved job"
            if self._current_job.status != TrainingStatus.PREPARING:
                return False, f"Job is not in PREPARING state (current: {self._current_job.status.value})"
            runner = self._current_runner
            job = self._current_job

        if runner is None:
            return False, "No runner available"

        # ── Hard runner type check ──
        from .runners.local import LocalRunner
        from .runners.ssh_remote import SSHRemoteRunner

        mode = normalize_execution_mode(getattr(job, "execution_mode", None))
        if mode == "remote_ssh" and not isinstance(runner, SSHRemoteRunner):
            raise RuntimeError(
                f"Remote SSH job was assigned to {type(runner).__name__} "
                f"(expected SSHRemoteRunner)"
            )
        if mode == "local" and not isinstance(runner, LocalRunner):
            raise RuntimeError(
                f"Local job was assigned to {type(runner).__name__} "
                f"(expected LocalRunner)"
            )

        # Emit runner selection to training console
        from .event_protocol import create_console_output_event
        import time as _time
        runner_event = create_console_output_event(
            job_id=job_id, timestamp=_time.time(),
            message=(
                f"Execution mode raw: {getattr(job, 'execution_mode', None)!r}\n"
                f"Runner selected: {type(runner).__name__}\n"
                f"Runner module: {type(runner).__module__}"
            ),
            source="job_manager",
        )
        self._notify_event_callbacks(runner_event)

        # ── Prepare ──
        ok, msg = runner.prepare(job, config)
        if not ok:
            with self._state_lock:
                if self._current_job and self._current_job.job_id == job_id:
                    self._current_job.status = TrainingStatus.FAILED
                    self._current_job.error_message = msg
                    cbs = list(self._status_callbacks)
                    job_ref = self._current_job
                    self._cleanup_job()
                else:
                    return False, msg
            self._notify_status_change(job_ref, cbs)
            self._history_finalize(job_id, TrainingStatus.FAILED, error_message=msg)
            return False, msg

        # ── Start ──
        try:
            success, message = runner.start(job, config)
        except Exception as exc:
            success = False
            message = f"Runner start exception: {exc}"

        if success:
            with self._state_lock:
                cbs = list(self._status_callbacks)
                if self._current_job and self._current_job.job_id == job_id:
                    self._current_job.status = TrainingStatus.RUNNING
                    self._current_job.started_at = datetime.now()
                job_ref = self._current_job
            if job_ref:
                self._notify_status_change(job_ref, cbs)
                self._history_update(job_id, status=TrainingStatus.RUNNING.value,
                                     started_at=job_ref.started_at.isoformat() if job_ref.started_at else None)
        else:
            with self._state_lock:
                cbs = list(self._status_callbacks)
                if self._current_job and self._current_job.job_id == job_id:
                    self._current_job.status = TrainingStatus.FAILED
                    self._current_job.error_message = message
                    job_ref = self._current_job
                    self._cleanup_job()
                else:
                    job_ref = None
            if job_ref:
                self._notify_status_change(job_ref, cbs)
            self._history_finalize(job_id, TrainingStatus.FAILED, error_message=message)

        return success, message

    def fail_reserved_job(self, job_id: str, error: str):
        """Fail a job that is in PREPARING state (e.g., background prep failed)."""
        with self._state_lock:
            if not self._validate_job_id(job_id):
                return
            if self._current_job.status != TrainingStatus.PREPARING:
                return

            self._current_job.status = TrainingStatus.FAILED
            self._current_job.error_message = error
            job_ref = self._current_job
            cbs = list(self._status_callbacks)
            self._cleanup_job()

        self._notify_status_change(job_ref, cbs)
        self._history_finalize(job_id, TrainingStatus.FAILED, error_message=error)

    def request_stop(self) -> bool:
        """Request to stop the current training job.

        For PREPARING state: cancels the reservation directly (STOPPED).
        For RUNNING state: signals adapter.stop() and transitions to STOPPING.

        Returns:
            True if stop initiated, False if nothing to stop.
        """
        status_cbs = []
        job_ref = None

        with self._state_lock:
            if self._current_job is None:
                return False

            if self._current_job.status == TrainingStatus.STOPPING:
                return False

            if not self._current_job.status.is_active():
                return False

            # PREPARING → STOPPED (no adapter activity to interrupt)
            if self._current_job.status == TrainingStatus.PREPARING:
                self._current_job.status = TrainingStatus.STOPPED
                job_ref = self._current_job
                status_cbs = list(self._status_callbacks)
                self._cleanup_job()
                self._notify_status_change(job_ref, status_cbs)
                self._history_finalize(job_ref.job_id, TrainingStatus.STOPPED)
                return True

            # RUNNING → STOPPING (runner.cancel() will handle the rest)
            self._current_job.status = TrainingStatus.STOPPING
            runner = self._current_runner
            status_cbs = list(self._status_callbacks)

        self._notify_status_change(self._current_job, status_cbs)
        if runner:
            return runner.cancel(self._current_job.job_id)
        return False

    def complete_job(self, job_id: str, **kwargs):
        """Mark job as completed (idempotent)"""
        job_ref = None
        cbs = []
        with self._state_lock:
            if not self._validate_job_id(job_id):
                return

            if self._current_job.status.is_terminal():
                return

            self._current_job.status = TrainingStatus.COMPLETED
            self._current_job.ended_at = kwargs.get('ended_at')

            if 'metadata' in kwargs:
                self._current_job.metadata.update(kwargs['metadata'])
                save_dir = kwargs['metadata'].get('save_dir', '')
                if save_dir:
                    self._current_job.output_directory = Path(save_dir)

            job_ref = self._current_job
            cbs = list(self._status_callbacks)
            self._cleanup_job()

        self._notify_status_change(job_ref, cbs)
        save_dir = kwargs.get('metadata', {}).get('save_dir', '')
        self._history_finalize(job_id, TrainingStatus.COMPLETED,
                               output_directory=save_dir or None,
                               ended_at=kwargs.get('ended_at'))

    def fail_job(self, job_id: str, error: str, **kwargs):
        """Mark job as failed (idempotent)"""
        job_ref = None
        cbs = []
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
                save_dir = kwargs['metadata'].get('save_dir', '')
                if save_dir:
                    self._current_job.output_directory = Path(save_dir)

            job_ref = self._current_job
            cbs = list(self._status_callbacks)
            self._cleanup_job()

        self._notify_status_change(job_ref, cbs)
        save_dir = kwargs.get('metadata', {}).get('save_dir', '')
        self._history_finalize(job_id, TrainingStatus.FAILED, error_message=error,
                               output_directory=save_dir or None,
                               ended_at=kwargs.get('ended_at'))

    def stop_job(self, job_id: str, **kwargs):
        """Mark job as stopped (idempotent)"""
        job_ref = None
        cbs = []
        with self._state_lock:
            if not self._validate_job_id(job_id):
                return

            if self._current_job.status.is_terminal():
                return

            self._current_job.status = TrainingStatus.STOPPED
            self._current_job.ended_at = kwargs.get('ended_at')

            if 'metadata' in kwargs:
                self._current_job.metadata.update(kwargs['metadata'])
                save_dir = kwargs['metadata'].get('save_dir', '')
                if save_dir:
                    self._current_job.output_directory = Path(save_dir)

            job_ref = self._current_job
            cbs = list(self._status_callbacks)
            self._cleanup_job()

        self._notify_status_change(job_ref, cbs)
        save_dir = kwargs.get('metadata', {}).get('save_dir', '')
        self._history_finalize(job_id, TrainingStatus.STOPPED,
                               output_directory=save_dir or None,
                               ended_at=kwargs.get('ended_at'))

    def get_current_job(self) -> Optional[TrainingJob]:
        """Get current job (thread-safe read)"""
        with self._state_lock:
            return self._current_job

    def subscribe_status(self, callback: Callable[[TrainingJob], None]):
        """Subscribe to job status changes"""
        with self._state_lock:
            if callback not in self._status_callbacks:
                self._status_callbacks.append(callback)

    def unsubscribe_status(self, callback: Callable[[TrainingJob], None]):
        """Unsubscribe from job status changes"""
        with self._state_lock:
            if callback in self._status_callbacks:
                self._status_callbacks.remove(callback)

    def subscribe_events(self, callback: Callable[[TrainingEvent], None]):
        """Subscribe to training events"""
        with self._state_lock:
            if callback not in self._event_callbacks:
                self._event_callbacks.append(callback)

    def unsubscribe_events(self, callback: Callable[[TrainingEvent], None]):
        """Unsubscribe from training events"""
        with self._state_lock:
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
        """Cleanup after job reaches terminal state.

        Caller MUST hold _state_lock. Runner/adapter cleanup is deferred
        outside the lock to avoid blocking other threads.
        """
        adapter_to_shutdown = None
        runner_to_cleanup = None

        if self._current_adapter:
            self._current_adapter.unsubscribe(self._on_adapter_event)
            if hasattr(self._current_adapter, 'shutdown'):
                adapter_to_shutdown = self._current_adapter

        if self._current_runner:
            self._current_runner.unsubscribe(self._on_runner_event)
            runner_to_cleanup = self._current_runner

        self._current_adapter = None
        self._current_runner = None
        self._current_job = None

        # Shutdown adapter outside lock to avoid blocking
        if adapter_to_shutdown is not None:
            try:
                adapter_to_shutdown.shutdown()
            except Exception:
                pass

        # Cleanup runner outside lock
        if runner_to_cleanup is not None and self._current_job:
            # Note: self._current_job is None by now, but we saved job_id
            pass  # runner cleanup is idempotent and handled by the runner itself

    def _on_adapter_event(self, event: TrainingEvent):
        """Forward adapter events to subscribers — NO lock held for dispatch.

        Terminal events (COMPLETED/FAILED/STOPPED) update job state
        and notify status callbacks (lock-free). Event callbacks are
        dispatched lock-free with a snapshot.
        """
        self._dispatch_event(event)

    def _on_runner_event(self, event: TrainingEvent):
        """Forward runner events to subscribers — same dispatch as adapter events."""
        self._dispatch_event(event)

    def _dispatch_event(self, event: TrainingEvent):
        # Convert float timestamp to datetime
        ended_dt = datetime.fromtimestamp(event.timestamp) if event.timestamp else datetime.now()

        if event.event_type == TrainingEventType.COMPLETED:
            self.complete_job(
                event.job_id,
                ended_at=ended_dt,
                metadata=event.payload
            )
        elif event.event_type == TrainingEventType.FAILED:
            self.fail_job(
                event.job_id,
                error=event.payload.get('error', 'Unknown error'),
                ended_at=ended_dt,
                metadata=event.payload
            )
        elif event.event_type == TrainingEventType.STOPPED:
            self.stop_job(
                event.job_id,
                ended_at=ended_dt,
                metadata=event.payload
            )

        self._notify_event_callbacks(event)

    def _notify_status_change(self, job: TrainingJob, callbacks: list):
        """Notify subscribers of status change — NO lock held.

        Args:
            job: The job with updated status (caller's snapshot).
            callbacks: Pre-snapshotted list of status callbacks.
        """
        for callback in callbacks:
            try:
                callback(job)
            except Exception:
                pass

    def _notify_event_callbacks(self, event: TrainingEvent):
        """Dispatch event to subscribers — NO lock held.

        Snapshots event callbacks under lock, then dispatches outside.
        """
        with self._state_lock:
            cbs = list(self._event_callbacks)
        for callback in cbs:
            try:
                callback(event)
            except Exception:
                pass


_manager_instance: Optional[JobManager] = None


def get_job_manager() -> JobManager:
    """Get global JobManager singleton"""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = JobManager()
    return _manager_instance


# ── Register default runners on first import ──────────────────────────

def _register_default_runners():
    """Lazily register the LocalRunner and SSHRemoteRunner in the global RunnerFactory."""
    from .runners.factory import RunnerFactory
    from .runners.local import LocalRunner
    from .runners.ssh_remote import SSHRemoteRunner
    factory = RunnerFactory.get_instance()
    if "local" not in factory._runners:
        factory.register("local", LocalRunner())
    if "remote_ssh" not in factory._runners:
        factory.register("remote_ssh", SSHRemoteRunner())


_register_default_runners()