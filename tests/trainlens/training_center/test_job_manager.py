"""Tests for Training Center Job Manager

Verifies mutual exclusion, state transitions, idempotency, and job ID validation.
"""

import pytest
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, MagicMock

from anylabeling.services.training_center.models import (
    TrainingMode,
    TrainingStatus,
    TrainingJob,
)
from anylabeling.services.training_center.event_protocol import (
    TrainingEventType,
    create_completed_event,
    create_failed_event,
    create_stopped_event,
)
from anylabeling.services.training_center.job_manager import (
    JobManager,
    get_job_manager,
)


@pytest.fixture
def job_manager():
    """Create fresh JobManager instance for each test"""
    manager = JobManager()
    manager._current_job = None
    manager._current_adapter = None
    manager._status_callbacks = []
    manager._event_callbacks = []
    return manager


@pytest.fixture
def mock_adapter():
    """Create mock adapter"""
    adapter = Mock()
    adapter.can_start.return_value = (True, "")
    adapter.start.return_value = (True, "Started")
    adapter.stop.return_value = True
    adapter.is_running.return_value = False
    adapter.subscribe = Mock()
    adapter.unsubscribe = Mock()
    return adapter


@pytest.fixture
def sample_job():
    """Create sample training job"""
    return TrainingJob(
        job_id="test-job-001",
        mode=TrainingMode.CUSTOM_SCRIPT,
        status=TrainingStatus.IDLE,
        created_at=datetime.now(),
        started_at=None,
        ended_at=None,
        workspace=Path("/tmp/workspace"),
        output_directory=None,
        display_name="Test Job",
        framework="custom",
        python_executable=Path("python"),
        command=["python", "train.py"],
        metadata={},
        error_message=None,
    )


class TestJobManagerSingleton:
    """Test singleton pattern"""

    def test_get_job_manager_singleton(self):
        """get_job_manager returns same instance"""
        manager1 = get_job_manager()
        manager2 = get_job_manager()
        assert manager1 is manager2

    def test_jobmanager_singleton(self):
        """JobManager() returns same instance"""
        manager1 = JobManager()
        manager2 = JobManager()
        assert manager1 is manager2


class TestMutualExclusion:
    """Test concurrent training prevention"""

    def test_single_job_can_start(self, job_manager, mock_adapter, sample_job):
        """First job can start when no active job"""
        success, message = job_manager.request_start(
            sample_job, mock_adapter, {"test": "config"}
        )

        assert success is True
        assert mock_adapter.start.called
        assert job_manager.get_current_job() is not None

    def test_second_job_blocked_while_first_active(
        self, job_manager, mock_adapter, sample_job
    ):
        """Second job cannot start while first is active"""
        job_manager.request_start(sample_job, mock_adapter, {})

        job2 = TrainingJob(
            job_id="test-job-002",
            mode=TrainingMode.GUIDED_ULTRALYTICS,
            status=TrainingStatus.IDLE,
            created_at=datetime.now(),
            started_at=None,
            ended_at=None,
            workspace=None,
            output_directory=None,
            display_name="Second Job",
            framework="ultralytics",
            python_executable=None,
            command=None,
            metadata={},
            error_message=None,
        )

        adapter2 = Mock()
        adapter2.can_start.return_value = (True, "")

        success, message = job_manager.request_start(job2, adapter2, {})

        assert success is False
        assert "already in progress" in message.lower()
        assert adapter2.start.called is False

    def test_sequential_jobs_after_completion(
        self, job_manager, mock_adapter, sample_job
    ):
        """New job can start after previous completes"""
        job_manager.request_start(sample_job, mock_adapter, {})

        # Capture job before completion
        job_ref = job_manager.get_current_job()

        job_manager.complete_job(sample_job.job_id)

        # After completion, current_job is cleared
        assert job_manager.get_current_job() is None
        assert job_ref.status == TrainingStatus.COMPLETED

        job2 = TrainingJob(
            job_id="test-job-003",
            mode=TrainingMode.CUSTOM_SCRIPT,
            status=TrainingStatus.IDLE,
            created_at=datetime.now(),
            started_at=None,
            ended_at=None,
            workspace=None,
            output_directory=None,
            display_name="Second Job",
            framework="custom",
            python_executable=None,
            command=None,
            metadata={},
            error_message=None,
        )

        adapter2 = Mock()
        adapter2.can_start.return_value = (True, "")
        adapter2.start.return_value = (True, "Started")

        success, message = job_manager.request_start(job2, adapter2, {})

        assert success is True
        assert adapter2.start.called


class TestStateTransitions:
    """Test job lifecycle state transitions"""

    def test_start_transitions_preparing_to_running(
        self, job_manager, mock_adapter, sample_job
    ):
        """Successful start: IDLE -> PREPARING -> RUNNING"""
        job_manager.request_start(sample_job, mock_adapter, {})

        current = job_manager.get_current_job()
        assert current.status == TrainingStatus.RUNNING

    def test_start_failure_transitions_to_failed(
        self, job_manager, mock_adapter, sample_job
    ):
        """Failed start transitions to FAILED"""
        mock_adapter.start.return_value = (False, "Failed to start")

        # Capture job reference before start (it will be set to PREPARING first)
        success, msg = job_manager.request_start(sample_job, mock_adapter, {})

        # After failed start, current_job is cleared
        assert success is False
        assert msg == "Failed to start"
        # Job was set to FAILED before cleanup
        assert sample_job.status == TrainingStatus.FAILED
        assert sample_job.error_message == "Failed to start"
        assert job_manager.get_current_job() is None

    def test_stop_transitions_to_stopping(
        self, job_manager, mock_adapter, sample_job
    ):
        """request_stop transitions RUNNING -> STOPPING"""
        job_manager.request_start(sample_job, mock_adapter, {})
        assert job_manager.get_current_job().status == TrainingStatus.RUNNING

        result = job_manager.request_stop()

        assert result is True
        assert job_manager.get_current_job().status == TrainingStatus.STOPPING
        assert mock_adapter.stop.called

    def test_complete_job_transitions_to_completed(
        self, job_manager, mock_adapter, sample_job
    ):
        """complete_job transitions to COMPLETED"""
        job_manager.request_start(sample_job, mock_adapter, {})

        # Capture job before terminal event
        job_ref = job_manager.get_current_job()

        job_manager.complete_job(sample_job.job_id)

        # After terminal event, current_job is cleared
        assert job_manager.get_current_job() is None
        # But the job reference we captured has correct status
        assert job_ref.status == TrainingStatus.COMPLETED
        assert job_ref.status.is_terminal()

    def test_fail_job_transitions_to_failed(
        self, job_manager, mock_adapter, sample_job
    ):
        """fail_job transitions to FAILED"""
        job_manager.request_start(sample_job, mock_adapter, {})

        # Capture job before terminal event
        job_ref = job_manager.get_current_job()

        job_manager.fail_job(sample_job.job_id, "Test error")

        # After terminal event, current_job is cleared
        assert job_manager.get_current_job() is None
        # But the job reference we captured has correct status
        assert job_ref.status == TrainingStatus.FAILED
        assert job_ref.error_message == "Test error"
        assert job_ref.status.is_terminal()

    def test_stop_job_transitions_to_stopped(
        self, job_manager, mock_adapter, sample_job
    ):
        """stop_job transitions to STOPPED"""
        job_manager.request_start(sample_job, mock_adapter, {})

        # Capture job before terminal event
        job_ref = job_manager.get_current_job()

        job_manager.stop_job(sample_job.job_id)

        # After terminal event, current_job is cleared
        assert job_manager.get_current_job() is None
        # But the job reference we captured has correct status
        assert job_ref.status == TrainingStatus.STOPPED
        assert job_ref.status.is_terminal()


class TestIdempotency:
    """Test idempotent terminal event handling"""

    def test_complete_job_idempotent(
        self, job_manager, mock_adapter, sample_job
    ):
        """Calling complete_job twice has no effect"""
        job_manager.request_start(sample_job, mock_adapter, {})

        # Capture job before first completion
        job_ref = job_manager.get_current_job()

        job_manager.complete_job(sample_job.job_id)
        first_status = job_ref.status

        # Second call should be no-op (job already cleared)
        job_manager.complete_job(sample_job.job_id)
        second_status = job_ref.status

        assert first_status == TrainingStatus.COMPLETED
        assert second_status == TrainingStatus.COMPLETED
        assert job_manager.get_current_job() is None

    def test_fail_job_after_completed_ignored(
        self, job_manager, mock_adapter, sample_job
    ):
        """fail_job after complete_job is ignored"""
        job_manager.request_start(sample_job, mock_adapter, {})

        # Capture job before completion
        job_ref = job_manager.get_current_job()

        job_manager.complete_job(sample_job.job_id)
        assert job_ref.status == TrainingStatus.COMPLETED

        # Try to fail after completion - should be ignored
        job_manager.fail_job(sample_job.job_id, "Late error")
        assert job_ref.status == TrainingStatus.COMPLETED
        assert job_manager.get_current_job() is None

    def test_stop_job_after_failed_ignored(
        self, job_manager, mock_adapter, sample_job
    ):
        """stop_job after fail_job is ignored"""
        job_manager.request_start(sample_job, mock_adapter, {})

        # Capture job before failure
        job_ref = job_manager.get_current_job()

        job_manager.fail_job(sample_job.job_id, "Error occurred")
        assert job_ref.status == TrainingStatus.FAILED

        # Try to stop after failure - should be ignored
        job_manager.stop_job(sample_job.job_id)
        assert job_ref.status == TrainingStatus.FAILED
        assert job_manager.get_current_job() is None


class TestJobIDValidation:
    """Test late event prevention via job_id validation"""

    def test_complete_job_wrong_id_ignored(
        self, job_manager, mock_adapter, sample_job
    ):
        """complete_job with wrong job_id is ignored"""
        job_manager.request_start(sample_job, mock_adapter, {})

        job_manager.complete_job("wrong-job-id")

        current = job_manager.get_current_job()
        assert current.status == TrainingStatus.RUNNING

    def test_fail_job_wrong_id_ignored(
        self, job_manager, mock_adapter, sample_job
    ):
        """fail_job with wrong job_id is ignored"""
        job_manager.request_start(sample_job, mock_adapter, {})

        job_manager.fail_job("wrong-job-id", "Error")

        current = job_manager.get_current_job()
        assert current.status == TrainingStatus.RUNNING

    def test_stop_job_wrong_id_ignored(
        self, job_manager, mock_adapter, sample_job
    ):
        """stop_job with wrong job_id is ignored"""
        job_manager.request_start(sample_job, mock_adapter, {})

        job_manager.stop_job("wrong-job-id")

        current = job_manager.get_current_job()
        assert current.status == TrainingStatus.RUNNING

    def test_late_event_after_job_completion(
        self, job_manager, mock_adapter, sample_job
    ):
        """Late events from old job don't affect new job"""
        job_manager.request_start(sample_job, mock_adapter, {})
        old_job_id = sample_job.job_id

        job_manager.complete_job(old_job_id)

        job2 = TrainingJob(
            job_id="test-job-new",
            mode=TrainingMode.CUSTOM_SCRIPT,
            status=TrainingStatus.IDLE,
            created_at=datetime.now(),
            started_at=None,
            ended_at=None,
            workspace=None,
            output_directory=None,
            display_name="New Job",
            framework="custom",
            python_executable=None,
            command=None,
            metadata={},
            error_message=None,
        )

        adapter2 = Mock()
        adapter2.can_start.return_value = (True, "")
        adapter2.start.return_value = (True, "Started")

        job_manager.request_start(job2, adapter2, {})

        job_manager.fail_job(old_job_id, "Late error from old job")

        current = job_manager.get_current_job()
        assert current.job_id == "test-job-new"
        assert current.status == TrainingStatus.RUNNING


class TestCallbackExceptionIsolation:
    """Test callback exception handling"""

    def test_status_callback_exception_isolated(
        self, job_manager, mock_adapter, sample_job
    ):
        """Exception in status callback doesn't break other callbacks"""
        results = []

        def bad_callback(job):
            raise ValueError("Bad callback")

        def good_callback(job):
            results.append(job.status)

        job_manager.subscribe_status(bad_callback)
        job_manager.subscribe_status(good_callback)

        job_manager.request_start(sample_job, mock_adapter, {})

        assert TrainingStatus.RUNNING in results

    def test_event_callback_exception_isolated(self, job_manager):
        """Exception in event callback doesn't break other callbacks"""
        results = []

        def bad_callback(event):
            raise ValueError("Bad callback")

        def good_callback(event):
            results.append(event.event_type)

        job_manager.subscribe_events(bad_callback)
        job_manager.subscribe_events(good_callback)

        event = create_completed_event("test-001", time.time(), source="test")
        job_manager._on_adapter_event(event)

        assert TrainingEventType.COMPLETED in results


class TestCallbackSubscription:
    """Test callback subscription/unsubscription"""

    def test_subscribe_status_callback(self, job_manager):
        """Can subscribe to status changes"""
        callback = Mock()
        job_manager.subscribe_status(callback)

        assert callback in job_manager._status_callbacks

    def test_unsubscribe_status_callback(self, job_manager):
        """Can unsubscribe from status changes"""
        callback = Mock()
        job_manager.subscribe_status(callback)
        job_manager.unsubscribe_status(callback)

        assert callback not in job_manager._status_callbacks

    def test_subscribe_event_callback(self, job_manager):
        """Can subscribe to training events"""
        callback = Mock()
        job_manager.subscribe_events(callback)

        assert callback in job_manager._event_callbacks

    def test_unsubscribe_event_callback(self, job_manager):
        """Can unsubscribe from training events"""
        callback = Mock()
        job_manager.subscribe_events(callback)
        job_manager.unsubscribe_events(callback)

        assert callback not in job_manager._event_callbacks


class TestTwoPhaseReserve:
    """Test reserve_job / start_reserved_job / fail_reserved_job lifecycle"""

    def test_reserve_occupies_job_manager(self, job_manager, mock_adapter, sample_job):
        """reserve_job sets PREPARING and get_current_job() returns the reserved job"""
        ok, msg = job_manager.reserve_job(sample_job, mock_adapter)

        assert ok is True
        assert "preparing" in msg.lower()
        assert sample_job.status == TrainingStatus.PREPARING

        current = job_manager.get_current_job()
        assert current is not None
        assert current.job_id == sample_job.job_id
        assert current.status == TrainingStatus.PREPARING

        # adapter.start() was NOT called
        assert mock_adapter.start.called is False

    def test_second_job_blocked_during_preparing(
        self, job_manager, mock_adapter, sample_job
    ):
        """Another job cannot reserve while one is PREPARING"""
        job_manager.reserve_job(sample_job, mock_adapter)

        job2 = TrainingJob(
            job_id="test-job-002",
            mode=TrainingMode.GUIDED_ULTRALYTICS,
            status=TrainingStatus.IDLE,
            created_at=datetime.now(),
            started_at=None,
            ended_at=None,
            workspace=None,
            output_directory=None,
            display_name="Second Job",
            framework="ultralytics",
            python_executable=None,
            command=None,
            metadata={},
            error_message=None,
        )

        adapter2 = Mock()
        adapter2.can_start.return_value = (True, "")

        ok, msg = job_manager.reserve_job(job2, adapter2)

        assert ok is False
        assert "already in progress" in msg.lower()
        assert adapter2.start.called is False

    def test_start_reserved_transitions_to_running(
        self, job_manager, mock_adapter, sample_job
    ):
        """start_reserved_job transitions PREPARING → RUNNING"""
        job_manager.reserve_job(sample_job, mock_adapter)

        mock_adapter.start.return_value = (True, "Started training")

        ok, msg = job_manager.start_reserved_job(sample_job.job_id, {"data": "test"})

        assert ok is True
        assert mock_adapter.start.called

        current = job_manager.get_current_job()
        assert current.status == TrainingStatus.RUNNING

    def test_start_reserved_wrong_id_fails(
        self, job_manager, mock_adapter, sample_job
    ):
        """start_reserved_job rejects wrong job_id"""
        job_manager.reserve_job(sample_job, mock_adapter)

        ok, msg = job_manager.start_reserved_job("wrong-job-id", {})

        assert ok is False
        assert "mismatch" in msg.lower()

    def test_start_reserved_not_preparing_fails(
        self, job_manager, mock_adapter, sample_job
    ):
        """start_reserved_job fails if job was already cleaned up"""
        job_manager.reserve_job(sample_job, mock_adapter)
        # Transition away from PREPARING (fail = terminal + cleanup)
        job_manager.fail_reserved_job(sample_job.job_id, "prep failed")

        ok, msg = job_manager.start_reserved_job(sample_job.job_id, {})

        assert ok is False
        # After cleanup, _current_job is None, so "mismatch" is returned
        assert "mismatch" in msg.lower()

    def test_fail_reserved_transitions_to_failed(
        self, job_manager, mock_adapter, sample_job
    ):
        """fail_reserved_job transitions PREPARING → FAILED and cleans up"""
        job_manager.reserve_job(sample_job, mock_adapter)

        job_ref = job_manager.get_current_job()
        job_manager.fail_reserved_job(sample_job.job_id, "Dataset creation error")

        assert job_ref.status == TrainingStatus.FAILED
        assert job_ref.error_message == "Dataset creation error"
        assert job_manager.get_current_job() is None

    def test_fail_reserved_wrong_id_ignored(
        self, job_manager, mock_adapter, sample_job
    ):
        """fail_reserved_job with wrong job_id is silently ignored"""
        job_manager.reserve_job(sample_job, mock_adapter)

        job_manager.fail_reserved_job("wrong-id", "error")

        current = job_manager.get_current_job()
        assert current.status == TrainingStatus.PREPARING

    def test_fail_reserved_not_preparing_ignored(
        self, job_manager, mock_adapter, sample_job
    ):
        """fail_reserved_job when not PREPARING is silently ignored"""
        job_manager.reserve_job(sample_job, mock_adapter)
        job_manager.fail_reserved_job(sample_job.job_id, "first error")
        # Already FAILED → not PREPARING

        job_manager.fail_reserved_job(sample_job.job_id, "second error")
        # No-op, but first error message preserved
        assert sample_job.error_message == "first error"

    def test_stop_during_preparing_transitions_to_stopped(
        self, job_manager, mock_adapter, sample_job
    ):
        """request_stop during PREPARING transitions directly to STOPPED"""
        job_manager.reserve_job(sample_job, mock_adapter)

        job_ref = job_manager.get_current_job()
        result = job_manager.request_stop()

        assert result is True
        # adapter.stop() was NOT called (nothing running)
        assert mock_adapter.stop.called is False
        assert job_ref.status == TrainingStatus.STOPPED
        assert job_manager.get_current_job() is None

    def test_sequential_jobs_after_prepare_stopped(
        self, job_manager, mock_adapter, sample_job
    ):
        """New job can reserve after previous PREPARING was stopped"""
        job_manager.reserve_job(sample_job, mock_adapter)
        job_manager.request_stop()  # STOPPED + cleanup

        job2 = TrainingJob(
            job_id="test-job-003",
            mode=TrainingMode.CUSTOM_SCRIPT,
            status=TrainingStatus.IDLE,
            created_at=datetime.now(),
            started_at=None,
            ended_at=None,
            workspace=None,
            output_directory=None,
            display_name="Second Job",
            framework="custom",
            python_executable=None,
            command=None,
            metadata={},
            error_message=None,
        )

        adapter2 = Mock()
        adapter2.can_start.return_value = (True, "")
        adapter2.start.return_value = (True, "Started")

        ok, msg = job_manager.reserve_job(job2, adapter2)

        assert ok is True
        assert job_manager.get_current_job().job_id == "test-job-003"
        assert job_manager.get_current_job().status == TrainingStatus.PREPARING

    def test_request_start_uses_two_phase_internally(
        self, job_manager, mock_adapter, sample_job
    ):
        """The one-shot request_start still works as before"""
        mock_adapter.start.return_value = (True, "Started training")

        ok, msg = job_manager.request_start(sample_job, mock_adapter, {"data": "test"})

        assert ok is True
        assert mock_adapter.start.called
        assert job_manager.get_current_job().status == TrainingStatus.RUNNING

    def test_request_start_failure_still_cleans(
        self, job_manager, mock_adapter, sample_job
    ):
        """request_start with failed adapter.start cleans up"""
        mock_adapter.start.return_value = (False, "Export error")

        ok, msg = job_manager.request_start(sample_job, mock_adapter, {})

        assert ok is False
        assert msg == "Export error"
        assert job_manager.get_current_job() is None
        assert sample_job.status == TrainingStatus.FAILED

    def test_prepare_then_start_other_mode_works(
        self, job_manager, mock_adapter, sample_job
    ):
        """Full two-phase guided workflow: reserve → start → complete"""
        # Phase 1: Reserve
        ok, _ = job_manager.reserve_job(sample_job, mock_adapter)
        assert ok
        assert job_manager.get_current_job().status == TrainingStatus.PREPARING

        # Phase 2: Start
        mock_adapter.start.return_value = (True, "Started")
        ok, _ = job_manager.start_reserved_job(sample_job.job_id, {"data": "test"})
        assert ok
        assert job_manager.get_current_job().status == TrainingStatus.RUNNING

        # Complete
        job_ref = job_manager.get_current_job()
        job_manager.complete_job(sample_job.job_id)
        assert job_ref.status == TrainingStatus.COMPLETED
        assert job_manager.get_current_job() is None

    def test_adapter_stop_not_called_during_preparing_stop(
        self, job_manager, mock_adapter, sample_job
    ):
        """request_stop during PREPARING does NOT call adapter.stop() — nothing to stop."""
        job_manager.reserve_job(sample_job, mock_adapter)

        result = job_manager.request_stop()

        assert result is True
        assert mock_adapter.stop.called is False
        assert sample_job.status == TrainingStatus.STOPPED
        assert job_manager.get_current_job() is None

    def test_adapter_start_not_called_after_preparing_stop(
        self, job_manager, mock_adapter, sample_job
    ):
        """start_reserved_job after PREPARING was stopped → rejected."""
        job_manager.reserve_job(sample_job, mock_adapter)
        job_manager.request_stop()  # STOPPED + cleanup

        mock_adapter.start.return_value = (True, "Should not reach")
        ok, msg = job_manager.start_reserved_job(sample_job.job_id, {})

        assert ok is False
        assert mock_adapter.start.called is False

    def test_fail_reserved_after_stop_ignored(
        self, job_manager, mock_adapter, sample_job
    ):
        """fail_reserved_job after PREPARING was stopped → silently ignored (no overwrite to FAILED)."""
        job_manager.reserve_job(sample_job, mock_adapter)
        job_manager.request_stop()  # STOPPED + cleanup

        job_manager.fail_reserved_job(sample_job.job_id, "late prep error")
        # Status remains STOPPED (not overwritten to FAILED)
        assert sample_job.status == TrainingStatus.STOPPED

    def test_adapter_shutdown_called_once_on_cleanup(
        self, job_manager, mock_adapter, sample_job
    ):
        """adapter.shutdown() is called exactly once during cleanup."""
        mock_adapter.shutdown = Mock()
        job_manager.reserve_job(sample_job, mock_adapter)
        job_manager.fail_reserved_job(sample_job.job_id, "test error")

        assert mock_adapter.shutdown.call_count == 1

    def test_adapter_shutdown_called_once_on_stop_preparing(
        self, job_manager, mock_adapter, sample_job
    ):
        """adapter.shutdown() called once when stopping during PREPARING."""
        mock_adapter.shutdown = Mock()
        job_manager.reserve_job(sample_job, mock_adapter)
        job_manager.request_stop()

        assert mock_adapter.shutdown.call_count == 1

    def test_adapter_shutdown_called_once_on_full_lifecycle(
        self, job_manager, mock_adapter, sample_job
    ):
        """adapter.shutdown() called exactly once in full reserve → start → complete lifecycle."""
        mock_adapter.shutdown = Mock()
        mock_adapter.start.return_value = (True, "Started")

        job_manager.reserve_job(sample_job, mock_adapter)
        job_manager.start_reserved_job(sample_job.job_id, {})
        job_manager.complete_job(sample_job.job_id)

        assert mock_adapter.shutdown.call_count == 1

    def test_status_never_reaches_running_after_fail_reserved(
        self, job_manager, mock_adapter, sample_job
    ):
        """After fail_reserved_job, status is FAILED and never becomes RUNNING."""
        job_manager.reserve_job(sample_job, mock_adapter)
        job_manager.fail_reserved_job(sample_job.job_id, "data error")

        assert sample_job.status == TrainingStatus.FAILED
        # start_reserved_job should be rejected
        mock_adapter.start.return_value = (True, "Too late")
        ok, _ = job_manager.start_reserved_job(sample_job.job_id, {})
        assert ok is False

    def test_stop_preparing_releases_slot_for_next_job(
        self, job_manager, mock_adapter, sample_job
    ):
        """After PREPARING → STOPPED, a new job can reserve immediately."""
        job_manager.reserve_job(sample_job, mock_adapter)
        assert job_manager.get_current_job() is not None

        job_manager.request_stop()
        assert job_manager.get_current_job() is None

        # New job can now reserve
        job2 = TrainingJob(
            job_id="test-job-new",
            mode=TrainingMode.CUSTOM_SCRIPT,
            status=TrainingStatus.IDLE,
            created_at=datetime.now(),
            started_at=None, ended_at=None,
            workspace=None, output_directory=None,
            display_name="New", framework="custom",
            python_executable=None, command=None, metadata={},
            error_message=None,
        )
        adapter2 = Mock()
        adapter2.can_start.return_value = (True, "")
        ok, _ = job_manager.reserve_job(job2, adapter2)
        assert ok is True
        assert job_manager.get_current_job().job_id == "test-job-new"

    def test_error_message_preserved_after_fail_reserved(
        self, job_manager, mock_adapter, sample_job
    ):
        """fail_reserved_job stores the real exception as error_message."""
        job_manager.reserve_job(sample_job, mock_adapter)
        job_manager.fail_reserved_job(
            sample_job.job_id, "create_yolo_dataset failed: Permission denied"
        )

        assert sample_job.error_message == "create_yolo_dataset failed: Permission denied"
        assert sample_job.status == TrainingStatus.FAILED


class TestLockReentrancy:
    """Test that callbacks can safely re-enter JobManager without deadlock."""

    def test_status_callback_can_call_get_current_job(
        self, job_manager, mock_adapter, sample_job
    ):
        """status callback calling get_current_job() does NOT deadlock."""
        captured = []

        def callback(job):
            # Simulate TrainingCenterWindow._update_status_bar pattern:
            # callback re-enters JobManager.get_current_job()
            current = job_manager.get_current_job()
            captured.append(current)

        job_manager.subscribe_status(callback)
        mock_adapter.start.return_value = (True, "Started")

        # This would deadlock with threading.Lock() (non-reentrant)
        job_manager.request_start(sample_job, mock_adapter, {"data": "test"})

        assert len(captured) > 0

    def test_status_callback_exception_does_not_block_others(
        self, job_manager, mock_adapter, sample_job
    ):
        """Exception in one callback doesn't prevent others from running."""
        results = []

        def bad_callback(job):
            raise RuntimeError("callback boom")

        def good_callback(job):
            results.append(job.status)

        job_manager.subscribe_status(bad_callback)
        job_manager.subscribe_status(good_callback)
        mock_adapter.start.return_value = (True, "Started")

        job_manager.request_start(sample_job, mock_adapter, {})

        assert TrainingStatus.RUNNING in results
