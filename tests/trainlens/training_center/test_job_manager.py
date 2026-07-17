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


class TestCallbacksOutsideLock:
    """Verify all callbacks execute OUTSIDE _state_lock."""

    def test_callback_get_current_job_no_deadlock(
        self, job_manager, mock_adapter, sample_job
    ):
        """status callback calling get_current_job() does NOT deadlock."""
        captured = []

        def callback(job):
            current = job_manager.get_current_job()
            captured.append(current)

        job_manager.subscribe_status(callback)
        mock_adapter.start.return_value = (True, "Started")
        job_manager.request_start(sample_job, mock_adapter, {"data": "test"})
        assert len(captured) > 0

    def test_callback_request_stop_no_deadlock(
        self, job_manager, mock_adapter, sample_job
    ):
        """status callback calling request_stop() does NOT deadlock."""
        mock_adapter.start.return_value = (True, "Started")
        mock_adapter.stop.return_value = True
        stopped = []

        def callback(job):
            if job.status == TrainingStatus.RUNNING:
                job_manager.request_stop()
                stopped.append(True)

        job_manager.subscribe_status(callback)
        job_manager.request_start(sample_job, mock_adapter, {})
        assert len(stopped) > 0

    def test_callback_unsubscribes_self_safely(
        self, job_manager, mock_adapter, sample_job
    ):
        """Callback that unsubscribes itself does NOT break iteration."""
        results = []

        def self_removing(job):
            results.append("self")
            job_manager.unsubscribe_status(self_removing)

        def other(job):
            results.append("other")

        job_manager.subscribe_status(self_removing)
        job_manager.subscribe_status(other)
        mock_adapter.start.return_value = (True, "Started")
        job_manager.request_start(sample_job, mock_adapter, {})

        assert "self" in results
        assert "other" in results

    def test_callback_registers_new_callback_not_called_this_round(
        self, job_manager, mock_adapter, sample_job
    ):
        """Callback registering a new subscriber: new CB not in current snapshot, called next round."""
        calls = []

        def first_cb(job):
            calls.append(("first", job.status))
            job_manager.subscribe_status(second_cb)

        def second_cb(job):
            calls.append(("second", job.status))

        job_manager.subscribe_status(first_cb)
        mock_adapter.start.return_value = (True, "Started")
        job_manager.request_start(sample_job, mock_adapter, {})

        # second_cb registered during PREPARING callback → appears in RUNNING snapshot
        # So it fires during RUNNING notification, not PREPARING
        assert ("first", TrainingStatus.PREPARING) in calls
        assert ("second", TrainingStatus.RUNNING) in calls
        # second was NOT called during PREPARING (not in that snapshot)
        assert ("second", TrainingStatus.PREPARING) not in calls

    def test_callback_exception_isolated_lock_free(
        self, job_manager, mock_adapter, sample_job
    ):
        """Exception in one callback does NOT prevent others from executing."""
        results = []

        def bad(job):
            raise RuntimeError("crash")

        def good(job):
            results.append(job.status)

        job_manager.subscribe_status(bad)
        job_manager.subscribe_status(good)
        mock_adapter.start.return_value = (True, "Started")
        job_manager.request_start(sample_job, mock_adapter, {})

        assert TrainingStatus.RUNNING in results

    def test_adapter_start_outside_lock(
        self, job_manager, mock_adapter, sample_job
    ):
        """adapter.start() is called while NO _state_lock is held."""
        lock_held_during_start = []

        original_start = mock_adapter.start

        def tracking_start(job, config):
            lock_held_during_start.append(job_manager._state_lock._is_owned())
            return original_start(job, config)

        mock_adapter.start = tracking_start
        mock_adapter.start.return_value = (True, "Started")
        mock_adapter.can_start.return_value = (True, "")

        job_manager.reserve_job(sample_job, mock_adapter)
        job_manager.start_reserved_job(sample_job.job_id, {})

        assert len(lock_held_during_start) > 0
        assert lock_held_during_start[0] is False, (
            "adapter.start() was called while _state_lock was held!"
        )

    def test_full_lifecycle_callbacks_outside_lock(
        self, job_manager, mock_adapter, sample_job
    ):
        """PREPARING → RUNNING → STOPPED: all callbacks outside lock."""
        statuses_seen = []
        lock_held_during_cb = []

        def tracking_cb(job):
            statuses_seen.append(job.status)
            lock_held_during_cb.append(job_manager._state_lock._is_owned())

        job_manager.subscribe_status(tracking_cb)
        mock_adapter.start.return_value = (True, "Started")
        mock_adapter.stop.return_value = True

        job_manager.request_start(sample_job, mock_adapter, {})
        job_manager.request_stop()

        assert len(lock_held_during_cb) > 0
        for i, held in enumerate(lock_held_during_cb):
            assert held is False, (
                f"Callback {i} ran while lock held (status={statuses_seen[i] if i < len(statuses_seen) else '?'})"
            )

    def test_slow_callback_does_not_block_other_thread(
        self, job_manager, mock_adapter, sample_job
    ):
        """A slow callback does NOT prevent another thread from querying."""
        import time, threading
        slow_started = []

        def slow_cb(job):
            slow_started.append(True)
            time.sleep(0.3)

        job_manager.subscribe_status(slow_cb)
        mock_adapter.start.return_value = (True, "Started")

        def trigger():
            job_manager.request_start(sample_job, mock_adapter, {})

        t = threading.Thread(target=trigger)
        t.start()
        time.sleep(0.05)

        start = time.time()
        job = job_manager.get_current_job()
        elapsed = time.time() - start

        t.join(timeout=2)
        assert elapsed < 0.5, f"get_current_job() blocked for {elapsed:.1f}s"
        assert len(slow_started) > 0


class TestHistoryIntegration:
    """Verify JobManager writes to HistoryStore on every lifecycle state transition."""

    @pytest.fixture
    def history_store(self, tmp_path):
        """Isolated HistoryStore using tmp_path."""
        from anylabeling.services.training_center.history import HistoryStore
        store = HistoryStore(history_dir=tmp_path / "history")
        return store

    @pytest.fixture
    def job_manager_with_history(self, history_store):
        """JobManager with isolated HistoryStore injected."""
        from anylabeling.services.training_center.job_manager import JobManager
        jm = JobManager()
        jm._current_job = None
        jm._current_adapter = None
        jm._status_callbacks = []
        jm._event_callbacks = []
        jm._history_store = history_store  # Inject isolated store
        return jm

    def test_reserve_writes_preparing_record(
        self, job_manager_with_history, mock_adapter, sample_job, history_store
    ):
        """reserve_job creates a PREPARING history record."""
        jm = job_manager_with_history
        sample_job.task = "Detect"
        sample_job.model = "yolov8n.pt"
        sample_job.data = "data.yaml"

        jm.reserve_job(sample_job, mock_adapter)

        record = history_store.get_job(sample_job.job_id)
        assert record is not None
        assert record.status == "preparing"
        assert record.mode == "custom_script"
        assert record.display_name == "Test Job"

    def test_start_updates_to_running(
        self, job_manager_with_history, mock_adapter, sample_job, history_store
    ):
        """start_reserved_job updates the same record to RUNNING."""
        jm = job_manager_with_history
        mock_adapter.start.return_value = (True, "Started")

        jm.reserve_job(sample_job, mock_adapter)
        jm.start_reserved_job(sample_job.job_id, {"data": "test"})

        record = history_store.get_job(sample_job.job_id)
        assert record is not None
        assert record.status == "running"

    def test_complete_finalizes_record(
        self, job_manager_with_history, mock_adapter, sample_job, history_store
    ):
        """complete_job finalizes the record as COMPLETED."""
        jm = job_manager_with_history
        mock_adapter.start.return_value = (True, "Started")

        jm.request_start(sample_job, mock_adapter, {})
        jm.complete_job(sample_job.job_id)

        record = history_store.get_job(sample_job.job_id)
        assert record is not None
        assert record.status == "completed"

    def test_stop_finalizes_record(
        self, job_manager_with_history, mock_adapter, sample_job, history_store
    ):
        """request_stop during RUNNING finalizes as STOPPED."""
        jm = job_manager_with_history
        mock_adapter.start.return_value = (True, "Started")
        mock_adapter.stop.return_value = True

        jm.request_start(sample_job, mock_adapter, {})
        jm.request_stop()
        jm.stop_job(sample_job.job_id)

        record = history_store.get_job(sample_job.job_id)
        assert record is not None
        assert record.status == "stopped"

    def test_fail_reserved_finalizes_record(
        self, job_manager_with_history, mock_adapter, sample_job, history_store
    ):
        """fail_reserved_job finalizes the record as FAILED."""
        jm = job_manager_with_history
        jm.reserve_job(sample_job, mock_adapter)
        jm.fail_reserved_job(sample_job.job_id, "dataset error")

        record = history_store.get_job(sample_job.job_id)
        assert record is not None
        assert record.status == "failed"
        assert record.error_message == "dataset error"

    def test_stop_preparing_finalizes_record(
        self, job_manager_with_history, mock_adapter, sample_job, history_store
    ):
        """request_stop during PREPARING finalizes as STOPPED."""
        jm = job_manager_with_history
        jm.reserve_job(sample_job, mock_adapter)
        jm.request_stop()

        record = history_store.get_job(sample_job.job_id)
        assert record is not None
        assert record.status == "stopped"

    def test_one_job_id_one_record(
        self, job_manager_with_history, mock_adapter, sample_job, history_store
    ):
        """A single job_id always maps to exactly one history record."""
        jm = job_manager_with_history
        mock_adapter.start.return_value = (True, "Started")

        jm.reserve_job(sample_job, mock_adapter)
        jm.start_reserved_job(sample_job.job_id, {})
        jm.complete_job(sample_job.job_id)

        jobs = history_store.list_jobs()
        matching = [j for j in jobs if j.job_id == sample_job.job_id]
        assert len(matching) == 1
        assert matching[0].status == "completed"

    def test_idle_not_written_to_history(
        self, job_manager_with_history, mock_adapter, sample_job, history_store
    ):
        """IDLE jobs are never written to history."""
        # Create a job, don't reserve it (stays IDLE)
        jobs_before = len(history_store.list_jobs())
        # Nothing should change — reserve_job is the entry point
        assert jobs_before == 0  # Fresh store has no records

    def test_orphan_recovery_on_startup(self, tmp_path, mock_adapter, sample_job):
        """Orphaned PREPARING/RUNNING records are marked FAILED on JobManager init."""
        from anylabeling.services.training_center.history import HistoryStore, JobHistoryRecord
        from anylabeling.services.training_center.job_manager import JobManager
        from datetime import datetime

        # Pre-populate history with an orphaned PREPARING record
        store = HistoryStore(history_dir=tmp_path / "history")
        orphan = JobHistoryRecord(
            job_id="orphan-1", mode="guided_ultralytics", status="running",
            created_at=datetime.now().isoformat(), display_name="Orphan Job",
        )
        store.append_job(orphan)

        # Creating a new JobManager should trigger recovery
        jm = JobManager()
        jm._history_store = store
        jm._recover_orphaned_jobs()

        recovered = store.get_job("orphan-1")
        assert recovered is not None
        assert recovered.status == "failed"
        assert "Application exited" in (recovered.error_message or "")

        # Cleanup singleton state
        JobManager._instance = None
