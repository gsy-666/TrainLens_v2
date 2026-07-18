"""Mutual exclusion tests — Guided Training vs Run Monitor (Custom script).

Verifies that both production GUI entry points share the same JobManager
singleton and that JobManager enforces only-one-active-job.
"""

import pytest
import sys
import os
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, PropertyMock

from PyQt6.QtWidgets import QApplication

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from anylabeling.services.training_center.job_manager import JobManager, get_job_manager
from anylabeling.services.training_center.models import (
    TrainingMode,
    TrainingStatus,
    TrainingJob,
)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def fresh_job_manager():
    """Clean JobManager for each test."""
    manager = JobManager()
    # Reset any leftover state from previous tests
    with manager._state_lock:
        manager._current_job = None
        manager._current_adapter = None
    manager._status_callbacks.clear()
    manager._event_callbacks.clear()
    return manager


@pytest.fixture
def sample_guided_job():
    return TrainingJob(
        job_id="guided-001",
        mode=TrainingMode.GUIDED_ULTRALYTICS,
        status=TrainingStatus.IDLE,
        created_at=datetime.now(),
        workspace=Path("/tmp/guided"),
        output_directory=Path("/tmp/guided/output"),
        display_name="Guided: test",
        framework="ultralytics",
        command=[],
        metadata={},
    )


@pytest.fixture
def sample_custom_job():
    return TrainingJob(
        job_id="custom-001",
        mode=TrainingMode.CUSTOM_SCRIPT,
        status=TrainingStatus.IDLE,
        created_at=datetime.now(),
        workspace=Path("/tmp/custom"),
        output_directory=None,
        display_name="Custom: train.py",
        framework="pytorch",
        python_executable=Path("python"),
        command=["python", "train.py"],
        metadata={},
    )


@pytest.fixture
def accepting_adapter():
    """Mock adapter that always says yes."""
    adapter = Mock()
    adapter.can_start.return_value = (True, "")
    adapter.start.return_value = (True, "Started OK")
    adapter.stop.return_value = True
    adapter.is_running.return_value = False
    adapter.subscribe = Mock()
    adapter.unsubscribe = Mock()
    return adapter


# ── Singleton sharing tests ────────────────────────────────────────────

class TestSharedJobManagerSingleton:
    """Both RunMonitor and GuidedTraining use the same JobManager instance."""

    def test_run_monitor_uses_job_manager_singleton(self, qapp):
        """RunMonitorWidget.job_manager IS get_job_manager()"""
        # Reset singleton for a clean test
        import anylabeling.services.training_center.job_manager as jm
        jm._manager_instance = None

        from anylabeling.views.run_monitor.run_monitor_widget import RunMonitorWidget
        widget = RunMonitorWidget(parent=None)

        assert widget.job_manager is get_job_manager()
        assert isinstance(widget.job_manager, JobManager)

    def test_guided_training_uses_job_manager_singleton(self, qapp):
        """GuidedTrainingWidget.job_manager IS get_job_manager()"""
        import anylabeling.services.training_center.job_manager as jm
        jm._manager_instance = None

        from anylabeling.views.training.guided_training_widget import GuidedTrainingWidget
        widget = GuidedTrainingWidget(
            parent=None,
            image_list=[],
            output_dir="/tmp/test",
            supported_shape=["rectangle"],
        )

        assert widget.job_manager is get_job_manager()
        assert isinstance(widget.job_manager, JobManager)

    def test_both_widgets_share_same_manager(self, qapp):
        """RunMonitorWidget and GuidedTrainingWidget share the same singleton."""
        import anylabeling.services.training_center.job_manager as jm
        jm._manager_instance = None

        from anylabeling.views.run_monitor.run_monitor_widget import RunMonitorWidget
        from anylabeling.views.training.guided_training_widget import GuidedTrainingWidget

        run_widget = RunMonitorWidget(parent=None)
        guided_widget = GuidedTrainingWidget(
            parent=None,
            image_list=[],
            output_dir="/tmp/test",
            supported_shape=["rectangle"],
        )

        assert run_widget.job_manager is guided_widget.job_manager
        assert run_widget.job_manager is get_job_manager()

    def test_reconstruct_keeps_singleton(self, qapp):
        """After closing and recreating widget, same singleton persists."""
        import anylabeling.services.training_center.job_manager as jm
        jm._manager_instance = None

        from anylabeling.views.training.guided_training_widget import GuidedTrainingWidget

        w1 = GuidedTrainingWidget(
            parent=None,
            image_list=[],
            output_dir="/tmp/test",
            supported_shape=["rectangle"],
        )
        jm1 = w1.job_manager

        # Close first widget (just delete reference)
        del w1

        w2 = GuidedTrainingWidget(
            parent=None,
            image_list=[],
            output_dir="/tmp/test",
            supported_shape=["rectangle"],
        )
        jm2 = w2.job_manager

        assert jm1 is jm2
        assert jm2 is get_job_manager()


# ── Mutual exclusion behaviour tests ───────────────────────────────────

class TestMutualExclusion:
    """JobManager enforces only one active job at a time."""

    def test_second_job_rejected_when_first_active(
        self, fresh_job_manager, sample_guided_job, sample_custom_job, accepting_adapter
    ):
        """Custom active → Guided rejected."""
        jm = fresh_job_manager

        # Start Custom first
        ok, msg = jm.request_start(sample_custom_job, accepting_adapter, {})
        assert ok, f"First job should start: {msg}"
        assert jm.get_current_job() is not None

        # Try Guided second
        adapter2 = Mock()
        adapter2.can_start.return_value = (True, "")
        adapter2.start.return_value = (True, "OK")
        adapter2.subscribe = Mock()
        adapter2.unsubscribe = Mock()

        ok2, msg2 = jm.request_start(sample_guided_job, adapter2, {})
        assert not ok2, "Second job should be rejected"
        assert "already in progress" in msg2.lower() or "already" in msg2.lower()
        adapter2.start.assert_not_called()

    def test_guided_active_custom_rejected(
        self, fresh_job_manager, sample_guided_job, sample_custom_job, accepting_adapter
    ):
        """Guided active → Custom rejected."""
        jm = fresh_job_manager

        # Start Guided first
        ok, msg = jm.request_start(sample_guided_job, accepting_adapter, {})
        assert ok, f"First job should start: {msg}"

        # Try Custom second
        adapter2 = Mock()
        adapter2.can_start.return_value = (True, "")
        adapter2.start.return_value = (True, "OK")
        adapter2.subscribe = Mock()
        adapter2.unsubscribe = Mock()

        ok2, msg2 = jm.request_start(sample_custom_job, adapter2, {})
        assert not ok2, "Second job should be rejected"
        adapter2.start.assert_not_called()

    def test_adapter_start_not_called_on_rejection(
        self, fresh_job_manager, sample_guided_job, sample_custom_job, accepting_adapter
    ):
        """Adapter.start() is never called when job is rejected."""
        jm = fresh_job_manager

        jm.request_start(sample_custom_job, accepting_adapter, {})

        adapter2 = Mock()
        adapter2.can_start.return_value = (True, "")
        adapter2.start = Mock(return_value=(True, "Should not be called"))
        adapter2.subscribe = Mock()
        adapter2.unsubscribe = Mock()

        jm.request_start(sample_guided_job, adapter2, {})
        adapter2.start.assert_not_called()

    def test_second_job_succeeds_after_first_completes(
        self, fresh_job_manager, sample_guided_job, sample_custom_job, accepting_adapter
    ):
        """After first job completes, second can start."""
        jm = fresh_job_manager

        jm.request_start(sample_custom_job, accepting_adapter, {})
        jm.complete_job(sample_custom_job.job_id)

        adapter2 = Mock()
        adapter2.can_start.return_value = (True, "")
        adapter2.start.return_value = (True, "OK")
        adapter2.subscribe = Mock()
        adapter2.unsubscribe = Mock()

        ok, msg = jm.request_start(sample_guided_job, adapter2, {})
        assert ok, f"Second job should start after first completed: {msg}"

    def test_second_job_succeeds_after_first_stopped(
        self, fresh_job_manager, sample_guided_job, sample_custom_job, accepting_adapter
    ):
        """After first job is stopped, second can start."""
        jm = fresh_job_manager

        jm.request_start(sample_custom_job, accepting_adapter, {})
        jm.stop_job(sample_custom_job.job_id)

        adapter2 = Mock()
        adapter2.can_start.return_value = (True, "")
        adapter2.start.return_value = (True, "OK")
        adapter2.subscribe = Mock()
        adapter2.unsubscribe = Mock()

        ok, msg = jm.request_start(sample_guided_job, adapter2, {})
        assert ok, f"Second job should start after first stopped: {msg}"
