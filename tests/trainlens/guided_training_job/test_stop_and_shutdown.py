"""Test stop functionality and shutdown behavior"""

import sys
import os
import pytest
from unittest.mock import Mock
from PyQt6.QtWidgets import QApplication

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))


@pytest.fixture(scope="session")
def qapp():
    """Create QApplication for real Qt tests"""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def mock_dependencies():
    """Mock JobManager, Adapter, HistoryStore"""
    job_manager = Mock()
    job_manager.get_current_job = Mock(return_value=None)
    job_manager.request_start = Mock(return_value=(True, ""))
    job_manager.request_stop = Mock(return_value=True)
    job_manager.subscribe_events = Mock()
    job_manager.unsubscribe_events = Mock()

    adapter = Mock()
    history_store = Mock()

    return job_manager, adapter, history_store


class TestStopAndShutdown:
    """Test stop request and widget shutdown"""

    def test_request_stop_is_idempotent(self, qapp, mock_dependencies):
        """request_stop can be called multiple times safely"""
        from anylabeling.views.training.guided_training_widget import GuidedTrainingWidget

        job_manager, adapter, history_store = mock_dependencies
        widget = GuidedTrainingWidget(
            parent=None,
            image_list=[],
            output_dir="/tmp/test",
            supported_shape=[],
            job_manager=job_manager,
            ultralytics_adapter=adapter,
            history_store=history_store
        )

        widget.training_status = "training"

        # First stop
        widget.request_stop()
        first_call_count = job_manager.request_stop.call_count

        # JobManager should make second call idempotent
        job_manager.request_stop = Mock(return_value=False)

        # Second stop
        widget.request_stop()

        # Both calls should go through
        assert job_manager.request_stop.call_count == 1

    def test_shutdown_unsubscribes_from_events(self, qapp, mock_dependencies):
        """shutdown() unsubscribes from JobManager events"""
        from anylabeling.views.training.guided_training_widget import GuidedTrainingWidget

        job_manager, adapter, history_store = mock_dependencies
        widget = GuidedTrainingWidget(
            parent=None,
            image_list=[],
            output_dir="/tmp/test",
            supported_shape=[],
            job_manager=job_manager,
            ultralytics_adapter=adapter,
            history_store=history_store
        )

        widget.shutdown()

        job_manager.unsubscribe_events.assert_called_once()

    def test_shutdown_only_stops_guided_jobs(self, qapp, mock_dependencies):
        """shutdown() only stops jobs belonging to guided mode"""
        from anylabeling.views.training.guided_training_widget import GuidedTrainingWidget
        from anylabeling.services.training_center.models import TrainingJob, TrainingMode, TrainingStatus

        job_manager, adapter, history_store = mock_dependencies

        # Mock a guided job
        guided_job = TrainingJob(
            job_id="guided_123",
            mode=TrainingMode.GUIDED_ULTRALYTICS,
            status=TrainingStatus.RUNNING
        )
        job_manager.get_current_job = Mock(return_value=guided_job)

        widget = GuidedTrainingWidget(
            parent=None,
            image_list=[],
            output_dir="/tmp/test",
            supported_shape=[],
            job_manager=job_manager,
            ultralytics_adapter=adapter,
            history_store=history_store
        )

        widget._current_job_id = "guided_123"
        widget.shutdown()

        # Should request stop for guided job
        job_manager.request_stop.assert_called_once()

    def test_shutdown_does_not_stop_custom_jobs(self, qapp, mock_dependencies):
        """shutdown() does not stop custom project jobs"""
        from anylabeling.views.training.guided_training_widget import GuidedTrainingWidget
        from anylabeling.services.training_center.models import TrainingJob, TrainingMode, TrainingStatus

        job_manager, adapter, history_store = mock_dependencies

        # Mock a custom job
        custom_job = TrainingJob(
            job_id="custom_456",
            mode=TrainingMode.CUSTOM_SCRIPT,
            status=TrainingStatus.RUNNING
        )
        job_manager.get_current_job = Mock(return_value=custom_job)

        widget = GuidedTrainingWidget(
            parent=None,
            image_list=[],
            output_dir="/tmp/test",
            supported_shape=[],
            job_manager=job_manager,
            ultralytics_adapter=adapter,
            history_store=history_store
        )

        widget._current_job_id = "guided_789"  # Different job
        widget.shutdown()

        # Should not request stop
        job_manager.request_stop.assert_not_called()

    def test_shutdown_is_idempotent(self, qapp, mock_dependencies):
        """shutdown() can be called multiple times safely"""
        from anylabeling.views.training.guided_training_widget import GuidedTrainingWidget

        job_manager, adapter, history_store = mock_dependencies
        widget = GuidedTrainingWidget(
            parent=None,
            image_list=[],
            output_dir="/tmp/test",
            supported_shape=[],
            job_manager=job_manager,
            ultralytics_adapter=adapter,
            history_store=history_store
        )

        # Should not raise exception
        widget.shutdown()
        widget.shutdown()
        widget.shutdown()

        # Unsubscribe should have been called multiple times but safely
        assert job_manager.unsubscribe_events.call_count >= 1
