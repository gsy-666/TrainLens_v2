"""Test guided training UI state updates"""

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
    adapter.can_start = Mock(return_value=(True, ""))

    history_store = Mock()
    history_store.save_record = Mock()

    return job_manager, adapter, history_store


class TestGuidedTrainingUIStates:
    """Test UI state transitions via TrainingEvent"""

    def test_preparing_state_disables_start_button(self, qapp, mock_dependencies):
        """PREPARING event disables start button and shows status"""
        from anylabeling.views.training.guided_training_widget import GuidedTrainingWidget
        from anylabeling.services.training_center.event_protocol import TrainingEvent, TrainingEventType

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

        widget._current_job_id = "guided_123"
        widget.ensure_train_tab_initialized()

        event = TrainingEvent(
            schema_version=1,
            job_id="guided_123",
            event_type=TrainingEventType.PREPARING,
            timestamp=1234567890.0,
            payload={}
        )

        widget._on_unified_training_event(event)

        assert widget.training_status == "preparing"
        assert not widget.start_training_button.isEnabled()

    def test_running_state_shows_stop_button(self, qapp, mock_dependencies):
        """PROCESS_STARTED event shows stop button and hides start"""
        from anylabeling.views.training.guided_training_widget import GuidedTrainingWidget
        from anylabeling.services.training_center.event_protocol import TrainingEvent, TrainingEventType

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

        widget._current_job_id = "guided_456"
        widget.ensure_train_tab_initialized()

        event = TrainingEvent(
            schema_version=1,
            job_id="guided_456",
            event_type=TrainingEventType.PROCESS_STARTED,
            timestamp=1234567890.0,
            payload={"total_epochs": 100}
        )

        widget._on_unified_training_event(event)

        assert widget.training_status == "training"
        # Use isHidden() instead of isVisible() - widget parent chain not shown in test
        assert widget.start_training_button.isHidden()  # start hidden
        assert not widget.stop_training_button.isHidden()  # stop not hidden (visible)
        assert widget.stop_training_button.isEnabled()

    def test_console_output_appends_once(self, qapp, mock_dependencies):
        """CONSOLE_OUTPUT appends message only once"""
        from anylabeling.views.training.guided_training_widget import GuidedTrainingWidget
        from anylabeling.services.training_center.event_protocol import TrainingEvent, TrainingEventType

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

        widget._current_job_id = "guided_789"
        widget.ensure_train_tab_initialized()
        widget.log_display.clear()

        event = TrainingEvent(
            schema_version=1,
            job_id="guided_789",
            event_type=TrainingEventType.CONSOLE_OUTPUT,
            timestamp=1234567890.0,
            payload={"message": "Epoch 1/100"}
        )

        widget._on_unified_training_event(event)
        log_text = widget.log_display.toPlainText()

        assert "Epoch 1/100" in log_text
        assert log_text.count("Epoch 1/100") == 1

    def test_completed_state_shows_export_button(self, qapp, mock_dependencies):
        """COMPLETED event shows export button and hides stop"""
        from anylabeling.views.training.guided_training_widget import GuidedTrainingWidget
        from anylabeling.services.training_center.event_protocol import TrainingEvent, TrainingEventType
        from anylabeling.services.training_center.models import TrainingJob, TrainingMode, TrainingStatus
        from datetime import datetime

        job_manager, adapter, history_store = mock_dependencies

        # Mock get_current_job to return a job
        mock_job = TrainingJob(
            job_id="guided_complete",
            mode=TrainingMode.GUIDED_ULTRALYTICS,
            status=TrainingStatus.COMPLETED,
            started_at=datetime.now(),
            ended_at=datetime.now(),
            metadata={"task": "detect", "model": "yolov8n.pt"}
        )
        job_manager.get_current_job = Mock(return_value=mock_job)

        widget = GuidedTrainingWidget(
            parent=None,
            image_list=[],
            output_dir="/tmp/test",
            supported_shape=[],
            job_manager=job_manager,
            ultralytics_adapter=adapter,
            history_store=history_store
        )

        widget._current_job_id = "guided_complete"
        widget.ensure_train_tab_initialized()

        event = TrainingEvent(
            schema_version=1,
            job_id="guided_complete",
            event_type=TrainingEventType.COMPLETED,
            timestamp=1234567890.0,
            payload={}
        )

        widget._on_unified_training_event(event)

        assert widget.training_status == "completed"
        # Use isHidden() instead of isVisible() - widget parent chain not shown in test
        assert widget.stop_training_button.isHidden()  # stop hidden
        assert not widget.export_button.isHidden()  # export not hidden (visible)
        assert widget._current_job_id is None  # Job cleared after terminal state

    def test_failed_state_shows_error(self, qapp, mock_dependencies):
        """FAILED event shows error message"""
        from anylabeling.views.training.guided_training_widget import GuidedTrainingWidget
        from anylabeling.services.training_center.event_protocol import TrainingEvent, TrainingEventType
        from anylabeling.services.training_center.models import TrainingJob, TrainingMode, TrainingStatus
        from datetime import datetime

        job_manager, adapter, history_store = mock_dependencies

        mock_job = TrainingJob(
            job_id="guided_fail",
            mode=TrainingMode.GUIDED_ULTRALYTICS,
            status=TrainingStatus.FAILED,
            ended_at=datetime.now(),
            metadata={}
        )
        job_manager.get_current_job = Mock(return_value=mock_job)

        widget = GuidedTrainingWidget(
            parent=None,
            image_list=[],
            output_dir="/tmp/test",
            supported_shape=[],
            job_manager=job_manager,
            ultralytics_adapter=adapter,
            history_store=history_store
        )

        widget._current_job_id = "guided_fail"
        widget.ensure_train_tab_initialized()
        widget.log_display.clear()

        event = TrainingEvent(
            schema_version=1,
            job_id="guided_fail",
            event_type=TrainingEventType.FAILED,
            timestamp=1234567890.0,
            payload={"error": "CUDA out of memory"}
        )

        widget._on_unified_training_event(event)

        assert widget.training_status == "error"
        assert "CUDA out of memory" in widget.log_display.toPlainText()

    def test_stopped_state_re_enables_buttons(self, qapp, mock_dependencies):
        """STOPPED event re-enables stop button for next run"""
        from anylabeling.views.training.guided_training_widget import GuidedTrainingWidget
        from anylabeling.services.training_center.event_protocol import TrainingEvent, TrainingEventType
        from anylabeling.services.training_center.models import TrainingJob, TrainingMode, TrainingStatus
        from datetime import datetime

        job_manager, adapter, history_store = mock_dependencies

        mock_job = TrainingJob(
            job_id="guided_stop",
            mode=TrainingMode.GUIDED_ULTRALYTICS,
            status=TrainingStatus.STOPPED,
            ended_at=datetime.now(),
            metadata={}
        )
        job_manager.get_current_job = Mock(return_value=mock_job)

        widget = GuidedTrainingWidget(
            parent=None,
            image_list=[],
            output_dir="/tmp/test",
            supported_shape=[],
            job_manager=job_manager,
            ultralytics_adapter=adapter,
            history_store=history_store
        )

        widget._current_job_id = "guided_stop"
        widget.ensure_train_tab_initialized()

        event = TrainingEvent(
            schema_version=1,
            job_id="guided_stop",
            event_type=TrainingEventType.STOPPED,
            timestamp=1234567890.0,
            payload={}
        )

        widget._on_unified_training_event(event)

        assert widget.training_status == "stop"
        assert widget.stop_training_button.isEnabled()  # Re-enabled
        assert widget._current_job_id is None

    def test_ignores_events_from_other_jobs(self, qapp, mock_dependencies):
        """Widget ignores events from other job_ids"""
        from anylabeling.views.training.guided_training_widget import GuidedTrainingWidget
        from anylabeling.services.training_center.event_protocol import TrainingEvent, TrainingEventType

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

        widget._current_job_id = "guided_mine"
        widget.ensure_train_tab_initialized()
        widget.training_status = "idle"

        # Event from different job
        event = TrainingEvent(
            schema_version=1,
            job_id="custom_other",
            event_type=TrainingEventType.PROCESS_STARTED,
            timestamp=1234567890.0,
            payload={}
        )

        widget._on_unified_training_event(event)

        # Status should not change
        assert widget.training_status == "idle"
