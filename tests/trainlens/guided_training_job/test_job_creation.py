"""Test guided training job lifecycle"""

import sys
import os
import pytest
from unittest.mock import Mock, MagicMock, patch
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
def mock_job_manager():
    """Mock JobManager"""
    manager = Mock()
    manager.get_current_job = Mock(return_value=None)
    manager.request_start = Mock(return_value=(True, ""))
    manager.request_stop = Mock(return_value=True)
    manager.subscribe_events = Mock()
    manager.unsubscribe_events = Mock()
    return manager


@pytest.fixture
def mock_adapter():
    """Mock UltralyticsAdapter"""
    adapter = Mock()
    adapter.can_start = Mock(return_value=(True, ""))
    adapter.start = Mock(return_value=(True, ""))
    adapter.stop = Mock(return_value=True)
    return adapter


@pytest.fixture
def mock_history_store():
    """Mock HistoryStore"""
    store = Mock()
    store.save_record = Mock()
    return store


class TestGuidedTrainingJobCreation:
    """Test TrainingJob creation from config"""

    def test_start_creates_training_job_with_correct_mode(self, qapp, mock_job_manager, mock_adapter, mock_history_store):
        """Start creates TrainingJob with GUIDED_ULTRALYTICS mode"""
        from anylabeling.views.training.guided_training_widget import GuidedTrainingWidget
        from anylabeling.services.training_center.models import TrainingMode

        widget = GuidedTrainingWidget(
            parent=None,
            image_list=[],
            output_dir="/tmp/test",
            supported_shape=[],
            job_manager=mock_job_manager,
            ultralytics_adapter=mock_adapter,
            history_store=mock_history_store
        )

        widget.selected_task_type = "Detect"
        widget.current_project_path = "/tmp/test/project"

        # Mock both get_current_config and get_training_args
        widget.get_current_config = Mock(return_value={
            "basic": {"project": "/tmp/test", "name": "project"}
        })
        widget.get_training_args = Mock(return_value={
            "model": "yolov8n.pt",
            "data": "/tmp/data.yaml",
            "epochs": 10,
            "batch": 16,
            "imgsz": 640,
            "device": "cpu",
            "project": "/tmp/test",
            "name": "project"
        })

        widget.start_training_from_train_tab()

        # Verify request_start was called
        mock_job_manager.request_start.assert_called_once()
        call_args = mock_job_manager.request_start.call_args
        job = call_args[1]['job']

        assert job.mode == TrainingMode.GUIDED_ULTRALYTICS
        assert job.job_id.startswith("guided_")

    def test_start_passes_complete_config_to_adapter(self, qapp, mock_job_manager, mock_adapter, mock_history_store):
        """Start passes complete training config to adapter"""
        from anylabeling.views.training.guided_training_widget import GuidedTrainingWidget

        widget = GuidedTrainingWidget(
            parent=None,
            image_list=[],
            output_dir="/tmp/test",
            supported_shape=[],
            job_manager=mock_job_manager,
            ultralytics_adapter=mock_adapter,
            history_store=mock_history_store
        )

        widget.selected_task_type = "Segment"
        widget.current_project_path = "/tmp/test/project"

        train_args = {
            "model": "yolov8n-seg.pt",
            "data": "/tmp/data.yaml",
            "epochs": 50,
            "batch": 32,
            "imgsz": 1024,
            "device": [0, 1],
            "project": "/tmp/test",
            "name": "seg_project",
            "lr0": 0.01,
            "optimizer": "SGD"
        }

        widget.get_current_config = Mock(return_value={
            "basic": {"project": "/tmp/test", "name": "seg_project"}
        })
        widget.get_training_args = Mock(return_value=train_args)
        widget.start_training_from_train_tab()

        # Verify config passed to adapter
        mock_job_manager.request_start.assert_called_once()
        call_args = mock_job_manager.request_start.call_args
        config = call_args[1]['config']

        assert config == train_args
        assert config["epochs"] == 50
        assert config["optimizer"] == "SGD"
