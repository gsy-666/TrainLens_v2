"""Tests for Ultralytics Adapter

Verifies event mapping from TrainingManager to unified protocol.
"""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
from pathlib import Path

from anylabeling.services.training_center.models import (
    TrainingMode,
    TrainingStatus,
    TrainingJob,
)
from anylabeling.services.training_center.event_protocol import TrainingEventType
from anylabeling.services.training_center.adapters.ultralytics_adapter import (
    UltralyticsAdapter,
)


@pytest.fixture
def mock_training_manager():
    """Mock Ultralytics TrainingManager"""
    manager = Mock()
    manager.is_training = False
    manager.callbacks = []
    manager.start_training = Mock(return_value=(True, "Started"))
    manager.stop_training = Mock(return_value=True)
    return manager


@pytest.fixture
def adapter(mock_training_manager):
    """Create UltralyticsAdapter with mocked TrainingManager"""
    with patch(
        'anylabeling.services.training_center.adapters.ultralytics_adapter.get_training_manager',
        return_value=mock_training_manager
    ):
        adapter = UltralyticsAdapter()
        adapter.manager = mock_training_manager
        return adapter


@pytest.fixture
def sample_job():
    """Create sample training job"""
    return TrainingJob(
        job_id="ultralytics-job-001",
        mode=TrainingMode.GUIDED_ULTRALYTICS,
        status=TrainingStatus.IDLE,
        created_at=datetime.now(),
        started_at=None,
        ended_at=None,
        workspace=Path("/tmp/workspace"),
        output_directory=None,
        display_name="YOLOv8 Training",
        framework="ultralytics",
        python_executable=None,
        command=None,
        metadata={},
        error_message=None,
    )


class TestUltralyticsAdapterInterface:
    """Test adapter interface implementation"""

    def test_can_start_when_idle(self, adapter, mock_training_manager):
        """can_start returns True when not training"""
        mock_training_manager.is_training = False

        can_start, reason = adapter.can_start()

        assert can_start is True
        assert reason == ""

    def test_can_start_when_busy(self, adapter, mock_training_manager):
        """can_start returns False when already training"""
        mock_training_manager.is_training = True

        can_start, reason = adapter.can_start()

        assert can_start is False
        assert "already in progress" in reason.lower()

    def test_start_training(self, adapter, sample_job, mock_training_manager):
        """start delegates to TrainingManager.start_training"""
        config = {"model": "yolov8n.pt", "epochs": 100}

        success, message = adapter.start(sample_job, config)

        assert success is True
        assert mock_training_manager.start_training.called
        mock_training_manager.start_training.assert_called_with(config)

    def test_start_with_invalid_config(self, adapter, sample_job):
        """start rejects non-dict config"""
        success, message = adapter.start(sample_job, "invalid")

        assert success is False
        assert "must be a dict" in message.lower()

    def test_stop_training(self, adapter, mock_training_manager):
        """stop delegates to TrainingManager.stop_training"""
        result = adapter.stop()

        assert result is True
        assert mock_training_manager.stop_training.called

    def test_is_running(self, adapter, mock_training_manager):
        """is_running reflects TrainingManager state"""
        mock_training_manager.is_training = False
        assert adapter.is_running() is False

        mock_training_manager.is_training = True
        assert adapter.is_running() is True


class TestEventMapping:
    """Test Ultralytics event mapping to unified protocol"""

    def test_training_started_mapped_to_process_started(self, adapter, sample_job):
        """training_started -> process_started"""
        events = []
        adapter.subscribe(lambda e: events.append(e))
        adapter._current_job_id = sample_job.job_id

        adapter._on_training_event("training_started", {"total_epochs": 100})

        assert len(events) == 1
        event = events[0]
        assert event.event_type == TrainingEventType.PROCESS_STARTED
        assert event.job_id == sample_job.job_id
        assert event.source == "ultralytics"
        assert event.payload.get("total_epochs") == 100

    def test_training_log_mapped_to_console_output(self, adapter, sample_job):
        """training_log -> console_output"""
        events = []
        adapter.subscribe(lambda e: events.append(e))
        adapter._current_job_id = sample_job.job_id

        adapter._on_training_event("training_log", {"message": "Epoch 1/100"})

        assert len(events) == 1
        event = events[0]
        assert event.event_type == TrainingEventType.CONSOLE_OUTPUT
        assert event.payload["message"] == "Epoch 1/100"
        assert event.payload["stream"] == "stdout"
        assert event.source == "ultralytics"

    def test_training_completed_mapped_to_completed(self, adapter, sample_job):
        """training_completed -> completed"""
        events = []
        adapter.subscribe(lambda e: events.append(e))
        adapter._current_job_id = sample_job.job_id

        adapter._on_training_event("training_completed", {"results": "Success"})

        assert len(events) == 1
        event = events[0]
        assert event.event_type == TrainingEventType.COMPLETED
        assert event.source == "ultralytics"
        assert event.payload.get("results") == "Success"

    def test_training_error_mapped_to_failed(self, adapter, sample_job):
        """training_error -> failed"""
        events = []
        adapter.subscribe(lambda e: events.append(e))
        adapter._current_job_id = sample_job.job_id

        adapter._on_training_event("training_error", {"error": "CUDA out of memory"})

        assert len(events) == 1
        event = events[0]
        assert event.event_type == TrainingEventType.FAILED
        assert event.payload["error"] == "CUDA out of memory"
        assert event.source == "ultralytics"

    def test_training_stopped_mapped_to_stopped(self, adapter, sample_job):
        """training_stopped -> stopped"""
        events = []
        adapter.subscribe(lambda e: events.append(e))
        adapter._current_job_id = sample_job.job_id

        adapter._on_training_event("training_stopped", {})

        assert len(events) == 1
        event = events[0]
        assert event.event_type == TrainingEventType.STOPPED
        assert event.source == "ultralytics"

    def test_unknown_event_type_ignored(self, adapter, sample_job):
        """Unknown event types are silently ignored"""
        events = []
        adapter.subscribe(lambda e: events.append(e))
        adapter._current_job_id = sample_job.job_id

        adapter._on_training_event("unknown_event", {"data": "test"})

        assert len(events) == 0

    def test_event_without_job_id_ignored(self, adapter):
        """Events without current_job_id are ignored"""
        events = []
        adapter.subscribe(lambda e: events.append(e))
        adapter._current_job_id = None

        adapter._on_training_event("training_log", {"message": "test"})

        assert len(events) == 0


class TestCallbackManagement:
    """Test callback subscription/unsubscription"""

    def test_subscribe_callback(self, adapter):
        """Can subscribe to events"""
        callback = Mock()
        adapter.subscribe(callback)

        assert callback in adapter._callbacks

    def test_unsubscribe_callback(self, adapter):
        """Can unsubscribe from events"""
        callback = Mock()
        adapter.subscribe(callback)
        adapter.unsubscribe(callback)

        assert callback not in adapter._callbacks

    def test_duplicate_subscription_prevented(self, adapter):
        """Subscribing same callback twice has no effect"""
        callback = Mock()
        adapter.subscribe(callback)
        adapter.subscribe(callback)

        assert adapter._callbacks.count(callback) == 1

    def test_callback_receives_events(self, adapter, sample_job):
        """Subscribed callbacks receive events"""
        received = []
        adapter.subscribe(lambda e: received.append(e))
        adapter._current_job_id = sample_job.job_id

        adapter._on_training_event("training_log", {"message": "test"})

        assert len(received) == 1

    def test_callback_exception_isolated(self, adapter, sample_job):
        """Exception in one callback doesn't break others"""
        results = []

        def bad_callback(event):
            raise ValueError("Bad callback")

        def good_callback(event):
            results.append(event.event_type)

        adapter.subscribe(bad_callback)
        adapter.subscribe(good_callback)
        adapter._current_job_id = sample_job.job_id

        adapter._on_training_event("training_log", {"message": "test"})

        assert TrainingEventType.CONSOLE_OUTPUT in results


class TestAdapterComposition:
    """Test adapter uses composition, not inheritance"""

    def test_adapter_wraps_training_manager(self, adapter, mock_training_manager):
        """Adapter holds reference to TrainingManager"""
        assert adapter.manager is mock_training_manager

    def test_adapter_registers_callback(self, adapter, mock_training_manager):
        """Adapter registers itself with TrainingManager callbacks"""
        assert adapter._on_training_event in mock_training_manager.callbacks

    def test_adapter_preserves_original_callbacks(self, adapter):
        """Adapter preserves original callback list"""
        assert hasattr(adapter, '_original_callbacks')
        assert isinstance(adapter._original_callbacks, list)
