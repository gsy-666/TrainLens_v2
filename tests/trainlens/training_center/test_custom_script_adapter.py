"""Tests for Custom Script Adapter

Verifies event mapping from ProcessManager to unified protocol.
"""

import pytest
import time
import sys
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
from pathlib import Path

from anylabeling.services.training_center.models import (
    TrainingMode,
    TrainingStatus,
    TrainingJob,
)
from anylabeling.services.training_center.event_protocol import TrainingEventType
from anylabeling.services.training_center.adapters.custom_script_adapter import (
    CustomScriptAdapter,
)


@pytest.fixture
def mock_process_manager():
    """Mock Run Monitor ProcessManager"""
    manager = Mock()
    manager.is_running = Mock(return_value=False)
    manager.start = Mock(return_value=True)
    manager.stop = Mock(return_value=True)
    manager._stop_requested = False  # Initialize flag for terminal state detection

    # Mock PyQt6 signals
    manager.process_started = Mock()
    manager.process_started.connect = Mock()
    manager.process_finished = Mock()
    manager.process_finished.connect = Mock()
    manager.stdout_ready = Mock()
    manager.stdout_ready.connect = Mock()
    manager.stderr_ready = Mock()
    manager.stderr_ready.connect = Mock()

    return manager


@pytest.fixture
def adapter(mock_process_manager):
    """Create CustomScriptAdapter with mocked ProcessManager"""
    with patch(
        'anylabeling.services.run_monitor.process_manager.ProcessManager',
        return_value=mock_process_manager
    ):
        adapter = CustomScriptAdapter()
        adapter.manager = mock_process_manager
        return adapter


@pytest.fixture
def sample_job():
    """Create sample training job"""
    return TrainingJob(
        job_id="custom-job-001",
        mode=TrainingMode.CUSTOM_SCRIPT,
        status=TrainingStatus.IDLE,
        created_at=datetime.now(),
        started_at=None,
        ended_at=None,
        workspace=Path("/tmp/workspace"),
        output_directory=None,
        display_name="Custom Training",
        framework="custom",
        python_executable=Path("/usr/bin/python3"),
        command=["python", "train.py"],
        metadata={},
        error_message=None,
    )


class TestCustomScriptAdapterInterface:
    """Test adapter interface implementation"""

    def test_can_start_when_idle(self, adapter, mock_process_manager):
        """can_start returns True when not running"""
        mock_process_manager.is_running.return_value = False

        can_start, reason = adapter.can_start()

        assert can_start is True
        assert reason == ""

    def test_can_start_when_busy(self, adapter, mock_process_manager):
        """can_start returns False when already running"""
        mock_process_manager.is_running.return_value = True

        can_start, reason = adapter.can_start()

        assert can_start is False
        assert "already in progress" in reason.lower()

    def test_start_with_valid_config(self, adapter, sample_job, mock_process_manager, tmp_path):
        """start creates Run and delegates to ProcessManager"""
        # Create real temp files for validation
        script_file = tmp_path / "train.py"
        script_file.write_text("print('training')")
        sample_job.workspace = tmp_path
        sample_job.python_executable = Path(sys.executable)

        config = {
            'script_path': str(script_file),
            'arguments': ['--epochs', '100']
        }

        with patch('anylabeling.services.run_monitor.models.Run') as MockRun:
            success, message = adapter.start(sample_job, config)

        assert success is True
        assert message == "Process started"
        assert mock_process_manager.start.called

    def test_start_without_script_path(self, adapter, sample_job):
        """start fails without script_path"""
        config = {'arguments': ['--epochs', '100']}

        success, message = adapter.start(sample_job, config)

        assert success is False
        assert "script_path" in message.lower()

    def test_start_with_non_dict_config(self, adapter, sample_job):
        """start rejects non-dict config"""
        success, message = adapter.start(sample_job, "invalid")

        assert success is False
        assert "must be a dict" in message.lower()

    def test_start_with_optional_arguments(self, adapter, sample_job, mock_process_manager, tmp_path):
        """start works without arguments field"""
        # Create real temp files for validation
        script_file = tmp_path / "train.py"
        script_file.write_text("print('training')")
        sample_job.workspace = tmp_path
        sample_job.python_executable = Path(sys.executable)

        config = {'script_path': str(script_file)}

        with patch('anylabeling.services.run_monitor.models.Run'):
            success, message = adapter.start(sample_job, config)

        assert success is True

    def test_start_failure_from_process_manager(self, adapter, sample_job, mock_process_manager, tmp_path):
        """start returns failure when ProcessManager.start fails"""
        # Create real temp files for validation
        script_file = tmp_path / "train.py"
        script_file.write_text("print('training')")
        sample_job.workspace = tmp_path
        sample_job.python_executable = Path(sys.executable)

        config = {'script_path': str(script_file)}
        mock_process_manager.start.return_value = False

        with patch('anylabeling.services.run_monitor.models.Run'):
            success, message = adapter.start(sample_job, config)

        assert success is False
        assert "failed to start" in message.lower()

    def test_stop_delegates_to_process_manager(self, adapter, mock_process_manager):
        """stop delegates to ProcessManager.stop"""
        result = adapter.stop()

        assert result is True
        assert mock_process_manager.stop.called

    def test_is_running(self, adapter, mock_process_manager):
        """is_running reflects ProcessManager state"""
        mock_process_manager.is_running.return_value = False
        assert adapter.is_running() is False

        mock_process_manager.is_running.return_value = True
        assert adapter.is_running() is True


class TestEventMapping:
    """Test ProcessManager event mapping to unified protocol"""

    def test_process_started_mapped(self, adapter, sample_job):
        """process_started signal -> process_started event"""
        events = []
        adapter.subscribe(lambda e: events.append(e))
        adapter._current_job_id = sample_job.job_id

        adapter._on_process_started(pid=1234)

        assert len(events) == 1
        event = events[0]
        assert event.event_type == TrainingEventType.PROCESS_STARTED
        assert event.job_id == sample_job.job_id
        assert event.source == "custom_script"
        assert event.payload["pid"] == 1234

    def test_process_finished_success_mapped_to_completed(self, adapter, sample_job):
        """process_finished(exit_code=0) -> completed"""
        events = []
        adapter.subscribe(lambda e: events.append(e))
        adapter._current_job_id = sample_job.job_id

        adapter._on_process_finished(pid=1234, exit_code=0)

        assert len(events) == 1
        event = events[0]
        assert event.event_type == TrainingEventType.COMPLETED
        assert event.source == "custom_script"
        assert event.payload["exit_code"] == 0

    def test_process_finished_failure_mapped_to_failed(self, adapter, sample_job):
        """process_finished(exit_code!=0) -> failed"""
        events = []
        adapter.subscribe(lambda e: events.append(e))
        adapter._current_job_id = sample_job.job_id

        adapter._on_process_finished(pid=1234, exit_code=1)

        assert len(events) == 1
        event = events[0]
        assert event.event_type == TrainingEventType.FAILED
        assert event.payload["error"] == "Process exited with code 1"
        assert event.payload["exit_code"] == 1

    def test_stdout_mapped_to_console_output(self, adapter, sample_job):
        """stdout_ready signal -> console_output event"""
        events = []
        adapter.subscribe(lambda e: events.append(e))
        adapter._current_job_id = sample_job.job_id

        adapter._on_stdout("Training started")

        assert len(events) == 1
        event = events[0]
        assert event.event_type == TrainingEventType.CONSOLE_OUTPUT
        assert event.payload["message"] == "Training started"
        assert event.payload["stream"] == "stdout"
        assert event.source == "custom_script"

    def test_stderr_mapped_to_console_output(self, adapter, sample_job):
        """stderr_ready signal -> console_output event"""
        events = []
        adapter.subscribe(lambda e: events.append(e))
        adapter._current_job_id = sample_job.job_id

        adapter._on_stderr("Warning: deprecated API")

        assert len(events) == 1
        event = events[0]
        assert event.event_type == TrainingEventType.CONSOLE_OUTPUT
        assert event.payload["message"] == "Warning: deprecated API"
        assert event.payload["stream"] == "stderr"
        assert event.source == "custom_script"

    def test_event_without_job_id_ignored(self, adapter):
        """Events without current_job_id are ignored"""
        events = []
        adapter.subscribe(lambda e: events.append(e))
        adapter._current_job_id = None

        adapter._on_stdout("Should be ignored")
        adapter._on_stderr("Should be ignored")
        adapter._on_process_started(pid=1234)
        adapter._on_process_finished(pid=1234, exit_code=0)

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

        adapter._on_stdout("test message")

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

        adapter._on_stdout("test")

        assert TrainingEventType.CONSOLE_OUTPUT in results


class TestAdapterComposition:
    """Test adapter uses composition, not inheritance"""

    def test_adapter_wraps_process_manager(self, adapter, mock_process_manager):
        """Adapter holds reference to ProcessManager"""
        assert adapter.manager is mock_process_manager

    def test_adapter_connects_to_signals(self, adapter, mock_process_manager):
        """Adapter connects to ProcessManager signals"""
        assert mock_process_manager.process_started.connect.called
        assert mock_process_manager.process_finished.connect.called
        assert mock_process_manager.stdout_ready.connect.called
        assert mock_process_manager.stderr_ready.connect.called


class TestRunObjectCreation:
    """Test Run object creation for ProcessManager"""

    def test_run_creation_with_full_config(self, adapter, sample_job, tmp_path):
        """Run object created with all fields"""
        # Create real temp files for validation
        script_file = tmp_path / "train.py"
        script_file.write_text("print('training')")
        sample_job.workspace = tmp_path
        sample_job.python_executable = Path(sys.executable)

        config = {
            'script_path': str(script_file),
            'arguments': ['--epochs', '100', '--batch-size', '32']
        }

        with patch('anylabeling.services.run_monitor.models.Run') as MockRun:
            adapter.start(sample_job, config)

        MockRun.assert_called_once()
        call_kwargs = MockRun.call_args[1]

        assert call_kwargs['run_id'] == sample_job.job_id
        assert call_kwargs['workspace_path'] == tmp_path
        assert call_kwargs['script_path'] == script_file
        assert call_kwargs['python_path'] == Path(sys.executable)
        assert call_kwargs['arguments'] == ['--epochs', '100', '--batch-size', '32']

    def test_run_creation_with_minimal_config(self, adapter, sample_job, tmp_path):
        """Run object created with minimal fields"""
        # Create real temp files for validation
        script_file = tmp_path / "train.py"
        script_file.write_text("print('training')")

        config = {'script_path': str(script_file)}
        sample_job.workspace = tmp_path
        sample_job.python_executable = Path(sys.executable)

        with patch('anylabeling.services.run_monitor.models.Run') as MockRun:
            adapter.start(sample_job, config)

        call_kwargs = MockRun.call_args[1]
        assert call_kwargs['arguments'] == []
