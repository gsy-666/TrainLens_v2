"""Tests for RunMonitorWidget

Verifies widget lifecycle, signal connections, callback invocation, and cleanup.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from anylabeling.views.run_monitor.run_monitor_widget import RunMonitorWidget
from anylabeling.services.run_monitor import Run, Workspace, DetectedScript, PythonEnvironment


@pytest.fixture(scope="session")
def qapp():
    """Create QApplication for tests"""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def widget(qapp):
    """Create RunMonitorWidget instance"""
    w = RunMonitorWidget()
    yield w
    w.cleanup()


class TestRunMonitorWidgetCreation:
    """Test widget creation and initialization"""

    def test_widget_creates_successfully(self, widget):
        """Widget can be created"""
        assert widget is not None

    def test_widget_has_job_manager(self, widget):
        """Widget initializes JobManager"""
        assert widget.job_manager is not None

    def test_widget_has_resource_monitor(self, widget):
        """Widget initializes ResourceMonitor"""
        assert widget.resource_monitor is not None

    def test_widget_has_ui_elements(self, widget):
        """Widget creates all UI elements"""
        assert widget.workspace_label is not None
        assert widget.open_workspace_btn is not None
        assert widget.scripts_label is not None
        assert widget.envs_label is not None
        assert widget.status_label is not None
        assert widget.script_combo is not None
        assert widget.python_combo is not None
        assert widget.args_edit is not None
        assert widget.console_output is not None
        assert widget.resources_label is not None
        assert widget.start_btn is not None
        assert widget.stop_btn is not None

    def test_start_button_initially_disabled(self, widget):
        """Start button is disabled until workspace is configured"""
        assert not widget.start_btn.isEnabled()

    def test_stop_button_initially_disabled(self, widget):
        """Stop button is disabled when not running"""
        assert not widget.stop_btn.isEnabled()


class TestRunMonitorWidgetCallbacks:
    """Test callback system"""

    def test_can_register_on_run_start_callback(self, widget):
        """Can register on_run_start callback"""
        callback = Mock()
        widget.on_run_start = callback

        assert widget.on_run_start == callback

    def test_can_register_on_run_stop_callback(self, widget):
        """Can register on_run_stop callback"""
        callback = Mock()
        widget.on_run_stop = callback

        assert widget.on_run_stop == callback

    def test_can_register_on_run_complete_callback(self, widget):
        """Can register on_run_complete callback"""
        callback = Mock()
        widget.on_run_complete = callback

        assert widget.on_run_complete == callback

    def test_on_run_start_callback_invoked_on_start(self, widget):
        """on_run_start callback is invoked when training starts"""
        callback = Mock()
        widget.on_run_start = callback

        # Mock workspace and environment
        widget.workspace = Mock(spec=Workspace)
        widget.workspace.path = Path("/tmp/workspace")
        widget.workspace.detected_scripts = []
        widget.workspace.detected_environments = []

        # Simulate successful start
        with patch('anylabeling.services.run_monitor.process_manager.ProcessManager.start', return_value=True):
            with patch('anylabeling.services.training_center.job_manager.JobManager.request_start', return_value=(True, "Started")):
                widget.script_combo.addItem("train.py", Mock(spec=DetectedScript, path=Path("/tmp/workspace/train.py"), framework="test"))
                widget.python_combo.addItem("Python 3.12", Mock(spec=PythonEnvironment, python_path=Path("/usr/bin/python3")))
                widget.script_combo.setEnabled(True)
                widget.python_combo.setEnabled(True)
                widget.start_btn.setEnabled(True)

                widget._on_start_training()

        assert callback.called
        assert callback.call_count == 1
        run = callback.call_args[0][0]
        assert isinstance(run, Run)

    def test_on_run_complete_callback_invoked_on_finish(self, widget):
        """on_run_complete callback is invoked when training completes"""
        from anylabeling.services.training_center.event_protocol import (
            TrainingEvent, TrainingEventType
        )
        from anylabeling.services.training_center.models import TrainingJob, TrainingMode
        from datetime import datetime
        import time

        callback = Mock()
        widget.on_run_complete = callback

        # Set up current job
        widget.current_job = TrainingJob(
            job_id="test-run-001",
            mode=TrainingMode.CUSTOM_SCRIPT,
            workspace=None,
            display_name="Test",
            framework="custom"
        )

        # Mock history_store.finalize_job to avoid KeyError
        with patch.object(widget.history_store, 'finalize_job'):
            # Simulate COMPLETED event
            event = TrainingEvent(
                schema_version=1,
                job_id="test-run-001",
                event_type=TrainingEventType.COMPLETED,
                timestamp=time.time(),
                payload={'exit_code': 0},
                source='test'
            )
            widget._on_training_event(event)

        assert callback.called
        assert callback.call_args[0][0] == 0


class TestRunMonitorWidgetState:
    """Test widget state management"""

    def test_is_running_reflects_process_manager_state(self, widget):
        """is_running() reflects JobManager state"""
        # Mock JobManager to return no active job
        with patch.object(widget.job_manager, 'get_current_job', return_value=None):
            assert widget.is_running() is False

        # Mock JobManager to return active job
        from anylabeling.services.training_center.models import TrainingJob, TrainingStatus
        from datetime import datetime
        active_job = TrainingJob(
            job_id="test-001",
            mode=Mock(),
            status=TrainingStatus.RUNNING,
            created_at=datetime.now(),
            workspace=Path("/tmp"),
            display_name="Test",
            framework="test",
            python_executable=None,
            command=None,
            metadata={}
        )
        with patch.object(widget.job_manager, 'get_current_job', return_value=active_job):
            assert widget.is_running() is True

    def test_workspace_initially_none(self, widget):
        """Workspace is None on init"""
        assert widget.workspace is None

    def test_current_run_initially_none(self, widget):
        """Current run is None on init"""
        assert widget.current_run is None


class TestRunMonitorWidgetCleanup:
    """Test cleanup and resource disposal"""

    def test_cleanup_stops_scanner_thread(self, widget):
        """cleanup stops scanner thread if running"""
        widget.scanner_thread = Mock()
        widget.scanner_thread.isRunning.return_value = True

        widget.cleanup()

        assert widget.scanner_thread.cancel.called
        assert widget.scanner_thread.wait.called

    def test_cleanup_stops_resource_monitor(self, widget):
        """cleanup stops resource monitor"""
        with patch.object(widget.resource_monitor, 'stop_monitoring') as mock_stop:
            widget.cleanup()

        assert mock_stop.called

    def test_cleanup_stops_process_manager(self, widget):
        """cleanup unsubscribes from JobManager"""
        with patch.object(widget.job_manager, 'unsubscribe_events') as mock_unsub:
            widget.cleanup()

        assert mock_unsub.called

    def test_cleanup_is_idempotent(self, widget):
        """cleanup can be called multiple times safely"""
        widget.cleanup()
        widget.cleanup()


class TestRunMonitorWidgetConsoleOutput:
    """Test console output handling"""

    def test_stdout_appends_to_console(self, widget):
        """stdout is appended to console output via CONSOLE_OUTPUT event"""
        from anylabeling.services.training_center.event_protocol import (
            TrainingEvent, TrainingEventType
        )
        from anylabeling.services.training_center.models import TrainingJob, TrainingMode
        import time

        initial_text = widget.console_output.toPlainText()

        # Set up current job
        widget.current_job = TrainingJob(
            job_id="test-001",
            mode=TrainingMode.CUSTOM_SCRIPT,
            workspace=None,
            display_name="Test",
            framework="custom"
        )

        # Simulate CONSOLE_OUTPUT event
        event = TrainingEvent(
            schema_version=1,
            job_id="test-001",
            event_type=TrainingEventType.CONSOLE_OUTPUT,
            timestamp=time.time(),
            payload={'message': 'test output line', 'stream': 'stdout'},
            source='test'
        )
        widget._on_training_event(event)

        final_text = widget.console_output.toPlainText()
        assert "test output line" in final_text
        assert len(final_text) > len(initial_text)

    def test_stderr_appends_to_console_with_error_prefix(self, widget):
        """stderr is appended with ERROR prefix via CONSOLE_OUTPUT event"""
        from anylabeling.services.training_center.event_protocol import (
            TrainingEvent, TrainingEventType
        )
        from anylabeling.services.training_center.models import TrainingJob, TrainingMode
        import time

        # Set up current job
        widget.current_job = TrainingJob(
            job_id="test-001",
            mode=TrainingMode.CUSTOM_SCRIPT,
            workspace=None,
            display_name="Test",
            framework="custom"
        )

        # Simulate CONSOLE_OUTPUT event with stderr
        event = TrainingEvent(
            schema_version=1,
            job_id="test-001",
            event_type=TrainingEventType.CONSOLE_OUTPUT,
            timestamp=time.time(),
            payload={'message': 'test error line', 'stream': 'stderr'},
            source='test'
        )
        widget._on_training_event(event)

        final_text = widget.console_output.toPlainText()
        assert "ERROR: test error line" in final_text


class TestRunMonitorWidgetStatusUpdates:
    """Test status display updates"""

    def test_update_status_changes_label(self, widget):
        """_update_status changes status label text"""
        from anylabeling.services.run_monitor import RunStatus

        widget._update_status(RunStatus.RUNNING)
        assert "Running" in widget.status_label.text()

        widget._update_status(RunStatus.COMPLETED)
        assert "Completed" in widget.status_label.text()

        widget._update_status(RunStatus.FAILED)
        assert "Failed" in widget.status_label.text()


class TestRunMonitorWidgetResourceDisplay:
    """Test resource monitoring display"""

    def test_resource_sample_updates_label(self, widget):
        """resource sample updates resources label"""
        sample = {
            "process": {
                "cpu_percent": 45.5,
                "memory_percent": 30.2,
            },
            "system": {
                "cpu_percent": 60.0,
                "memory_percent": 50.0,
            }
        }

        widget._on_resource_sample(sample)

        text = widget.resources_label.text()
        assert "45.5%" in text
        assert "30.2%" in text

    def test_gpu_resource_displayed(self, widget):
        """GPU resources are displayed when available"""
        sample = {
            "gpu": {
                "gpus": [
                    {
                        "index": 0,
                        "utilization": 85.0,
                        "memory_used": 8000,
                        "memory_total": 16000,
                    }
                ]
            }
        }

        widget._on_resource_sample(sample)

        text = widget.resources_label.text()
        assert "GPU 0" in text
        assert "85.0%" in text
