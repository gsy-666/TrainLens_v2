"""Tests for Environment Wizard integration with RunMonitorWidget."""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from PyQt6.QtWidgets import QApplication

from anylabeling.views.run_monitor.run_monitor_widget import RunMonitorWidget
from anylabeling.services.training_center.environment.models import (
    EnvironmentInfo,
    EnvironmentStatus,
)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def widget(qapp):
    w = RunMonitorWidget()
    yield w
    w.cleanup()


class TestEnvironmentWidgetIntegration:
    """Tests for environment UI integration."""

    def test_environment_section_created(self, widget):
        """Widget has environment section UI elements."""
        assert widget.python_path_edit is not None
        assert widget.python_browse_btn is not None
        assert widget.env_detect_btn is not None
        assert widget.env_create_venv_btn is not None
        assert widget.env_install_req_btn is not None
        assert widget.env_status_label is not None
        assert widget.env_summary_label is not None
        assert widget.env_log is not None

    def test_detect_button_enabled_only_with_python(self, widget):
        """Detect button is disabled when no Python is set."""
        widget.python_path_edit.setText("")
        widget._update_env_buttons()
        assert not widget.env_detect_btn.isEnabled()

        widget.python_path_edit.setText(sys.executable)
        widget._update_env_buttons()
        assert widget.env_detect_btn.isEnabled()

    def test_create_venv_no_project_disabled(self, widget):
        """Create .venv is disabled when no workspace selected."""
        widget.workspace = None
        widget._update_env_buttons()
        assert not widget.env_create_venv_btn.isEnabled()

    def test_install_requirements_no_file_disabled(self, widget):
        """Install Requirements is disabled when requirements.txt missing."""
        widget.python_path_edit.setText(sys.executable)
        widget.workspace = Mock()
        widget.workspace.path = Path("/tmp/no_req")
        widget._update_env_buttons()
        assert not widget.env_install_req_btn.isEnabled()

    def test_error_status_prevents_start(self, widget):
        """ERROR status disables Start button."""
        widget._env_info = EnvironmentInfo(
            python_path="/bad/python",
            executable_valid=False,
            status=EnvironmentStatus.ERROR,
            error_message="Bad Python",
        )
        widget._update_start_button()
        assert not widget.start_btn.isEnabled()

    def test_warning_status_allows_start(self, widget):
        """WARNING status allows Start."""
        widget._env_info = EnvironmentInfo(
            python_path=sys.executable,
            executable_valid=True,
            pip_available=True,
            status=EnvironmentStatus.WARNING,
        )
        widget.script_combo.addItem("test.py", Mock(path=Path("test.py"), framework="test"))
        widget.python_path_edit.setText(sys.executable)
        widget._update_start_button()
        assert widget.start_btn.isEnabled()

    def test_start_uses_selected_python(self, widget):
        """Start training uses the Python from the text field."""
        widget.python_path_edit.setText(sys.executable)
        widget.workspace = Mock()
        widget.workspace.path = Path("/tmp/ws")
        widget.workspace.detected_scripts = []
        widget.script_combo.addItem("test.py", Mock(path=Path("/tmp/ws/test.py"), framework="test"))

        with patch("subprocess.run", return_value=Mock(returncode=0)):
            with patch("anylabeling.services.training_center.job_manager.JobManager.request_start") as mock_start:
                mock_start.return_value = (True, "OK")
                widget._env_info = EnvironmentInfo(
                    python_path=sys.executable,
                    executable_valid=True,
                    pip_available=True,
                    status=EnvironmentStatus.READY,
                )
                widget._on_start_training()
                job = mock_start.call_args[1]["job"]
                assert job.python_executable == Path(sys.executable)

    def test_key_normalization_consistent(self, widget):
        """Same project with different casing produces same binding key."""
        from anylabeling.views.run_monitor.run_monitor_widget import _make_binding_key
        k1 = _make_binding_key("C:\\MyProject")
        k2 = _make_binding_key("c:\\myproject")
        assert k1 == k2
