"""Run Monitor Widget - Reusable Component

Extracted from RunMonitorWindow to be embeddable in other containers.
Preserves all functionality: workspace scanning, script detection, environment detection,
configuration, start/stop, console output, and resource monitoring.
"""

import hashlib
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QMessageBox,
    QSplitter,
    QPlainTextEdit,
    QComboBox,
    QLineEdit,
    QGroupBox,
    QFormLayout,
    QTextEdit,
)

from anylabeling.services.run_monitor import (
    Run,
    RunStatus,
    Workspace,
    DetectedScript,
    PythonEnvironment,
)
from anylabeling.services.run_monitor.workspace_scanner import WorkspaceScanner
from anylabeling.services.run_monitor.workspace_scanner_thread import WorkspaceScannerThread
from anylabeling.services.run_monitor.process_manager import ProcessManager
from anylabeling.services.run_monitor.resource_monitor import ResourceMonitor
from anylabeling.services.run_monitor.run_storage import RunStorage
from anylabeling.services.run_monitor.event_protocol import (
    EventProtocol,
    create_run_created_event,
    create_process_started_event,
    create_process_completed_event,
    create_process_failed_event,
)

# Training Center integration
from anylabeling.services.training_center.models import (
    TrainingJob,
    TrainingMode,
    TrainingStatus,
)
from anylabeling.services.training_center.job_manager import get_job_manager
from anylabeling.services.training_center.event_protocol import TrainingEvent, TrainingEventType

# Environment Wizard
from anylabeling.services.training_center.environment import (
    EnvironmentInfo,
    EnvironmentStatus,
    EnvironmentWorker,
    EnvironmentTaskController,
    find_project_python,
)


_ENV_BINDING_PREFIX = "trainlens/env_binding/"

def _make_binding_key(project_dir: str) -> str:
    """Create a QSettings key from a normalized project path.

    Uses os.path.normcase for Windows case-insensitive matching.
    """
    import os
    norm = os.path.normcase(os.path.normpath(os.path.abspath(str(project_dir))))
    norm = norm.replace("\\", "/")
    return _ENV_BINDING_PREFIX + hashlib.sha256(norm.encode()).hexdigest()[:16]


class RunMonitorWidget(QWidget):
    """Reusable Run Monitor widget

    Can be embedded in TrainingCenterWindow or used standalone in RunMonitorWindow.

    Callbacks:
        on_run_start: Called when training starts with (run: Run)
        on_run_stop: Called when training stops
        on_run_complete: Called when training completes with (exit_code: int)
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # State
        self.workspace: Optional[Workspace] = None
        self.current_run: Optional[Run] = None
        self.current_job: Optional[TrainingJob] = None  # Unified training job
        self.storage: Optional[RunStorage] = None
        self.scanner_thread: Optional[WorkspaceScannerThread] = None

        # Services
        self.resource_monitor = ResourceMonitor()
        self.job_manager = get_job_manager()

        # Environment wizard state
        self._env_info: Optional[EnvironmentInfo] = None
        self._env_controller: Optional[EnvironmentTaskController] = None
        self._env_task_id: Optional[str] = None  # current task id (for UI signal disconnect)
        self._env_generation: int = 0

        # Callbacks
        self.on_run_start: Optional[Callable[[Run], None]] = None
        self.on_run_stop: Optional[Callable[[], None]] = None
        self.on_run_complete: Optional[Callable[[int], None]] = None

        # Connect resource monitor
        self.resource_monitor.resource_sample.connect(self._on_resource_sample)

        # Subscribe to Training Center events
        self.job_manager.subscribe_events(self._on_training_event)

        # Initialize UI
        self._init_ui()

    def _init_ui(self):
        """Initialize UI components"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # Top bar: workspace selection
        top_bar = self._create_top_bar()
        layout.addWidget(top_bar)

        # Main content: split view
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel: detected scripts and environments
        left_panel = self._create_left_panel()
        splitter.addWidget(left_panel)

        # Right panel: configuration, status, resources
        right_panel = self._create_right_panel()
        splitter.addWidget(right_panel)

        splitter.setSizes([300, 900])
        layout.addWidget(splitter, stretch=1)

        # Console output
        console_group = self._create_console_panel()
        layout.addWidget(console_group, stretch=1)

        # Bottom buttons
        bottom_buttons = self._create_bottom_buttons()
        layout.addLayout(bottom_buttons)

    def _create_top_bar(self) -> QWidget:
        """Create top bar with workspace path and open button"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Workspace:"))

        self.workspace_label = QLabel("No workspace selected")
        self.workspace_label.setStyleSheet("color: gray;")
        layout.addWidget(self.workspace_label, stretch=1)

        self.open_workspace_btn = QPushButton("Open Workspace")
        self.open_workspace_btn.clicked.connect(self._on_open_workspace)
        layout.addWidget(self.open_workspace_btn)

        return widget

    def _create_left_panel(self) -> QWidget:
        """Create left panel with detected scripts and environments"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # Scripts group
        scripts_group = QGroupBox("Detected Training Scripts")
        scripts_layout = QVBoxLayout(scripts_group)
        self.scripts_label = QLabel("No scripts detected")
        self.scripts_label.setStyleSheet("color: gray;")
        self.scripts_label.setWordWrap(True)
        scripts_layout.addWidget(self.scripts_label)
        scripts_layout.addStretch()
        layout.addWidget(scripts_group)

        # Environments group
        envs_group = QGroupBox("Python Environments")
        envs_layout = QVBoxLayout(envs_group)
        self.envs_label = QLabel("No environments detected")
        self.envs_label.setStyleSheet("color: gray;")
        self.envs_label.setWordWrap(True)
        envs_layout.addWidget(self.envs_label)
        envs_layout.addStretch()
        layout.addWidget(envs_group)

        return widget

    def _create_right_panel(self) -> QWidget:
        """Create right panel with configuration, environment, and status"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # Status
        status_group = QGroupBox("Status")
        status_layout = QHBoxLayout(status_group)
        self.status_label = QLabel("Idle")
        self.status_label.setStyleSheet("font-weight: bold; color: gray;")
        status_layout.addWidget(self.status_label)
        self.elapsed_label = QLabel("")
        status_layout.addWidget(self.elapsed_label)
        status_layout.addStretch()
        layout.addWidget(status_group)

        # Run Configuration
        config_group = QGroupBox("Run Configuration")
        config_layout = QFormLayout(config_group)

        self.script_combo = QComboBox()
        self.script_combo.setEnabled(False)
        config_layout.addRow("Script:", self.script_combo)

        self.args_edit = QLineEdit()
        self.args_edit.setPlaceholderText("Additional arguments...")
        self.args_edit.setEnabled(False)
        config_layout.addRow("Arguments:", self.args_edit)

        layout.addWidget(config_group)

        # Environment section
        env_group = self._create_environment_section()
        layout.addWidget(env_group)

        # Resources
        resources_group = QGroupBox("Resources")
        resources_layout = QVBoxLayout(resources_group)
        self.resources_label = QLabel("No active run")
        self.resources_label.setStyleSheet("color: gray; font-family: monospace;")
        self.resources_label.setWordWrap(True)
        resources_layout.addWidget(self.resources_label)
        layout.addWidget(resources_group)

        # Metrics dashboard
        from anylabeling.views.training.metrics import TrainingMetricsDashboard
        self.metrics_dashboard = TrainingMetricsDashboard()
        layout.addWidget(self.metrics_dashboard, stretch=2)

        layout.addStretch()

        return widget

    def _create_environment_section(self) -> QGroupBox:
        """Create the environment wizard section."""
        group = QGroupBox("Environment")
        layout = QVBoxLayout(group)

        # Python executable row
        py_row = QHBoxLayout()
        py_row.addWidget(QLabel("Python Executable:"))
        self.python_path_edit = QLineEdit()
        self.python_path_edit.setPlaceholderText("Select a project first...")
        self.python_path_edit.setReadOnly(True)
        self.python_path_edit.editingFinished.connect(self._on_python_path_edited)
        py_row.addWidget(self.python_path_edit, stretch=1)
        self.python_browse_btn = QPushButton("Browse")
        self.python_browse_btn.clicked.connect(self._on_browse_python)
        py_row.addWidget(self.python_browse_btn)
        layout.addLayout(py_row)

        # Status display
        self.env_status_label = QLabel("Not checked")
        self.env_status_label.setStyleSheet("color: gray;")
        layout.addWidget(self.env_status_label)

        # Environment summary
        self.env_summary_label = QLabel("")
        self.env_summary_label.setStyleSheet("font-family: monospace; font-size: 9pt;")
        self.env_summary_label.setWordWrap(True)
        layout.addWidget(self.env_summary_label)

        # Buttons row
        btn_row = QHBoxLayout()
        self.env_detect_btn = QPushButton("Detect")
        self.env_detect_btn.clicked.connect(self._on_detect_environment)
        self.env_detect_btn.setEnabled(False)
        btn_row.addWidget(self.env_detect_btn)

        self.env_create_venv_btn = QPushButton("Create .venv")
        self.env_create_venv_btn.clicked.connect(self._on_create_venv)
        self.env_create_venv_btn.setEnabled(False)
        btn_row.addWidget(self.env_create_venv_btn)

        self.env_install_req_btn = QPushButton("Install Requirements")
        self.env_install_req_btn.clicked.connect(self._on_install_requirements)
        self.env_install_req_btn.setEnabled(False)
        btn_row.addWidget(self.env_install_req_btn)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Environment log header row
        self._env_log_header = QWidget()
        self._env_log_header.hide()  # hidden until log shown
        log_header_layout = QHBoxLayout(self._env_log_header)
        log_header_layout.setContentsMargins(0, 2, 0, 0)
        log_header_layout.addWidget(QLabel("Environment Log"))
        log_header_layout.addStretch()
        self.env_clear_log_btn = QPushButton("Clear")
        self.env_clear_log_btn.clicked.connect(lambda: self.env_log.clear())
        log_header_layout.addWidget(self.env_clear_log_btn)
        layout.addWidget(self._env_log_header)

        # Show/Hide Log toggle (in button row)
        self._env_toggle_log_btn = QPushButton("Show Log")
        self._env_toggle_log_btn.clicked.connect(self._toggle_env_log)
        btn_row.addWidget(self._env_toggle_log_btn)
        btn_row.addStretch()

        # Environment log (QPlainTextEdit, default hidden)
        self.env_log = QPlainTextEdit()
        self.env_log.setReadOnly(True)
        self.env_log.setMinimumHeight(140)
        self.env_log.setMaximumBlockCount(2000)
        self.env_log.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.env_log.setStyleSheet(
            "font-family: Consolas, 'Courier New', monospace; font-size: 10pt;"
        )
        self.env_log.setPlaceholderText("Environment operation log...")
        self.env_log.hide()
        layout.addWidget(self.env_log, stretch=1)

        return group

    def _create_console_panel(self) -> QGroupBox:
        """Create console output panel"""
        group = QGroupBox("Console")
        layout = QVBoxLayout(group)

        self.console_output = QPlainTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setStyleSheet(
            "font-family: 'Consolas', 'Courier New', monospace; font-size: 10pt;"
        )
        self.console_output.setPlainText("Console output will appear here...")
        layout.addWidget(self.console_output)

        return group

    def _create_bottom_buttons(self) -> QHBoxLayout:
        """Create bottom button bar"""
        layout = QHBoxLayout()
        layout.addStretch()

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop_training)
        layout.addWidget(self.stop_btn)

        self.start_btn = QPushButton("Start Training")
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._on_start_training)
        layout.addWidget(self.start_btn)

        return layout

    def _on_open_workspace(self):
        """Handle open workspace button click"""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Training Workspace"
        )
        if not folder:
            return

        workspace_path = Path(folder)
        self.workspace_label.setText(str(workspace_path))
        self.workspace_label.setStyleSheet("color: black;")

        # Clear console
        self.console_output.clear()
        self.console_output.appendPlainText(f"Scanning workspace: {workspace_path}")

        # Disable UI during scan
        self.open_workspace_btn.setEnabled(False)

        # Start scan in background thread
        if self.scanner_thread and self.scanner_thread.isRunning():
            self.scanner_thread.cancel()
            self.scanner_thread.wait()

        self.scanner_thread = WorkspaceScannerThread(workspace_path, self)
        self.scanner_thread.progress.connect(self._on_scan_progress)
        self.scanner_thread.finished.connect(self._on_scan_finished)
        self.scanner_thread.error.connect(self._on_scan_error)
        self.scanner_thread.start()

    def _on_scan_progress(self, current: int, total: int, message: str):
        """Handle scan progress updates"""
        if total > 0:
            self.console_output.appendPlainText(f"[{current}/{total}] {message}")
        else:
            self.console_output.appendPlainText(message)

    def _on_scan_finished(self, workspace: Workspace):
        """Handle scan completion"""
        self.workspace = workspace
        self._update_detected_items()

        # Initialize storage
        try:
            self.storage = RunStorage(workspace.path)
            self.storage.initialize()
            self.storage.save_workspace(self.workspace)

            self.console_output.appendPlainText(
                f"Found {len(self.workspace.detected_scripts)} training scripts"
            )
            self.console_output.appendPlainText(
                f"Found {len(self.workspace.detected_environments)} Python environments"
            )
        except Exception as e:
            self.console_output.appendPlainText(f"Warning: Storage initialization failed: {e}")

        # Re-enable UI
        self.open_workspace_btn.setEnabled(True)

    def _on_scan_error(self, error_message: str):
        """Handle scan error"""
        self.console_output.appendPlainText(f"Error: {error_message}")
        QMessageBox.critical(self, "Scan Error", f"Failed to scan workspace: {error_message}")
        self.open_workspace_btn.setEnabled(True)

    def _update_detected_items(self):
        """Update UI with detected scripts and environments"""
        if not self.workspace:
            return

        # Update scripts
        if self.workspace.detected_scripts:
            scripts_text = []
            self.script_combo.clear()
            for script in self.workspace.detected_scripts:
                rel_path = script.path.relative_to(self.workspace.path)
                framework = script.framework or "unknown"
                scripts_text.append(
                    f"{rel_path} ({framework}, confidence: {script.confidence:.2f})"
                )
                self.script_combo.addItem(str(rel_path), script)

            self.scripts_label.setText("\n".join(scripts_text))
            self.scripts_label.setStyleSheet("color: black;")
            self.script_combo.setEnabled(True)
        else:
            self.scripts_label.setText("No training scripts detected")
            self.scripts_label.setStyleSheet("color: orange;")

        # Update environments display (left panel)
        if self.workspace.detected_environments:
            envs_text = []
            for env in self.workspace.detected_environments:
                env_type = env.env_type
                version = env.version
                envs_text.append(f"{env_type} (Python {version})")
            self.envs_label.setText("\n".join(envs_text))
            self.envs_label.setStyleSheet("color: black;")
        else:
            self.envs_label.setText("No Python environments detected")
            self.envs_label.setStyleSheet("color: orange;")

        # Auto-discover project Python
        self._auto_discover_python()

        # Enable controls if scripts available
        if self.workspace.detected_scripts:
            self.args_edit.setEnabled(True)
            self._update_start_button()

    # ── Environment Wizard methods ──────────────────────────────────────

    def _auto_discover_python(self):
        """Auto-discover project Python after workspace selection."""
        if not self.workspace:
            return

        project_dir = self.workspace.path

        # Check saved binding first
        saved = self._load_python_binding(project_dir)
        if saved and Path(saved).exists():
            self._set_python_path(saved)
            self._on_detect_environment()
            return

        # Try to find project venv
        found = find_project_python(project_dir)
        if found:
            self._set_python_path(str(found))
            self._save_python_binding(project_dir, str(found))
            self._on_detect_environment()
        else:
            self._set_python_path("")
            self._set_env_status(EnvironmentStatus.NOT_FOUND, "No environment found")
            self._update_env_buttons()

    def _on_browse_python(self):
        """Browse for a Python executable."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Python Executable",
            filter="Python (python.exe python3 python);;All Files (*)"
        )
        if path:
            self._set_python_path(path)
            if self.workspace:
                self._save_python_binding(self.workspace.path, path)
            self._on_detect_environment()

    def _on_python_path_edited(self):
        """Handle manual Python path edit."""
        path = self.python_path_edit.text().strip()
        if path and Path(path).exists():
            if self.workspace:
                self._save_python_binding(self.workspace.path, path)
            self._on_detect_environment()

    def _on_detect_environment(self):
        """Start environment detection in background thread."""
        python_path = self.python_path_edit.text().strip()
        if not python_path:
            return

        self._env_generation += 1
        gen = self._env_generation
        self._set_env_status(EnvironmentStatus.CHECKING, "Detecting...")
        self._auto_show_env_log()
        self._set_env_controls_enabled(False)

        worker = EnvironmentWorker()
        worker.request_detect(python_path, str(self.workspace.path) if self.workspace else "", gen)
        # Result signal connected before start_task (must be on worker before thread runs)
        worker.detection_done.connect(lambda info: self._on_detection_done(info, gen))

        self._start_env_worker(worker)

    def _on_detection_done(self, info: EnvironmentInfo, generation: int):
        """Handle detection result."""
        if generation != self._env_generation:
            return

        self._env_info = info
        self._display_env_info(info)
        self._finish_env_ui_operation()
        self._set_env_controls_enabled(True)
        self._update_env_buttons()
        self._update_start_button()

    def _on_create_venv(self):
        """Create .venv in project directory."""
        if not self.workspace:
            return

        self._env_generation += 1
        gen = self._env_generation
        self._set_env_status(EnvironmentStatus.CREATING, "Creating .venv ...")
        self._env_log_clear()
        self._auto_show_env_log()
        self._set_env_controls_enabled(False)

        worker = EnvironmentWorker()
        worker.request_create_venv(str(self.workspace.path), gen)
        worker.venv_created.connect(lambda ok, path, msg: self._on_venv_created(ok, path, msg, gen))

        self._start_env_worker(worker)

    def _on_venv_created(self, success: bool, venv_path: str, message: str, generation: int):
        """Handle venv creation result."""
        if generation != self._env_generation:
            return

        self._env_log_append(message)
        self._finish_env_ui_operation()
        self._set_env_controls_enabled(True)

        if success and venv_path:
            self._set_python_path(venv_path)
            if self.workspace:
                self._save_python_binding(self.workspace.path, venv_path)
            self._on_detect_environment()
        else:
            self._auto_show_env_log()
            self._set_env_status(EnvironmentStatus.ERROR, message)
            self._update_env_buttons()
            self._update_start_button()

    def _on_install_requirements(self):
        """Install requirements.txt."""
        python_path = self.python_path_edit.text().strip()
        if not python_path:
            return
        if not self.workspace:
            return

        req_path = self.workspace.path / "requirements.txt"
        if not req_path.exists():
            return

        self._env_generation += 1
        gen = self._env_generation
        self._set_env_status(EnvironmentStatus.INSTALLING, "Installing requirements...")
        self._env_log_clear()
        self._auto_show_env_log()
        self._set_env_controls_enabled(False)

        worker = EnvironmentWorker()
        worker.request_install_requirements(python_path, str(req_path), gen)
        worker.requirements_done.connect(lambda ok, msg: self._on_requirements_done(ok, msg, gen))

        self._start_env_worker(worker)

    def _start_env_worker(self, worker: EnvironmentWorker):
        """Start an environment worker via the stable controller.

        Internal lifecycle stays with the controller.
        Only connects UI signals here.
        Previous task is not forcibly disconnected — it completes naturally.
        """
        ctrl = self._get_controller()
        task_id = ctrl.start_task(worker)
        self._env_task_id = task_id

        # Register UI signals (disconnected only by controller on task finish
        # or by _cleanup_env_thread on widget destruction)
        ctrl.connect_result(task_id, "log_message", self._env_log_append)
        ctrl.connect_result(task_id, "status_changed", self._set_env_status_text)

    def _get_controller(self) -> EnvironmentTaskController:
        """Get or create the shared environment task controller.

        The controller lives on the widget's parent chain or is created
        once per widget. Its lifetime is independent of widget destruction.
        """
        if self._env_controller is None:
            self._env_controller = EnvironmentTaskController()
        return self._env_controller

    def _is_env_task_running(self) -> bool:
        """Check if the current env task is still running."""
        if not self._env_task_id or not self._env_controller:
            return False
        return self._env_controller.is_running(self._env_task_id)

    def _finish_env_ui_operation(self):
        """Clear widget state after normal task completion.

        Does NOT call disconnect_all_ui — the worker may already be deleted
        by the time queued signals reach the GUI thread. Simply clears our
        local task_id so the next operation can start fresh.
        """
        self._env_task_id = None

    def _cleanup_env_thread(self):
        """Disconnect UI signals — ONLY used during widget destruction.

        Normal task completion uses _finish_env_ui_operation() instead
        to avoid accessing workers that may already be deleted.
        """
        if self._env_task_id and self._env_controller:
            self._env_controller.disconnect_all_ui(self._env_task_id)
        self._env_task_id = None

    def _on_requirements_done(self, success: bool, message: str, generation: int):
        """Handle requirements installation result."""
        if generation != self._env_generation:
            return

        self._env_log_append(message)
        self._finish_env_ui_operation()
        self._set_env_controls_enabled(True)

        if success:
            self._on_detect_environment()
        else:
            self._auto_show_env_log()
            self._set_env_status(EnvironmentStatus.ERROR, message)
            self._update_env_buttons()
            self._update_start_button()

    # ── Environment UI helpers ──────────────────────────────────────────

    # ── Environment UI helpers ──────────────────────────────────────────

    def _set_python_path(self, path: str):
        """Set Python path in the text field."""
        self.python_path_edit.setText(path)

    def _set_env_status(self, status: EnvironmentStatus, message: str = ""):
        """Set environment status display."""
        self._set_env_status_text(status.value)
        color = {"ready": "green", "warning": "orange", "error": "red",
                 "not_found": "red", "checking": "blue", "creating": "blue",
                 "installing": "blue", "not_checked": "gray"}.get(status.value, "gray")
        display = message or status.value.replace("_", " ").title()
        self.env_status_label.setText(display)
        self.env_status_label.setStyleSheet(f"font-weight: bold; color: {color};")

    def _set_env_status_text(self, status_value: str):
        """Update status label from status string."""
        color_map = {"ready": "green", "warning": "orange", "error": "red",
                     "not_found": "red", "checking": "blue", "creating": "blue",
                     "installing": "blue", "not_checked": "gray"}
        color = color_map.get(status_value, "gray")
        self.env_status_label.setText(status_value.replace("_", " ").title())
        self.env_status_label.setStyleSheet(f"font-weight: bold; color: {color};")

    def _display_env_info(self, info: EnvironmentInfo):
        """Display environment detection results."""
        self._set_env_status(info.status, info.error_message or info.warning_message)
        self.env_summary_label.setText(info.summary)

    def _update_env_buttons(self):
        """Update environment button states."""
        has_project = self.workspace is not None
        has_python = bool(self.python_path_edit.text().strip())
        has_venv = has_project and (self.workspace.path / ".venv").exists()
        has_req = has_project and (self.workspace.path / "requirements.txt").exists()
        task_running = self._is_env_task_running()
        training_running = (
            self.current_job is not None
            and self.current_job.status.is_active()
        ) if self.current_job else False

        self.env_detect_btn.setEnabled(has_python and not task_running)
        self.env_create_venv_btn.setEnabled(
            has_project and not has_venv and not task_running and not training_running
        )
        self.env_install_req_btn.setEnabled(
            has_python and has_req and not task_running and not training_running
        )
        self.python_browse_btn.setEnabled(not task_running and not training_running)

    def _set_env_controls_enabled(self, enabled: bool):
        """Enable/disable environment controls."""
        self._update_env_buttons()

    def _toggle_env_log(self):
        """Toggle environment log visibility."""
        if self.env_log.isVisible():
            self._hide_env_log()
        else:
            self._show_env_log()

    def _show_env_log(self):
        """Show environment log with header."""
        self.env_log.show()
        self._env_log_header.show()
        self._env_toggle_log_btn.setText("Hide Log")

    def _hide_env_log(self):
        """Hide environment log and header."""
        self.env_log.hide()
        self._env_log_header.hide()
        self._env_toggle_log_btn.setText("Show Log")

    def _auto_show_env_log(self):
        """Auto-expand log on operation start/error (idempotent)."""
        if not self.env_log.isVisible():
            self._show_env_log()

    def _env_log_append(self, message: str):
        """Append a message to the environment log with auto-scroll."""
        self.env_log.appendPlainText(message.rstrip())
        # Auto-scroll to bottom unless user scrolled up
        sb = self.env_log.verticalScrollBar()
        if sb and sb.value() >= sb.maximum() - 10:
            sb.setValue(sb.maximum())

    def _env_log_clear(self):
        """Clear environment log."""
        self.env_log.clear()

    def _update_start_button(self):
        """Update Start button enabled state based on environment and scripts."""
        has_script = self.script_combo.count() > 0
        python_path = self.python_path_edit.text().strip()
        has_python = bool(python_path) and Path(python_path).exists()

        if not has_script:
            self.start_btn.setEnabled(False)
            return

        if self._env_info is None:
            # Not checked yet — allow but warn
            self.start_btn.setEnabled(has_python)
            return

        if self._env_info.status in (EnvironmentStatus.ERROR,):
            self.start_btn.setEnabled(False)
        else:
            self.start_btn.setEnabled(has_python)

    # ── Project-python binding persistence ──────────────────────────────

    def _save_python_binding(self, project_dir: Path, python_path: str):
        """Save project-to-python binding via QSettings."""
        try:
            from PyQt6 import QtCore
            settings = QtCore.QSettings("anylabeling", "anylabeling")
            key = _make_binding_key(str(project_dir))
            settings.setValue(key, python_path)
        except Exception:
            pass

    def _load_python_binding(self, project_dir: Path) -> Optional[str]:
        """Load saved python binding for a project."""
        try:
            from PyQt6 import QtCore
            settings = QtCore.QSettings("anylabeling", "anylabeling")
            key = _make_binding_key(str(project_dir))
            val = settings.value(key)
            return val if val else None
        except Exception:
            return None

    def _on_start_training(self):
        """Handle start training - integrated with JobManager"""
        if not self.workspace:
            return

        # ── Early mutual exclusion check ──
        current = self.job_manager.get_current_job()
        if current is not None and current.status.is_active():
            QMessageBox.critical(
                self, "Training Busy",
                f"Another training job is already running:\n"
                f"{current.display_name}\n\n"
                f"Please stop it or wait for it to finish before starting a new one."
            )
            return

        script_idx = self.script_combo.currentIndex()
        if script_idx < 0:
            QMessageBox.warning(self, "Warning", "Please select a training script")
            return

        python_path_str = self.python_path_edit.text().strip()
        if not python_path_str:
            QMessageBox.warning(self, "Warning", "Please select a Python environment")
            return

        python_path = Path(python_path_str)
        if not python_path.exists():
            QMessageBox.critical(
                self, "Error",
                f"Python executable not found:\n{python_path}"
            )
            return

        # Quick validation
        import subprocess
        try:
            result = subprocess.run(
                [str(python_path), "--version"],
                capture_output=True, timeout=10,
            )
            if result.returncode != 0:
                QMessageBox.critical(
                    self, "Error",
                    f"Python does not execute:\n{python_path}\n\n{result.stderr}"
                )
                return
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot run Python:\n{python_path}\n\n{e}")
            return

        selected_script: DetectedScript = self.script_combo.itemData(script_idx)

        args_text = self.args_edit.text().strip()
        arguments = args_text.split() if args_text else []

        # Create unified TrainingJob
        job_id = f"custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_job = TrainingJob(
            job_id=job_id,
            mode=TrainingMode.CUSTOM_SCRIPT,
            status=TrainingStatus.IDLE,
            created_at=datetime.now(),
            started_at=None,
            ended_at=None,
            workspace=self.workspace.path,
            output_directory=None,
            display_name=f"Custom: {selected_script.path.name}",
            framework=selected_script.framework or "custom",
            python_executable=python_path,
            command=[str(selected_script.path)] + arguments,
            metadata={
                "script_path": str(selected_script.path),
                "arguments": arguments,
            },
            error_message=None,
        )

        # Create legacy Run for backward compatibility
        self.current_run = Run(
            run_id=job_id,
            workspace_path=self.workspace.path,
            script_path=selected_script.path,
            python_path=python_path,
            arguments=arguments,
            framework=selected_script.framework,
        )

        self.console_output.clear()
        self.console_output.appendPlainText(f"[{job_id}] Starting training via JobManager...")
        self.console_output.appendPlainText(
            f"Script: {selected_script.path.relative_to(self.workspace.path)}"
        )
        self.console_output.appendPlainText(f"Python: {python_path}")
        if arguments:
            self.console_output.appendPlainText(f"Arguments: {' '.join(arguments)}")
        self.console_output.appendPlainText("")

        # Save to RunStorage (legacy)
        if self.storage:
            self.storage.save_run(self.current_run)
            event = create_run_created_event(
                run_id=job_id,
                timestamp=time.time(),
                script=str(selected_script.path),
                python=str(python_path),
                framework=selected_script.framework,
            )
            self.storage.save_event(event)

        # Request start via JobManager (sole HistoryStore authority)
        from anylabeling.services.training_center.adapters.custom_script_adapter import CustomScriptAdapter

        adapter = CustomScriptAdapter()

        config = {
            'script_path': str(selected_script.path),
            'arguments': arguments,
        }

        success, message = self.job_manager.request_start(
            job=self.current_job,
            adapter=adapter,
            config=config
        )

        if not success:
            QMessageBox.critical(self, "Error", f"Failed to start training: {message}")
            self.current_run = None
            self.current_job = None
            return

        self.console_output.appendPlainText(f"[System] {message}")

        self._update_status(RunStatus.RUNNING)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.script_combo.setEnabled(False)
        self.args_edit.setEnabled(False)
        self.open_workspace_btn.setEnabled(False)
        self._set_env_controls_enabled(False)

        # Bind metrics dashboard
        od = str(self.workspace.path) if self.workspace else None
        self.metrics_dashboard.bind_job(job_id, od)

        # Callback
        if self.on_run_start:
            self.on_run_start(self.current_run)

    def _on_stop_training(self):
        """Handle stop training - integrated with JobManager"""
        current_job = self.job_manager.get_current_job()
        if not current_job or not current_job.status.is_active():
            return

        reply = QMessageBox.question(
            self,
            "Confirm Stop",
            "Are you sure you want to stop the training process?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.console_output.appendPlainText("\n[System] Stopping training via JobManager...")

            # Disable stop button immediately to prevent double-click
            self.stop_btn.setEnabled(False)
            self._update_status(RunStatus.STOPPING)

            # Request stop via JobManager
            success = self.job_manager.request_stop()

            if success:
                self.console_output.appendPlainText("[System] Stop signal sent")
            else:
                self.console_output.appendPlainText("[System] Stop request failed or already stopping")
                # Keep button disabled - will be re-enabled on terminal state

            # Callback
            if self.on_run_stop:
                self.on_run_stop()

    def _on_resource_sample(self, sample: dict):
        """Handle resource sample"""
        lines = []

        if "process" in sample:
            cpu = sample["process"].get("cpu_percent", 0)
            mem = sample["process"].get("memory_percent", 0)
            lines.append(f"Process CPU: {cpu:.1f}%")
            lines.append(f"Process Memory: {mem:.1f}%")

        if "system" in sample:
            cpu = sample["system"].get("cpu_percent", 0)
            mem = sample["system"].get("memory_percent", 0)
            lines.append(f"System CPU: {cpu:.1f}%")
            lines.append(f"System Memory: {mem:.1f}%")

        if "gpu" in sample and "gpus" in sample["gpu"]:
            for gpu in sample["gpu"]["gpus"]:
                idx = gpu.get("index", 0)
                util = gpu.get("utilization", 0)
                mem_used = gpu.get("memory_used", 0)
                mem_total = gpu.get("memory_total", 1)
                mem_pct = (mem_used / mem_total * 100) if mem_total > 0 else 0
                lines.append(f"GPU {idx}: {util:.1f}% (VRAM: {mem_pct:.1f}%)")

        if lines:
            self.resources_label.setText("\n".join(lines))
            self.resources_label.setStyleSheet("color: black; font-family: monospace;")
        else:
            self.resources_label.setText("No resource data")

        if self.storage and self.current_run:
            self.storage.save_resource_sample(self.current_run.run_id, sample)

    def _update_status(self, status: RunStatus):
        """Update status display"""
        if status == RunStatus.IDLE:
            self.status_label.setText("Idle")
            self.status_label.setStyleSheet("font-weight: bold; color: gray;")
        elif status == RunStatus.RUNNING:
            self.status_label.setText("Running")
            self.status_label.setStyleSheet("font-weight: bold; color: green;")
        elif status == RunStatus.COMPLETED:
            self.status_label.setText("Completed")
            self.status_label.setStyleSheet("font-weight: bold; color: blue;")
        elif status == RunStatus.FAILED:
            self.status_label.setText("Failed")
            self.status_label.setStyleSheet("font-weight: bold; color: red;")
        elif status == RunStatus.STOPPED:
            self.status_label.setText("Stopped")
            self.status_label.setStyleSheet("font-weight: bold; color: orange;")
        elif status == RunStatus.STOPPING:
            self.status_label.setText("Stopping...")
            self.status_label.setStyleSheet("font-weight: bold; color: orange;")

    def is_running(self) -> bool:
        """Check if training is currently running"""
        current_job = self.job_manager.get_current_job()
        return current_job is not None and current_job.status.is_active()

    def _on_training_event(self, event: TrainingEvent):
        """Handle unified training events from JobManager

        Maps TrainingEvent to UI updates (display only).
        HistoryStore writes are handled by JobManager.
        """
        if not self.current_job or event.job_id != self.current_job.job_id:
            return

        if event.event_type == TrainingEventType.PROCESS_STARTED:
            pid = event.payload.get('pid', 0)
            self.console_output.appendPlainText(f"[System] Process started (PID: {pid})")

            # Start resource monitoring
            if pid:
                self.resource_monitor.start_monitoring(pid, interval_ms=1000)

        elif event.event_type == TrainingEventType.CONSOLE_OUTPUT:
            # Display console output
            stream = event.payload.get('stream', 'stdout')
            message = event.payload.get('message', '')

            if stream == 'stderr':
                timestamp = datetime.now().strftime("%H:%M:%S")
                self.console_output.appendPlainText(f"[{timestamp}] ERROR: {message}")
            else:
                timestamp = datetime.now().strftime("%H:%M:%S")
                self.console_output.appendPlainText(f"[{timestamp}] {message}")

        elif event.event_type == TrainingEventType.EPOCH_METRICS:
            self.metrics_dashboard.on_metric_event(event.job_id, event.payload)

        elif event.event_type == TrainingEventType.COMPLETED:
            exit_code = event.payload.get('exit_code', 0)
            self.console_output.appendPlainText(
                f"\n[System] Process completed (exit code: {exit_code})"
            )

            self.resource_monitor.stop_monitoring()
            self._update_status(RunStatus.COMPLETED)
            self._reset_ui_after_completion()
            self.metrics_dashboard.on_run_completed(self.current_job.job_id)

            # Callback
            if self.on_run_complete:
                self.on_run_complete(exit_code)

        elif event.event_type == TrainingEventType.FAILED:
            error_msg = event.payload.get("error", "Unknown error")
            exit_code = event.payload.get('exit_code', 1)
            self.console_output.appendPlainText(
                f"\n[System] Process failed (exit code: {exit_code})"
            )
            self.console_output.appendPlainText(f"[System] Error: {error_msg}")

            self.resource_monitor.stop_monitoring()
            self._update_status(RunStatus.FAILED)
            self._reset_ui_after_completion()
            self.metrics_dashboard.on_run_stopped(self.current_job.job_id)

            # Callback
            if self.on_run_complete:
                self.on_run_complete(exit_code)

        elif event.event_type == TrainingEventType.STOPPED:
            self.console_output.appendPlainText(f"\n[System] Process stopped by user")

            self.resource_monitor.stop_monitoring()
            self._update_status(RunStatus.STOPPED)
            self._reset_ui_after_completion()
            self.metrics_dashboard.on_run_stopped(self.current_job.job_id)

            # Callback
            if self.on_run_complete:
                self.on_run_complete(0)

    def _reset_ui_after_completion(self):
        """Reset UI controls after job completion"""
        self.stop_btn.setEnabled(False)
        self._update_start_button()
        self.script_combo.setEnabled(True)
        self.args_edit.setEnabled(True)
        self.open_workspace_btn.setEnabled(True)
        self._update_env_buttons()

        # Reset resources display
        self.resources_label.setText("No active run")
        self.resources_label.setStyleSheet("color: gray;")

    def cleanup(self):
        """Clean up resources (call before destroying widget).

        Disconnects UI signals from running workers but does NOT
        block on thread completion. Internal lifecycle is managed
        by EnvironmentTaskController — never touched here.
        """
        self.metrics_dashboard.cleanup()
        # Disconnect UI signals from current env task (safe)
        self._cleanup_env_thread()
        # Controller survives — its tasks continue independently

        if self.scanner_thread and self.scanner_thread.isRunning():
            self.scanner_thread.cancel()
            self.scanner_thread.wait()

        self.resource_monitor.stop_monitoring()

        # Unsubscribe from JobManager events
        self.job_manager.unsubscribe_events(self._on_training_event)

        # Don't force stop - let JobManager handle graceful shutdown
