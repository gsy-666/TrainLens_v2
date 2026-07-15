"""Run Monitor Widget - Reusable Component

Extracted from RunMonitorWindow to be embeddable in other containers.
Preserves all functionality: workspace scanning, script detection, environment detection,
configuration, start/stop, console output, and resource monitoring.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

from PyQt6.QtCore import Qt, QTimer
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
        self.storage: Optional[RunStorage] = None
        self.scanner_thread: Optional[WorkspaceScannerThread] = None

        # Services
        self.process_manager = ProcessManager()
        self.resource_monitor = ResourceMonitor()

        # Callbacks
        self.on_run_start: Optional[Callable[[Run], None]] = None
        self.on_run_stop: Optional[Callable[[], None]] = None
        self.on_run_complete: Optional[Callable[[int], None]] = None

        # Connect signals
        self.process_manager.process_started.connect(self._on_process_started)
        self.process_manager.process_finished.connect(self._on_process_finished)
        self.process_manager.stdout_ready.connect(self._on_stdout)
        self.process_manager.stderr_ready.connect(self._on_stderr)
        self.resource_monitor.resource_sample.connect(self._on_resource_sample)

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
        """Create right panel with configuration and status"""
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

        # Configuration
        config_group = QGroupBox("Run Configuration")
        config_layout = QFormLayout(config_group)

        self.script_combo = QComboBox()
        self.script_combo.setEnabled(False)
        config_layout.addRow("Script:", self.script_combo)

        self.python_combo = QComboBox()
        self.python_combo.setEnabled(False)
        config_layout.addRow("Python:", self.python_combo)

        self.args_edit = QLineEdit()
        self.args_edit.setPlaceholderText("Additional arguments...")
        self.args_edit.setEnabled(False)
        config_layout.addRow("Arguments:", self.args_edit)

        layout.addWidget(config_group)

        # Resources
        resources_group = QGroupBox("Resources")
        resources_layout = QVBoxLayout(resources_group)
        self.resources_label = QLabel("No active run")
        self.resources_label.setStyleSheet("color: gray; font-family: monospace;")
        self.resources_label.setWordWrap(True)
        resources_layout.addWidget(self.resources_label)
        layout.addWidget(resources_group)

        # Metrics placeholder
        metrics_group = QGroupBox("Training Metrics")
        metrics_layout = QVBoxLayout(metrics_group)
        self.metrics_label = QLabel(
            "Training metrics will be displayed when the script outputs structured events."
        )
        self.metrics_label.setStyleSheet("color: gray;")
        self.metrics_label.setWordWrap(True)
        metrics_layout.addWidget(self.metrics_label)
        layout.addWidget(metrics_group)

        layout.addStretch()

        return widget

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
                    f"✓ {rel_path} ({framework}, confidence: {script.confidence:.2f})"
                )
                self.script_combo.addItem(str(rel_path), script)

            self.scripts_label.setText("\n".join(scripts_text))
            self.scripts_label.setStyleSheet("color: black;")
            self.script_combo.setEnabled(True)
        else:
            self.scripts_label.setText("No training scripts detected")
            self.scripts_label.setStyleSheet("color: orange;")

        # Update environments
        if self.workspace.detected_environments:
            envs_text = []
            self.python_combo.clear()
            for env in self.workspace.detected_environments:
                env_type = env.env_type
                version = env.version
                envs_text.append(f"• {env_type} (Python {version})")
                display_text = f"{env_type} - Python {version}"
                self.python_combo.addItem(display_text, env)

            self.envs_label.setText("\n".join(envs_text))
            self.envs_label.setStyleSheet("color: black;")
            self.python_combo.setEnabled(True)
        else:
            self.envs_label.setText("No Python environments detected")
            self.envs_label.setStyleSheet("color: orange;")

        # Enable controls if both available
        if self.workspace.detected_scripts and self.workspace.detected_environments:
            self.args_edit.setEnabled(True)
            self.start_btn.setEnabled(True)

    def _on_start_training(self):
        """Handle start training"""
        if not self.workspace:
            return

        script_idx = self.script_combo.currentIndex()
        python_idx = self.python_combo.currentIndex()

        if script_idx < 0 or python_idx < 0:
            QMessageBox.warning(
                self, "Warning", "Please select script and Python environment"
            )
            return

        selected_script: DetectedScript = self.script_combo.itemData(script_idx)
        selected_env: PythonEnvironment = self.python_combo.itemData(python_idx)

        args_text = self.args_edit.text().strip()
        arguments = args_text.split() if args_text else []

        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_run = Run(
            run_id=run_id,
            workspace_path=self.workspace.path,
            script_path=selected_script.path,
            python_path=selected_env.python_path,
            arguments=arguments,
            framework=selected_script.framework,
        )

        self.console_output.clear()
        self.console_output.appendPlainText(f"[{run_id}] Starting training...")
        self.console_output.appendPlainText(
            f"Script: {selected_script.path.relative_to(self.workspace.path)}"
        )
        self.console_output.appendPlainText(f"Python: {selected_env.python_path}")
        if arguments:
            self.console_output.appendPlainText(f"Arguments: {' '.join(arguments)}")
        self.console_output.appendPlainText("")

        if self.storage:
            self.storage.save_run(self.current_run)
            event = create_run_created_event(
                run_id=run_id,
                timestamp=time.time(),
                script=str(selected_script.path),
                python=str(selected_env.python_path),
                framework=selected_script.framework,
            )
            self.storage.save_event(event)

        success = self.process_manager.start(self.current_run)
        if not success:
            QMessageBox.critical(self, "Error", "Failed to start training process")
            self.current_run = None
            return

        self._update_status(RunStatus.RUNNING)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.script_combo.setEnabled(False)
        self.python_combo.setEnabled(False)
        self.args_edit.setEnabled(False)
        self.open_workspace_btn.setEnabled(False)

        # Callback
        if self.on_run_start:
            self.on_run_start(self.current_run)

    def _on_stop_training(self):
        """Handle stop training"""
        if not self.process_manager.is_running():
            return

        reply = QMessageBox.question(
            self,
            "Confirm Stop",
            "Are you sure you want to stop the training process?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.console_output.appendPlainText("\n[System] Stopping training...")
            self.process_manager.stop()
            self._update_status(RunStatus.STOPPING)

            # Callback
            if self.on_run_stop:
                self.on_run_stop()

    def _on_process_started(self, pid: int):
        """Handle process started"""
        self.console_output.appendPlainText(f"[System] Process started (PID: {pid})")

        if self.current_run and self.storage:
            event = create_process_started_event(
                run_id=self.current_run.run_id,
                timestamp=time.time(),
                pid=pid,
            )
            self.storage.save_event(event)

        self.resource_monitor.start_monitoring(pid, interval_ms=1000)

    def _on_process_finished(self, pid: int, exit_code: int):
        """Handle process finished"""
        self.console_output.appendPlainText(
            f"\n[System] Process finished (PID: {pid}, exit code: {exit_code})"
        )

        self.resource_monitor.stop_monitoring()

        if self.current_run and self.storage:
            duration = 0.0
            if self.current_run.start_time and self.current_run.end_time:
                duration = (
                    self.current_run.end_time - self.current_run.start_time
                ).total_seconds()

            if exit_code == 0:
                event = create_process_completed_event(
                    run_id=self.current_run.run_id,
                    timestamp=time.time(),
                    exit_code=exit_code,
                    duration=duration,
                )
            else:
                event = create_process_failed_event(
                    run_id=self.current_run.run_id,
                    timestamp=time.time(),
                    exit_code=exit_code,
                    error=f"Process exited with code {exit_code}",
                )
            self.storage.save_event(event)
            self.storage.save_run(self.current_run)

        if exit_code == 0:
            self._update_status(RunStatus.COMPLETED)
        else:
            self._update_status(RunStatus.FAILED)

        self.stop_btn.setEnabled(False)
        self.start_btn.setEnabled(True)
        self.script_combo.setEnabled(True)
        self.python_combo.setEnabled(True)
        self.args_edit.setEnabled(True)
        self.open_workspace_btn.setEnabled(True)

        # Callback
        if self.on_run_complete:
            self.on_run_complete(exit_code)

    def _on_stdout(self, line: str):
        """Handle stdout"""
        event = EventProtocol.parse_line(line)
        if event:
            self.console_output.appendPlainText(f"[Event] {event.event}")
            if self.storage and self.current_run:
                self.storage.save_event(event)
        else:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.console_output.appendPlainText(f"[{timestamp}] {line}")

        if self.storage and self.current_run:
            self.storage.save_console_line(self.current_run.run_id, line)

    def _on_stderr(self, line: str):
        """Handle stderr"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.console_output.appendPlainText(f"[{timestamp}] ERROR: {line}")

        if self.storage and self.current_run:
            self.storage.save_console_line(self.current_run.run_id, f"ERROR: {line}")

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
        return self.process_manager.is_running()

    def cleanup(self):
        """Clean up resources (call before destroying widget)"""
        if self.scanner_thread and self.scanner_thread.isRunning():
            self.scanner_thread.cancel()
            self.scanner_thread.wait()

        self.resource_monitor.stop_monitoring()

        if self.process_manager.is_running():
            self.process_manager.stop()
