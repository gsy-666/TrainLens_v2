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

# Training Center integration
from anylabeling.services.training_center.models import (
    TrainingJob,
    TrainingMode,
    TrainingStatus,
)
from anylabeling.services.training_center.job_manager import get_job_manager
from anylabeling.services.training_center.history import get_history_store, JobHistoryRecord
from anylabeling.services.training_center.event_protocol import TrainingEvent, TrainingEventType


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
        self.history_store = get_history_store()

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
        """Handle start training - integrated with JobManager"""
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
            python_executable=selected_env.python_path,
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
            python_path=selected_env.python_path,
            arguments=arguments,
            framework=selected_script.framework,
        )

        self.console_output.clear()
        self.console_output.appendPlainText(f"[{job_id}] Starting training via JobManager...")
        self.console_output.appendPlainText(
            f"Script: {selected_script.path.relative_to(self.workspace.path)}"
        )
        self.console_output.appendPlainText(f"Python: {selected_env.python_path}")
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
                python=str(selected_env.python_path),
                framework=selected_script.framework,
            )
            self.storage.save_event(event)

        # Save to HistoryStore (unified)
        history_record = JobHistoryRecord(
            job_id=job_id,
            mode=TrainingMode.CUSTOM_SCRIPT.value,
            status=TrainingStatus.IDLE.value,
            created_at=self.current_job.created_at.isoformat(),
            workspace=str(self.workspace.path),
            display_name=self.current_job.display_name,
            framework=self.current_job.framework,
            python_executable=str(selected_env.python_path),
            command=[str(selected_script.path)] + arguments,
            metadata=self.current_job.metadata,
        )
        self.history_store.append_job(history_record)

        # Request start via JobManager
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
        self.python_combo.setEnabled(False)
        self.args_edit.setEnabled(False)
        self.open_workspace_btn.setEnabled(False)

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

        Maps TrainingEvent to UI updates and HistoryStore writes.
        """
        if not self.current_job or event.job_id != self.current_job.job_id:
            return

        # Update HistoryStore based on event type
        if event.event_type == TrainingEventType.PROCESS_STARTED:
            pid = event.payload.get('pid', 0)
            self.console_output.appendPlainText(f"[System] Process started (PID: {pid})")

            # Start resource monitoring
            if pid:
                self.resource_monitor.start_monitoring(pid, interval_ms=1000)

            self.history_store.update_job(
                event.job_id,
                status=TrainingStatus.RUNNING.value,
                started_at=datetime.fromtimestamp(event.timestamp).isoformat(),
            )

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

        elif event.event_type == TrainingEventType.COMPLETED:
            exit_code = event.payload.get('exit_code', 0)
            self.console_output.appendPlainText(
                f"\n[System] Process completed (exit code: {exit_code})"
            )

            self.resource_monitor.stop_monitoring()
            self._update_status(RunStatus.COMPLETED)
            self._reset_ui_after_completion()

            self.history_store.finalize_job(
                event.job_id,
                status=TrainingStatus.COMPLETED,
                ended_at=datetime.fromtimestamp(event.timestamp),
                output_directory=event.payload.get('save_dir', ''),
            )

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

            self.history_store.finalize_job(
                event.job_id,
                status=TrainingStatus.FAILED,
                ended_at=datetime.fromtimestamp(event.timestamp),
                error_message=error_msg,
                output_directory=event.payload.get('save_dir', ''),
            )

            # Callback
            if self.on_run_complete:
                self.on_run_complete(exit_code)

        elif event.event_type == TrainingEventType.STOPPED:
            self.console_output.appendPlainText(f"\n[System] Process stopped by user")

            self.resource_monitor.stop_monitoring()
            self._update_status(RunStatus.STOPPED)
            self._reset_ui_after_completion()

            self.history_store.finalize_job(
                event.job_id,
                status=TrainingStatus.STOPPED,
                ended_at=datetime.fromtimestamp(event.timestamp),
                output_directory=event.payload.get('save_dir', ''),
            )

            # Callback
            if self.on_run_complete:
                self.on_run_complete(0)

    def _reset_ui_after_completion(self):
        """Reset UI controls after job completion"""
        self.stop_btn.setEnabled(False)
        self.start_btn.setEnabled(True)
        self.script_combo.setEnabled(True)
        self.python_combo.setEnabled(True)
        self.args_edit.setEnabled(True)
        self.open_workspace_btn.setEnabled(True)

        # Reset resources display
        self.resources_label.setText("No active run")
        self.resources_label.setStyleSheet("color: gray;")

    def cleanup(self):
        """Clean up resources (call before destroying widget)"""
        if self.scanner_thread and self.scanner_thread.isRunning():
            self.scanner_thread.cancel()
            self.scanner_thread.wait()

        self.resource_monitor.stop_monitoring()

        # Unsubscribe from JobManager events
        self.job_manager.unsubscribe_events(self._on_training_event)

        # Don't force stop - let JobManager handle graceful shutdown
