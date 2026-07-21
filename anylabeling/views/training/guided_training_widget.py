import csv
import datetime
import glob
import json
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

from PyQt6.QtCore import Qt, QObject, QProcess, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from anylabeling.config import get_config
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.qt import new_icon
from anylabeling.views.labeling.utils.theme import get_theme
from anylabeling.views.training.widgets.ultralytics_widgets import *
from anylabeling.views.training.guided_training_widget_events import handle_unified_training_event
from anylabeling.services.auto_training.ultralytics._io import *
from anylabeling.services.auto_training.ultralytics.config import *
from anylabeling.services.auto_training.ultralytics.exporter import (
    ExportEventRedirector,
    ExportLogRedirector,
    get_export_manager,
)
from anylabeling.services.auto_training.ultralytics.general import (
    create_yolo_dataset,
    format_classes_display,
    parse_string_to_digit_list,
)
from anylabeling.services.auto_training.ultralytics.style import *
from anylabeling.services.auto_training.ultralytics.trainer import (
    TrainingEventRedirector,
    TrainingLogRedirector,
    get_training_manager,
)
from anylabeling.views.training.metrics import TrainingMetricsDashboard
from anylabeling.services.training_center.preflight.models import (
    GuidedPreflightContext,
)
from anylabeling.services.training_center.preflight.worker import PreflightWorker
from anylabeling.views.training.preflight_dialog import PreflightDialog
from anylabeling.services.auto_training.ultralytics.utils import *
from anylabeling.services.auto_training.ultralytics.validators import (
    validate_basic_config,
    validate_data_file,
    validate_task_requirements,
)
from anylabeling.services.training_center.job_manager import get_job_manager
from anylabeling.services.training_center.models import TrainingJob, TrainingMode, TrainingStatus
from anylabeling.services.training_center.adapters.ultralytics_adapter import UltralyticsAdapter
from anylabeling.services.training_center.event_protocol import TrainingEventType

def _get_combo_training_value(combo):
    """Extract training_value string from a QComboBox item.

    Handles both DeviceInfo objects (new) and bare strings (legacy).
    Returns the training_value string suitable for config dicts.
    """
    data = combo.currentData()
    if data is None:
        return "auto"
    if hasattr(data, 'training_value'):
        return data.training_value
    return str(data)


class _TrainingPrepWorker(QObject):
    """Background worker for dataset creation + training args preparation.

    Runs on a QThread to keep the GUI responsive during
    create_yolo_dataset() — which copies images, parses labels,
    and writes YAML — all of which are synchronous file I/O.
    """

    finished = pyqtSignal(dict)   # train_args on success
    error = pyqtSignal(str)       # error message on failure

    def __init__(self, widget, config):
        super().__init__()
        self.widget = widget
        self.config = config

    def run(self):
        """Execute get_training_args() on background thread."""
        try:
            train_args = self.widget.get_training_args(self.config)
            self.finished.emit(train_args)
        except Exception as e:
            self.error.emit(str(e))


class GuidedTrainingWidget(QWidget):
    def __init__(
        self,
        parent=None,
        image_list=None,
        output_dir=None,
        supported_shape=None,
        open_folder_callback=None,
        image_list_getter=None,
        job_manager=None,           # Accepted for compat, unused (singleton)
        ultralytics_adapter=None,   # Accepted for compat, unused
        history_store=None,         # Accepted for compat, unused (singleton)
    ):
        super().__init__(parent)

        # Store callbacks for explicit dependency injection (no parent() chain)
        self._open_folder_callback = open_folder_callback
        self._image_list_getter = image_list_getter

        # Accept parameters directly or from parent (backward compat)
        if parent is not None and hasattr(parent, 'image_list'):
            self.image_list = parent.image_list
            self.output_dir = parent.output_dir
            self.supported_shape = parent.supported_shape
        else:
            self.image_list = image_list if image_list is not None else []
            self.output_dir = output_dir if output_dir is not None else ""
            self.supported_shape = supported_shape if supported_shape is not None else []
        self.selected_task_type = None
        self.config_widgets = {}
        self._classification_cache = None
        self._detection_cache = None
        self._valid_image_count_cache = {}
        self._summary_view_mode = None
        self._config_tab_initialized = False
        self._train_tab_initialized = False
        self.task_type_buttons = {}
        self.names = []

        # Training related attributes - keep for backward compatibility
        self.log_redirector = TrainingLogRedirector()
        self.log_redirector.log_signal.connect(
            self.append_training_log, Qt.ConnectionType.QueuedConnection
        )
        self.event_redirector = TrainingEventRedirector()
        self.event_redirector.training_event_signal.connect(
            self.on_training_event, Qt.ConnectionType.QueuedConnection
        )
        self.training_manager = get_training_manager()
        self.training_manager.callbacks = [
            self.event_redirector.emit_training_event
        ]

        # Integrated JobManager for mutual exclusion with Run Monitor
        self.job_manager = get_job_manager()
        self.job_manager.subscribe_events(self._on_training_event_from_job)
        self.job_manager.subscribe_status(self._on_job_status_change)

        # Two-phase prepare start (background dataset creation)
        self._prep_thread = None
        self._prep_worker = None
        self._prep_job_id = None
        self._prep_adapter = None
        self._pending_project_path = None
        self._pending_name = None

        # Export related attributes
        self.export_log_redirector = ExportLogRedirector()
        self.export_log_redirector.log_signal.connect(
            self.append_training_log, Qt.ConnectionType.QueuedConnection
        )
        self.export_event_redirector = ExportEventRedirector()
        self.export_event_redirector.export_event_signal.connect(
            self.on_export_event, Qt.ConnectionType.QueuedConnection
        )
        self.export_manager = get_export_manager()
        self.export_manager.callbacks = [
            self.export_event_redirector.emit_export_event
        ]

        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_training_progress)
        self.image_timer = QTimer()
        self.image_timer.timeout.connect(self.update_training_images)
        self.current_project_path = None

        # Metrics dashboard (lazy-init on first use)
        self._metrics_dashboard = None  # TrainingMetricsDashboard

        # Preflight state
        self._preflight_worker: Optional[PreflightWorker] = None
        self._preflight_thread: Optional[QThread] = None
        self._preflight_result = None
        self._preflight_running = False

        # Prepared dataset (generated from loaded images)
        self._prepared_dataset_dir: Optional[str] = None
        self._prepared_yaml_path: Optional[str] = None
        self._dataset_preparing = False
        self.training_status = "idle"  # idle, training, completed, error
        self.current_epochs = 0

        try:
            app_config = get_config()
            self.project_readonly = (
                app_config.get("training", {})
                .get("ultralytics", {})
                .get("project_readonly", False)
            )
        except Exception:
            # Fallback for tests or when config is unavailable
            self.project_readonly = False

        self.init_ui()
        self.setStyleSheet(get_ultralytics_dialog_style())
        self.refresh_dataset_summary()
        self.update_labeled_images_hint()

    def init_ui(self):
        self.data_tab = QWidget()
        self.config_tab = QWidget()
        self.train_tab = QWidget()
        self.metrics_tab = QWidget()  # placeholder, dashboard lazy-init

        self.tab_widget = QTabWidget()
        self.tab_widget.addTab(self.data_tab, self.tr("Data"))
        self.tab_widget.addTab(self.config_tab, self.tr("Config"))
        self.tab_widget.addTab(self.train_tab, self.tr("Train"))
        self.tab_widget.addTab(self.metrics_tab, self.tr("Metrics"))
        self.tab_widget.setTabVisible(3, False)  # hidden until training starts

        # Stage gating: Data starts enabled, Config/Train require data check
        self._data_check_passed = False
        self._config_completed = False
        self._update_stage_gates()

        # Config controls: disabled during PREPARING / RUNNING / STOPPING
        self._training_config_controls = []  # populated on first use
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.tab_widget)

        self.init_data_tab()

    def _update_stage_gates(self):
        """Enable/disable tabs based on current stage.

        Data tab always enabled. Config enabled after Data Check passes.
        Train enabled after Config is completed. Metrics shown during training.
        Next button always enabled — navigation is gated by click handler.
        """
        data_ok = self._data_check_passed
        config_ok = self._config_completed

        self.tab_widget.setTabEnabled(0, True)   # Data always enabled
        self.tab_widget.setTabEnabled(1, data_ok)  # Config: data must pass
        self.tab_widget.setTabEnabled(2, data_ok and config_ok)  # Train: both

    def ensure_config_tab_initialized(self):
        if self._config_tab_initialized:
            return
        self.init_config_tab()
        self._config_tab_initialized = True

    def ensure_train_tab_initialized(self):
        if self._train_tab_initialized:
            return
        self.init_train_tab()
        self._train_tab_initialized = True

    # ── Config control management ───────────────────────────────────

    def _gather_config_controls(self) -> list:
        """Return all controls that modify training config (cached after first call)."""
        if self._training_config_controls:
            return self._training_config_controls

        controls = []

        # Data tab: task type buttons
        for btn in getattr(self, 'task_type_buttons', {}).values():
            if hasattr(btn, 'isEnabled'):
                controls.append(btn)

        # Data tab: load images button
        if hasattr(self, 'load_images_button'):
            controls.append(self.load_images_button)

        # Config tab: all config_widgets
        for w in getattr(self, 'config_widgets', {}).values():
            if hasattr(w, 'isEnabled'):
                controls.append(w)

        # Config tab: import/save buttons
        for attr in ('_import_btn', '_save_config_btn', '_previous_btn', '_train_btn'):
            w = getattr(self, attr, None)
            if w is not None and hasattr(w, 'isEnabled'):
                controls.append(w)

        # Train tab: start button, previous button
        for attr in ('start_training_button', 'previous_button'):
            w = getattr(self, attr, None)
            if w is not None and hasattr(w, 'isEnabled'):
                controls.append(w)

        self._training_config_controls = [c for c in controls if c is not None]
        return self._training_config_controls

    def _set_config_controls_enabled(self, enabled: bool):
        """Enable/disable only configuration-changing controls.

        Does NOT touch: tab bar, stop button, console, metrics dashboard,
        export button, or Training Center top-level tabs.
        """
        for control in self._gather_config_controls():
            try:
                control.setEnabled(enabled)
            except RuntimeError:
                pass  # control was deleted

    def _ensure_metrics_dashboard(self) -> TrainingMetricsDashboard:
        """Lazy-init the metrics dashboard on the Metrics tab."""
        if self._metrics_dashboard is not None:
            return self._metrics_dashboard
        from anylabeling.views.training.metrics import TrainingMetricsDashboard

        self._metrics_dashboard = TrainingMetricsDashboard()
        layout = QVBoxLayout(self.metrics_tab)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self._metrics_dashboard)
        return self._metrics_dashboard

    def save_training_logs_to_file(self):
        """Save training logs to a local file with timestamp"""
        if (
            not hasattr(self, "log_display")
            or not self.log_display.toPlainText().strip()
        ):
            return

        if not os.path.exists(self.current_project_path):
            return
        log_dir_path = os.path.join(self.current_project_path, "logs")
        os.makedirs(log_dir_path, exist_ok=True)

        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_log_{self.training_status}_{timestamp}.txt"
            log_file_path = os.path.join(log_dir_path, filename)

            with open(log_file_path, "w", encoding="utf-8") as f:
                f.write(self.log_display.toPlainText())

            logger.info(f"Training logs saved to: {log_file_path}")

        except Exception as e:
            logger.error(f"Failed to save training logs: {str(e)}")

    def has_active_training(self) -> bool:
        """Check if training is currently active"""
        return self.training_status == "training"

    def request_stop(self) -> None:
        """Request to stop the current training"""
        if self.training_status == "training":
            success = self.job_manager.request_stop()
            if success:
                self.append_training_log(self.tr("Stopping training..."))
            else:
                self.append_training_log(self.tr("Cancel to stop training"))

    def shutdown(self) -> None:
        """Shutdown widget and cleanup resources (idempotent)"""
        # Stop background prep thread if running
        if hasattr(self, '_prep_thread') and self._prep_thread is not None:
            try:
                self._prep_thread.quit()
                self._prep_thread.wait(1000)
            except Exception:
                pass
            self._prep_thread = None
            self._prep_worker = None
            self._prep_job_id = None
            self._prep_adapter = None

        # Save logs if needed
        if self.training_status in ["completed", "error", "stop"]:
            self.save_training_logs_to_file()

        # Stop timers
        if hasattr(self, 'progress_timer'):
            self.progress_timer.stop()
        if hasattr(self, 'image_timer'):
            self.image_timer.stop()

        # Disconnect signals
        try:
            if hasattr(self, 'log_redirector'):
                self.log_redirector.log_signal.disconnect()
            if hasattr(self, 'event_redirector'):
                self.event_redirector.training_event_signal.disconnect()
            if hasattr(self, 'export_log_redirector'):
                self.export_log_redirector.log_signal.disconnect()
            if hasattr(self, 'export_event_redirector'):
                self.export_event_redirector.export_event_signal.disconnect()
        except Exception:
            pass  # Signals may already be disconnected

        # Clear cache
        self.clear_cache()

    def closeEvent(self, event):
        """Handle widget close - business logic only"""
        # Widget doesn't own the close event, but can cleanup
        self.shutdown()
        super().closeEvent(event)

    def go_to_specific_tab(self, index):
        """Go to specific tab by index"""
        if index >= 1:
            self.ensure_config_tab_initialized()
        if index >= 2:
            self.ensure_train_tab_initialized()
        self._update_stage_gates()  # refresh gate state before switching
        self.tab_widget.setCurrentIndex(index)
        # Refresh dataset samples when navigating to Train tab
        if index == 2 and self.training_status in ("idle", "completed"):
            self._display_dataset_samples()

    # Data Tab
    def show_pose_config(self):
        """Show the pose config field"""
        if hasattr(self, "pose_config_label"):
            self.pose_config_label.setVisible(True)
            self.config_widgets["pose_config"].setVisible(True)

            for i in range(self.pose_config_layout.count()):
                widget = self.pose_config_layout.itemAt(i).widget()
                if widget:
                    widget.setVisible(True)

    def hide_pose_config(self):
        """Hide the pose config field"""
        if hasattr(self, "pose_config_label"):
            self.pose_config_label.setVisible(False)
            self.config_widgets["pose_config"].setVisible(False)

            for i in range(self.pose_config_layout.count()):
                widget = self.pose_config_layout.itemAt(i).widget()
                if widget:
                    widget.setVisible(False)

    def on_task_type_selected(self, task_type):
        normalized_task_type = None
        for task in TASK_TYPES:
            if task.lower() == task_type.lower():
                normalized_task_type = task
                break

        if normalized_task_type is None:
            logger.warning(f"Unknown task type: {task_type}")
            return

        task_type = normalized_task_type

        if task_type not in self.task_type_buttons:
            logger.warning(f"Task type button not found: {task_type}")
            return

        if self.selected_task_type == task_type:
            self.selected_task_type = None
            self.task_type_buttons[task_type].set_selected(False)
            self.hide_pose_config()
        else:
            if self.selected_task_type:
                self.task_type_buttons[self.selected_task_type].set_selected(
                    False
                )
            self.selected_task_type = task_type
            self.task_type_buttons[task_type].set_selected(True)

            if task_type.lower() == "pose":
                self.show_pose_config()
            else:
                self.hide_pose_config()

        self.refresh_dataset_summary()
        self.update_labeled_images_hint()
        self._invalidate_data_check()

    def create_task_handler(self, task_type):
        def handler():
            self.on_task_type_selected(task_type)

        return handler

    def init_task_configuration(self, parent_layout):
        config_widget = QWidget()
        config_layout = QVBoxLayout(config_widget)

        task_type_layout = QHBoxLayout()
        task_type_layout.addWidget(QLabel(self.tr("Task Type:")))
        for task_type in TASK_TYPES:
            button = CustomQPushButton(task_type)
            button.clicked.connect(self.create_task_handler(task_type))
            task_type_layout.addWidget(button)
            self.task_type_buttons[task_type] = button

        task_type_layout.addStretch()
        self.labeled_images_hint = QLabel()
        self.labeled_images_hint.setVisible(False)
        self.labeled_images_hint.setStyleSheet(
            f"color: {get_theme()['text_secondary']}; font-size: 10px;"
        )
        task_type_layout.addWidget(self.labeled_images_hint)
        config_layout.addLayout(task_type_layout)
        parent_layout.addWidget(config_widget)

    def update_labeled_images_hint(self):
        if not self.selected_task_type:
            self.labeled_images_hint.setVisible(False)
            return

        if self.selected_task_type not in self._valid_image_count_cache:
            self._valid_image_count_cache[self.selected_task_type] = (
                get_task_valid_images(
                    self.image_list,
                    self.selected_task_type,
                    self.output_dir,
                )
            )

        valid_images = self._valid_image_count_cache[self.selected_task_type]
        theme = get_theme()
        color = (
            theme["success"]
            if valid_images >= MIN_LABELED_IMAGES_THRESHOLD
            else theme["error"]
        )
        self.labeled_images_hint.setText(
            f'{self.tr("Valid Images:")} {valid_images} | '
            f'{self.tr("Required:")} <span style="color: {color};">'
            f"{MIN_LABELED_IMAGES_THRESHOLD}</span>"
        )
        self.labeled_images_hint.setVisible(True)

    def refresh_dataset_summary(self):
        if not self.image_list:
            self.summary_table.clear()
            self._summary_view_mode = None
            return

        summary_view_mode = (
            "classify" if self.selected_task_type == "Classify" else "detect"
        )
        if self._summary_view_mode == summary_view_mode:
            return

        if summary_view_mode == "classify":
            table_data = self._get_classification_table_data()
        else:
            table_data = self._get_detection_table_data()

        self.summary_table.load_data(table_data)
        self._summary_view_mode = summary_view_mode

    def _get_classification_table_data(self):
        if self._classification_cache is None:
            self._classification_cache = self._compute_classification_data()
        return self._classification_cache

    def _get_detection_table_data(self):
        if self._detection_cache is None:
            self._detection_cache = self._compute_detection_data()
        return self._detection_cache

    def _compute_classification_data(self):
        headers = ["Label"] + self.supported_shape + ["Total"]

        # Get classification statistics
        classify_shapes = TASK_SHAPE_MAPPINGS.get("Classify", ["flags"])
        label_infos = get_label_infos(
            self.image_list, classify_shapes, self.output_dir
        )
        if not label_infos:
            return [headers]

        table_data = [headers]
        total_counts = [0] * len(self.supported_shape)
        total_images = 0

        for label, infos in sorted(label_infos.items()):
            # All shape columns are 0 for classification
            shape_counts = [0] * len(self.supported_shape)
            image_count = infos.get("_total", 0)
            total_images += image_count

            row = [label] + [str(c) for c in shape_counts] + [str(image_count)]
            table_data.append(row)

        total_row = (
            ["Total"] + [str(c) for c in total_counts] + [str(total_images)]
        )
        table_data.append(total_row)

        return table_data

    def _compute_detection_data(self):
        return get_statistics_table_data(
            self.image_list, self.supported_shape, self.output_dir
        )

    def clear_cache(self):
        self._classification_cache = None
        self._detection_cache = None
        self._valid_image_count_cache.clear()
        self._summary_view_mode = None
        self._invalidate_data_check()

    def _invalidate_data_check(self):
        """Invalidate Data Check result AND dataset cache when data/task changes."""
        if hasattr(self, '_data_check_passed') and self._data_check_passed:
            self._data_check_passed = False
            self._config_completed = False
            self._update_stage_gates()
        # Invalidate cached dataset (source changed)
        self._prepared_dataset_dir = None
        self._prepared_yaml_path = None
        self._update_dataset_status(None)

    def _run_data_check(self):
        """Run Data Check — only data-related checks (no model/device/output).

        Stores result in _preflight_result and updates stage gates.
        Shows a non-blocking summary via QMessageBox if UI is interactive.
        """
        from anylabeling.services.training_center.preflight.models import (
            PreflightResult, PreflightIssue, PreflightSeverity, GuidedPreflightContext,
        )
        from anylabeling.services.training_center.preflight.guided_yaml import (
            check_yaml_dataset_paths, check_yaml_structure, read_yaml_safe,
        )
        from anylabeling.services.training_center.preflight.guided_detect_labels import (
            check_detect_labels,
        )

        task_type = self.selected_task_type or ""
        image_count = len(self.image_list) if self.image_list else 0

        result = PreflightResult(mode="data_check")

        # Image count
        if image_count == 0:
            result.add(PreflightIssue(
                code="NO_IMAGES_LOADED", severity=PreflightSeverity.ERROR,
                title="No images loaded",
                message="Please load images before checking the dataset.",
            ))

        # Valid labeled images
        if task_type and image_count > 0:
            from anylabeling.services.auto_training.ultralytics.config import MIN_LABELED_IMAGES_THRESHOLD
            from anylabeling.services.auto_training.ultralytics.validators import get_task_valid_images
            valid = get_task_valid_images(self.image_list, task_type, self.output_dir)
            if valid < MIN_LABELED_IMAGES_THRESHOLD:
                result.add(PreflightIssue(
                    code="INSUFFICIENT_VALID_IMAGES", severity=PreflightSeverity.ERROR,
                    title=f"Only {valid} valid labeled images found",
                    message=f"At least {MIN_LABELED_IMAGES_THRESHOLD} valid labeled images required. Found: {valid}.",
                    suggestion="Add more labeled images or check label format.",
                ))
            else:
                result.add(PreflightIssue(
                    code="VALID_IMAGES_OK", severity=PreflightSeverity.PASS,
                    title=f"{valid} valid labeled images",
                    message=f"Valid labeled images: {valid} (required: {MIN_LABELED_IMAGES_THRESHOLD}).",
                ))

        # Dataset YAML + label checks (if config available)
        # Prefer prepared YAML; fall back to config data field
        if hasattr(self, 'config_widgets') and self.config_widgets:
            config = self.get_current_config()
            yaml_path = (getattr(self, '_prepared_yaml_path', None)
                         or config.get("basic", {}).get("data", ""))
            if yaml_path and os.path.isfile(yaml_path):
                yaml_data, yaml_error = read_yaml_safe(yaml_path)
                if yaml_error:
                    result.add(PreflightIssue(
                        code="YAML_READ_ERROR", severity=PreflightSeverity.ERROR,
                        title="Cannot read dataset YAML",
                        message=yaml_error, path=yaml_path,
                    ))
                elif yaml_data:
                    check_yaml_structure(result, yaml_path, yaml_data)
                    check_yaml_dataset_paths(result, yaml_path, yaml_data)
                    check_detect_labels(result, yaml_path, yaml_data, task_type)

        self._preflight_result = result

        # Update gates
        if result.can_start:
            self._data_check_passed = True
            self._update_stage_gates()
            QMessageBox.information(
                self, self.tr("Data Check Passed"),
                self.tr(f"Data check completed successfully.\n{result.summary()}"),
            )
        else:
            self._data_check_passed = False
            self._update_stage_gates()
            # Show errors as QMessageBox
            error_msgs = "\n".join(
                f"• [{i.severity.value.upper()}] {i.title}" for i in result.errors()
            )
            if error_msgs:
                QMessageBox.critical(
                    self, self.tr("Data Check Failed"),
                    self.tr(f"Data check found issues:\n\n{error_msgs}\n\n"
                            f"Please fix these issues and re-run Check Dataset."),
                )

    def load_images(self):
        """Load images via the host's folder dialog callback.

        Uses explicit callback injection — no parent() chain traversal.
        """
        if not callable(self._open_folder_callback):
            QMessageBox.information(
                self,
                self.tr("Load Images"),
                self.tr("Image loading is not available in standalone mode."),
            )
            return

        previous_images = list(self.image_list) if self.image_list else []

        # Invoke the host's folder dialog
        self._open_folder_callback()

        # Sync latest image list from host
        if callable(self._image_list_getter):
            latest_images = self._image_list_getter()
            if latest_images:
                self.image_list = list(latest_images)
            elif previous_images:
                # User cancelled — restore previous list
                self.image_list = previous_images
        elif previous_images:
            self.image_list = previous_images

        self.clear_cache()
        self.refresh_dataset_summary()
        self.update_labeled_images_hint()

    def sync_image_list_from_host(self):
        """Pull latest image_list from the host via image_list_getter.

        Returns:
            True if sync succeeded, False otherwise.
        """
        if not callable(self._image_list_getter):
            return False

        images = self._image_list_getter()
        if images is None:
            return False

        self.image_list = list(images)
        self.clear_cache()
        self.refresh_dataset_summary()
        self.update_labeled_images_hint()
        return True

    def init_dataset_summary(self, parent_layout):
        summary_widget = QWidget()
        summary_layout = QVBoxLayout(summary_widget)
        summary_layout.addWidget(QLabel(self.tr("Dataset Summary:")))

        self.summary_table = CustomTable()
        summary_layout.addWidget(self.summary_table)
        parent_layout.addWidget(summary_widget, 1)

    def proceed_to_config(self):
        """Handle Next button click with clear feedback for each state.

        Gating order:
        1. No images loaded → tell user to load images
        2. Images loaded but < required → show current vs required count
        3. Data Check not done → prompt to run Check Dataset
        4. Data Check has ERROR → re-show last result
        5. Data Check PASS → navigate to Config
        """
        from anylabeling.services.auto_training.ultralytics.config import MIN_LABELED_IMAGES_THRESHOLD

        task_type = self.selected_task_type or ""

        # 1. No images loaded
        if not self.image_list:
            self._show_data_gate_message(
                self.tr("No Images Loaded"),
                self.tr("Load images before continuing.\n\n"
                        f"At least {MIN_LABELED_IMAGES_THRESHOLD} valid labeled images are required."),
            )
            return

        # 2. Count valid images
        from anylabeling.services.auto_training.ultralytics.validators import get_task_valid_images
        if task_type:
            valid = get_task_valid_images(self.image_list, task_type, self.output_dir)
        else:
            valid = 0

        if valid < MIN_LABELED_IMAGES_THRESHOLD:
            self._show_data_gate_message(
                self.tr("Not Enough Valid Images"),
                self.tr(f"Current valid images: {valid}\n"
                        f"Required valid images: {MIN_LABELED_IMAGES_THRESHOLD}\n\n"
                        f"Load or label more images before continuing."),
            )
            return

        # 3. Data Check not performed yet
        if not self._data_check_passed:
            # Auto-run Data Check
            self._run_data_check()
            if self._data_check_passed:
                # Check passed → proceed
                pass  # Fall through to navigation
            elif self._preflight_result and self._preflight_result.has_errors:
                # ERROR → re-show last result
                self._show_data_gate_message(
                    self.tr("Data Check Failed"),
                    self.tr("The dataset check found errors:\n\n") +
                    "\n".join(f"• {i.title}" for i in self._preflight_result.errors()),
                )
                return
            else:
                # Warnings or cancelled → let user try again
                return

        # 4. Navigate to Config
        is_valid, error_message = validate_task_requirements(
            self.selected_task_type, self.image_list, self.output_dir
        )
        if not is_valid:
            self._show_data_gate_message(
                self.tr("Validation Error"), error_message,
            )
            return

        self.ensure_config_tab_initialized()
        current_project = self.config_widgets["project"].text().strip()
        if not current_project:
            current_project = os.path.join(
                get_default_project_dir(), self.selected_task_type.lower()
            )
            self.config_widgets["project"].setText(current_project)
        self.config_widgets["project"].setReadOnly(self.project_readonly)

        self.go_to_specific_tab(1)

    def _show_data_gate_message(self, title: str, message: str):
        """Show a data gate message to the user.

        Production: uses QMessageBox.warning.
        Tests can patch this method to avoid modal dialogs blocking.
        """
        QMessageBox.warning(self, title, message)

    def init_actions(self, parent_layout):
        actions_layout = QHBoxLayout()

        self.load_images_button = SecondaryButton(self.tr("Load Images"))
        self.load_images_button.clicked.connect(self.load_images)
        actions_layout.addWidget(self.load_images_button)

        self.check_dataset_button = SecondaryButton(self.tr("Check Dataset"))
        self.check_dataset_button.clicked.connect(self._run_data_check)
        actions_layout.addWidget(self.check_dataset_button)

        actions_layout.addStretch()

        self.next_button = PrimaryButton(self.tr("Next"))
        self.next_button.clicked.connect(self.proceed_to_config)
        actions_layout.addWidget(self.next_button)
        parent_layout.addLayout(actions_layout)

    def init_data_tab(self):
        layout = QVBoxLayout(self.data_tab)

        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        self.init_task_configuration(scroll_layout)
        self.init_dataset_summary(scroll_layout)

        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)

        self.init_actions(layout)

    # Config Tab
    def browse_model_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Select Model File"),
            "",
            "Model Files (*.pt);;All Files (*)",
        )
        if file_path:
            self.config_widgets["model"].setText(file_path)

    def browse_project_dir(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, self.tr("Select Project Directory"), ""
        )
        if dir_path:
            self.config_widgets["project"].setText(dir_path)

    def browse_data_file(self):
        if self.selected_task_type == "Classify":
            dir_path = QFileDialog.getExistingDirectory(
                self, self.tr("Select Classification Dataset Directory"), ""
            )
            if dir_path:
                self.config_widgets["data"].setText(dir_path)
        else:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                self.tr("Select Data File"),
                "",
                "Text Files (*.yaml);;All Files (*)",
            )
            if file_path:
                is_valid, result = validate_data_file(file_path)
                if is_valid:
                    self.config_widgets["data"].setText(file_path)
                    self.names = result
                    logger.info(f"Data file loaded successfully: {file_path}")
                else:
                    QMessageBox.warning(
                        self, self.tr("Invalid Data File"), result
                    )
                    self.config_widgets["data"].clear()
                    self.names = []

    def browse_pose_config_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Select Pose Config File"),
            "",
            "YAML Files (*.yaml *.yml);;All Files (*)",
        )
        if file_path:
            self.config_widgets["pose_config"].setText(file_path)

    def setup_cuda_checkboxes(self, device_count):
        if not hasattr(self, "_cuda_layout") or not self._cuda_layout:
            if self.device_checkboxes.layout() is None:
                self._cuda_layout = QHBoxLayout(self.device_checkboxes)
            else:
                self._cuda_layout = self.device_checkboxes.layout()
            self._cuda_layout.setContentsMargins(0, 0, 0, 0)
            self._cuda_layout.setSpacing(5)
        else:
            while self._cuda_layout.count():
                child = self._cuda_layout.takeAt(0)
                if child.widget():
                    child.widget().setParent(None)

        for i in range(device_count):
            checkbox = CustomCheckBox(f"GPU {i}")
            checkbox.setMaximumHeight(20)
            checkbox.setChecked(True)  # Default check all GPUs
            self._cuda_layout.addWidget(checkbox)

    def _populate_device_combo(self):
        """Populate device QComboBox with Auto + CPU + detected GPUs.

        Each GPU item stores the full DeviceInfo as userData,
        including runtime_id and runtime_python for external CUDA envs.
        """
        import logging
        _log = logging.getLogger(__name__)
        from anylabeling.services.training_center.device_service import detect_local_devices, DeviceInfo
        combo = self.config_widgets["device"]
        current_val = combo.currentData() or "auto"

        combo.blockSignals(True)
        combo.clear()

        # Auto (always first) — no runtime binding
        combo.addItem("Auto", DeviceInfo(
            backend="auto", index=0, display_name="Auto",
            training_value="auto", available=True,
        ))
        _log.debug("Device combo: added Auto / auto")

        # Detected GPUs
        self._detected_gpus = []
        try:
            devices = detect_local_devices()
            _log.debug("detect_local_devices returned %d devices", len(devices))
            for d in devices:
                _log.debug("  Device: backend=%s idx=%d name=%s avail=%s runtime_py=%s",
                           d.backend, d.index, d.display_name, d.available,
                           d.runtime_python or "(none)")
                if d.backend == "cuda" and d.available:
                    self._detected_gpus.append(d)
                    combo.addItem(d.display_name, d)
                    _log.info("Device combo: added %s / %s runtime=%s",
                              d.display_name, d.training_value,
                              d.runtime_python or "(none)")
                elif d.backend == "cpu" and d.available:
                    combo.addItem("CPU", d)
                    _log.debug("Device combo: added CPU / cpu")
        except Exception as e:
            _log.warning("Device detection failed: %s", e)
            combo.addItem("CPU", DeviceInfo(
                backend="cpu", index=0, display_name="CPU",
                training_value="cpu", available=True,
            ))

        # Restore previous selection or default to Auto
        for i in range(combo.count()):
            if combo.itemData(i) and getattr(combo.itemData(i), 'training_value', None) == getattr(current_val, 'training_value', current_val):
                combo.setCurrentIndex(i)
                break
        else:
            combo.setCurrentIndex(0)

        combo.blockSignals(False)
        _log.info("Device combo: %d items, selected: training_value=%s runtime_py=%s",
                  combo.count(),
                  getattr(combo.currentData(), 'training_value', combo.currentData()) if combo.currentData() else None,
                  getattr(combo.currentData(), 'runtime_python', '') if combo.currentData() else '')

    def _on_refresh_devices(self):
        """Re-scan GPUs and refresh the device combo."""
        self._populate_device_combo()
        self._update_gpu_info_label()
        self._update_test_gpu_btn()
        self.append_training_log("Devices refreshed.")

    def _on_execution_mode_changed(self, idx: int):
        """Toggle between Local and Remote SSH execution modes."""
        mode = self.config_widgets["execution_mode"].currentData()
        is_remote = mode == "remote_ssh"
        self.remote_profile_widget.setVisible(is_remote)

    def _get_selected_training_device(self) -> dict:
        """Return the selected device's full training configuration.

        Reads from the device combo's userData (DeviceInfo) directly.
        No re-scanning, no guessing from display text.

        Returns dict with keys:
            requested_device, resolved_device, runtime_id,
            runtime_python, device_name, execution_mode
        """
        combo = self.config_widgets.get("device")
        if not combo:
            return {"requested_device": "auto", "resolved_device": "cpu",
                    "runtime_id": "", "runtime_python": "", "device_name": "",
                    "execution_mode": "local"}

        item_data = combo.currentData()
        if item_data is None:
            return {"requested_device": "auto", "resolved_device": "cpu",
                    "runtime_id": "", "runtime_python": "", "device_name": "",
                    "execution_mode": "local"}

        # New path: DeviceInfo dataclass in userData
        if hasattr(item_data, 'training_value'):
            d = item_data
            requested_device = d.training_value
            resolved_device = self._resolve_ultralytics_device(requested_device)
            return {
                "requested_device": requested_device,
                "resolved_device": resolved_device,
                "runtime_id": d.runtime_id,
                "runtime_python": d.runtime_python,
                "device_name": d.device_name or d.display_name,
                "execution_mode": d.execution_location,
            }

        # Legacy: bare string in userData
        tv = str(item_data)
        return {"requested_device": tv, "resolved_device": self._resolve_ultralytics_device(tv),
                "runtime_id": "", "runtime_python": "", "device_name": "",
                "execution_mode": "local"}

    def _resolve_ultralytics_device(self, device_value: str) -> str:
        """Convert internal device value to Ultralytics-compatible format."""
        from anylabeling.services.training_center.device_service import resolve_training_device
        return resolve_training_device(device_value)

    def _get_selected_runtime_python(self) -> str:
        """Get the runtime Python path for the currently selected device."""
        info = self._get_selected_training_device()
        return info.get("runtime_python", "") or ""

    def _on_test_gpu(self):
        """Run a quick CUDA tensor test — uses external runtime if registered."""
        from anylabeling.services.training_center.device_service import test_gpu_quick
        combo = self.config_widgets["device"]
        device_value = combo.currentData() or "auto"

        self.test_gpu_btn.setEnabled(False)
        self.test_gpu_btn.setText("Testing...")

        # Try external runtime first
        runtime_python = self._get_selected_runtime_python()
        if runtime_python and runtime_python != sys.executable:
            # Test via external runtime subprocess
            from anylabeling.services.training_center.preflight.guided_checks import _query_runtime_cuda
            info = _query_runtime_cuda(runtime_python)
            if info:
                self._show_gpu_test_result({
                    "status": "PASS" if info.get("cuda_available") else "FAILED",
                    "device_name": ", ".join(info.get("gpu_names", ["Unknown"])),
                    "total_memory_gb": 0,
                    "free_memory_gb": 0,
                    "cuda_version": info.get("torch_cuda_version", "N/A"),
                    "torch_version": info.get("torch_version", "N/A"),
                    "elapsed_ms": 0,
                    "error": None if info.get("cuda_available") else "CUDA not available in runtime",
                })
            else:
                self._show_gpu_test_result({
                    "status": "FAILED",
                    "device_name": "N/A",
                    "total_memory_gb": 0,
                    "free_memory_gb": 0,
                    "cuda_version": None,
                    "torch_version": "N/A",
                    "elapsed_ms": 0,
                    "error": f"Failed to query runtime: {runtime_python}",
                })
        else:
            # In-process test
            result = test_gpu_quick(device_value)
            self._show_gpu_test_result(result)

        self.test_gpu_btn.setEnabled(True)
        self.test_gpu_btn.setText("Test GPU")

    def _get_selected_runtime_python(self) -> str:
        """Get the runtime Python path for the currently selected device."""
        combo = self.config_widgets.get("device")
        if not combo:
            return ""
        device_value = combo.currentData() or ""
        if device_value in ("cpu", "auto", ""):
            return ""
        try:
            from anylabeling.services.training_center.environment_scanner import (
                get_registered_envs,
            )
            for reg in get_registered_envs():
                if str(reg.get("verification_status", "")).strip().lower() == "ready":
                    return reg.get("python_path", "")
        except Exception:
            pass
        return ""

    def _show_gpu_test_result(self, result: dict):
        """Display GPU test result in a message box."""
        if result["status"] == "PASS":
            msg = (
                f"GPU Test PASSED ✓\n\n"
                f"Device: {result['device_name']}\n"
                f"Total VRAM: {result['total_memory_gb']:.1f} GB\n"
                f"Free VRAM: {result['free_memory_gb']:.1f} GB\n"
                f"CUDA: {result['cuda_version']}\n"
                f"PyTorch: {result['torch_version']}\n"
                f"Test time: {result['elapsed_ms']:.1f} ms"
            )
            QMessageBox.information(self, "GPU Test", msg)
        else:
            msg = (
                f"GPU Test FAILED ✗\n\n"
                f"Error: {result.get('error', 'Unknown')}\n\n"
                f"PyTorch: {result['torch_version']}\n"
                f"CUDA: {result['cuda_version'] or 'N/A'}"
            )
            QMessageBox.warning(self, "GPU Test Failed", msg)

    def _update_gpu_info_label(self):
        """Update the GPU info label below the device selector.

        Checks both in-process torch and registered external runtimes
        to provide accurate GPU availability information.
        """
        if not hasattr(self, 'gpu_info_label'):
            return
        from anylabeling.services.training_center.device_service import get_device_diagnostics

        # Check if external CUDA runtime is ready
        has_external_runtime = False
        try:
            from anylabeling.services.training_center.environment_scanner import (
                get_registered_envs,
            )
            for reg in get_registered_envs():
                if str(reg.get("verification_status", "")).strip().lower() == "ready":
                    has_external_runtime = True
                    break
        except Exception:
            pass

        diag = get_device_diagnostics()

        if has_external_runtime:
            # External CUDA runtime is available — GPU training is ready
            gpu_info = ", ".join(
                f"{g['name']} · {g['total_memory_gb']:.1f} GB"
                for g in diag.get("gpus", [])
            ) if diag.get("gpus") else "GPU detected"
            self.gpu_info_label.setText(
                f"✓ GPU training ready (external runtime)"
                + (f" — {gpu_info}" if gpu_info else "")
            )
            self.gpu_info_label.setStyleSheet("color: green; font-size: 11px;")
        elif diag["pytorch_cpu_only"]:
            self.gpu_info_label.setText(
                "⚠ Installed PyTorch is CPU-only. Install a CUDA-enabled PyTorch build for NVIDIA GPU training."
            )
            self.gpu_info_label.setStyleSheet("color: #e67e22; font-size: 11px;")
        elif not diag["cuda_available"]:
            self.gpu_info_label.setText(
                "⚠ CUDA GPU not available (check NVIDIA driver and CUDA toolkit)."
            )
            self.gpu_info_label.setStyleSheet("color: #e67e22; font-size: 11px;")
        elif diag["gpu_count"] > 0:
            parts = []
            for g in diag["gpus"]:
                parts.append(f"{g['name']} · {g['total_memory_gb']:.1f} GB")
            self.gpu_info_label.setText("✓ " + " | ".join(parts))
            self.gpu_info_label.setStyleSheet("color: green; font-size: 11px;")
        else:
            self.gpu_info_label.setText("")
        self.gpu_info_label.setVisible(bool(self.gpu_info_label.text()))

    def _update_test_gpu_btn(self):
        """Enable/disable Test GPU button based on CUDA availability."""
        if not hasattr(self, 'test_gpu_btn'):
            return
        from anylabeling.services.training_center.device_service import get_device_diagnostics
        diag = get_device_diagnostics()
        has_gpu = diag["cuda_available"] and diag["gpu_count"] > 0
        self.test_gpu_btn.setEnabled(has_gpu)
        if not has_gpu and not diag["pytorch_cpu_only"]:
            self.test_gpu_btn.setEnabled(True)  # Still allow testing to get error details

    def on_device_changed(self, device_text):
        """Handle device selection change."""
        self.device_checkboxes.setVisible(False)
        self._update_gpu_info_label()

    # ── GPU Environment Status Card ──────────────────────────────────

    def _update_gpu_env_card(self):
        """Update the GPU environment status card based on current diagnosis."""
        if not hasattr(self, 'gpu_env_card'):
            return
        from anylabeling.services.training_center.device_service import (
            diagnose_environment, EnvironmentState,
        )
        from anylabeling.services.training_center.environment_scanner import (
            get_registered_envs,
        )
        diag = diagnose_environment()

        # Check external runtimes for GPU readiness
        has_external_gpu = False
        external_info = None
        try:
            for reg in get_registered_envs():
                if str(reg.get("verification_status", "")).strip().lower() == "ready":
                    has_external_gpu = True
                    external_info = reg
                    break
        except Exception:
            pass

        if diag.state == EnvironmentState.GPU_READY or has_external_gpu:
            # In-process or external GPU ready
            if has_external_gpu and external_info:
                gpu_name = ", ".join(external_info.get("gpu_names", ["Unknown"]))
                rt_python = external_info.get("python_path", "")
                rt_torch = external_info.get("torch_version", "?")
                rt_cuda = external_info.get("torch_cuda_version", "?")
                self.gpu_env_status_label.setText(
                    f"Detected GPU: {gpu_name}\n"
                    f"GUI PyTorch: {diag.torch_version}\n"
                    f"Training Runtime: {rt_python}\n"
                    f"Runtime PyTorch: {rt_torch} · CUDA {rt_cuda}\n"
                    f"Status: GPU training ready"
                )
            else:
                self.gpu_env_status_label.setText(
                    f"✓ {diag.gpu_name}\nPyTorch {diag.torch_version} · CUDA {diag.torch_cuda_version}"
                )
            self.gpu_env_status_label.setStyleSheet("color: green; font-size: 11px;")
            self.enable_gpu_btn.setVisible(False)
            self.continue_cpu_btn.setVisible(False)
        elif diag.state == EnvironmentState.GPU_PRESENT_CPU_TORCH:
            self.gpu_env_status_label.setText(
                f"Detected: {diag.gpu_name}\n"
                f"Current PyTorch: {diag.torch_version} (CPU-only)\n"
                "GPU training is not enabled."
            )
            self.gpu_env_status_label.setStyleSheet("color: #e67e22; font-size: 11px;")
            self.enable_gpu_btn.setVisible(True)
            self.continue_cpu_btn.setVisible(True)
        elif diag.state == EnvironmentState.NO_NVIDIA_GPU:
            self.gpu_env_status_label.setText("No NVIDIA GPU detected.")
            self.gpu_env_status_label.setStyleSheet("color: gray; font-size: 11px;")
            self.enable_gpu_btn.setVisible(False)
            self.continue_cpu_btn.setVisible(False)
        else:
            self.gpu_env_status_label.setText(diag.diagnostic_message or "GPU status unknown.")
            self.gpu_env_status_label.setStyleSheet("color: #e67e22; font-size: 11px;")
            self.enable_gpu_btn.setVisible(False)
            self.continue_cpu_btn.setVisible(False)

    def _on_enable_gpu_clicked(self):
        """Show GPU install confirmation, then start background installation."""
        from anylabeling.services.training_center.device_service import diagnose_environment
        from anylabeling.services.training_center.runtime_installer import NVIDIA_RUNTIME_DIR
        diag = diagnose_environment()

        msg = (
            f"Install GPU Training Runtime\n\n"
            f"GPU: {diag.gpu_name}\n"
            f"Current PyTorch: {diag.torch_version} (CPU-only)\n\n"
            f"Will install CUDA-enabled PyTorch into:\n{NVIDIA_RUNTIME_DIR}\n\n"
            f"Packages: torch, torchvision, torchaudio, ultralytics\n"
            f"Source: PyPI (pypi.org)\n\n"
            f"Estimated disk: ~3-5 GB. May take 5-15 min.\n\n"
            f"Continue?"
        )
        reply = QMessageBox.question(
            self, "Enable GPU Training", msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._start_gpu_install()

    def _on_continue_cpu(self):
        self.gpu_env_card.setVisible(False)
        self.append_training_log("Continuing with CPU training.")

    def _on_view_diagnostics(self):
        from anylabeling.services.training_center.device_service import (
            diagnose_environment,
        )
        diag = diagnose_environment()
        msg = (
            f"=== GPU Diagnostics ===\n\n"
            f"State: {diag.state.value}\n"
            f"GPU: {diag.gpu_name or 'N/A'}\nCount: {diag.gpu_count}\n"
            f"Driver: {diag.driver_version or 'N/A'}\n"
            f"Driver CUDA: {diag.driver_cuda_version or 'N/A'}\n"
            f"PyTorch: {diag.torch_version or 'N/A'}\n"
            f"Torch CUDA: {diag.torch_cuda_version or 'N/A'}\n"
            f"CUDA available: {diag.cuda_available}\n\n"
            f"Action: {diag.recommended_action}"
        )
        QMessageBox.information(self, "GPU Diagnostics", msg)

    def _on_rescan_environments(self):
        """Background scan for CUDA training environments."""
        from anylabeling.services.training_center.environment_scanner import (
            EnvironmentScannerWorker, find_best_env, rank_environments,
            register_external_env, EnvStatus,
        )

        self.rescan_envs_btn.setEnabled(False)
        self.rescan_envs_btn.setText("Scanning...")
        self.gpu_install_progress.setVisible(True)
        self.gpu_install_progress.setText("Scanning environments...")
        self.gpu_install_progress.setStyleSheet("color: #2196F3; font-size: 11px;")
        self.append_training_log("Environment scan started...")

        self._scan_thread = QThread(self)
        self._scan_worker = EnvironmentScannerWorker()
        self._scan_worker.moveToThread(self._scan_thread)

        self._scan_worker.progress_text.connect(self._on_scan_progress)
        self._scan_worker.finished.connect(self._on_scan_finished)
        self._scan_thread.started.connect(self._scan_worker.run)
        self._scan_thread.finished.connect(self._scan_thread.deleteLater)
        self._scan_thread.start()

    def _on_scan_progress(self, text: str):
        self.gpu_install_progress.setText(text[:200])

    def _on_scan_finished(self, results: list):
        from anylabeling.services.training_center.environment_scanner import (
            find_best_env, rank_environments, register_external_env, EnvStatus,
        )

        self.rescan_envs_btn.setEnabled(True)
        self.rescan_envs_btn.setText("Rescan Environments")
        self.gpu_install_progress.setVisible(False)

        # Log detailed results
        ready_count = sum(1 for e in results if e.is_cuda_ready)
        missing_pkg_count = sum(1 for e in results if e.needs_packages)
        cpu_count = sum(1 for e in results if e.status == EnvStatus.CPU_ONLY)
        self.append_training_log(
            f"Environment scan: {len(results)} total, "
            f"{ready_count} READY, {missing_pkg_count} CUDA+missing pkgs, {cpu_count} CPU-only"
        )
        for e in rank_environments(results):
            self.append_training_log(
                f"  [{e.status}] {e.env_name}: torch={e.torch_version} cuda={e.torch_cuda_version} "
                f"utl={e.ultralytics_version or 'missing'} gpu={e.gpu_names}"
            )

        best = find_best_env(results)

        # ── No CUDA at all ──
        if best is None and missing_pkg_count == 0 and ready_count == 0:
            QMessageBox.information(
                self, "No CUDA Environment",
                "No CUDA-capable training environment was found.\n\n"
                "You can:\n"
                "  • Install a new GPU Runtime\n"
                "  • Continue with CPU training"
            )
            return

        # ── Safety net: if find_best_env missed something, take first MISSING_PACKAGES ──
        if best is None and missing_pkg_count > 0:
            for e in results:
                if e.needs_packages:
                    best = e
                    break

        # ── Safety net: if still None, take first READY ──
        if best is None and ready_count > 0:
            for e in results:
                if e.is_cuda_ready:
                    best = e
                    break

        if best is None:
            # Should not happen, but handle gracefully
            self.append_training_log("ERROR: find_best_env returned None despite available envs")
            return

        if best.is_cuda_ready:
            register_external_env(best)
            self._populate_device_combo()
            self._update_gpu_env_card()
            QMessageBox.information(
                self, "CUDA Environment Found",
                f"Ready CUDA environment registered:\n\n"
                f"  {best.env_name}\n"
                f"  Python {best.python_version}\n"
                f"  PyTorch {best.torch_version} · CUDA {best.torch_cuda_version}\n"
                f"  GPU: {', '.join(best.gpu_names)}\n"
                f"  Ultralytics: {best.ultralytics_version}\n\n"
                f"Device list has been updated."
            )
        elif best.needs_packages:
            # ── CUDA ready but missing packages ──
            gpu_info = ", ".join(
                f"{name} · {mem} GB" for name, mem in zip(best.gpu_names, best.gpu_memory_gb)
            ) if best.gpu_memory_gb else ", ".join(best.gpu_names)

            msg = (
                f"CUDA Environment Found\n\n"
                f"Python:\n{best.python_path}\n\n"
                f"PyTorch:\n{best.torch_version} · CUDA {best.torch_cuda_version}\n\n"
                f"GPU:\n{gpu_info}\n\n"
                f"Missing:\nUltralytics (required for training)\n\n"
                f"Install missing packages into this environment?"
            )
            reply = QMessageBox.question(
                self, "Install Missing Packages", msg,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._install_missing_packages(best)
        else:
            QMessageBox.information(
                self, "No Ready Environment",
                f"Best available: {best.status}\n\n"
                f"  {best.env_name}\n"
                f"  Python: {best.python_version}\n"
                f"  PyTorch: {best.torch_version or 'N/A'}"
            )

    def _install_missing_packages(self, env_info):
        """Install missing packages (ultralytics) via QProcess — async, non-blocking."""
        from PyQt6.QtCore import QProcess

        python_exe = env_info.python_path

        # ── Guard: prevent double-install ──
        if hasattr(self, '_package_install_process') and self._package_install_process is not None:
            QMessageBox.information(self, "Installation in Progress",
                                    "Package installation is already running.")
            return

        # ── Create non-modal progress dialog ──
        self._package_install_dialog = QDialog(self)
        self._package_install_dialog.setWindowTitle("Installing Missing Packages")
        self._package_install_dialog.setMinimumSize(550, 350)
        dlg_layout = QVBoxLayout(self._package_install_dialog)

        info_label = QLabel(
            f"Environment: {env_info.env_name}\n"
            f"Python: {python_exe}\n"
            f"Package: ultralytics==8.4.96\n\n"
            f"Stage: Installing"
        )
        info_label.setWordWrap(True)
        dlg_layout.addWidget(info_label)
        self._package_install_info_label = info_label

        self._package_install_log = QTextEdit()
        self._package_install_log.setReadOnly(True)
        self._package_install_log.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        dlg_layout.addWidget(self._package_install_log)

        btn_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self._on_cancel_package_install)
        btn_layout.addWidget(cancel_btn)
        btn_layout.addStretch()
        dlg_layout.addLayout(btn_layout)

        self._package_install_dialog.finished.connect(self._on_package_install_dialog_closed)
        self._package_install_dialog.show()  # non-modal!

        # ── Disable rescan/install buttons during install ──
        self.rescan_envs_btn.setEnabled(False)
        self.enable_gpu_btn.setEnabled(False)
        self.gpu_install_progress.setVisible(True)
        self.gpu_install_progress.setText("Installing ultralytics...")
        self.gpu_install_progress.setStyleSheet("color: #2196F3; font-size: 11px;")

        # ── Create QProcess ──
        self._package_install_process = QProcess(self)
        self._package_install_process.setProgram(python_exe)
        self._package_install_process.setArguments(["-m", "pip", "install", "ultralytics==8.4.96"])
        self._package_install_process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)

        self._package_install_process.readyReadStandardOutput.connect(
            self._on_package_install_output
        )
        self._package_install_process.finished.connect(
            self._on_package_install_finished
        )
        self._package_install_process.errorOccurred.connect(
            self._on_package_install_error
        )

        self._package_install_log_buffer = []
        self._package_install_env_info = env_info
        self._package_install_process.start()

    def _on_package_install_output(self):
        """Read pip stdout in real-time."""
        if not hasattr(self, '_package_install_process') or self._package_install_process is None:
            return
        data = self._package_install_process.readAllStandardOutput()
        try:
            text = bytes(data).decode(os.getfilesystemencoding(), errors="replace")
        except Exception:
            text = bytes(data).decode("utf-8", errors="replace")
        self._package_install_log_buffer.append(text)
        if hasattr(self, '_package_install_log'):
            self._package_install_log.append(text.strip())
            # Auto-scroll
            sb = self._package_install_log.verticalScrollBar()
            sb.setValue(sb.maximum())

    def _on_package_install_finished(self, exit_code: int, exit_status):
        """Handle pip completion — start async verification."""
        from PyQt6.QtCore import QProcess
        from anylabeling.services.training_center.environment_scanner import (
            register_external_env,
        )

        proc = self._package_install_process
        self._package_install_process = None
        env_info = self._package_install_env_info

        if exit_status != QProcess.ExitStatus.NormalExit or exit_code != 0:
            # ── Pip failed ──
            log_text = "\n".join(self._package_install_log_buffer[-20:])
            if hasattr(self, '_package_install_info_label'):
                self._package_install_info_label.setText(
                    f"Stage: FAILED (exit code {exit_code})"
                )
            self.gpu_install_progress.setText(f"✗ pip failed (exit {exit_code})")
            self.gpu_install_progress.setStyleSheet("color: red; font-size: 11px;")
            QMessageBox.warning(
                self._package_install_dialog or self,
                "Install Failed",
                f"pip exited with code {exit_code}.\n\nLast output:\n{log_text[:500]}"
            )
            self._finish_package_install(success=False)
            return

        # ── Pip succeeded → verify ──
        if hasattr(self, '_package_install_info_label'):
            self._package_install_info_label.setText("Stage: VERIFYING")
        self.gpu_install_progress.setText("Verifying CUDA runtime...")

        # Run diagnosis asynchronously via QProcess
        verify_script = """
import json, sys, traceback
result = {"torch_version":"","cuda_version":"","cuda_available":False,"gpu_name":"","tensor_test":"FAIL","ultralytics_version":"","ultralytics_ok":False,"error":""}
try:
    import torch
    result["torch_version"] = torch.__version__
    result["cuda_version"] = getattr(torch.version,"cuda","") or ""
    result["cuda_available"] = torch.cuda.is_available()
    if result["cuda_available"]:
        result["gpu_name"] = torch.cuda.get_device_name(0)
        t = torch.randn(100,100,device="cuda:0")
        t = t @ t.T
        torch.cuda.synchronize()
        del t
        torch.cuda.empty_cache()
        result["tensor_test"] = "PASS"
    import ultralytics
    result["ultralytics_ok"] = True
    result["ultralytics_version"] = ultralytics.__version__
except Exception as e:
    result["error"] = str(e)
print(json.dumps(result, ensure_ascii=False))
"""
        self._package_verify_process = QProcess(self)
        self._package_verify_process.setProgram(env_info.python_path)
        self._package_verify_process.setArguments(["-c", verify_script])
        self._package_verify_process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._package_verify_process.finished.connect(self._on_package_verify_finished)
        self._package_verify_process.errorOccurred.connect(
            lambda err: self._on_package_verify_error(str(err))
        )
        self._package_verify_process.start()

    def _on_package_verify_finished(self, exit_code: int, exit_status):
        """Handle verification result."""
        import json
        from anylabeling.services.training_center.environment_scanner import (
            register_external_env,
        )
        proc = self._package_verify_process
        self._package_verify_process = None

        if exit_status != QProcess.ExitStatus.NormalExit or exit_code != 0:
            self.gpu_install_progress.setText("✗ Verification script failed")
            self.gpu_install_progress.setStyleSheet("color: red; font-size: 11px;")
            self._finish_package_install(success=False)
            return

        output = bytes(proc.readAllStandardOutput()).decode("utf-8", errors="replace").strip()
        try:
            data = json.loads(output)
        except json.JSONDecodeError:
            self.gpu_install_progress.setText("✗ Invalid verification output")
            self.gpu_install_progress.setStyleSheet("color: red; font-size: 11px;")
            self._finish_package_install(success=False)
            return

        tensor_ok = data.get("tensor_test") == "PASS"
        utl_ok = data.get("ultralytics_ok", False)

        if not tensor_ok or not utl_ok:
            reason = []
            if not tensor_ok:
                reason.append("CUDA tensor test: FAILED")
            if not utl_ok:
                reason.append("Ultralytics import: FAILED")
            self.gpu_install_progress.setText(f"✗ Verification: {', '.join(reason)}")
            self.gpu_install_progress.setStyleSheet("color: red; font-size: 11px;")
            QMessageBox.warning(
                self._package_install_dialog or self,
                "Verification Failed",
                f"Installation succeeded but environment is not ready:\n\n"
                f"  {', '.join(reason)}\n\n"
                f"Error: {data.get('error', 'N/A')}"
            )
            self._finish_package_install(success=False)
            return

        # ── Success! Register and refresh ──
        env_info = self._package_install_env_info
        env_info.ultralytics_installed = True
        env_info.ultralytics_version = data.get("ultralytics_version", "")
        env_info.status = "ready"
        register_external_env(env_info)

        self._populate_device_combo()
        self._update_gpu_env_card()
        self.gpu_install_progress.setText("✓ Environment is READY!")
        self.gpu_install_progress.setStyleSheet("color: green; font-size: 11px; font-weight: bold;")
        if hasattr(self, '_package_install_info_label'):
            self._package_install_info_label.setText("Stage: COMPLETED ✓")
        QMessageBox.information(
            self._package_install_dialog or self,
            "Success",
            f"Ultralytics {data['ultralytics_version']} installed.\n"
            f"CUDA test: PASS\n"
            f"Device list has been updated.\n\n"
            f"GPU: {data.get('gpu_name', 'N/A')}"
        )
        self._finish_package_install(success=True)

    def _on_package_verify_error(self, error_str: str):
        self.gpu_install_progress.setText(f"✗ Verify error: {error_str}")
        self.gpu_install_progress.setStyleSheet("color: red; font-size: 11px;")
        self._finish_package_install(success=False)

    def _on_package_install_error(self, error):
        """QProcess failed to start."""
        self.gpu_install_progress.setText(f"✗ Failed to start: {error}")
        self.gpu_install_progress.setStyleSheet("color: red; font-size: 11px;")
        self._finish_package_install(success=False)

    def _on_cancel_package_install(self):
        if hasattr(self, '_package_install_process') and self._package_install_process:
            self._package_install_process.terminate()
            # If still running after 3s, kill
            QTimer.singleShot(3000, self._force_kill_install_process)
        self.gpu_install_progress.setText("Cancelled")
        self._finish_package_install(success=False)

    def _force_kill_install_process(self):
        if hasattr(self, '_package_install_process') and self._package_install_process:
            try:
                self._package_install_process.kill()
            except Exception:
                pass

    def _on_package_install_dialog_closed(self):
        """Cleanup when dialog is closed."""
        self._package_install_dialog = None
        self._finish_package_install(success=False)

    def _finish_package_install(self, success: bool):
        """Cleanup after package install (success or failure)."""
        self._package_install_process = None
        self._package_verify_process = None
        self._package_install_env_info = None
        self._package_install_log_buffer = []

        self.rescan_envs_btn.setEnabled(True)
        self.enable_gpu_btn.setEnabled(True)

        if not success and hasattr(self, 'gpu_install_progress'):
            self.gpu_install_progress.setVisible(False)

    def _start_gpu_install(self):
        from anylabeling.services.training_center.runtime_installer import (
            RuntimeInstallerWorker, NVIDIA_RUNTIME_DIR,
        )
        self.enable_gpu_btn.setEnabled(False)
        self.gpu_install_progress.setVisible(True)
        self.gpu_install_progress.setText("Starting installation...")
        self.gpu_install_progress.setStyleSheet("color: #2196F3; font-size: 11px;")
        self.append_training_log("GPU runtime installation started...")

        self._install_thread = QThread(self)
        self._install_worker = RuntimeInstallerWorker(str(NVIDIA_RUNTIME_DIR))
        self._install_worker.moveToThread(self._install_thread)

        self._install_worker.progress_text.connect(self._on_install_progress)
        self._install_worker.finished.connect(self._on_install_finished)
        self._install_thread.started.connect(self._install_worker.run)
        self._install_thread.finished.connect(self._install_thread.deleteLater)
        self._install_thread.start()

    def _on_install_progress(self, text: str):
        self.gpu_install_progress.setText(text[:200])
        self.append_training_log(f"[GPU Install] {text}")

    def _on_install_finished(self, success: bool, message: str):
        if success:
            self.gpu_install_progress.setText(
                "✓ GPU runtime ready!"
            )
            self.gpu_install_progress.setStyleSheet("color: green; font-size: 11px; font-weight: bold;")
            QMessageBox.information(
                self, "GPU Runtime Ready",
                f"GPU runtime installed and verified!\n\n"
                f"Device list has been updated.\n\n{message}"
            )
        else:
            self.gpu_install_progress.setText(f"✗ Failed.\n{message[:200]}")
            self.gpu_install_progress.setStyleSheet("color: red; font-size: 11px;")
            QMessageBox.warning(
                self, "Installation Failed",
                f"GPU runtime installation failed.\n\n{message}\n\nCPU training remains available."
            )
        self.enable_gpu_btn.setEnabled(True)
        self._update_gpu_env_card()
        self._populate_device_combo()

    def init_basic_settings(self, parent_layout):
        group = QGroupBox(self.tr("Basic Settings"))
        layout = QFormLayout(group)
        layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )
        layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)

        project_layout = QHBoxLayout()
        self.config_widgets["project"] = CustomLineEdit()
        selected_task_type = (
            self.selected_task_type.lower()
            if self.selected_task_type
            else "detect"
        )
        text_project = os.path.join(
            get_default_project_dir(), selected_task_type
        )
        self.config_widgets["project"].setText(text_project)
        project_browse_btn = SecondaryButton("Browse")
        project_browse_btn.clicked.connect(self.browse_project_dir)
        project_layout.addWidget(self.config_widgets["project"])
        project_layout.addWidget(project_browse_btn)
        layout.addRow("Project:", project_layout)

        self.config_widgets["name"] = CustomLineEdit()
        self.config_widgets["name"].setText("exp")
        layout.addRow("Name:", self.config_widgets["name"])

        model_layout = QHBoxLayout()
        self.config_widgets["model"] = CustomLineEdit()
        model_browse_btn = SecondaryButton("Browse")
        model_browse_btn.clicked.connect(self.browse_model_file)
        model_layout.addWidget(self.config_widgets["model"])
        model_layout.addWidget(model_browse_btn)
        layout.addRow("Model:", model_layout)

        data_layout = QHBoxLayout()
        self.config_widgets["data"] = CustomLineEdit()
        data_browse_btn = SecondaryButton("Browse")
        data_browse_btn.clicked.connect(self.browse_data_file)
        data_layout.addWidget(self.config_widgets["data"])
        data_layout.addWidget(data_browse_btn)
        layout.addRow("Data:", data_layout)

        pose_config_layout = QHBoxLayout()
        self.config_widgets["pose_config"] = CustomLineEdit()
        pose_config_browse_btn = SecondaryButton("Browse")
        pose_config_browse_btn.clicked.connect(self.browse_pose_config_file)
        pose_config_layout.addWidget(self.config_widgets["pose_config"])
        pose_config_layout.addWidget(pose_config_browse_btn)

        self.pose_config_label = QLabel("Pose Config:")
        layout.addRow(self.pose_config_label, pose_config_layout)
        self.pose_config_layout = pose_config_layout

        self.pose_config_label.setVisible(False)
        self.config_widgets["pose_config"].setVisible(False)
        pose_config_browse_btn.setVisible(False)

        # ── Device (Auto / CPU / GPU detection) ──
        device_layout = QHBoxLayout()
        self.config_widgets["device"] = CustomComboBox()
        self._populate_device_combo()
        self.device_checkboxes = QWidget()
        self.device_checkboxes.setVisible(False)
        self.config_widgets["device"].currentTextChanged.connect(
            self.on_device_changed
        )
        device_layout.addWidget(self.config_widgets["device"])
        # Refresh Devices button
        self.refresh_devices_btn = QPushButton("↻")
        self.refresh_devices_btn.setFixedWidth(32)
        self.refresh_devices_btn.setToolTip("Refresh Devices")
        self.refresh_devices_btn.clicked.connect(self._on_refresh_devices)
        device_layout.addWidget(self.refresh_devices_btn)
        # Test GPU button
        self.test_gpu_btn = QPushButton("Test GPU")
        self.test_gpu_btn.setToolTip("Run quick CUDA tensor test")
        self.test_gpu_btn.clicked.connect(self._on_test_gpu)
        self._update_test_gpu_btn()
        device_layout.addWidget(self.test_gpu_btn)
        device_layout.addWidget(self.device_checkboxes)
        layout.addRow("Device:", device_layout)
        self.on_device_changed(self.config_widgets["device"].currentText())

        # GPU info label
        self.gpu_info_label = QLabel()
        self.gpu_info_label.setStyleSheet("color: gray; font-size: 11px;")
        self.gpu_info_label.setWordWrap(True)
        layout.addRow("", self.gpu_info_label)
        self._update_gpu_info_label()

        # ── Execution mode (Local / Remote SSH) ──
        exec_layout = QHBoxLayout()
        self.config_widgets["execution_mode"] = QComboBox()
        self.config_widgets["execution_mode"].addItem("Local", "local")
        self.config_widgets["execution_mode"].addItem("Remote SSH", "remote_ssh")
        self.config_widgets["execution_mode"].currentIndexChanged.connect(
            self._on_execution_mode_changed
        )
        exec_layout.addWidget(QLabel("Execution:"))
        exec_layout.addWidget(self.config_widgets["execution_mode"], 1)
        layout.addRow("Execution:", exec_layout)

        # Remote profile widget (hidden for local mode)
        from anylabeling.views.training.remote.profile_widget import RemoteProfileWidget
        self.remote_profile_widget = RemoteProfileWidget()
        self.remote_profile_widget.setVisible(False)
        layout.addRow(self.remote_profile_widget)

        dataset_layout = QHBoxLayout()
        self.config_widgets["dataset_ratio"] = CustomSlider(
            Qt.Orientation.Horizontal
        )
        self.config_widgets["dataset_ratio"].setRange(5, 95)
        self.config_widgets["dataset_ratio"].setValue(80)
        self.dataset_ratio_label = QLabel("0.8")
        self.config_widgets["dataset_ratio"].valueChanged.connect(
            lambda v: self.dataset_ratio_label.setText(str(v / 100.0))
        )
        dataset_layout.addWidget(self.config_widgets["dataset_ratio"])
        dataset_layout.addWidget(self.dataset_ratio_label)
        layout.addRow("Dataset Ratio:", dataset_layout)

        # ── GPU Environment Status Card ──
        self.gpu_env_card = QGroupBox("GPU Environment")
        gpu_env_layout = QVBoxLayout(self.gpu_env_card)
        self.gpu_env_status_label = QLabel()
        self.gpu_env_status_label.setWordWrap(True)
        gpu_env_layout.addWidget(self.gpu_env_status_label)

        gpu_env_btns = QHBoxLayout()
        self.enable_gpu_btn = QPushButton("Enable GPU Training")
        self.enable_gpu_btn.clicked.connect(self._on_enable_gpu_clicked)
        self.enable_gpu_btn.setVisible(False)
        gpu_env_btns.addWidget(self.enable_gpu_btn)

        self.continue_cpu_btn = QPushButton("Continue with CPU")
        self.continue_cpu_btn.clicked.connect(self._on_continue_cpu)
        self.continue_cpu_btn.setVisible(False)
        gpu_env_btns.addWidget(self.continue_cpu_btn)

        self.view_diag_btn = QPushButton("View Diagnostics")
        self.view_diag_btn.clicked.connect(self._on_view_diagnostics)
        gpu_env_btns.addWidget(self.view_diag_btn)

        self.rescan_envs_btn = QPushButton("Rescan Environments")
        self.rescan_envs_btn.clicked.connect(self._on_rescan_environments)
        gpu_env_btns.addWidget(self.rescan_envs_btn)

        gpu_env_btns.addStretch()
        gpu_env_layout.addLayout(gpu_env_btns)

        self.gpu_install_progress = QLabel()
        self.gpu_install_progress.setVisible(False)
        self.gpu_install_progress.setWordWrap(True)
        gpu_env_layout.addWidget(self.gpu_install_progress)

        layout.addRow(self.gpu_env_card)
        self._update_gpu_env_card()

        parent_layout.addWidget(group)

    def toggle_advanced_settings(self):
        """Toggle the visibility of advanced settings"""
        if self.advanced_content_widget.isVisible():
            self.advanced_content_widget.setVisible(False)
            self.advanced_toggle_btn.setIcon(
                QIcon(new_icon("caret-down", "svg"))
            )
        else:
            self.advanced_content_widget.setVisible(True)
            self.advanced_toggle_btn.setIcon(
                QIcon(new_icon("caret-up", "svg"))
            )

    def init_train_settings(self, parent_layout):
        group = QGroupBox(self.tr("Train Settings"))
        layout = QVBoxLayout(group)

        # Basic settings
        basic_group = QGroupBox(self.tr("Basic"))
        basic_layout = QHBoxLayout(basic_group)
        basic_layout.addWidget(QLabel("Epochs:"))
        self.config_widgets["epochs"] = CustomSpinBox()
        self.config_widgets["epochs"].setRange(1, 10000)
        self.config_widgets["epochs"].setValue(
            DEFAULT_TRAINING_CONFIG["epochs"]
        )
        basic_layout.addWidget(self.config_widgets["epochs"])

        basic_layout.addWidget(QLabel("Batch:"))
        self.config_widgets["batch"] = CustomSpinBox()
        self.config_widgets["batch"].setRange(-1, 8192)
        self.config_widgets["batch"].setValue(DEFAULT_TRAINING_CONFIG["batch"])
        basic_layout.addWidget(self.config_widgets["batch"])

        basic_layout.addWidget(QLabel("Image Size:"))
        self.config_widgets["imgsz"] = CustomSpinBox()
        self.config_widgets["imgsz"].setRange(32, 8192)
        self.config_widgets["imgsz"].setValue(DEFAULT_TRAINING_CONFIG["imgsz"])
        basic_layout.addWidget(self.config_widgets["imgsz"])

        basic_layout.addWidget(QLabel("Workers:"))
        self.config_widgets["workers"] = CustomSpinBox()
        self.config_widgets["workers"].setRange(0, NUM_WORKERS)
        self.config_widgets["workers"].setValue(
            DEFAULT_TRAINING_CONFIG["workers"]
        )
        basic_layout.addWidget(self.config_widgets["workers"])

        basic_layout.addWidget(QLabel("Classes:"))
        self.config_widgets["classes"] = CustomLineEdit()
        self.config_widgets["classes"].setText(
            DEFAULT_TRAINING_CONFIG["classes"]
        )
        self.config_widgets["classes"].setPlaceholderText(
            self.tr("Class indices (e.g., 0,1,2) or leave empty for all")
        )
        basic_layout.addWidget(self.config_widgets["classes"])

        self.config_widgets["single_cls"] = CustomCheckBox("Single Class")
        self.config_widgets["single_cls"].setChecked(
            DEFAULT_TRAINING_CONFIG["single_cls"]
        )
        basic_layout.addWidget(self.config_widgets["single_cls"])

        basic_layout.addStretch()
        layout.addWidget(basic_group)

        # Advanced settings
        advanced_container = QWidget()
        advanced_container_layout = QVBoxLayout(advanced_container)
        advanced_container_layout.setContentsMargins(0, 0, 0, 0)

        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(5)

        advanced_label = QLabel(self.tr("Advanced Settings"))
        advanced_label.setStyleSheet("font-weight: bold;")
        header_layout.addWidget(advanced_label)

        # Collapse/Expand button
        self.advanced_toggle_btn = QPushButton()
        self.advanced_toggle_btn.setFixedSize(*ICON_SIZE_NORMAL)
        self.advanced_toggle_btn.setStyleSheet(get_advanced_toggle_btn_style())
        self.advanced_toggle_btn.setIcon(QIcon(new_icon("caret-down", "svg")))
        self.advanced_toggle_btn.clicked.connect(self.toggle_advanced_settings)
        header_layout.addWidget(self.advanced_toggle_btn)
        header_layout.addStretch()
        advanced_container_layout.addWidget(header_widget)

        self.advanced_content_widget = QWidget()
        self.advanced_content_widget.setVisible(False)
        advanced_layout = QVBoxLayout(self.advanced_content_widget)

        # 1. Training Strategy
        strategy_group = QGroupBox("Training Strategy")
        strat_layout = QHBoxLayout(strategy_group)
        strat_layout.addWidget(QLabel("Time (h):"))
        self.config_widgets["time"] = CustomDoubleSpinBox()
        self.config_widgets["time"].setValue(DEFAULT_TRAINING_CONFIG["time"])
        self.config_widgets["time"].setSpecialValueText("None")
        strat_layout.addWidget(self.config_widgets["time"])

        strat_layout.addWidget(QLabel("Patience:"))
        self.config_widgets["patience"] = CustomSpinBox()
        self.config_widgets["patience"].setRange(1, 10000)
        self.config_widgets["patience"].setValue(
            DEFAULT_TRAINING_CONFIG["patience"]
        )
        strat_layout.addWidget(self.config_widgets["patience"])

        strat_layout.addWidget(QLabel("Close Mosaic:"))
        self.config_widgets["close_mosaic"] = CustomSpinBox()
        self.config_widgets["close_mosaic"].setRange(0, 1000)
        self.config_widgets["close_mosaic"].setValue(
            DEFAULT_TRAINING_CONFIG["close_mosaic"]
        )
        strat_layout.addWidget(self.config_widgets["close_mosaic"])

        strat_layout.addWidget(QLabel("Optimizer:"))
        self.config_widgets["optimizer"] = CustomComboBox()
        self.config_widgets["optimizer"].addItems(OPTIMIZER_OPTIONS)
        strat_layout.addWidget(self.config_widgets["optimizer"])

        self.config_widgets["cos_lr"] = CustomCheckBox("Cosine LR")
        self.config_widgets["cos_lr"].setChecked(
            DEFAULT_TRAINING_CONFIG["cos_lr"]
        )
        strat_layout.addWidget(self.config_widgets["cos_lr"])
        self.config_widgets["amp"] = CustomCheckBox("AMP")
        self.config_widgets["amp"].setChecked(DEFAULT_TRAINING_CONFIG["amp"])
        strat_layout.addWidget(self.config_widgets["amp"])
        self.config_widgets["multi_scale"] = CustomCheckBox("Multi Scale")
        self.config_widgets["multi_scale"].setChecked(
            DEFAULT_TRAINING_CONFIG["multi_scale"]
        )
        strat_layout.addWidget(self.config_widgets["multi_scale"])
        strat_layout.addStretch()
        advanced_layout.addWidget(strategy_group)

        # 2. Learning Rate
        lr_group = QGroupBox("Learning Rate")
        lr_layout = QHBoxLayout(lr_group)
        lr_layout.addWidget(QLabel("LR0:"))
        self.config_widgets["lr0"] = CustomDoubleSpinBox()
        self.config_widgets["lr0"].setDecimals(6)
        self.config_widgets["lr0"].setValue(DEFAULT_TRAINING_CONFIG["lr0"])
        lr_layout.addWidget(self.config_widgets["lr0"])

        lr_layout.addWidget(QLabel("LRF:"))
        self.config_widgets["lrf"] = CustomDoubleSpinBox()
        self.config_widgets["lrf"].setDecimals(6)
        self.config_widgets["lrf"].setValue(DEFAULT_TRAINING_CONFIG["lrf"])
        lr_layout.addWidget(self.config_widgets["lrf"])

        lr_layout.addWidget(QLabel("Momentum:"))
        self.config_widgets["momentum"] = CustomDoubleSpinBox()
        self.config_widgets["momentum"].setDecimals(3)
        self.config_widgets["momentum"].setValue(
            DEFAULT_TRAINING_CONFIG["momentum"]
        )
        lr_layout.addWidget(self.config_widgets["momentum"])

        lr_layout.addWidget(QLabel("Weight Decay:"))
        self.config_widgets["weight_decay"] = CustomDoubleSpinBox()
        self.config_widgets["weight_decay"].setDecimals(6)
        self.config_widgets["weight_decay"].setValue(
            DEFAULT_TRAINING_CONFIG["weight_decay"]
        )
        lr_layout.addWidget(self.config_widgets["weight_decay"])
        lr_layout.addStretch()
        advanced_layout.addWidget(lr_group)

        # 3. Warmup Parameters
        warmup_group = QGroupBox("Warmup Parameters")
        warmup_layout = QHBoxLayout(warmup_group)
        warmup_layout.addWidget(QLabel("Warmup Epochs:"))
        self.config_widgets["warmup_epochs"] = CustomDoubleSpinBox()
        self.config_widgets["warmup_epochs"].setDecimals(1)
        self.config_widgets["warmup_epochs"].setValue(
            DEFAULT_TRAINING_CONFIG["warmup_epochs"]
        )
        warmup_layout.addWidget(self.config_widgets["warmup_epochs"])

        warmup_layout.addWidget(QLabel("Warmup Momentum:"))
        self.config_widgets["warmup_momentum"] = CustomDoubleSpinBox()
        self.config_widgets["warmup_momentum"].setDecimals(3)
        self.config_widgets["warmup_momentum"].setValue(
            DEFAULT_TRAINING_CONFIG["warmup_momentum"]
        )
        warmup_layout.addWidget(self.config_widgets["warmup_momentum"])

        warmup_layout.addWidget(QLabel("Warmup Bias LR:"))
        self.config_widgets["warmup_bias_lr"] = CustomDoubleSpinBox()
        self.config_widgets["warmup_bias_lr"].setDecimals(3)
        self.config_widgets["warmup_bias_lr"].setValue(
            DEFAULT_TRAINING_CONFIG["warmup_bias_lr"]
        )
        warmup_layout.addWidget(self.config_widgets["warmup_bias_lr"])
        warmup_layout.addStretch()
        advanced_layout.addWidget(warmup_group)

        # 4. Augmentation Settings
        augment_group = QGroupBox("Augmentation Settings")
        augment_layout = QVBoxLayout(augment_group)
        augment_params = [
            (
                "hsv_h",
                "HSV Hue:",
                DEFAULT_TRAINING_CONFIG["hsv_h"],
                0.0,
                1.0,
                3,
            ),
            (
                "hsv_s",
                "HSV Saturation:",
                DEFAULT_TRAINING_CONFIG["hsv_s"],
                0.0,
                1.0,
                3,
            ),
            (
                "hsv_v",
                "HSV Value:",
                DEFAULT_TRAINING_CONFIG["hsv_v"],
                0.0,
                1.0,
                3,
            ),
            (
                "degrees",
                "Rotation Degrees:",
                DEFAULT_TRAINING_CONFIG["degrees"],
                -180.0,
                180.0,
                1,
            ),
            (
                "translate",
                "Translate:",
                DEFAULT_TRAINING_CONFIG["translate"],
                0.0,
                1.0,
                3,
            ),
            ("scale", "Scale:", DEFAULT_TRAINING_CONFIG["scale"], 0.0, 2.0, 3),
            (
                "shear",
                "Shear:",
                DEFAULT_TRAINING_CONFIG["shear"],
                -45.0,
                45.0,
                1,
            ),
            (
                "perspective",
                "Perspective:",
                DEFAULT_TRAINING_CONFIG["perspective"],
                0.0,
                0.001,
                6,
            ),
        ]

        grid_layout = QGridLayout()
        grid_layout.setHorizontalSpacing(10)
        grid_layout.setVerticalSpacing(5)
        for i, (
            param,
            label,
            default,
            min_val,
            max_val,
            decimals,
        ) in enumerate(augment_params):
            row = i // 4
            col = (i % 4) * 2

            label_widget = QLabel(label)
            label_widget.setMinimumWidth(80)
            grid_layout.addWidget(label_widget, row, col)

            widget = CustomDoubleSpinBox()
            widget.setRange(min_val, max_val)
            widget.setDecimals(decimals)
            widget.setValue(default)
            widget.setMinimumWidth(80)
            self.config_widgets[param] = widget
            grid_layout.addWidget(widget, row, col + 1)

        for col in range(8, 10):
            grid_layout.setColumnStretch(col, 1)
        augment_layout.addLayout(grid_layout)
        advanced_layout.addWidget(augment_group)

        # 5. Regularization
        reg_group = QGroupBox("Regularization")
        reg_layout = QHBoxLayout(reg_group)
        reg_layout.addWidget(QLabel("Dropout:"))
        self.config_widgets["dropout"] = CustomDoubleSpinBox()
        self.config_widgets["dropout"].setDecimals(3)
        self.config_widgets["dropout"].setValue(
            DEFAULT_TRAINING_CONFIG["dropout"]
        )
        reg_layout.addWidget(self.config_widgets["dropout"])

        reg_layout.addWidget(QLabel("Fraction:"))
        self.config_widgets["fraction"] = CustomDoubleSpinBox()
        self.config_widgets["fraction"].setDecimals(3)
        self.config_widgets["fraction"].setValue(
            DEFAULT_TRAINING_CONFIG["fraction"]
        )
        reg_layout.addWidget(self.config_widgets["fraction"])

        self.config_widgets["rect"] = CustomCheckBox("Rectangular")
        self.config_widgets["rect"].setChecked(DEFAULT_TRAINING_CONFIG["rect"])
        reg_layout.addWidget(self.config_widgets["rect"])
        reg_layout.addStretch()
        advanced_layout.addWidget(reg_group)

        # 6. Loss Weights
        loss_group = QGroupBox("Loss Weights")
        loss_layout = QHBoxLayout(loss_group)
        loss_layout.addWidget(QLabel("Box:"))
        self.config_widgets["box"] = CustomDoubleSpinBox()
        self.config_widgets["box"].setDecimals(2)
        self.config_widgets["box"].setValue(DEFAULT_TRAINING_CONFIG["box"])
        loss_layout.addWidget(self.config_widgets["box"])

        loss_layout.addWidget(QLabel("Cls:"))
        self.config_widgets["cls"] = CustomDoubleSpinBox()
        self.config_widgets["cls"].setDecimals(2)
        self.config_widgets["cls"].setValue(DEFAULT_TRAINING_CONFIG["cls"])
        loss_layout.addWidget(self.config_widgets["cls"])

        loss_layout.addWidget(QLabel("DFL:"))
        self.config_widgets["dfl"] = CustomDoubleSpinBox()
        self.config_widgets["dfl"].setDecimals(2)
        self.config_widgets["dfl"].setValue(DEFAULT_TRAINING_CONFIG["dfl"])
        loss_layout.addWidget(self.config_widgets["dfl"])

        loss_layout.addWidget(QLabel("Pose:"))
        self.config_widgets["pose"] = CustomDoubleSpinBox()
        self.config_widgets["pose"].setDecimals(2)
        self.config_widgets["pose"].setValue(DEFAULT_TRAINING_CONFIG["pose"])
        loss_layout.addWidget(self.config_widgets["pose"])

        loss_layout.addWidget(QLabel("Kobj:"))
        self.config_widgets["kobj"] = CustomDoubleSpinBox()
        self.config_widgets["kobj"].setDecimals(2)
        self.config_widgets["kobj"].setValue(DEFAULT_TRAINING_CONFIG["kobj"])
        loss_layout.addWidget(self.config_widgets["kobj"])
        loss_layout.addStretch()
        advanced_layout.addWidget(loss_group)

        # 7. Checkpoint and Validation
        ckpt_group = QGroupBox("Checkpoint and Validation")
        ckpt_layout = QHBoxLayout(ckpt_group)
        ckpt_layout.addWidget(QLabel("Save Period:"))
        self.config_widgets["save_period"] = CustomSpinBox()
        self.config_widgets["save_period"].setRange(-1, 1000)
        self.config_widgets["save_period"].setValue(
            DEFAULT_TRAINING_CONFIG["save_period"]
        )
        self.config_widgets["save_period"].setSpecialValueText("Disabled")
        ckpt_layout.addWidget(self.config_widgets["save_period"])

        self.config_widgets["val"] = CustomCheckBox("Validation")
        self.config_widgets["val"].setChecked(DEFAULT_TRAINING_CONFIG["val"])
        ckpt_layout.addWidget(self.config_widgets["val"])
        self.config_widgets["plots"] = CustomCheckBox("Plots")
        self.config_widgets["plots"].setChecked(
            DEFAULT_TRAINING_CONFIG["plots"]
        )
        ckpt_layout.addWidget(self.config_widgets["plots"])
        self.config_widgets["save"] = CustomCheckBox("Save")
        self.config_widgets["save"].setChecked(DEFAULT_TRAINING_CONFIG["save"])
        ckpt_layout.addWidget(self.config_widgets["save"])
        self.config_widgets["resume"] = CustomCheckBox("Resume")
        self.config_widgets["resume"].setChecked(
            DEFAULT_TRAINING_CONFIG["resume"]
        )
        ckpt_layout.addWidget(self.config_widgets["resume"])
        self.config_widgets["cache"] = CustomCheckBox("Cache")
        self.config_widgets["cache"].setChecked(
            DEFAULT_TRAINING_CONFIG["cache"]
        )
        ckpt_layout.addWidget(self.config_widgets["cache"])
        self.config_widgets["skip_empty_files"] = CustomCheckBox(
            "Skip Empty Files"
        )
        self.config_widgets["skip_empty_files"].setChecked(False)
        ckpt_layout.addWidget(self.config_widgets["skip_empty_files"])
        self.config_widgets["only_checked_files"] = CustomCheckBox(
            "Only Checked Files"
        )
        self.config_widgets["only_checked_files"].setChecked(False)
        ckpt_layout.addWidget(self.config_widgets["only_checked_files"])
        ckpt_layout.addStretch()
        advanced_layout.addWidget(ckpt_group)

        advanced_container_layout.addWidget(self.advanced_content_widget)
        layout.addWidget(advanced_container)
        parent_layout.addWidget(group)

    def load_config_to_ui(self, config):
        def set_widget_value(key, value):
            if key not in self.config_widgets:
                return

            widget = self.config_widgets[key]
            widget_type = type(widget).__name__

            try:
                if widget_type == "CustomLineEdit":
                    if key == "classes":
                        widget.setText(format_classes_display(value))
                    else:
                        widget.setText(str(value) if value is not None else "")
                elif widget_type in ["CustomSpinBox", "CustomDoubleSpinBox"]:
                    widget.setValue(value)
                elif widget_type == "CustomComboBox":
                    if isinstance(value, str):
                        index = widget.findText(value)
                        if index >= 0:
                            widget.setCurrentIndex(index)
                    else:
                        widget.setCurrentIndex(value)
                elif widget_type == "CustomCheckBox":
                    widget.setChecked(bool(value))
                elif widget_type == "CustomSlider":
                    widget.setValue(value)
            except Exception as e:
                logger.warning(f"Failed to set value for widget {key}: {e}")

        sections_to_process = [
            "basic",
            "train",
            "augment",
            "strategy",
            "learning_rate",
            "warmup",
            "regularization",
            "loss_weights",
            "checkpoint",
        ]
        for section in sections_to_process:
            if section in config:
                for key, value in config[section].items():
                    if key == "dataset_ratio":
                        # Legacy migration: detect unreasonable ratios (< 0.5)
                        if 0 <= value <= 1:
                            if value < 0.5:
                                logger.warning(
                                    f"Migrated legacy split setting ({value}) → train_ratio=0.8"
                                )
                                value = 0.8
                            self.config_widgets[key].setValue(int(value * 100))
                            self.dataset_ratio_label.setText(str(value))
                        else:
                            if value < 50:
                                logger.warning(
                                    f"Migrated legacy split setting ({value}) → train_ratio=80"
                                )
                                value = 80
                            self.config_widgets[key].setValue(int(value))
                            self.dataset_ratio_label.setText(
                                str(value / 100.0)
                            )
                    elif key == "device":
                        from anylabeling.services.training_center.device_service import migrate_legacy_device
                        migrated = migrate_legacy_device(str(value))
                        # Find matching item by training_value (works for both string and DeviceInfo)
                        idx = -1
                        for i in range(self.config_widgets[key].count()):
                            item = self.config_widgets[key].itemData(i)
                            training_val = item.training_value if hasattr(item, 'training_value') else str(item)
                            if training_val == migrated:
                                idx = i
                                break
                        if idx >= 0:
                            self.config_widgets[key].setCurrentIndex(idx)
                        elif migrated.startswith("cuda:"):
                            # GPU not found — fall back to Auto
                            self.config_widgets[key].setCurrentIndex(0)
                            logger.warning(
                                "Saved device %s (%s) not available, falling back to Auto",
                                value, migrated,
                            )
                        else:
                            self.config_widgets[key].setCurrentIndex(0)  # Auto
                        self.on_device_changed(self.config_widgets[key].currentText())
                    elif key == "optimizer":
                        index = self.config_widgets[key].findText(str(value))
                        if index >= 0:
                            self.config_widgets[key].setCurrentIndex(index)
                    elif key == "pose_config":
                        if value:
                            self.config_widgets[key].setText(value)
                    elif key in (
                        "skip_empty_files",
                        "only_checked_files",
                    ):
                        set_widget_value(key, value)
                    else:
                        set_widget_value(key, value)

        for key, value in config.items():
            if key not in sections_to_process and key in self.config_widgets:
                set_widget_value(key, value)

    def import_config(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Import Config"),
            "",
            "JSON Files (*.json);;All Files (*)",
        )
        if file_path:
            config = load_config_from_file(file_path)
            if config:
                self.load_config_to_ui(config)
                QMessageBox.information(
                    self,
                    self.tr("Success"),
                    self.tr("Config imported successfully"),
                )
            else:
                QMessageBox.warning(
                    self, self.tr("Error"), self.tr("Failed to import config")
                )

    def get_current_config(self):
        def get_widget_value(key):
            if key not in self.config_widgets:
                return None
            widget = self.config_widgets[key]
            widget_type = type(widget).__name__
            try:
                if widget_type == "CustomLineEdit":
                    return widget.text()
                elif widget_type in ["CustomSpinBox", "CustomDoubleSpinBox"]:
                    return widget.value()
                elif widget_type == "CustomComboBox":
                    return widget.currentText()
                elif widget_type == "CustomCheckBox":
                    return widget.isChecked()
                elif widget_type == "CustomSlider":
                    return widget.value()
            except Exception:
                return None
            return None

        config = {
            "basic": {
                "project": get_widget_value("project") or "",
                "name": get_widget_value("name") or "",
                "model": (get_widget_value("model") or "").strip('"'),
                "data": (get_widget_value("data") or "").strip('"'),
                "device": (
                    _get_combo_training_value(self.config_widgets["device"])
                    if "device" in self.config_widgets
                    and hasattr(self.config_widgets["device"], "currentData")
                    else get_widget_value("device") or "auto"
                ),
                "dataset_ratio": (
                    get_widget_value("dataset_ratio") / 100.0
                    if get_widget_value("dataset_ratio") is not None
                    else 0.8
                ),
                "pose_config": get_widget_value("pose_config"),
            },
            "train": {
                "epochs": get_widget_value("epochs"),
                "batch": get_widget_value("batch"),
                "imgsz": get_widget_value("imgsz"),
                "workers": get_widget_value("workers"),
                "single_cls": get_widget_value("single_cls"),
                "classes": parse_string_to_digit_list(
                    get_widget_value("classes") or ""
                ),
            },
            "strategy": {
                "time": get_widget_value("time"),
                "patience": get_widget_value("patience"),
                "close_mosaic": get_widget_value("close_mosaic"),
                "optimizer": get_widget_value("optimizer"),
                "cos_lr": get_widget_value("cos_lr"),
                "amp": get_widget_value("amp"),
                "multi_scale": get_widget_value("multi_scale"),
            },
            "learning_rate": {
                "lr0": get_widget_value("lr0"),
                "lrf": get_widget_value("lrf"),
                "momentum": get_widget_value("momentum"),
                "weight_decay": get_widget_value("weight_decay"),
            },
            "warmup": {
                "warmup_epochs": get_widget_value("warmup_epochs"),
                "warmup_momentum": get_widget_value("warmup_momentum"),
                "warmup_bias_lr": get_widget_value("warmup_bias_lr"),
            },
            "augment": {
                "hsv_h": get_widget_value("hsv_h"),
                "hsv_s": get_widget_value("hsv_s"),
                "hsv_v": get_widget_value("hsv_v"),
                "degrees": get_widget_value("degrees"),
                "translate": get_widget_value("translate"),
                "scale": get_widget_value("scale"),
                "shear": get_widget_value("shear"),
                "perspective": get_widget_value("perspective"),
            },
            "regularization": {
                "dropout": get_widget_value("dropout"),
                "fraction": get_widget_value("fraction"),
                "rect": get_widget_value("rect"),
            },
            "loss_weights": {
                "box": get_widget_value("box"),
                "cls": get_widget_value("cls"),
                "dfl": get_widget_value("dfl"),
                "pose": get_widget_value("pose"),
                "kobj": get_widget_value("kobj"),
            },
            "checkpoint": {
                "save_period": get_widget_value("save_period"),
                "val": get_widget_value("val"),
                "plots": get_widget_value("plots"),
                "save": get_widget_value("save"),
                "resume": get_widget_value("resume"),
                "cache": get_widget_value("cache"),
                "skip_empty_files": get_widget_value("skip_empty_files"),
                "only_checked_files": get_widget_value("only_checked_files"),
            },
        }

        return config

    def save_current_config(self):
        try:
            save_config(self.get_current_config())
            template = self.tr("Configuration saved successfully to %s")
            msg_test = template % get_settings_config_path()
            QMessageBox.information(self, self.tr("Success"), msg_test)
        except Exception as e:
            QMessageBox.warning(
                self, self.tr("Error"), f"Failed to save config: {str(e)}"
            )

    def start_training(self):
        if self.training_status == "training":
            QMessageBox.warning(
                self,
                self.tr("Training in Progress"),
                self.tr(
                    "Training is currently in progress. Please stop the training first if you need to reconfigure."
                ),
            )
            return

        config = self.get_current_config()
        is_valid, error_message = validate_basic_config(config)
        if is_valid == "directory_exists":
            project_dir = error_message
            potential_model_path = os.path.join(
                project_dir, "weights", "best.pt"
            )

            if os.path.exists(potential_model_path):
                reply = QMessageBox.question(
                    self,
                    self.tr("Existing Model Detected"),
                    self.tr(
                        "A trained model already exists at this location.\n\n"
                        "Do you want to:\n"
                        "Yes - Export the existing model directly\n"
                        "No - Continue to retrain (will overwrite)"
                    ),
                    QMessageBox.StandardButton.Yes
                    | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes,
                )

                if reply == QMessageBox.StandardButton.Yes:
                    self.current_project_path = project_dir
                    self.training_status = "completed"
                    save_config(config)
                    self.go_to_specific_tab(2)
                    self.update_training_status_display()
                    self.start_training_button.setVisible(False)
                    self.export_button.setVisible(True)
                    self.previous_button.setVisible(True)
                    self.update_training_images()
                    self.append_training_log(
                        f"Loaded existing model from: {potential_model_path}"
                    )
                    return

            reply = QMessageBox.question(
                self,
                self.tr("Directory Exists"),
                self.tr(
                    "Project directory already exists! Do you want to overwrite it?\nIf not, please manually modify the `Name` field value."
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )

            if reply == QMessageBox.StandardButton.Yes:
                try:
                    shutil.rmtree(error_message)
                    self.append_training_log(
                        f"Removed existing directory: {error_message}"
                    )
                except Exception as e:
                    error_msg = f"Failed to remove directory: {str(e)}"
                    logger.error(error_msg)
                    return
            else:
                return
        elif not is_valid:
            QMessageBox.warning(
                self, self.tr("Validation Error"), error_message
            )
            self.append_training_log(f"Validation Error: {error_message}")
            return

        if not self.selected_task_type:
            QMessageBox.warning(
                self,
                self.tr("Error"),
                self.tr("Please select a task type first"),
            )
            return

        if self.selected_task_type.lower() == "pose":
            pose_config = config["basic"].get("pose_config", "")
            if not pose_config or not os.path.exists(pose_config):
                QMessageBox.warning(
                    self,
                    self.tr("Error"),
                    self.tr(
                        "Please select a valid pose configuration file for pose detection tasks"
                    ),
                )
                return

        if self.training_status in ["completed", "error"]:
            reply = QMessageBox.question(
                self,
                self.tr("Reset Training"),
                self.tr(
                    "Training traces detected. Do you want to reset the training tab?"
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.reset_train_tab()
        elif self.training_status == "stop":
            self.start_training_button.setVisible(True)
            self.training_status == "idle"

        save_config(config)
        self._config_completed = True
        self._update_stage_gates()
        self.go_to_specific_tab(2)

    def init_config_buttons(self, parent_layout):
        button_layout = QHBoxLayout()

        self._import_btn = SecondaryButton(self.tr("Import Config"))
        self._import_btn.clicked.connect(self.import_config)
        button_layout.addWidget(self._import_btn)

        self._save_config_btn = SecondaryButton(self.tr("Save Config"))
        self._save_config_btn.clicked.connect(self.save_current_config)
        button_layout.addWidget(self._save_config_btn)
        button_layout.addStretch()

        self._previous_btn = SecondaryButton(self.tr("Previous"))
        self._previous_btn.clicked.connect(lambda: self.go_to_specific_tab(0))
        button_layout.addWidget(self._previous_btn)

        self._train_btn = PrimaryButton(self.tr("Next"))
        self._train_btn.clicked.connect(self.start_training)
        button_layout.addWidget(self._train_btn)

        parent_layout.addLayout(button_layout)

    def load_default_config(self):
        config = load_config()
        self.load_config_to_ui(config)

    def init_config_tab(self):
        layout = QVBoxLayout(self.config_tab)

        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        self.init_basic_settings(scroll_layout)
        self.init_train_settings(scroll_layout)

        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)

        self.init_config_buttons(layout)
        self.load_default_config()

    # Train tab
    def update_training_status_display(self):
        if not hasattr(self, 'status_label'):
            return
        color = TRAINING_STATUS_COLORS.get(self.training_status, "#6c757d")
        text = self.tr(
            TRAINING_STATUS_TEXTS.get(self.training_status, "Unknown status")
        )
        self.status_label.setText(text)
        self.status_label.setStyleSheet(get_status_label_style(color))

    def update_training_progress(self):
        if not self.current_project_path:
            return

        results_file = os.path.join(self.current_project_path, "results.csv")
        if os.path.exists(results_file):
            try:
                with open(results_file, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    if len(rows) > 1:  # Skip header
                        self.current_epochs = len(rows) - 1
                        progress = min(
                            100,
                            int(
                                (self.current_epochs / self.total_epochs) * 100
                            ),
                        )
                        self.progress_bar.setValue(progress)
                        self.progress_bar.setFormat(
                            f"{self.current_epochs}/{self.total_epochs}"
                        )
            except Exception as e:
                logger.warning(f"Failed to read results.csv: {e}")

    def update_training_images(self):
        """Phase-aware training image refresh (runs on QTimer, non-blocking).

        Phase 1 (idle/completed, no running job): show dataset samples.
        Phase 2 (training in progress): show train_batch images.
        Phase 3 (just completed): show val predictions + curves.
        """
        if not self.current_project_path and not self._prepared_dataset_dir:
            return

        # ── Phase 3: just completed → val predictions ──
        if self.training_status == "completed":
            self._display_completed_images()
            return

        # ── Phase 2: training in progress → train_batch ──
        if self.training_status == "training" and self.current_project_path:
            self._display_train_batch_images()
            return

        # ── Phase 1: idle → dataset samples ──
        self._display_dataset_samples()

    def _display_dataset_samples(self):
        """Show sample images from the prepared dataset on the Train tab."""
        dataset_dir = self._prepared_dataset_dir
        if not dataset_dir or not os.path.isdir(dataset_dir):
            # Reset all image slots
            for i, label in enumerate(self.image_labels):
                label.clear()
                label.setText(self.tr("No image"))
                self.image_paths[i] = None
            return

        # Collect sample images from train split
        train_img_dir = os.path.join(dataset_dir, "images", "train")
        samples = []
        if os.path.isdir(train_img_dir):
            for f in sorted(os.listdir(train_img_dir)):
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    samples.append(os.path.join(train_img_dir, f))
        # Take first 4 from train, then 2 from val
        sample_paths = samples[:4]
        val_img_dir = os.path.join(dataset_dir, "images", "val")
        if os.path.isdir(val_img_dir):
            val_samples = []
            for f in sorted(os.listdir(val_img_dir)):
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    val_samples.append(os.path.join(val_img_dir, f))
            sample_paths.extend(val_samples[:2])

        # Fill up to 6 slots
        sample_paths = sample_paths[:6]

        for i, image_label in enumerate(self.image_labels):
            if i < len(sample_paths):
                path = sample_paths[i]
                try:
                    pixmap = QPixmap(path)
                    if not pixmap.isNull():
                        scaled = pixmap.scaled(
                            150, 150,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation,
                        )
                        image_label.setPixmap(scaled)
                        image_label.setText("")
                        image_label.setToolTip(f"[Sample] {os.path.basename(path)}")
                        self.image_paths[i] = path
                    else:
                        image_label.clear()
                        image_label.setText(self.tr("No image"))
                        self.image_paths[i] = None
                except Exception:
                    image_label.clear()
                    image_label.setText(self.tr("No image"))
                    self.image_paths[i] = None
            else:
                image_label.clear()
                image_label.setText(self.tr("No image"))
                self.image_paths[i] = None

    def _display_train_batch_images(self):
        """Show train_batch images during active training."""
        if not self.current_project_path:
            return

        def find_images_by_pattern(patterns, max_count=3):
            found_files = []
            for pattern in patterns:
                matches = glob.glob(
                    os.path.join(self.current_project_path, pattern)
                )
                matches.sort()
                found_files.extend(matches)
                if len(found_files) >= max_count:
                    break
            return found_files[:max_count]

        # Train batch images in first 3 slots
        train_batches = find_images_by_pattern(["train_batch*.jpg"], 3)
        # Val/curve images in last 3 slots
        if self.selected_task_type == "Classify":
            val_patterns = ["val_batch0_labels.jpg", "val_batch0_pred.jpg", "results.png"]
        else:
            val_patterns = ["*PR_curve.png", "*F1_curve.png", "results.png"]
        val_batches = find_images_by_pattern(val_patterns, 3)

        all_paths = train_batches + val_batches

        for i, image_label in enumerate(self.image_labels):
            if i < len(all_paths):
                path = all_paths[i]
                try:
                    pixmap = QPixmap(path)
                    if not pixmap.isNull():
                        scaled = pixmap.scaled(
                            150, 150,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation,
                        )
                        image_label.setPixmap(scaled)
                        image_label.setText("")
                        image_label.setToolTip(os.path.basename(path))
                        self.image_paths[i] = path
                    else:
                        image_label.clear()
                        image_label.setText(self.tr("No image"))
                        self.image_paths[i] = None
                except Exception:
                    image_label.clear()
                    image_label.setText(self.tr("No image"))
                    self.image_paths[i] = None
            else:
                image_label.clear()
                image_label.setText(self.tr("No image"))
                self.image_paths[i] = None

    def _display_completed_images(self):
        """Show val predictions and curves after training completes."""
        if not self.current_project_path:
            return

        def find_images_by_pattern(patterns, max_count=6):
            found_files = []
            for pattern in patterns:
                matches = glob.glob(
                    os.path.join(self.current_project_path, pattern)
                )
                matches.sort()
                found_files.extend(matches)
                if len(found_files) >= max_count:
                    break
            return found_files[:max_count]

        # Priority: val_batch pred/labels → curves → results
        patterns = [
            "val_batch0_pred.jpg",
            "val_batch0_labels.jpg",
            "val_batch1_pred.jpg",
            "val_batch1_labels.jpg",
            "train_batch0.jpg",
            "train_batch1.jpg",
            "*PR_curve.png",
            "*F1_curve.png",
            "results.png",
            "confusion_matrix.png",
            "labels.jpg",
        ]
        all_paths = find_images_by_pattern(patterns, 6)

        for i, image_label in enumerate(self.image_labels):
            if i < len(all_paths):
                path = all_paths[i]
                try:
                    pixmap = QPixmap(path)
                    if not pixmap.isNull():
                        scaled = pixmap.scaled(
                            150, 150,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation,
                        )
                        image_label.setPixmap(scaled)
                        image_label.setText("")
                        basename = os.path.basename(path)
                        # Prepend phase tag for clarity
                        if "pred" in basename:
                            tag = "[Pred]"
                        elif "labels" in basename:
                            tag = "[GT]"
                        elif "train_batch" in basename:
                            tag = "[Train]"
                        else:
                            tag = ""
                        image_label.setToolTip(f"{tag} {basename}".strip())
                        self.image_paths[i] = path
                    else:
                        image_label.clear()
                        image_label.setText(self.tr("No image"))
                        self.image_paths[i] = None
                except Exception:
                    image_label.clear()
                    image_label.setText(self.tr("No image"))
                    self.image_paths[i] = None
            else:
                image_label.clear()
                image_label.setText(self.tr("No image"))
                self.image_paths[i] = None

    def _on_unified_training_event(self, event):
        """Handle unified TrainingEvent from JobManager"""
        handle_unified_training_event(self, event)

    def on_training_event(self, event_type, data):
        if event_type == "training_started":
            self.training_status = "training"
            self.total_epochs = data["total_epochs"]
            self.current_epochs = 0
            # Guard: train tab may not be initialized yet (off-screen tests)
            if hasattr(self, 'progress_bar'):
                self.progress_bar.setValue(0)
                self.progress_bar.setFormat(f"0/{self.total_epochs}")
            self.update_training_status_display()

            # Disable ONLY config-changing controls, NOT tabs/metrics/console/stop
            self._set_config_controls_enabled(False)
            self.start_training_button.setVisible(False)
            self.stop_training_button.setVisible(True)
            self.export_button.setVisible(False)
            self.previous_button.setVisible(False)
            self.progress_timer.start(1000)
            self.image_timer.start(5000)
            self.append_training_log(self.tr("Worker process started"))

            # Show Metrics tab and bind dashboard
            dashboard = self._ensure_metrics_dashboard()
            self.tab_widget.setTabVisible(3, True)
            save_dir = data.get("save_dir") or self.current_project_path
            dashboard.bind_job("guided", save_dir)
            # Auto-switch to Train tab so user sees progress
            self.tab_widget.setCurrentIndex(2)

        elif event_type == "training_completed":
            self.training_status = "completed"
            real_save_dir = data.get("save_dir", "")
            if real_save_dir and os.path.isdir(real_save_dir):
                self.current_project_path = real_save_dir
                # Update dashboard with real save_dir (may differ from predicted path)
                if self._metrics_dashboard:
                    self._metrics_dashboard.update_output_dir("guided", real_save_dir)
            self._set_config_controls_enabled(True)
            self.update_training_status_display()
            self.stop_training_button.setVisible(False)
            self.start_training_button.setVisible(False)
            self.previous_button.setVisible(True)
            self.export_button.setVisible(True)
            self.progress_timer.stop()
            self.image_timer.stop()
            self.update_training_progress()
            self.update_training_images()
            self.append_training_log(
                self.tr("Training completed successfully!")
            )
            if self._metrics_dashboard:
                self._metrics_dashboard.on_run_completed("guided")

        elif event_type == "training_error":
            self.training_status = "error"
            real_save_dir = data.get("save_dir", "")
            if real_save_dir and os.path.isdir(real_save_dir):
                self.current_project_path = real_save_dir
                if self._metrics_dashboard:
                    self._metrics_dashboard.update_output_dir("guided", real_save_dir)
            self._set_config_controls_enabled(True)
            self.update_training_status_display()
            self.start_training_button.setVisible(False)
            self.previous_button.setVisible(True)
            self.stop_training_button.setVisible(False)
            self.export_button.setVisible(False)
            self.progress_timer.stop()
            self.image_timer.stop()
            error_msg = data.get("error", "Unknown error occurred")
            self.append_training_log(f"ERROR: {error_msg}")
            if self._metrics_dashboard:
                self._metrics_dashboard.on_run_stopped("guided")

        elif event_type == "training_stopped":
            self.training_status = "stop"
            real_save_dir = data.get("save_dir", "")
            if real_save_dir and os.path.isdir(real_save_dir):
                self.current_project_path = real_save_dir
                if self._metrics_dashboard:
                    self._metrics_dashboard.update_output_dir("guided", real_save_dir)
            self._set_config_controls_enabled(True)
            self.update_training_status_display()
            self.start_training_button.setVisible(True)
            self.start_training_button.setEnabled(True)
            self.previous_button.setVisible(True)
            self.stop_training_button.setVisible(False)
            self.export_button.setVisible(False)
            self.progress_timer.stop()
            self.image_timer.stop()
            self.append_training_log(self.tr("Training stopped by user"))
            if self._metrics_dashboard:
                self._metrics_dashboard.on_run_stopped("guided")

        elif event_type == "epoch_metrics":
            # Forward structured metrics to dashboard (real-time)
            if self._metrics_dashboard:
                self._metrics_dashboard.on_metric_event("guided", data)

        elif event_type == "training_log":
            log_message = data.get("message", "")
            if log_message:
                self.append_training_log(log_message)

    def append_training_log(self, text):
        def clean_ansi_codes(text: str) -> str:
            ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
            return ansi_escape.sub("", text)

        if hasattr(self, "log_display"):
            text = clean_ansi_codes(text)
            self.log_display.append(text.strip())

    def _on_training_event_from_job(self, event):
        """Handle unified training events from JobManager.

        Forwards all event types to handle_unified_training_event()
        which handles COMPLETED, FAILED, STOPPED, WORKER_READY, etc.
        """
        from anylabeling.views.training.guided_training_widget_events import (
            handle_unified_training_event,
        )
        handle_unified_training_event(self, event)

    def _on_job_status_change(self, job):
        """Handle job status changes from JobManager.

        Updates UI state and sets training_status string for display.
        Terminal statuses are handled here; non-terminal events go
        through handle_unified_training_event.
        """
        if job.status == TrainingStatus.RUNNING:
            self.training_status = "training"
            self.start_training_button.setVisible(False)
            self.stop_training_button.setVisible(True)
            self.stop_training_button.setEnabled(True)
            self.export_button.setVisible(False)
            self.previous_button.setVisible(False)
            self.update_training_status_display()
        elif job.status == TrainingStatus.COMPLETED:
            self.training_status = "completed"
            self.update_training_status_display()
            self.stop_training_button.setVisible(False)
            self.start_training_button.setVisible(True)
            self.start_training_button.setEnabled(True)
            self.previous_button.setVisible(True)
            self.export_button.setVisible(True)
            self.progress_timer.stop()
            self.image_timer.stop()
        elif job.status == TrainingStatus.FAILED:
            self.training_status = "error"
            self.update_training_status_display()
            self.start_training_button.setVisible(True)
            self.start_training_button.setEnabled(True)
            self.stop_training_button.setVisible(False)
            self.export_button.setVisible(False)
            self.previous_button.setVisible(True)
            self.progress_timer.stop()
            self.image_timer.stop()
        elif job.status == TrainingStatus.STOPPED:
            self.training_status = "stop"
            self.update_training_status_display()
            self.start_training_button.setVisible(True)
            self.start_training_button.setEnabled(True)
            self.stop_training_button.setVisible(False)
            self.export_button.setVisible(False)
            self.previous_button.setVisible(True)
            self.progress_timer.stop()
            self.image_timer.stop()
        elif job.status == TrainingStatus.PREPARING:
            self.training_status = "preparing"
            self.update_training_status_display()
            self.start_training_button.setEnabled(False)

    def init_training_status(self, parent_layout):
        status_group = QGroupBox(self.tr("Training Status"))
        status_layout = QVBoxLayout(status_group)

        self.status_label = QLabel(self.tr("Ready to train"))
        self.status_label.setStyleSheet(get_status_label_style())
        status_layout.addWidget(self.status_label)

        progress_layout = QHBoxLayout()
        progress_layout.addWidget(QLabel(self.tr("Progress:")))
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("0/0")
        self.progress_bar.setStyleSheet(get_progress_bar_style())
        progress_layout.addWidget(self.progress_bar)
        status_layout.addLayout(progress_layout)

        self.dataset_status_label = QLabel(self.tr("Dataset: Not prepared"))
        self.dataset_status_label.setStyleSheet("color: gray;")
        status_layout.addWidget(self.dataset_status_label)

        parent_layout.addWidget(status_group)

    def clear_training_logs(self):
        if hasattr(self, "log_display"):
            reply = QMessageBox.question(
                self,
                self.tr("Clear Logs"),
                self.tr("Are you sure you want to clear all training logs?"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.log_display.clear()

    def copy_training_logs(self):
        if hasattr(self, "log_display"):
            text = self.log_display.toPlainText()
            if text:
                clipboard = QApplication.clipboard()
                clipboard.setText(text)

    def init_training_logs(self, parent_layout):
        logs_group = QGroupBox(self.tr("Training Logs"))
        logs_layout = QVBoxLayout(logs_group)

        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setMinimumHeight(250)
        self.log_display.setStyleSheet(get_log_display_style())
        logs_layout.addWidget(self.log_display)

        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.clear_logs_button = SecondaryButton(self.tr("Clear"))
        self.clear_logs_button.clicked.connect(self.clear_training_logs)
        button_layout.addWidget(self.clear_logs_button)

        self.copy_logs_button = SecondaryButton(self.tr("Copy"))
        self.copy_logs_button.clicked.connect(self.copy_training_logs)
        button_layout.addWidget(self.copy_logs_button)

        logs_layout.addLayout(button_layout)
        parent_layout.addWidget(logs_group)

    def init_training_images(self, parent_layout):
        images_group = QGroupBox(self.tr("Training Images"))
        images_layout = QVBoxLayout(images_group)
        images_layout.setContentsMargins(5, 5, 5, 5)

        self.image_labels = []
        self.image_paths = [None] * 6
        self.images_widget = QWidget()
        images_row_layout = QHBoxLayout(self.images_widget)
        images_row_layout.setSpacing(10)
        images_row_layout.setContentsMargins(0, 0, 0, 0)

        for i in range(6):
            image_label = QLabel()
            image_label.setMinimumSize(150, 150)
            image_label.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
            )
            image_label.setStyleSheet(get_image_label_style())
            image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            image_label.setText(self.tr("No image"))
            image_label.setScaledContents(False)
            image_label.mousePressEvent = (
                lambda event, idx=i: self.on_image_clicked(idx)
            )
            self.image_labels.append(image_label)
            images_row_layout.addWidget(image_label, 1)

        images_layout.addWidget(self.images_widget, 1)
        parent_layout.addWidget(images_group, 1)

    def on_image_clicked(self, index):
        if self.image_paths[index]:
            self.open_image_file(self.image_paths[index])

    def open_image_file(self, image_path):
        try:
            is_wsl2 = False
            try:
                if (
                    hasattr(os, "uname")
                    and "microsoft" in os.uname().release.lower()
                ):
                    is_wsl2 = True
            except (AttributeError, OSError):
                pass

            if is_wsl2:  # WSL2
                windows_path = (
                    subprocess.check_output(["wslpath", "-w", image_path])
                    .decode()
                    .strip()
                )
                subprocess.run(
                    [
                        "powershell.exe",
                        "-c",
                        f'Start-Process "{windows_path}"',
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            elif os.name == "nt":  # Windows
                os.startfile(image_path)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", image_path])
            elif os.name == "posix":  # Linux
                subprocess.run(["xdg-open", image_path])
        except Exception as e:
            logger.warning(f"Failed to open image {image_path}: {e}")

    def open_training_directory(self):
        if self.current_project_path and os.path.exists(
            self.current_project_path
        ):
            try:
                is_wsl2 = False
                try:
                    if (
                        hasattr(os, "uname")
                        and "microsoft" in os.uname().release.lower()
                    ):
                        is_wsl2 = True
                except (AttributeError, OSError):
                    pass

                if is_wsl2:  # WSL2
                    wsl_path = self.current_project_path
                    windows_path = (
                        subprocess.check_output(["wslpath", "-w", wsl_path])
                        .decode()
                        .strip()
                    )
                    subprocess.run(["explorer.exe", windows_path])
                elif os.name == "nt":  # Windows
                    os.startfile(self.current_project_path)
                elif platform.system() == "Darwin":  # macOS
                    subprocess.run(["open", self.current_project_path])
                elif os.name == "posix":  # Linux
                    subprocess.run(["xdg-open", self.current_project_path])
            except Exception as e:
                self.append_training_log(f"Failed to open directory: {str(e)}")
                QMessageBox.information(
                    self,
                    self.tr("Info"),
                    f"Directory path: {self.current_project_path}",
                )
        else:
            QMessageBox.information(
                self,
                self.tr("Info"),
                self.tr("No training directory available"),
            )

    def stop_training(self):
        reply = QMessageBox.question(
            self,
            self.tr("Confirm Stop"),
            self.tr("Are you sure you want to stop the training?"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            # For PREPARING: cancel the background prep thread first
            current = self.job_manager.get_current_job()
            if current is not None and current.status == TrainingStatus.PREPARING:
                self._cancel_prep_thread()

            success = self.job_manager.request_stop()
            if success:
                if hasattr(self, 'stop_training_button'):
                    self.stop_training_button.setEnabled(False)
                self.append_training_log(self.tr("Stopping training..."))
            else:
                self.append_training_log(self.tr("Cancel to stop training"))

    def _cancel_prep_thread(self):
        """Cancel the background preparation thread and invalidate the reservation."""
        if self._prep_thread is None:
            return
        # Clear job_id first so late queued signals are dropped
        self._prep_job_id = None
        # Disconnect signals so finished/error don't fire after stop
        if self._prep_worker is not None:
            self._prep_worker.finished.disconnect(self._on_prep_finished)
            self._prep_worker.error.disconnect(self._on_prep_error)
        self._prep_thread.quit()
        if not self._prep_thread.wait(3000):
            self._prep_thread.terminate()
            self._prep_thread.wait()
        self._prep_thread = None
        self._prep_worker = None

    def get_training_args(self, config):
        try:
            # ── Use prepared dataset if available, else create it ──
            if hasattr(self, '_prepared_yaml_path') and self._prepared_yaml_path:
                if self.selected_task_type == "Classify":
                    data_path = self._prepared_dataset_dir or self._prepared_yaml_path
                else:
                    data_path = self._prepared_yaml_path
                self.append_training_log(
                    f"Using prepared dataset: {data_path}"
                )
            elif self.selected_task_type == "Classify" and os.path.isdir(
                config["basic"]["data"]
            ):
                data_path = config["basic"]["data"]
                self.append_training_log(
                    f"Using existing dataset: {data_path}"
                )
            else:
                temp_dir = create_yolo_dataset(
                    self.image_list,
                    self.selected_task_type,
                    config["basic"]["dataset_ratio"],
                    config["basic"]["data"],
                    self.output_dir,
                    config["basic"].get("pose_config"),
                    config["checkpoint"].get("skip_empty_files", False),
                    config["checkpoint"].get("only_checked_files", False),
                )
                logger.info(f"Successfully created YOLO dataset at {temp_dir}")
                self.append_training_log(f"Created dataset: {temp_dir}")

                if self.selected_task_type == "Classify":
                    data_path = temp_dir
                else:
                    data_path = os.path.join(temp_dir, "data.yaml")

            device_value = config["basic"]["device"]
            # Convert internal device format to Ultralytics-compatible
            from anylabeling.services.training_center.device_service import (
                resolve_training_device, migrate_legacy_device,
            )
            requested_device = migrate_legacy_device(str(device_value or "auto"))
            ul_device = resolve_training_device(requested_device)
            self._requested_device = requested_device
            self._resolved_device = ul_device

            train_args = {
                "data": data_path,
                "model": config["basic"]["model"],
                "project": config["basic"]["project"],
                "name": config["basic"]["name"],
                "device": ul_device,
            }

            # Add advanced parameters
            advanced_params = {}
            for section in [
                "train",
                "strategy",
                "learning_rate",
                "warmup",
                "augment",
                "regularization",
                "loss_weights",
                "checkpoint",
            ]:
                advanced_params.update(config.get(section, {}))
            # Exclude X-AnyLabeling specific parameters not recognized by ultralytics
            xany_params_to_exclude = {"skip_empty_files", "only_checked_files"}
            for key, value in advanced_params.items():
                if key not in xany_params_to_exclude:
                    train_args[key] = value
            self.total_epochs = train_args.get("epochs", 100)

            # Log training arguments (not a fake yolo CLI)
            self.append_training_log(
                f"Training arguments: " +
                ", ".join(f"{k}={v}" for k, v in sorted(train_args.items())
                          if k not in xany_params_to_exclude)
            )

            return train_args

        except Exception as e:
            self.append_training_log(
                f"Error preparing training args: {str(e)}"
            )
            raise

    # ── Preflight ───────────────────────────────────────────────────

    def _build_guided_preflight_context(self) -> GuidedPreflightContext:
        """Build immutable preflight snapshot from current UI state."""
        config = self.get_current_config()
        project = config["basic"].get("project", "")
        name = config["basic"].get("name", "")
        output_dir = os.path.join(project, name) if project and name else project

        # Resolve runtime info from the device combo (single source of truth)
        dev_info = self._get_selected_training_device()
        runtime_id = dev_info["runtime_id"]
        runtime_python = dev_info["runtime_python"]
        requested_device = dev_info["requested_device"]
        resolved_device = dev_info["resolved_device"]
        device_name = dev_info["device_name"]

        # Use prepared YAML if available, else config field
        dataset_yaml = self._prepared_yaml_path or config["basic"].get("data", "")
        if dataset_yaml and not os.path.isfile(dataset_yaml):
            dataset_yaml = ""
        return GuidedPreflightContext(
            task_type=self.selected_task_type or "",
            model_path=config["basic"].get("model", ""),
            dataset_yaml=dataset_yaml,
            epochs=config["train"].get("epochs", 100),
            batch=config["train"].get("batch", 16),
            imgsz=config["train"].get("imgsz", 640),
            device=requested_device,
            output_dir=output_dir,
            job_name=name,
            runtime_id=runtime_id,
            runtime_python=runtime_python,
            requested_device=requested_device,
            resolved_device=resolved_device,
            device_name=device_name,
        )

    # ── Dataset cache constants ──────────────────────────────────────
    _MANIFEST_VERSION = 2
    _CONVERTER_VERSION = 4  # bumped: fix train_ratio passthrough + source_dir
    _SPLIT_SEED = 42

    def _prepare_dataset(self) -> bool:
        """Create (or reuse) YOLO dataset from loaded images.

        Returns True on success. False on failure.
        Cache: validates manifest integrity before reuse.
        """
        if self._dataset_preparing:
            return False
        if not self.image_list or not self.selected_task_type:
            return False

        self._dataset_preparing = True
        if hasattr(self, 'start_training_button'):
            self.start_training_button.setEnabled(False)
        if hasattr(self, 'run_check_button'):
            self.run_check_button.setEnabled(False)

        config = self.get_current_config()
        ratio = config["basic"].get("dataset_ratio", 0.8)
        try:
            fp = self._compute_source_fingerprint()
            manifest = self._find_cached_manifest(fp, ratio)
            if (manifest and not getattr(self, '_force_rebuild_dataset', False)
                    and self._validate_manifest_integrity(manifest)):
                # Merge class IDs (preserve existing, append new)
                merged = self._merge_class_mapping(
                    manifest.get("class_to_id", {}), self._extract_labels_from_source()
                )
                manifest["class_to_id"] = merged
                manifest["classes"] = [k for k, _ in sorted(merged.items(), key=lambda x: x[1])]
                self._prepared_dataset_dir = manifest["dataset_dir"]
                self._prepared_yaml_path = manifest["yaml_path"]
                self._update_dataset_status(manifest)
                self._force_rebuild_dataset = False
                self.append_training_log(self.tr("Reusing prepared dataset."))
                return True

            self.append_training_log(self.tr("Preparing dataset..."))
            final_dir = create_yolo_dataset(
                self.image_list, self.selected_task_type, ratio,
                "",  # Empty data_file → force auto-extraction from JSON (not coco8)
                self.output_dir,
                config["basic"].get("pose_config"),
                skip_empty_files=True,
                only_checked_files=config["checkpoint"].get("only_checked_files", False),
            )

            yaml_path = os.path.join(final_dir, "data.yaml") if self.selected_task_type != "Classify" else final_dir
            self._prepared_dataset_dir = final_dir
            self._prepared_yaml_path = yaml_path
            self._force_rebuild_dataset = False

            manifest = self._save_manifest(final_dir, yaml_path, fp, ratio)

            # ── Hard validation: refuse coco8 / stale outputs ──
            if not self._validate_prepared_output(final_dir, yaml_path, manifest):
                # Validation failed — clean up and return
                self._prepared_dataset_dir = None
                self._prepared_yaml_path = None
                self._update_dataset_status(None)
                return False

            self._update_dataset_status(manifest)
            self.append_training_log(self.tr(f"Dataset prepared: {yaml_path}"))
            return True
        except Exception as e:
            self._prepared_dataset_dir = None
            self._prepared_yaml_path = None
            self._update_dataset_status(None)
            self.append_training_log(f"Dataset preparation failed: {e}")
            QMessageBox.critical(
                self, self.tr("Dataset Preparation Failed"),
                self.tr(f"Failed to prepare dataset:\n{e}"),
            )
            return False
        finally:
            self._dataset_preparing = False
            if hasattr(self, 'start_training_button'):
                self.start_training_button.setEnabled(True)
            if hasattr(self, 'run_check_button'):
                self.run_check_button.setEnabled(True)

    # ── Fingerprint ──────────────────────────────────────────────────

    def _compute_source_fingerprint(self) -> str:
        """SHA256 of sorted image paths, sizes, mtimes, JSON content hashes, task, ratio, seed."""
        import hashlib
        h = hashlib.sha256()
        h.update((self.selected_task_type or "").encode())
        config = self.get_current_config()
        ratio = config["basic"].get("dataset_ratio", 0.8)
        h.update(f":ratio={ratio}:seed={self._SPLIT_SEED}:cv={self._CONVERTER_VERSION}:".encode())
        for img in sorted(self.image_list):
            rel = os.path.relpath(img, self.output_dir) if self.output_dir else img
            h.update(rel.encode())
            try:
                st = os.stat(img)
                h.update(f":{st.st_size}:{st.st_mtime_ns}".encode())
            except OSError:
                h.update(b":missing")
            json_path = os.path.splitext(img)[0] + ".json"
            if os.path.isfile(json_path):
                try:
                    h.update(open(json_path, "rb").read())
                except OSError:
                    h.update(b":json_unreadable")
            else:
                h.update(b":no_json")
        return h.hexdigest()[:24]

    # ── Label extraction ─────────────────────────────────────────────

    def _extract_labels_from_source(self) -> list:
        """Extract sorted unique labels from all source JSON annotations."""
        labels = set()
        for img in self.image_list:
            json_path = os.path.splitext(img)[0] + ".json"
            if not os.path.isfile(json_path):
                continue
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    jd = json.load(f)
                for s in jd.get("shapes", []):
                    lbl = s.get("label")
                    if lbl and str(lbl).strip():
                        labels.add(str(lbl).strip())
            except Exception:
                pass
        return sorted(labels)

    # ── Manifest I/O ─────────────────────────────────────────────────

    def _find_cached_manifest(self, fingerprint: str, ratio: float):
        """Find a cached manifest matching fingerprint, ratio, and converter_version."""
        try:
            from anylabeling.services.auto_training.ultralytics.config import get_dataset_path
            base = get_dataset_path()
            task_dir = os.path.join(base, (self.selected_task_type or "detect").lower())
            if not os.path.isdir(task_dir):
                return None
            for entry in os.listdir(task_dir):
                mf = os.path.join(task_dir, entry, "dataset_manifest.json")
                if os.path.isfile(mf):
                    try:
                        with open(mf, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        if (data.get("manifest_version") == self._MANIFEST_VERSION
                                and data.get("converter_version") == self._CONVERTER_VERSION
                                and data.get("fingerprint") == fingerprint
                                and abs(data.get("train_ratio", data.get("split_ratio", 0.8)) - ratio) < 0.001
                                and data.get("task_type") == (self.selected_task_type or "detect").lower()):
                            return data
                    except Exception:
                        pass
        except Exception:
            pass
        return None

    def _save_manifest(self, dataset_dir: str, yaml_path: str,
                       fingerprint: str, ratio: float) -> dict:
        """Save dataset_manifest.json."""
        labels = self._extract_labels_from_source()
        class_to_id = {lbl: i for i, lbl in enumerate(labels)}
        # Count actual valid labeled images (not total loaded images)
        try:
            from anylabeling.services.auto_training.ultralytics.validators import get_task_valid_images
            valid_count = get_task_valid_images(
                self.image_list, self.selected_task_type or "detect", self.output_dir
            )
        except Exception:
            valid_count = len(self.image_list)
        # Count actual train/val images from output
        train_count = 0
        val_count = 0
        for split_name, counter in [("train", "train_count"), ("val", "val_count")]:
            img_dir = os.path.join(dataset_dir, "images", split_name)
            if os.path.isdir(img_dir):
                count = len([
                    f for f in os.listdir(img_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
                ])
                if split_name == "train":
                    train_count = count
                else:
                    val_count = count
        manifest = {
            "manifest_version": self._MANIFEST_VERSION,
            "converter_version": self._CONVERTER_VERSION,
            "dataset_dir": dataset_dir,
            "yaml_path": yaml_path,
            "source_fingerprint": fingerprint,
            "fingerprint": fingerprint,
            "train_ratio": ratio,
            "split_seed": self._SPLIT_SEED,
            "task_type": (self.selected_task_type or "detect").lower(),
            "source_dir": str(self.output_dir) if self.output_dir else os.path.dirname(self.image_list[0]) if self.image_list else "",
            "classes": labels,
            "class_to_id": class_to_id,
            "valid_images": valid_count,
            "train_count": train_count,
            "val_count": val_count,
            "created_at": datetime.datetime.now().isoformat(),
        }
        mf_path = os.path.join(dataset_dir, "dataset_manifest.json")
        with open(mf_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        return manifest

    # ── Output validation ─────────────────────────────────────────────

    def _validate_prepared_output(self, dataset_dir: str, yaml_path: str,
                                  manifest: dict) -> bool:
        """Hard validation: refuse coco8 / stale / inconsistent outputs.

        Returns True if output passes all checks.
        """
        import re

        dir_basename = os.path.basename(dataset_dir)

        # 1. Reject coco8 directory names
        if re.match(r'^coco\d+[_\-]', dir_basename):
            self.append_training_log(
                self.tr(f"ERROR: Prepared dataset dir is stale coco: {dir_basename}")
            )
            return False

        # 2. Manifest must exist
        mf_path = os.path.join(dataset_dir, "dataset_manifest.json")
        if not os.path.isfile(mf_path):
            self.append_training_log(
                self.tr("ERROR: dataset_manifest.json missing after preparation")
            )
            return False

        # 3. YAML must exist and be parseable
        if not os.path.isfile(yaml_path):
            self.append_training_log(
                self.tr(f"ERROR: YAML missing: {yaml_path}")
            )
            return False

        try:
            from anylabeling.services.auto_training.ultralytics._io import load_yaml_config
            yaml_data = load_yaml_config(yaml_path)
        except Exception as e:
            self.append_training_log(f"ERROR: Cannot parse YAML: {e}")
            return False

        if not yaml_data or not isinstance(yaml_data, dict):
            self.append_training_log(self.tr("ERROR: YAML is empty or invalid"))
            return False

        # 4. No stale coco markers
        if "download" in yaml_data:
            self.append_training_log(self.tr("ERROR: YAML contains stale 'download' key"))
            return False

        # 5. nc must match actual classes
        actual_nc = len(manifest.get("classes", []))
        yaml_nc = yaml_data.get("nc", 0)
        if yaml_nc != actual_nc:
            self.append_training_log(
                f"ERROR: YAML nc={yaml_nc} does not match actual classes={actual_nc}"
            )
            return False

        # 6. names dict must match manifest classes
        yaml_names = yaml_data.get("names", {})
        if isinstance(yaml_names, dict):
            manifest_classes = manifest.get("classes", [])
            yaml_class_names = [yaml_names[i] for i in sorted(yaml_names.keys())]
            if yaml_class_names != manifest_classes:
                self.append_training_log(
                    f"ERROR: YAML names does not match manifest classes"
                )
                return False

            # 7. YAML class IDs must be 0..nc-1
            expected_ids = set(range(actual_nc))
            actual_ids = set(yaml_names.keys())
            if actual_ids != expected_ids:
                self.append_training_log(
                    f"ERROR: YAML class IDs {sorted(actual_ids)} != expected {sorted(expected_ids)}"
                )
                return False

        # 8. Label TXT class IDs must be in range 0..nc-1
        for split in ("train", "val"):
            lbl_dir = os.path.join(dataset_dir, "labels", split)
            if not os.path.isdir(lbl_dir):
                continue
            for f in os.listdir(lbl_dir):
                fp = os.path.join(lbl_dir, f)
                if os.path.getsize(fp) == 0:
                    continue
                try:
                    with open(fp, "r") as fh:
                        for line in fh:
                            parts = line.strip().split()
                            if parts:
                                cid = int(float(parts[0]))
                                if cid < 0 or cid >= actual_nc:
                                    self.append_training_log(
                                        f"ERROR: Label {f} has out-of-range class ID {cid}"
                                    )
                                    return False
                except Exception:
                    pass

        # 9. Valid image count must match actual exported images
        expected_valid = manifest.get("valid_images", 0)
        actual_train = 0
        actual_val = 0
        for split in ("train", "val"):
            img_dir = os.path.join(dataset_dir, "images", split)
            if os.path.isdir(img_dir):
                count = len([
                    f for f in os.listdir(img_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
                ])
                if split == "train":
                    actual_train = count
                else:
                    actual_val = count
        actual_images = actual_train + actual_val
        if actual_images != expected_valid:
            self.append_training_log(
                f"ERROR: Exported images ({actual_images}) != valid count ({expected_valid})"
            )
            return False

        # 10. Train/val split hard check (strict: exact match)
        ratio = manifest.get("train_ratio", manifest.get("split_ratio", 0.8))
        expected_train = round(expected_valid * ratio)
        expected_val = expected_valid - expected_train
        if actual_train != expected_train:
            self.append_training_log(
                f"ERROR: train count ({actual_train}) != expected ({expected_train}) "
                f"for train_ratio={ratio}, valid={expected_valid}"
            )
            return False
        if actual_val != expected_val:
            self.append_training_log(
                f"ERROR: val count ({actual_val}) != expected ({expected_val})"
            )
            return False
        # Also enforce train > val when ratio > 0.5
        if ratio > 0.5 and actual_train <= actual_val:
            self.append_training_log(
                f"ERROR: train_ratio={ratio} but train ({actual_train}) <= val ({actual_val})"
            )
            return False

        # Update manifest with actual counts if they differ
        manifest["train_count"] = actual_train
        manifest["val_count"] = actual_val

        return True

    # ── Integrity validation ─────────────────────────────────────────

    def _validate_manifest_integrity(self, manifest: dict) -> bool:
        """Validate cached dataset is complete before reuse. Returns True if valid."""
        try:
            ds = manifest.get("dataset_dir", "")
            yaml_path = manifest.get("yaml_path", "")
            if not ds or not os.path.isdir(ds):
                return False
            if not yaml_path or not os.path.isfile(yaml_path):
                return False

            # Validate YAML
            from anylabeling.services.auto_training.ultralytics._io import load_yaml_config
            yaml_data = load_yaml_config(yaml_path)
            if not yaml_data or not isinstance(yaml_data, dict):
                return False
            if "download" in yaml_data:  # stale coco8
                return False

            names = yaml_data.get("names", {})
            if not names:
                return False
            nc = yaml_data.get("nc", 0)
            if nc != len(names):
                return False

            # Check train/val paths exist with images
            for split in ("train", "val"):
                img_dir = os.path.join(ds, "images", split)
                lbl_dir = os.path.join(ds, "labels", split)
                if not os.path.isdir(img_dir):
                    return False
                images = set(os.path.splitext(f)[0] for f in os.listdir(img_dir)
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')))
                if not os.path.isdir(lbl_dir):
                    return False
                labels = set(os.path.splitext(f)[0] for f in os.listdir(lbl_dir)
                             if f.endswith('.txt') and os.path.getsize(os.path.join(lbl_dir, f)) > 0)
                if not images:
                    return False
                # Each image must have a label with content
                for stem in images:
                    if stem not in labels:
                        return False

            # Validate all class_ids in labels are in range
            for split in ("train", "val"):
                lbl_dir = os.path.join(ds, "labels", split)
                for f in os.listdir(lbl_dir):
                    fp = os.path.join(lbl_dir, f)
                    if os.path.getsize(fp) == 0:
                        continue
                    with open(fp, "r") as fh:
                        for line in fh:
                            parts = line.strip().split()
                            if parts:
                                try:
                                    cid = int(float(parts[0]))
                                    if cid < 0 or cid >= nc:
                                        return False
                                except ValueError:
                                    return False

            return True
        except Exception:
            return False

    def _merge_class_mapping(self, existing: dict, new_labels: list) -> dict:
        """Merge new labels into existing mapping, preserving existing IDs."""
        next_id = max(existing.values()) + 1 if existing else 0
        merged = dict(existing)  # preserve old IDs
        for lbl in new_labels:
            if lbl not in merged:
                merged[lbl] = next_id
                next_id += 1
        return merged

    # ── UI actions ───────────────────────────────────────────────────

    def _rebuild_dataset(self):
        """Force rebuild dataset on next prepare."""
        self._force_rebuild_dataset = True
        self._prepared_dataset_dir = None
        self._prepared_yaml_path = None
        self._update_dataset_status(None)
        self.append_training_log(self.tr("Will rebuild dataset on next Start."))
        if not self._prepare_dataset():
            self.append_training_log(self.tr("Dataset rebuild failed."))

    def _clear_dataset_cache(self):
        """Delete all cached datasets for current task type from disk."""
        try:
            from anylabeling.services.auto_training.ultralytics.config import get_dataset_path
            import shutil
            task_dir = os.path.join(get_dataset_path(), (self.selected_task_type or "detect").lower())
            if os.path.isdir(task_dir):
                shutil.rmtree(task_dir)
            self._prepared_dataset_dir = None
            self._prepared_yaml_path = None
            self._preflight_result = None
            self._force_rebuild_dataset = True
            # Clear config["basic"]["data"] if it points into the deleted cache
            config = self.get_current_config()
            data_val = config["basic"].get("data", "")
            if data_val:
                cache_root = os.path.abspath(get_dataset_path())
                try:
                    if os.path.abspath(data_val).startswith(cache_root):
                        if hasattr(self, 'config_widgets') and self.config_widgets and "data" in self.config_widgets:
                            self.config_widgets["data"].setText("")
                except Exception:
                    pass
            self._update_dataset_status(None)
            self.append_training_log(self.tr("Dataset cache cleared from disk."))
        except Exception as e:
            self.append_training_log(f"Failed to clear cache: {e}")

    def _update_dataset_status(self, manifest: dict | None):
        """Update the dataset status label on the Train page."""
        if not hasattr(self, 'dataset_status_label'):
            return
        if manifest is None:
            self.dataset_status_label.setText(self.tr("Dataset: Not prepared"))
            self.dataset_status_label.setStyleSheet("color: gray;")
        else:
            classes = manifest.get("classes", [])
            self.dataset_status_label.setText(
                self.tr(f"Dataset: Ready | Classes: {len(classes)} | "
                        f"Valid: {manifest.get('valid_images', '?')}")
            )
            self.dataset_status_label.setStyleSheet("color: green; font-weight: bold;")

    def _run_full_preflight(self):
        """Run preflight background check and show result dialog."""
        if self._preflight_running:
            return  # Prevent double-click

        # Ensure dataset is prepared (may have been cleared)
        if not self._prepared_dataset_dir:
            if not self._prepare_dataset():
                return  # Preparation failed — don't run preflight

        ctx = self._build_guided_preflight_context()
        is_active = self.job_manager.get_current_job() is not None

        self._preflight_running = True
        self._preflight_result = None

        # Create worker + thread
        self._preflight_worker = PreflightWorker()
        self._preflight_thread = QThread()
        self._preflight_worker.moveToThread(self._preflight_thread)

        # Dialog (shown immediately in "checking" mode)
        from anylabeling.services.training_center.preflight import PreflightResult
        dialog = PreflightDialog(PreflightResult(mode="guided"), self)
        dialog.set_checking(True)

        # Wire signals
        self._preflight_worker.progress.connect(dialog.set_progress)
        self._preflight_worker.finished.connect(
            lambda result: self._on_preflight_finished(result, dialog)
        )
        self._preflight_thread.started.connect(
            lambda: self._preflight_worker.run_guided(ctx, is_active)
        )

        # Internal lifecycle
        self._preflight_worker.finished.connect(self._preflight_thread.quit)
        self._preflight_thread.finished.connect(self._preflight_thread.deleteLater)

        self._preflight_thread.start()
        result = dialog.exec()

        # If user closed/cancelled, invalidate result
        if result != QDialog.DialogCode.Accepted:
            self._preflight_result = None
            self._preflight_running = False
        if self._preflight_worker:
            self._preflight_worker.cancel()

    def _on_preflight_finished(self, result, dialog):
        """Handle preflight completion — update dialog and store result."""
        self._preflight_running = False
        self._preflight_result = result
        dialog.update_result(result)

        # Re-enable start button state
        if hasattr(self, 'start_training_button'):
            self.start_training_button.setEnabled(True)

    def start_training_from_train_tab(self, skip_preflight: bool = False):
        # Ensure config and train tabs are initialized
        self.ensure_config_tab_initialized()
        self.ensure_train_tab_initialized()

        if self._preflight_running:
            return

        # Gate: dataset preparation already in progress
        if self._dataset_preparing:
            return

        # Gate: must have passed Data Check
        if not self._data_check_passed:
            QMessageBox.warning(
                self, self.tr("Check Required"),
                self.tr("Please complete 'Check Dataset' on the Data page first."),
            )
            return

        # Gate: config must be completed
        if not self._config_completed:
            QMessageBox.warning(
                self, self.tr("Config Required"),
                self.tr("Please complete the Config page first."),
            )
            return

        # Step 1: Prepare dataset from loaded images (before preflight)
        if not self._prepared_dataset_dir:
            if not self._prepare_dataset():
                return  # Preparation failed

        # Step 2: Run full preflight
        if not skip_preflight:
            self._run_full_preflight()
            if self._preflight_result is None:
                return  # Dialog cancelled
            if self._preflight_result.has_errors:
                return  # Cannot start with errors
            if self._preflight_result.has_warnings and not self._preflight_result.can_start:
                return  # User didn't confirm
        
        # Fall through to original start logic
        # Ensure config and train tabs are initialized
        self.ensure_config_tab_initialized()
        self.ensure_train_tab_initialized()

        # ── Read UI config (GUI thread, fast) ──
        try:
            config = self.get_current_config()
        except Exception as e:
            QMessageBox.critical(self, self.tr("Config Error"), str(e))
            return

        project_path = config["basic"]["project"]
        name = config["basic"]["name"]
        self.current_project_path = os.path.join(project_path, name)

        # ── Phase 1: Reserve job manager slot atomically ──
        job_id = f"guided_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        # Use prepared YAML as the actual dataset (not config's coco8.yaml)
        actual_data = self._prepared_yaml_path or config["basic"].get("data", "")
        # Normalize task display name
        task_display = (self.selected_task_type or "").strip().capitalize()

        # ── Resolve GPU runtime (single source of truth: device combo) ──
        dev_info = self._get_selected_training_device()
        runtime_id = dev_info["runtime_id"]
        runtime_python = dev_info["runtime_python"]
        requested_device = dev_info["requested_device"]
        resolved_device = dev_info["resolved_device"]
        device_name = dev_info["device_name"]
        execution_mode = dev_info["execution_mode"]

        # Override execution_mode from UI selector (Local/Remote SSH)
        if "execution_mode" in self.config_widgets:
            ui_mode = self.config_widgets["execution_mode"].currentData()
            if ui_mode and ui_mode == "remote_ssh":
                execution_mode = "remote_ssh"

        # Remote mode: validate profile (profile may stay None for local)
        remote_profile = None
        if execution_mode == "remote_ssh":
            profile_widget = getattr(self, 'remote_profile_widget', None)
            if not profile_widget:
                QMessageBox.critical(self, self.tr("Remote Not Configured"),
                    self.tr("Please configure a remote SSH profile first."))
                return
            remote_profile = profile_widget.get_profile()
            if not remote_profile:
                QMessageBox.critical(self, self.tr("No Profile Selected"),
                    self.tr("Please select or create a remote SSH profile."))
                return
            if not remote_profile.host:
                QMessageBox.critical(self, self.tr("Profile Incomplete"),
                    self.tr("Remote host is not configured."))
                return
            # Store remote info on job
            runtime_id = remote_profile.profile_id
            runtime_python = remote_profile.remote_python
            device_name = remote_profile.host
            self.append_training_log(
                f"Execution mode: Remote SSH\n"
                f"Remote host: {remote_profile.host}:{remote_profile.port}\n"
                f"Remote Python: {remote_profile.remote_python}\n"
                f"Remote workspace: {remote_profile.remote_workspace}"
            )

        # Hard validation: GPU tasks MUST have a runtime Python
        if requested_device.startswith("cuda") or (resolved_device and resolved_device not in ("cpu", "")):
            self.append_training_log(
                f"Selected device: {requested_device}\n"
                f"Selected runtime id: {runtime_id or '(none)'}\n"
                f"Selected runtime python: {runtime_python or '(none)'}"
            )
            if not runtime_python:
                QMessageBox.critical(
                    self, self.tr("GPU Runtime Missing"),
                    self.tr("Selected GPU runtime is not bound to this device.\n\n"
                            "Device: {device}\n\n"
                            "Possible causes:\n"
                            "• No CUDA-enabled Python environment registered\n"
                            "• Registered environment is not in 'ready' state\n"
                            "• Run 'Scan Environments' to detect CUDA installations\n\n"
                            "Falling back to CPU training is NOT allowed for GPU selection.").format(device=requested_device)
                )
                return
            if not os.path.isfile(runtime_python):
                QMessageBox.critical(
                    self, self.tr("Runtime Python Missing"),
                    self.tr(
                        "Selected GPU runtime Python does not exist:\n\n"
                        "{path}\n\n"
                        "The environment may have been moved or uninstalled.\n"
                        "Please re-scan environments and select a valid CUDA Python."
                    ).format(path=runtime_python)
                )
                return

            # Log the training command info
            self.append_training_log(
                f"Worker Python: {runtime_python}\n"
                f"Training arguments: device={resolved_device}"
            )
        else:
            # CPU or Auto without external runtime — safe to use GUI Python
            from anylabeling.services.training_center.device_service import resolve_training_device
            resolved_device = resolve_training_device(requested_device)
            # Preserve remote_ssh if already set (don't overwrite to local)
            if execution_mode != "remote_ssh":
                execution_mode = "local"

        self.current_job = TrainingJob(
            job_id=job_id,
            mode=TrainingMode.GUIDED_ULTRALYTICS,
            status=TrainingStatus.IDLE,
            created_at=datetime.datetime.now(),
            started_at=None,
            ended_at=None,
            workspace=project_path,
            output_directory=Path(self.current_project_path),
            display_name=f"Guided: {name}",
            framework="ultralytics",
            command=[],
            metadata={},
            error_message=None,
            task=task_display,
            model=config["basic"].get("model", ""),
            data=actual_data,
            project=project_path,
            name=name,
            runtime_id=runtime_id,
            runtime_python=runtime_python,
            requested_device=requested_device,
            resolved_device=resolved_device,
            execution_mode=execution_mode or "local",
            # Remote SSH fields
            remote_profile_id=remote_profile.profile_id if remote_profile else None,
            remote_host=remote_profile.host if remote_profile else None,
            remote_port=remote_profile.port if remote_profile else None,
            remote_username=remote_profile.username if remote_profile else None,
            remote_workspace=remote_profile.remote_workspace if remote_profile else None,
            remote_python=remote_profile.remote_python if remote_profile else None,
            # Session password (memory only, never persisted)
            _session_password=(
                profile_widget.get_session_password()
                if (remote_profile and remote_profile.auth_method.value == "password")
                else None
            ),
        )

        # ── Hard assertion: execution_mode must be preserved ──
        selected_mode = self.config_widgets["execution_mode"].currentData() if "execution_mode" in self.config_widgets else "local"
        job_mode = self.current_job.execution_mode or "local"
        self.append_training_log(
            f"Guided selection mode: {selected_mode!r}\n"
            f"Job execution mode before reserve: {job_mode!r}"
        )
        if selected_mode != job_mode:
            QMessageBox.critical(
                self, self.tr("Internal Error"),
                self.tr(
                    "Execution mode lost while creating job:\n"
                    f"Selected: {selected_mode!r}\n"
                    f"Job: {job_mode!r}"
                )
            )
            self.current_job = None
            return

        # Remote mode: validate required fields
        if job_mode == "remote_ssh":
            required_fields = {
                "remote_profile_id": self.current_job.remote_profile_id,
                "remote_host": self.current_job.remote_host,
                "remote_username": self.current_job.remote_username,
                "remote_workspace": self.current_job.remote_workspace,
                "remote_python": self.current_job.remote_python,
            }
            missing = [k for k, v in required_fields.items() if not v]
            if missing:
                QMessageBox.critical(
                    self, self.tr("Remote Job Incomplete"),
                    self.tr(f"Remote job is missing fields: {', '.join(missing)}")
                )
                self.current_job = None
                return

        adapter = UltralyticsAdapter()
        ok, msg = self.job_manager.reserve_job(self.current_job, adapter)
        if not ok:
            self.append_training_log(f"Failed to reserve job: {msg}")
            QMessageBox.critical(self, self.tr("Training Busy"), msg)
            self.current_job = None
            return

        # Store for prep thread / late signal validation
        self._prep_job_id = job_id
        self._current_job_id = job_id  # for event handler guard
        self._prep_adapter = adapter

        # ── UI: preparing state ──
        self.training_status = "preparing"
        self.update_training_status_display()
        if execution_mode == "remote_ssh":
            self.append_training_log(self.tr("Preparing remote job..."))
        else:
            self.append_training_log(self.tr("Preparing training..."))
        self.start_training_button.setEnabled(False)
        self.stop_training_button.setVisible(True)
        self.export_button.setVisible(False)
        self.previous_button.setVisible(False)

        # ── Phase 2: Run dataset creation in background thread ──
        self._prep_thread = QThread(self)
        self._prep_worker = _TrainingPrepWorker(self, config)
        self._prep_worker.moveToThread(self._prep_thread)

        self._prep_thread.started.connect(self._prep_worker.run)
        self._prep_worker.finished.connect(self._on_prep_finished)
        self._prep_worker.error.connect(self._on_prep_error)
        self._prep_thread.finished.connect(self._prep_thread.deleteLater)

        # Store config for adapter creation later
        self._pending_project_path = project_path
        self._pending_name = name

        self._prep_thread.start()

    def _on_prep_finished(self, train_args):
        """Called on GUI thread after background preparation completes.

        Validates that the job is still in PREPARING state (handles late
        signals from cancelled threads) before starting via JobManager.
        """
        self._prep_thread.quit()
        self._prep_thread.wait()
        self._prep_thread = None
        self._prep_worker = None

        # Late-signal guard: only proceed if still reserved
        if self._prep_job_id is None:
            return

        current = self.job_manager.get_current_job()
        if current is None or current.job_id != self._prep_job_id:
            self._prep_job_id = None
            return
        if current.status != TrainingStatus.PREPARING:
            self._prep_job_id = None
            return

        success, message = self.job_manager.start_reserved_job(
            job_id=self._prep_job_id,
            config=train_args,
        )
        self._prep_job_id = None

        if not success:
            self.append_training_log(f"Failed to start training: {message}")
            QMessageBox.critical(self, self.tr("Training Error"), message)
            self._reset_start_ui()
            return

        # Guard: only show waiting if job is still active (not already FAILED)
        current = self.job_manager.get_current_job()
        if current and current.status.is_terminal():
            return

        mode = getattr(self.current_job, 'execution_mode', 'local') or 'local'
        if mode == 'remote_ssh':
            self.append_training_log(self.tr("Remote worker process started, waiting for ready..."))
        else:
            self.append_training_log(self.tr("Job reserved, waiting for worker..."))

    def _on_prep_error(self, error_msg):
        """Called on GUI thread when background preparation fails."""
        self._prep_thread.quit()
        self._prep_thread.wait()
        self._prep_thread = None
        self._prep_worker = None

        # Late-signal guard
        if self._prep_job_id is None:
            return

        current = self.job_manager.get_current_job()
        if current is not None and current.job_id == self._prep_job_id:
            self.job_manager.fail_reserved_job(self._prep_job_id, error_msg)
        self._prep_job_id = None

        self.append_training_log(f"ERROR: {error_msg}")
        QMessageBox.critical(self, self.tr("Training Error"), error_msg)
        self._reset_start_ui()

    def _reset_start_ui(self):
        """Restore Start button state after preparation/startup failure."""
        self.training_status = "idle"
        self.update_training_status_display()
        self.start_training_button.setEnabled(True)
        self.start_training_button.setVisible(True)
        self.stop_training_button.setVisible(False)
        self.export_button.setVisible(False)
        self.previous_button.setVisible(True)

    def init_training_actions(self, parent_layout):
        actions_layout = QHBoxLayout()

        self.open_dir_button = SecondaryButton(self.tr("Open Directory"))
        self.open_dir_button.clicked.connect(self.open_training_directory)
        actions_layout.addWidget(self.open_dir_button)

        self.run_check_button = SecondaryButton(self.tr("Run Check"))
        self.run_check_button.clicked.connect(self._run_full_preflight)
        actions_layout.addWidget(self.run_check_button)

        self.rebuild_dataset_button = SecondaryButton(self.tr("Rebuild Dataset"))
        self.rebuild_dataset_button.clicked.connect(self._rebuild_dataset)
        actions_layout.addWidget(self.rebuild_dataset_button)

        self.clear_cache_button = SecondaryButton(self.tr("Clear Cache"))
        self.clear_cache_button.clicked.connect(self._clear_dataset_cache)
        actions_layout.addWidget(self.clear_cache_button)

        actions_layout.addStretch()

        self.stop_training_button = SecondaryButton(self.tr("Stop Training"))
        self.stop_training_button.clicked.connect(self.stop_training)
        self.stop_training_button.setVisible(False)
        actions_layout.addWidget(self.stop_training_button)

        self.previous_button = SecondaryButton(self.tr("Previous"))
        self.previous_button.clicked.connect(
            lambda: self.go_to_specific_tab(1)
        )
        self.previous_button.setVisible(True)
        actions_layout.addWidget(self.previous_button)

        self.start_training_button = PrimaryButton(self.tr("Start Training"))
        self.start_training_button.clicked.connect(
            self.start_training_from_train_tab
        )
        actions_layout.addWidget(self.start_training_button)

        self.export_button = PrimaryButton(self.tr("Export"))
        self.export_button.clicked.connect(self.start_export)
        self.export_button.setVisible(False)
        actions_layout.addWidget(self.export_button)

        parent_layout.addLayout(actions_layout)

    def init_train_tab(self):
        layout = QVBoxLayout(self.train_tab)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        self.init_training_status(scroll_layout)
        self.init_training_logs(scroll_layout)
        self.init_training_images(scroll_layout)

        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)

        self.init_training_actions(layout)

        # Show dataset samples immediately if prepared
        self._display_dataset_samples()

    def on_export_event(self, event_type, data):
        if event_type == "export_started":
            self.append_training_log(self.tr("Export started..."))
            self.export_button.setEnabled(False)
        elif event_type == "export_completed":
            exported_path = data.get("exported_path", "")
            export_format = data.get("format", "onnx")
            self.append_training_log(
                self.tr(
                    f"Export completed successfully! File saved to: {exported_path}"
                )
            )
            QMessageBox.information(
                self,
                self.tr("Export Successful"),
                self.tr(
                    f"Model successfully exported to {export_format.upper()} format:\n{exported_path}"
                ),
            )
            self.export_button.setEnabled(True)
        elif event_type == "export_error":
            error_msg = data.get("error", "Unknown error occurred")
            self.append_training_log(f"ERROR: {error_msg}")
            QMessageBox.warning(self, self.tr("Export Error"), error_msg)
            self.export_button.setEnabled(True)
        elif event_type == "export_log":
            log_message = data.get("message", "")
            if log_message:
                self.append_training_log(log_message)

    def start_export(self):
        if not self.current_project_path:
            QMessageBox.warning(
                self,
                self.tr("Error"),
                self.tr("No training project available for export"),
            )
            return

        weights_path = os.path.join(
            self.current_project_path, "weights", "best.pt"
        )
        if not os.path.exists(weights_path):
            QMessageBox.warning(
                self,
                self.tr("Model Not Found"),
                self.tr(f"Model weights not found at: {weights_path}"),
            )
            return

        export_dialog = ExportFormatDialog(self)
        if export_dialog.exec() == QDialog.DialogCode.Accepted:
            export_format = export_dialog.get_selected_format()
            success, message = self.export_manager.start_export(
                self.current_project_path, export_format
            )
            if not success:
                QMessageBox.critical(self, self.tr("Export Error"), message)
                self.append_training_log(f"Failed to start export: {message}")

    def reset_train_tab(self):
        self.training_status = "idle"
        self.current_project_path = None
        self.current_epochs = 0
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("0/0")
        self.update_training_status_display()

        if hasattr(self, "log_display"):
            self.log_display.clear()

        for i, image_label in enumerate(self.image_labels):
            image_label.clear()
            image_label.setText(self.tr("No image"))
            image_label.setToolTip("")
            self.image_paths[i] = None

        self.previous_button.setVisible(True)
        self.start_training_button.setVisible(True)
        self.export_button.setVisible(False)
        self.stop_training_button.setVisible(False)
