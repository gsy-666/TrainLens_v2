"""Unified Training Center Window

Combines Guided Training, Custom Project (Run Monitor), and History
into a single non-modal window with shared JobManager and HistoryStore.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from anylabeling.services.training_center.job_manager import get_job_manager
from anylabeling.services.training_center.history import get_history_store
from anylabeling.services.training_center.models import TrainingStatus
from anylabeling.views.training.guided_training_widget import GuidedTrainingWidget
from anylabeling.views.run_monitor.run_monitor_widget import RunMonitorWidget
from anylabeling.views.training.training_history_widget import TrainingHistoryWidget


# Module-level singleton for the TrainingCenterWindow
_training_center_window = None

# Tab indices
TAB_GUIDED = 0
TAB_CUSTOM = 1
TAB_HISTORY = 2

TAB_NAMES = {
    "guided": TAB_GUIDED,
    "custom": TAB_CUSTOM,
    "history": TAB_HISTORY,
}


def open_training_center(parent=None, tab="guided", open_folder_callback=None, image_list_getter=None):
    """Open or raise the unified TrainingCenterWindow.

    Args:
        parent: Parent widget (typically the LabelingWidget)
        tab: Initial tab to show ("guided", "custom", "history")
        open_folder_callback: Callable for Load Images button
        image_list_getter: Callable returning current image list

    Returns:
        TrainingCenterWindow instance
    """
    global _training_center_window

    # Check if existing window is still valid
    if _training_center_window is not None:
        try:
            _training_center_window.isVisible()
        except RuntimeError:
            _training_center_window = None

    if _training_center_window is None:
        _training_center_window = TrainingCenterWindow(
            parent=parent,
            open_folder_callback=open_folder_callback,
            image_list_getter=image_list_getter,
        )
    else:
        # Update callbacks on existing window (host may have changed)
        if open_folder_callback is not None:
            _training_center_window.guided_widget._open_folder_callback = open_folder_callback
        if image_list_getter is not None:
            _training_center_window.guided_widget._image_list_getter = image_list_getter
            # Re-sync images when reusing existing window
            _training_center_window.guided_widget.sync_image_list_from_host()

    # Switch to requested tab
    tab_idx = TAB_NAMES.get(tab, TAB_GUIDED)
    _training_center_window.tab_widget.setCurrentIndex(tab_idx)

    # Auto-sync when opening Guided tab
    if tab_idx == TAB_GUIDED:
        _training_center_window.guided_widget.sync_image_list_from_host()

    _training_center_window.show()
    _training_center_window.raise_()
    _training_center_window.activateWindow()

    return _training_center_window


class TrainingCenterWindow(QMainWindow):
    """Unified training center with Guided, Custom, and History tabs."""

    def __init__(self, parent=None, open_folder_callback=None, image_list_getter=None):
        super().__init__(parent)

        # Shared services
        self.job_manager = get_job_manager()
        self.history_store = get_history_store()

        # Window setup
        self.setWindowTitle("TrainLens Training Center")
        self.setMinimumSize(1200, 760)
        self.resize(1200, 760)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Status bar at top
        self._create_status_bar(main_layout)

        # Tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.currentChanged.connect(self._on_tab_changed)

        # Tab 0: Guided Training — inject host callbacks
        self.guided_widget = GuidedTrainingWidget(
            parent=None,
            image_list=[],
            output_dir="",
            supported_shape=["rectangle", "polygon"],
            open_folder_callback=open_folder_callback,
            image_list_getter=image_list_getter,
        )
        self.tab_widget.addTab(self.guided_widget, "Guided Training")

        # Tab 1: Custom Project (Run Monitor)
        self.run_monitor_widget = RunMonitorWidget(parent=None)
        self.tab_widget.addTab(self.run_monitor_widget, "Custom Project")

        # Tab 2: History
        self.history_widget = TrainingHistoryWidget(parent=None)
        self.tab_widget.addTab(self.history_widget, "History")

        main_layout.addWidget(self.tab_widget, stretch=1)

        # Subscribe to job status for active job display
        self.job_manager.subscribe_status(self._update_status_bar)
        self.job_manager.subscribe_events(self._on_job_event)

        # Initial status
        self._update_status_bar_from_current()

    def _create_status_bar(self, parent_layout):
        """Create a thin status bar at the top."""
        self.status_frame = QFrame()
        self.status_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.status_frame.setStyleSheet(
            "QFrame { background-color: #f5f5f5; padding: 4px 8px; }"
        )

        status_layout = QHBoxLayout(self.status_frame)
        status_layout.setContentsMargins(8, 2, 8, 2)

        self.status_label = QLabel("Idle")
        self.status_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()

        parent_layout.addWidget(self.status_frame)

    def _update_status_bar_from_current(self):
        """Update status bar text from current job."""
        job = self.job_manager.get_current_job()
        if job is not None and job.status.is_active():
            name = job.display_name or job.job_id
            self.status_label.setText(f"Running: {name}")
            self.status_label.setStyleSheet(
                "font-weight: bold; font-size: 12px; color: #1976D2;"
            )
        else:
            self.status_label.setText("Idle")
            self.status_label.setStyleSheet(
                "font-weight: bold; font-size: 12px; color: #4CAF50;"
            )

    def _update_status_bar(self, job):
        """Update status bar when job status changes — uses passed job to avoid lock reentry."""
        if job is not None and job.status.is_active():
            name = job.display_name or job.job_id
            self.status_label.setText(f"Running: {name}")
            self.status_label.setStyleSheet(
                "font-weight: bold; font-size: 12px; color: #1976D2;"
            )
        else:
            self.status_label.setText("Idle")
            self.status_label.setStyleSheet(
                "font-weight: bold; font-size: 12px; color: #4CAF50;"
            )

    def _on_job_event(self, event):
        """Handle job events for History auto-refresh."""
        from anylabeling.services.training_center.event_protocol import TrainingEventType
        if event.event_type in (
            TrainingEventType.COMPLETED,
            TrainingEventType.FAILED,
            TrainingEventType.STOPPED,
        ):
            # Auto-refresh history when a terminal event arrives
            if hasattr(self, 'history_widget'):
                self.history_widget.refresh()

    def _on_tab_changed(self, index):
        """Handle tab changes — auto-sync images for Guided tab."""
        if index == TAB_GUIDED and hasattr(self, 'guided_widget'):
            self.guided_widget.sync_image_list_from_host()
        if index == TAB_HISTORY and hasattr(self, 'history_widget'):
            self.history_widget.refresh()

    def closeEvent(self, event):
        """Closing the window does NOT stop running tasks."""
        # Just hide - don't stop tasks, don't destroy shared services
        event.accept()

    def shutdown(self):
        """Clean shutdown: stop active job, unsubscribe, but keep singleton alive."""
        try:
            self.job_manager.unsubscribe_status(self._update_status_bar)
            self.job_manager.unsubscribe_events(self._on_job_event)
        except Exception:
            pass

        # Stop any active job
        try:
            current = self.job_manager.get_current_job()
            if current is not None and current.status.is_active():
                self.job_manager.request_stop()
        except Exception:
            pass

        # Shutdown sub-widgets
        for widget_name in ('guided_widget', 'run_monitor_widget'):
            widget = getattr(self, widget_name, None)
            if widget is not None and hasattr(widget, 'shutdown'):
                try:
                    widget.shutdown()
                except Exception:
                    pass
