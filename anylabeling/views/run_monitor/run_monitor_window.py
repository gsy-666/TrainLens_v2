"""Run Monitor Main Window

Thin wrapper around RunMonitorWidget for standalone usage.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QMessageBox

from .run_monitor_widget import RunMonitorWidget


class RunMonitorWindow(QDialog):
    """Standalone Run Monitor window

    Wraps RunMonitorWidget with window chrome.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Run Monitor")
        self.resize(1200, 800)
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowCloseButtonHint
        )

        # Create widget
        self.widget = RunMonitorWidget(self)

        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(self.widget)

    def closeEvent(self, event):
        """Handle window close"""
        if self.widget.is_running():
            reply = QMessageBox.question(
                self,
                "Training in Progress",
                "Training is still running. Stop training before closing?",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No
                | QMessageBox.StandardButton.Cancel,
            )

            if reply == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
            elif reply == QMessageBox.StandardButton.Yes:
                self.widget.cleanup()

        self.widget.cleanup()
        event.accept()
