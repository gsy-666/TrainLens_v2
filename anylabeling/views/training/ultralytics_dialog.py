import csv
import datetime
import glob
import os
import platform
import re
import shutil
import subprocess

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QMessageBox,
)

from anylabeling.views.training.guided_training_widget import GuidedTrainingWidget
from anylabeling.services.auto_training.ultralytics.config import (
    DEFAULT_WINDOW_TITLE,
    DEFAULT_WINDOW_SIZE,
)
from anylabeling.services.auto_training.ultralytics.style import *


class UltralyticsDialog(QDialog):
    """Thin wrapper providing QDialog interface for GuidedTrainingWidget"""

    def __init__(
        self,
        parent=None,
        job_manager=None,
        ultralytics_adapter=None,
        history_store=None
    ):
        super().__init__(parent)

        # Window setup
        self.setWindowTitle(DEFAULT_WINDOW_TITLE)
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        self.resize(*DEFAULT_WINDOW_SIZE)
        self.setMinimumSize(*DEFAULT_WINDOW_SIZE)

        # Extract callbacks from parent (typically LabelWidget)
        host = parent
        open_folder_callback = None
        get_image_list_callback = None

        if host and hasattr(host, 'open_folder_dialog'):
            open_folder_callback = host.open_folder_dialog

        if host and hasattr(host, 'image_list'):
            # image_list is a property, wrap in lambda
            get_image_list_callback = lambda: host.image_list

        # Create the training widget with dependency injection
        self.training_widget = GuidedTrainingWidget(
            parent=parent,
            job_manager=job_manager,
            ultralytics_adapter=ultralytics_adapter,
            history_store=history_store,
            open_folder_callback=open_folder_callback,
            get_image_list_callback=get_image_list_callback
        )

        # Layout with no margins
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.training_widget)

        self.setStyleSheet(get_ultralytics_dialog_style())

    def closeEvent(self, event):
        """Handle window close with training check"""
        if self.training_widget.has_active_training():
            QMessageBox.warning(
                self,
                self.tr("Training in Progress"),
                self.tr(
                    "Cannot close window while training is in progress. Please stop training first."
                ),
            )
            event.ignore()
            return

        # Shutdown the widget
        self.training_widget.shutdown()
        super().closeEvent(event)

    # Proxy properties for backward compatibility
    @property
    def training_manager(self):
        return self.training_widget.training_manager

    @property
    def export_manager(self):
        return self.training_widget.export_manager

    @property
    def image_list(self):
        return self.training_widget.image_list

    @image_list.setter
    def image_list(self, value):
        self.training_widget.image_list = value

    @property
    def output_dir(self):
        return self.training_widget.output_dir

    @output_dir.setter
    def output_dir(self, value):
        self.training_widget.output_dir = value
