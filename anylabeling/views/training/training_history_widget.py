"""Training History Widget

Displays training job history from the shared HistoryStore.
"""

import os
import subprocess
import sys
from datetime import datetime

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from anylabeling.services.training_center.history import get_history_store
from anylabeling.services.training_center.models import TrainingStatus
from anylabeling.services.training_center.event_protocol import TrainingEventType


# Status display helpers
STATUS_COLORS = {
    "completed": QColor(76, 175, 80),   # Green
    "failed": QColor(244, 67, 54),      # Red
    "stopped": QColor(255, 152, 0),     # Orange
    "running": QColor(33, 150, 243),    # Blue
    "preparing": QColor(156, 39, 176),  # Purple
    "stopping": QColor(255, 152, 0),    # Orange
    "idle": QColor(158, 158, 158),      # Grey
}

STATUS_LABELS = {
    "completed": "COMPLETED",
    "failed": "FAILED",
    "stopped": "STOPPED",
    "running": "Running...",
    "preparing": "Preparing...",
    "stopping": "Stopping...",
    "idle": "Idle",
}

MODE_LABELS = {
    "guided_ultralytics": "Guided",
    "custom_script": "Custom",
}


class TrainingHistoryWidget(QWidget):
    """Displays training job history from HistoryStore."""

    COLUMNS = ["Status", "Mode", "Name", "Task / Script", "Started At", "Duration", "Output"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.history_store = get_history_store()

        self._init_ui()
        self.refresh()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Top bar
        top_bar = QHBoxLayout()
        title = QLabel("Training History")
        title.setStyleSheet("font-size: 14px; font-weight: bold;")
        top_bar.addWidget(title)
        top_bar.addStretch()

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh)
        top_bar.addWidget(refresh_btn)
        layout.addLayout(top_bar)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.table, stretch=1)

        # Detail panel
        detail_layout = QHBoxLayout()

        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setMaximumHeight(120)
        self.detail_text.setPlaceholderText("Select a job to view details...")
        detail_layout.addWidget(self.detail_text)

        action_buttons = QVBoxLayout()
        self.open_dir_btn = QPushButton("Open Directory")
        self.open_dir_btn.clicked.connect(self._open_output_directory)
        self.open_dir_btn.setEnabled(False)
        action_buttons.addWidget(self.open_dir_btn)
        action_buttons.addStretch()
        detail_layout.addLayout(action_buttons)

        layout.addLayout(detail_layout)

    def refresh(self):
        """Reload history from store."""
        jobs = self.history_store.list_jobs()
        self.table.setRowCount(len(jobs))

        for row, job in enumerate(jobs):
            self._set_row(row, job)

        self.table.resizeColumnsToContents()

    def _set_row(self, row, job):
        # Status
        status_text = STATUS_LABELS.get(job.status, job.status.upper())
        status_item = QTableWidgetItem(status_text)
        color = STATUS_COLORS.get(job.status, QColor(158, 158, 158))
        status_item.setForeground(color)
        status_item.setData(Qt.ItemDataRole.UserRole, job.job_id)
        self.table.setItem(row, 0, status_item)

        # Mode
        mode_text = MODE_LABELS.get(job.mode, job.mode)
        self.table.setItem(row, 1, QTableWidgetItem(mode_text))

        # Name
        self.table.setItem(row, 2, QTableWidgetItem(job.display_name or ""))

        # Task / Script
        task_text = ""
        if job.command:
            task_text = " ".join(str(c) for c in job.command)
            if len(task_text) > 60:
                task_text = task_text[:57] + "..."
        self.table.setItem(row, 3, QTableWidgetItem(task_text))

        # Started At
        started = job.started_at or job.created_at
        started_text = ""
        if started:
            try:
                dt = datetime.fromisoformat(started)
                started_text = dt.strftime("%Y-%m-%d %H:%M")
            except (ValueError, TypeError):
                started_text = str(started)[:16]
        self.table.setItem(row, 4, QTableWidgetItem(started_text))

        # Duration
        duration_text = ""
        if job.duration_seconds is not None and job.duration_seconds > 0:
            mins = int(job.duration_seconds // 60)
            secs = int(job.duration_seconds % 60)
            duration_text = f"{mins}m {secs}s"
        self.table.setItem(row, 5, QTableWidgetItem(duration_text))

        # Output directory
        output_text = job.output_directory or ""
        self.table.setItem(row, 6, QTableWidgetItem(output_text))

    def _on_selection_changed(self):
        selected = self.table.selectedItems()
        if not selected:
            self.detail_text.clear()
            self.open_dir_btn.setEnabled(False)
            return

        row = selected[0].row()
        job_id = self.table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        job = self.history_store.get_job(job_id)
        if job is None:
            self.detail_text.clear()
            self.open_dir_btn.setEnabled(False)
            return

        # Build detail text
        lines = []
        lines.append(f"Job ID: {job.job_id}")
        lines.append(f"Status: {STATUS_LABELS.get(job.status, job.status.upper())}")
        lines.append(f"Mode: {MODE_LABELS.get(job.mode, job.mode)}")
        lines.append(f"Name: {job.display_name or 'N/A'}")
        lines.append(f"Framework: {job.framework or 'N/A'}")

        if job.started_at:
            lines.append(f"Started: {job.started_at[:19]}")
        if job.ended_at:
            lines.append(f"Ended: {job.ended_at[:19]}")
        if job.duration_seconds is not None and job.duration_seconds > 0:
            mins = int(job.duration_seconds // 60)
            secs = int(job.duration_seconds % 60)
            lines.append(f"Duration: {mins}m {secs}s")

        lines.append(f"Output: {job.output_directory or 'N/A'}")

        if job.error_message:
            lines.append(f"Error: {job.error_message}")

        self.detail_text.setPlainText("\n".join(lines))
        self.open_dir_btn.setEnabled(bool(job.output_directory))

    def _open_output_directory(self):
        selected = self.table.selectedItems()
        if not selected:
            return
        row = selected[0].row()
        job_id = self.table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        job = self.history_store.get_job(job_id)
        if job is None or not job.output_directory:
            QMessageBox.information(self, "Info", "No output directory recorded for this job.")
            return

        path = job.output_directory
        if not os.path.exists(path):
            QMessageBox.warning(
                self, "Directory Not Found",
                f"The output directory no longer exists:\n{path}"
            )
            return

        # Open in system file manager
        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.run(["open", path])
        else:
            subprocess.run(["xdg-open", path])
