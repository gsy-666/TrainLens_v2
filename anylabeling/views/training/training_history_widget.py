"""Training History Widget

Displays training job history from the shared HistoryStore.
"""

import os
import subprocess
import sys
from datetime import datetime

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPalette
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
    QDialog,
)

from anylabeling.services.training_center.history import (
    get_history_store, format_duration,
)
from anylabeling.services.training_center.job_manager import get_job_manager
from anylabeling.services.training_center.models import TrainingStatus
from anylabeling.services.training_center.event_protocol import TrainingEventType
from anylabeling.views.training.metrics import TrainingMetricsDashboard


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

    COLUMNS = [
        "Started", "Task", "Model", "Dataset", "Status",
        "Epochs", "Best Metric", "Duration",
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.history_store = get_history_store()
        self.job_manager = get_job_manager()

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

        # Light blue selection via palette (preserves per-cell setForeground)
        palette = self.table.palette()
        palette.setColor(QPalette.ColorRole.Highlight, QColor("#DCEBFF"))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#111111"))
        self.table.setPalette(palette)
        # Hover only via stylesheet (doesn't affect selection)
        self.table.setStyleSheet(
            "QTableWidget::item:hover { background-color: #F3F8FF; }"
        )

        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.table, stretch=1)

        # Detail panel
        detail_layout = QHBoxLayout()

        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setMaximumHeight(140)
        self.detail_text.setPlaceholderText("Select a job to view details...")
        detail_layout.addWidget(self.detail_text)

        # Action buttons (2-column grid)
        action_layout = QVBoxLayout()
        action_layout.setSpacing(4)

        self.open_dir_btn = QPushButton("Open Output Folder")
        self.open_dir_btn.clicked.connect(self._open_output_directory)
        self.open_dir_btn.setEnabled(False)
        action_layout.addWidget(self.open_dir_btn)

        self.open_log_btn = QPushButton("Open Log")
        self.open_log_btn.clicked.connect(self._open_log)
        self.open_log_btn.setEnabled(False)
        action_layout.addWidget(self.open_log_btn)

        self.open_csv_btn = QPushButton("Open results.csv")
        self.open_csv_btn.clicked.connect(self._open_results_csv)
        self.open_csv_btn.setEnabled(False)
        action_layout.addWidget(self.open_csv_btn)

        self.open_best_pt_btn = QPushButton("Open best.pt Folder")
        self.open_best_pt_btn.clicked.connect(self._open_best_pt_folder)
        self.open_best_pt_btn.setEnabled(False)
        action_layout.addWidget(self.open_best_pt_btn)

        self.view_metrics_btn = QPushButton("View Metrics")
        self.view_metrics_btn.clicked.connect(self._view_metrics)
        self.view_metrics_btn.setEnabled(False)
        action_layout.addWidget(self.view_metrics_btn)

        self.delete_btn = QPushButton("Delete Record")
        self.delete_btn.clicked.connect(self._delete_record)
        self.delete_btn.setEnabled(False)
        self.delete_btn.setStyleSheet("color: #d32f2f;")
        action_layout.addWidget(self.delete_btn)

        action_layout.addStretch()
        detail_layout.addLayout(action_layout)

        layout.addLayout(detail_layout)

    def refresh(self):
        """Reload history from store. Filter IDLE; show active job from JobManager."""
        jobs = self.history_store.list_jobs()
        active_job = self.job_manager.get_current_job()

        # Filter out IDLE records that aren't the current active job
        filtered = []
        for job in jobs:
            if job.status == "idle":
                if active_job and job.job_id == active_job.job_id:
                    pass  # Keep if it's the current active job (shouldn't happen)
                else:
                    continue  # Skip IDLE records
            filtered.append(job)

        self.table.setRowCount(len(filtered))

        for row, job in enumerate(filtered):
            self._set_row(row, job, active_job)

        self.table.resizeColumnsToContents()

    def _set_row(self, row, job, active_job=None):
        # Determine effective status: JobManager's active job overrides stale history
        effective_status = job.status
        if active_job and job.job_id == active_job.job_id:
            effective_status = active_job.status.value

        # 0. Started
        started = job.started_at or job.created_at
        started_text = ""
        if started:
            try:
                dt = datetime.fromisoformat(started)
                started_text = dt.strftime("%Y-%m-%d %H:%M")
            except (ValueError, TypeError):
                started_text = str(started)[:16]
        started_item = QTableWidgetItem(started_text)
        started_item.setData(Qt.ItemDataRole.UserRole, job.job_id)
        self.table.setItem(row, 0, started_item)

        # 1. Task
        task_text = job.task or (job.metadata.get("task", "") if isinstance(job.metadata, dict) else "") or ""
        # Normalize task capitalization
        if task_text:
            task_text = task_text.strip().capitalize()
        self.table.setItem(row, 1, QTableWidgetItem(task_text or "—"))

        # 2. Model — show basename in table, full path in detail
        model_text = job.model_name or job.model or (job.metadata.get("model", "") if isinstance(job.metadata, dict) else "")
        if not model_text and job.command:
            for part in job.command:
                if part.endswith('.pt') or part.endswith('.yaml'):
                    model_text = part
                    break
        # Table: basename only
        model_basename = os.path.basename(model_text) if model_text else ""
        self.table.setItem(row, 2, QTableWidgetItem(model_basename or "—"))

        # 3. Dataset — show basename in table, full path in detail
        dataset_text = job.dataset_yaml or job.data or (job.metadata.get("data", "") if isinstance(job.metadata, dict) else "")
        if dataset_text:
            dataset_text = os.path.basename(dataset_text)
        self.table.setItem(row, 3, QTableWidgetItem(dataset_text or "—"))

        # 4. Status
        status_text = STATUS_LABELS.get(effective_status, effective_status.upper())
        status_item = QTableWidgetItem(status_text)
        color = STATUS_COLORS.get(effective_status, QColor(158, 158, 158))
        status_item.setForeground(color)
        self.table.setItem(row, 4, status_item)

        # 5. Epochs
        epochs_text = "—"
        completed = job.completed_epochs or job.final_epoch
        requested = job.requested_epochs or job.total_epochs
        if completed is not None:
            if requested:
                epochs_text = f"{completed} / {requested}"
            else:
                epochs_text = str(completed)
        self.table.setItem(row, 5, QTableWidgetItem(epochs_text))

        # 6. Best Metric
        metric_text = "—"
        if job.best_metric_name and job.best_metric_value is not None:
            short_name = job.best_metric_name.replace("metrics/", "")
            metric_text = f"{short_name} {job.best_metric_value:.3f}"
        elif job.best_map50 is not None:
            metric_text = f"mAP50 {job.best_map50:.3f}"
        elif job.best_map50_95 is not None:
            metric_text = f"mAP50-95 {job.best_map50_95:.3f}"
        self.table.setItem(row, 6, QTableWidgetItem(metric_text))

        # 7. Duration
        duration_text = format_duration(job.duration_seconds)
        self.table.setItem(row, 7, QTableWidgetItem(duration_text))

    # ── Selection & Detail ───────────────────────────────────────

    def _on_selection_changed(self):
        selected = self.table.selectedItems()
        if not selected:
            self._clear_detail()
            return

        row = selected[0].row()
        job_id = self.table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        job = self.history_store.get_job(job_id)
        if job is None:
            self._clear_detail()
            return

        self._show_detail(job)

    def _clear_detail(self):
        self.detail_text.clear()
        for btn in [self.open_dir_btn, self.open_log_btn, self.open_csv_btn,
                     self.open_best_pt_btn, self.view_metrics_btn, self.delete_btn]:
            btn.setEnabled(False)

    def _show_detail(self, job):
        lines = []
        status_label = STATUS_LABELS.get(job.status, job.status.upper())
        lines.append(f"Status: {status_label}")

        # Start / Finish
        if job.started_at:
            lines.append(f"Started: {job.started_at[:19]}")
        if job.ended_at:
            lines.append(f"Finished: {job.ended_at[:19]}")
        duration = format_duration(job.duration_seconds)
        if duration:
            lines.append(f"Duration: {duration}")

        task = job.task or "—"
        if task:
            task = task.strip().capitalize()
        lines.append(f"Task: {task}")
        model_full = job.model_name or job.model or "—"
        lines.append(f"Model: {model_full}")
        dataset_full = job.dataset_yaml or job.data or "—"
        lines.append(f"Dataset: {dataset_full}")

        # Epochs
        completed = job.completed_epochs or job.final_epoch
        requested = job.requested_epochs or job.total_epochs
        if completed is not None:
            if requested:
                lines.append(f"Epochs: {completed} / {requested}")
            else:
                lines.append(f"Epochs: {completed}")
        else:
            lines.append("Epochs: —")

        # Best metric
        if job.best_metric_name and job.best_metric_value is not None:
            lines.append(f"Best: {job.best_metric_name} = {job.best_metric_value:.4f}")
        elif job.best_map50 is not None:
            lines.append(f"Best: mAP50 = {job.best_map50:.4f}")
        else:
            lines.append("Best: —")

        # Paths
        out = job.output_directory or job.project_path or "—"
        lines.append(f"Project: {out}")
        lines.append(f"best.pt: {job.best_weights_path or 'Not available'}")
        lines.append(f"last.pt: {job.last_weights_path or 'Not available'}")
        lines.append(f"results.csv: {job.results_csv_path or 'Not available'}")

        if job.error_message:
            lines.append(f"Error: {job.error_message}")

        self.detail_text.setPlainText("\n".join(lines))

        # Enable/disable buttons
        has_output = bool(job.output_directory) and os.path.isdir(job.output_directory or "")
        self.open_dir_btn.setEnabled(has_output)
        self.open_log_btn.setEnabled(has_output)

        has_csv = bool(job.results_csv_path) and os.path.isfile(job.results_csv_path or "")
        self.open_csv_btn.setEnabled(has_csv)

        has_best_pt = bool(job.best_weights_path) and os.path.isfile(job.best_weights_path or "")
        self.open_best_pt_btn.setEnabled(has_best_pt)

        self.view_metrics_btn.setEnabled(has_output or has_csv)
        self.delete_btn.setEnabled(True)

    # ── Actions ──────────────────────────────────────────────────

    def _selected_job(self):
        """Get currently selected job record."""
        selected = self.table.selectedItems()
        if not selected:
            return None
        row = selected[0].row()
        job_id = self.table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        return self.history_store.get_job(job_id)

    def _open_path(self, path, not_found_msg="Path not available."):
        """Safely open a file or directory in the system file manager."""
        if not path:
            QMessageBox.information(self, "Info", not_found_msg)
            return
        if os.path.isfile(path):
            # Open containing folder and select file
            if sys.platform == "win32":
                subprocess.run(["explorer", "/select,", os.path.abspath(path)])
            elif sys.platform == "darwin":
                subprocess.run(["open", "-R", path])
            else:
                subprocess.run(["xdg-open", os.path.dirname(path)])
        elif os.path.isdir(path):
            if sys.platform == "win32":
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.run(["open", path])
            else:
                subprocess.run(["xdg-open", path])
        else:
            QMessageBox.warning(self, "Not Found", f"Path does not exist:\n{path}")

    def _open_output_directory(self):
        job = self._selected_job()
        if job is None:
            return
        path = job.output_directory or job.project_path
        if not path or not os.path.isdir(path or ""):
            QMessageBox.information(self, "Info", "Output directory no longer exists.")
            return
        self._open_path(path)

    def _open_log(self):
        job = self._selected_job()
        if job is None:
            return
        out_dir = job.output_directory
        if not out_dir or not os.path.isdir(out_dir):
            QMessageBox.information(self, "Info", "Output directory not available.")
            return
        # Try to find a log file
        log_candidates = [
            os.path.join(out_dir, "train.log"),
            os.path.join(out_dir, "log.txt"),
            os.path.join(out_dir, "events.out.tfevents."),
        ]
        for candidate in log_candidates:
            if os.path.isfile(candidate):
                self._open_path(candidate)
                return
        # Fall back to opening the output dir
        QMessageBox.information(
            self, "No Log Found",
            "No log file found in the output directory.\nOpening the output folder instead."
        )
        self._open_path(out_dir)

    def _open_results_csv(self):
        job = self._selected_job()
        if job is None:
            return
        path = job.results_csv_path
        if not path or not os.path.isfile(path or ""):
            QMessageBox.information(self, "Info", "results.csv not available for this job.")
            return
        self._open_path(path)

    def _open_best_pt_folder(self):
        job = self._selected_job()
        if job is None:
            return
        path = job.best_weights_path
        if not path or not os.path.isfile(path or ""):
            QMessageBox.information(self, "Info", "best.pt not available for this job.")
            return
        self._open_path(path)

    def _view_metrics(self):
        """Open a metrics dashboard dialog for the selected historical job."""
        job = self._selected_job()
        if job is None:
            return
        path = job.output_directory or job.project_path
        if not path or not os.path.isdir(path or ""):
            QMessageBox.information(self, "Info", "No metrics data available for this job.")
            return

        results_csv = job.results_csv_path or os.path.join(path, "results.csv")
        metrics_jsonl = os.path.join(path, "metrics.jsonl")
        if not os.path.isfile(results_csv) and not os.path.isfile(metrics_jsonl):
            QMessageBox.information(
                self, "No Metrics Data",
                "This job's output directory does not contain metrics data.\n\n"
                f"Expected: {results_csv}"
            )
            return

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Metrics — {job.display_name or job.job_id}")
        dialog.resize(900, 600)
        dialog.setMinimumSize(600, 400)
        layout = QVBoxLayout(dialog)
        dashboard = TrainingMetricsDashboard()
        layout.addWidget(dashboard)
        dashboard.load_history(job.job_id, path)

        # Close button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        dialog.exec()
        dashboard.cleanup()

    def _delete_record(self):
        """Delete the selected history record (database only, not training files)."""
        job = self._selected_job()
        if job is None:
            return

        reply = QMessageBox.question(
            self, "Delete Record",
            f"Delete history record for '{job.display_name or job.job_id}'?\n\n"
            "This only removes the database record, not the training files.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            self.history_store.delete_job(job.job_id)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to delete record: {e}")
            return

        self.refresh()
