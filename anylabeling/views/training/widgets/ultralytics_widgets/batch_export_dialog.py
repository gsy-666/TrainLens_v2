"""Batch multi-format export dialog with format checkboxes and progress table."""

import os
import platform
from datetime import datetime
from typing import List

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QCheckBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QGroupBox,
    QScrollArea,
    QWidget,
    QMessageBox,
    QAbstractItemView,
    QProgressBar,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QBrush

from anylabeling.services.auto_training.ultralytics.style import (
    get_ultralytics_dialog_style,
)
from anylabeling.services.auto_training.ultralytics.export_formats import (
    EXPORT_FORMATS,
    FORMAT_BY_CODE,
    CATEGORY_ORDER,
    FORMAT_BY_CATEGORY,
    FormatInfo,
    FormatStatus,
    get_format_status,
    get_missing_pip_packages,
    get_large_missing_packages,
)
from anylabeling.services.auto_training.ultralytics.batch_exporter import (
    BatchExportRequest,
    ExportJob,
    get_batch_export_manager,
)
from anylabeling.views.labeling.utils.theme import get_theme
from .custom_widgets import PrimaryButton, SecondaryButton


STATUS_LABELS = {
    FormatStatus.READY: "Available",
    FormatStatus.MISSING_DEPENDENCY: "Missing Dependency",
    FormatStatus.UNSUPPORTED_PLATFORM: "Unsupported Platform",
    FormatStatus.UNSUPPORTED_DEVICE: "Unsupported Device",
    FormatStatus.NOT_IMPLEMENTED: "Not Implemented",
}

STATUS_COLORS = {
    FormatStatus.READY: "#4caf50",
    FormatStatus.MISSING_DEPENDENCY: "#ff9800",
    FormatStatus.UNSUPPORTED_PLATFORM: "#f44336",
    FormatStatus.UNSUPPORTED_DEVICE: "#f44336",
    FormatStatus.NOT_IMPLEMENTED: "#9e9e9e",
}

JOB_STATUS_COLORS = {
    "pending": "#9e9e9e",
    "running": "#2196f3",
    "completed": "#4caf50",
    "failed": "#f44336",
    "skipped": "#ff9800",
    "cancelled": "#757575",
}


class FormatCheckBox(QCheckBox):
    """Checkbox with attached FormatInfo metadata."""

    def __init__(self, info: FormatInfo, parent=None):
        super().__init__(info.display_name, parent)
        self._info = info
        self._status = get_format_status(info)
        self._update_tooltip()

    @property
    def info(self) -> FormatInfo:
        return self._info

    @property
    def status(self) -> str:
        return self._status

    def _update_tooltip(self):
        lines = [
            f"{self._info.display_name} ({self._info.format_code})",
            f"Output: {self._info.output_path_template}",
            f"Usage: {self._info.description}",
            f"Status: {STATUS_LABELS.get(self._status, self._status)}",
        ]
        if self._status == FormatStatus.MISSING_DEPENDENCY:
            missing = get_missing_pip_packages(self._info)
            lines.append(f"Missing: {', '.join(missing)}")
        if self._status == FormatStatus.UNSUPPORTED_PLATFORM:
            lines.append(
                f"Requires: {', '.join(self._info.capability.supported_platforms)}"
            )
        self.setToolTip("\n".join(lines))


class BatchExportDialog(QDialog):
    """Multi-format batch export dialog."""

    # Signal for package install query (emitted from UI thread)
    install_query_response = pyqtSignal(str)  # "accept", "skip", "cancel"

    def __init__(
        self,
        source_model: str,
        project_path: str,
        imgsz: int = 640,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Batch Model Export"))
        self.setMinimumSize(780, 620)
        self.setModal(True)
        self.setStyleSheet(get_ultralytics_dialog_style())

        self._source_model = source_model
        self._project_path = project_path
        self._imgsz = imgsz
        self._selected_formats: List[str] = []
        self._checkboxes: List[FormatCheckBox] = []
        self._all_jobs: List[ExportJob] = []
        self._is_exporting = False

        self._manager = get_batch_export_manager()

        # ── Build UI ────────────────────────────────────────────────────
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(16, 16, 16, 16)

        # Header
        header = QLabel(
            self.tr(
                f"<h3>Export Model: {os.path.basename(source_model)}</h3>"
                "<p>Select formats and click 'Start Batch Export'. "
                "Exports run one at a time to avoid resource conflicts.</p>"
            )
        )
        header.setWordWrap(True)
        main_layout.addWidget(header)

        # Scrollable format selection area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(6)

        t = get_theme()
        for category in CATEGORY_ORDER:
            formats_in_cat = FORMAT_BY_CATEGORY.get(category, [])
            if not formats_in_cat:
                continue

            group = QGroupBox(category)
            group.setStyleSheet(
                f"QGroupBox {{ font-weight: bold; color: {t['text']}; "
                f"border: 1px solid {t.get('border', '#444')}; "
                f"border-radius: 4px; margin-top: 8px; padding-top: 16px; }}"
                f"QGroupBox::title {{ padding: 0 8px; }}"
            )
            group_layout = QVBoxLayout(group)
            group_layout.setSpacing(2)

            for info in formats_in_cat:
                cb = FormatCheckBox(info, self)
                self._checkboxes.append(cb)

                # Disable formats that can't work
                status = cb.status
                if status in (
                    FormatStatus.UNSUPPORTED_PLATFORM,
                    FormatStatus.UNSUPPORTED_DEVICE,
                    FormatStatus.NOT_IMPLEMENTED,
                ):
                    cb.setEnabled(False)
                    cb.setChecked(False)

                group_layout.addWidget(cb)

            scroll_layout.addWidget(group)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        main_layout.addWidget(scroll, 1)

        # Batch options row
        options_layout = QHBoxLayout()
        options_layout.setSpacing(12)

        self.select_all_btn = QPushButton(self.tr("Select All Compatible"))
        self.select_all_btn.clicked.connect(self._on_select_all)
        options_layout.addWidget(self.select_all_btn)

        self.deselect_all_btn = QPushButton(self.tr("Deselect All"))
        self.deselect_all_btn.clicked.connect(self._on_deselect_all)
        options_layout.addWidget(self.deselect_all_btn)

        options_layout.addStretch()

        self.selected_count_label = QLabel(self.tr("0 formats selected"))
        options_layout.addWidget(self.selected_count_label)

        main_layout.addLayout(options_layout)

        # Progress table
        self.progress_table = QTableWidget(0, 4)
        self.progress_table.setHorizontalHeaderLabels(
            [self.tr("Format"), self.tr("Status"), self.tr("Output"), self.tr("Error")]
        )
        self.progress_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.progress_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.progress_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch
        )
        self.progress_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.Stretch
        )
        self.progress_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.progress_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.progress_table.verticalHeader().setVisible(False)
        self.progress_table.setVisible(False)
        main_layout.addWidget(self.progress_table, 2)

        # Overall progress bar
        self.overall_progress = QProgressBar()
        self.overall_progress.setVisible(False)
        self.overall_progress.setFormat(
            self.tr("%v / %m formats completed")
        )
        main_layout.addWidget(self.overall_progress)

        # Summary label
        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        self.summary_label.setVisible(False)
        main_layout.addWidget(self.summary_label)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.cancel_btn = SecondaryButton(self.tr("Cancel"))
        self.cancel_btn.clicked.connect(self._on_cancel)
        button_layout.addWidget(self.cancel_btn)

        self.open_folder_btn = QPushButton(self.tr("Open Export Folder"))
        self.open_folder_btn.clicked.connect(self._on_open_folder)
        self.open_folder_btn.setVisible(False)
        button_layout.addWidget(self.open_folder_btn)

        self.start_btn = PrimaryButton(self.tr("Start Batch Export"))
        self.start_btn.clicked.connect(self._on_start)
        button_layout.addWidget(self.start_btn)

        main_layout.addLayout(button_layout)

        # Connect manager signals
        self._manager.job_updated.connect(self._on_job_updated)
        self._manager.all_completed.connect(self._on_all_completed)
        self._manager.export_log.connect(self._on_export_log)
        self._manager.package_install_query.connect(self._on_package_query)

        # Initialize checkbox selection
        self._update_selected_count()

    def _get_selected_infos(self) -> List[FormatInfo]:
        """Get FormatInfos for checked and enabled checkboxes."""
        return [cb.info for cb in self._checkboxes if cb.isChecked() and cb.isEnabled()]

    def _update_selected_count(self):
        count = len(self._get_selected_infos())
        self.selected_count_label.setText(
            self.tr(f"{count} format(s) selected")
        )
        self.start_btn.setEnabled(count > 0)

    def _on_select_all(self):
        for cb in self._checkboxes:
            if cb.isEnabled():
                cb.setChecked(True)
        self._update_selected_count()

    def _on_deselect_all(self):
        for cb in self._checkboxes:
            cb.setChecked(False)
        self._update_selected_count()

    def _on_start(self):
        """Start the batch export."""
        selected = self._get_selected_infos()
        if not selected:
            QMessageBox.warning(
                self, self.tr("No Formats"), self.tr("Please select at least one format.")
            )
            return

        format_codes = [info.format_code for info in selected]

        # Build timestamped output root
        from datetime import datetime

        model_name = os.path.splitext(os.path.basename(self._source_model))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = os.path.join(
            os.path.dirname(self._source_model),
            "exports",
            f"{model_name}_{timestamp}",
        )

        request = BatchExportRequest(
            source_model=self._source_model,
            formats=format_codes,
            output_root=output_root,
            imgsz=self._imgsz,
        )

        # Copy source model to output
        source_dir = os.path.join(output_root, "source")
        os.makedirs(source_dir, exist_ok=True)
        import shutil

        shutil.copy2(self._source_model, os.path.join(source_dir, "best.pt"))

        # Show progress table
        self.progress_table.setVisible(True)
        self.overall_progress.setVisible(True)
        self.progress_table.setRowCount(0)

        # Disable format selection
        for cb in self._checkboxes:
            cb.setEnabled(False)

        self.select_all_btn.setEnabled(False)
        self.deselect_all_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.cancel_btn.setText(self.tr("Cancel Export"))

        self._is_exporting = True

        # Start batch
        success = self._manager.start_batch(request)
        if not success:
            QMessageBox.critical(
                self, self.tr("Error"), self.tr("Export is already in progress.")
            )
            self._reset_ui()

    def _on_job_updated(self, job: ExportJob):
        """Handle job status change."""
        self._all_jobs.append(job)

        # Update or add row in progress table
        found = False
        for row in range(self.progress_table.rowCount()):
            item = self.progress_table.item(row, 0)
            if item and item.data(Qt.ItemDataRole.UserRole) == job.job_id:
                found = True
                self._update_table_row(row, job)
                break

        if not found:
            row = self.progress_table.rowCount()
            self.progress_table.insertRow(row)
            self._update_table_row(row, job)

        # Update overall progress
        total = len(self._manager.jobs)
        completed = sum(
            1 for j in self._manager.jobs if j.status in ("completed", "failed", "skipped", "cancelled")
        )
        self.overall_progress.setMaximum(total)
        self.overall_progress.setValue(completed)

    def _update_table_row(self, row: int, job: ExportJob):
        """Update a specific row in the progress table."""
        t = get_theme()

        # Format name
        info = FORMAT_BY_CODE.get(job.format_code)
        display = info.display_name if info else job.format_code
        fmt_item = QTableWidgetItem(display)
        fmt_item.setData(Qt.ItemDataRole.UserRole, job.job_id)
        self.progress_table.setItem(row, 0, fmt_item)

        # Status
        status_text = job.status.capitalize()
        status_item = QTableWidgetItem(status_text)
        color = JOB_STATUS_COLORS.get(job.status, "#9e9e9e")
        status_item.setForeground(QBrush(QColor(color)))
        self.progress_table.setItem(row, 1, status_item)

        # Output path
        output_text = job.output_path if job.output_path else "—"
        output_item = QTableWidgetItem(output_text)
        self.progress_table.setItem(row, 2, output_item)

        # Error
        error_text = job.error_message if job.error_message else ""
        error_item = QTableWidgetItem(error_text)
        error_item.setForeground(QBrush(QColor(t.get("error", "#f44336"))))
        self.progress_table.setItem(row, 3, error_item)

    def _on_all_completed(self, summary: dict):
        """Handle batch completion."""
        self._is_exporting = False

        # Generate manifest
        if self._manager._request:
            manifest_path = self._manager.generate_manifest(
                self._manager._request.output_root
            )
            # Record export manifest in training job history
            if manifest_path:
                self._record_export_history(
                    self._manager._request.output_root, manifest_path
                )
                self.summary_label.setText(
                    self.tr(
                        f"<b>Export complete:</b> {summary['completed']} succeeded, "
                        f"{summary['failed']} failed, {summary['skipped']} skipped, "
                        f"{summary['cancelled']} cancelled.<br>"
                        f"Output: {self._manager._request.output_root}<br>"
                        f"Manifest: {manifest_path}"
                    )
                )
            else:
                self.summary_label.setText(
                    self.tr(
                        f"<b>Export complete:</b> {summary['completed']} succeeded, "
                        f"{summary['failed']} failed, {summary['skipped']} skipped, "
                        f"{summary['cancelled']} cancelled."
                    )
                )
        else:
            self.summary_label.setText(
                self.tr(
                    f"<b>Export complete:</b> {summary['completed']} succeeded."
                )
            )

        self.summary_label.setVisible(True)
        self.open_folder_btn.setVisible(True)
        self.cancel_btn.setText(self.tr("Close"))
        self.start_btn.setVisible(False)
        self.overall_progress.setVisible(False)

    def _record_export_history(self, output_root: str, manifest_path: str):
        """Record export results in the training job history."""
        try:
            from anylabeling.services.training_center.job_manager import get_job_manager
            from anylabeling.services.training_center.history import get_history_store

            job_manager = get_job_manager()
            history_store = get_history_store()

            # Find the most recent completed/finished training job for this project
            jobs = job_manager.list_jobs()
            for job in jobs:
                if job.project_path and os.path.normpath(job.project_path) == os.path.normpath(
                    self._project_path
                ):
                    try:
                        history_store.update_job(
                            job.job_id,
                            export_path=output_root,
                            metadata={
                                **(job.metadata or {}),
                                "export_manifest": manifest_path,
                                "export_completed_at": datetime.now().isoformat(),
                            },
                        )
                    except KeyError:
                        pass  # job not in history store
                    break
        except Exception:
            pass  # Best-effort: don't break the dialog for history failures

    def _on_export_log(self, message: str):
        """Handle log messages — shown in the summary label during export."""
        self.summary_label.setText(message)
        self.summary_label.setVisible(True)

    def _on_package_query(self, all_packages: list, large_packages: list):
        """User needs to confirm installation of large packages."""
        msg = QMessageBox(self)
        msg.setWindowTitle(self.tr("Large Dependencies Required"))
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setText(
            self.tr(
                "The following large packages are needed for the selected formats:\n\n"
                f"{chr(10).join(f'  • {p}' for p in large_packages)}\n\n"
                "These may take significant time and disk space to install.\n"
                "How would you like to proceed?"
            )
        )
        msg.setStandardButtons(
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No
            | QMessageBox.StandardButton.Cancel
        )
        msg.button(QMessageBox.StandardButton.Yes).setText(
            self.tr("Install (non-blocking)")
        )
        msg.button(QMessageBox.StandardButton.No).setText(
            self.tr("Skip these formats")
        )
        msg.button(QMessageBox.StandardButton.Cancel).setText(
            self.tr("Cancel all")
        )

        result = msg.exec()

        if result == QMessageBox.StandardButton.Yes:
            self._manager.on_install_confirmed(True)
        elif result == QMessageBox.StandardButton.No:
            self._manager.on_install_skip_format()
        else:
            self._manager.on_install_cancel_all()

    def _on_cancel(self):
        """Cancel button — either cancel export or close dialog."""
        if self._is_exporting:
            reply = QMessageBox.question(
                self,
                self.tr("Cancel Export"),
                self.tr(
                    "Cancel the current batch export?\n"
                    "The current format will finish, remaining formats will be cancelled."
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._manager.cancel_batch()
        else:
            self.reject()

    def _on_open_folder(self):
        """Open the export output folder."""
        if self._manager._request:
            output_root = self._manager._request.output_root
            if os.path.exists(output_root):
                import subprocess
                import sys

                if sys.platform == "win32":
                    os.startfile(output_root)
                elif sys.platform == "darwin":
                    subprocess.run(["open", output_root])
                else:
                    subprocess.run(["xdg-open", output_root])

    def _reset_ui(self):
        """Reset UI to initial state."""
        for cb in self._checkboxes:
            cb.setEnabled(True)
        self.select_all_btn.setEnabled(True)
        self.deselect_all_btn.setEnabled(True)
        self.start_btn.setEnabled(True)
        self.cancel_btn.setText(self.tr("Cancel"))
        self.progress_table.setVisible(False)
        self.overall_progress.setVisible(False)
        self.summary_label.setVisible(False)
        self.open_folder_btn.setVisible(False)
        self._update_selected_count()
