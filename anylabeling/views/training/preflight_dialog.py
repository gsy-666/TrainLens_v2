"""TrainLens Preflight — reusable result dialog."""

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QPushButton, QTextEdit, QVBoxLayout, QWidget,
)

from anylabeling.services.training_center.preflight.models import (
    PreflightResult, PreflightSeverity,
)


class PreflightDialog(QDialog):
    """Non-blocking preflight result dialog.

    Shows:
    - Summary (pass/warning/error counts)
    - Issue list (clickable for details)
    - Action buttons (Back / Start Anyway / Start Training / Cancel)
    """

    def __init__(self, result: PreflightResult, parent=None):
        super().__init__(parent)
        self._result = result
        self._user_choice = None  # "start_anyway", "start", "back", "cancel"
        self._checking = False  # True while worker is running

        self.setWindowTitle("Training Preflight Check")
        self.setMinimumSize(650, 500)
        self.resize(700, 550)

        self._build_ui()
        self._populate(result)

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("Training Preflight Check")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        layout.addWidget(title)

        # Summary cards
        self._summary_widget = QWidget()
        summary_layout = QHBoxLayout(self._summary_widget)
        summary_layout.setContentsMargins(0, 4, 0, 4)

        self._pass_label = self._make_count_label("Passed", QColor(76, 175, 80))
        self._warn_label = self._make_count_label("Warnings", QColor(255, 152, 0))
        self._err_label = self._make_count_label("Errors", QColor(244, 67, 54))

        summary_layout.addWidget(self._pass_label)
        summary_layout.addWidget(self._warn_label)
        summary_layout.addWidget(self._err_label)
        summary_layout.addStretch()
        layout.addWidget(self._summary_widget)

        # Progress label (shown during checking)
        self._progress_label = QLabel("")
        self._progress_label.setStyleSheet("color: #1976D2; font-style: italic;")
        self._progress_label.hide()
        layout.addWidget(self._progress_label)

        # Issue list
        self._issue_list = QListWidget()
        self._issue_list.currentRowChanged.connect(self._on_issue_selected)
        layout.addWidget(self._issue_list, stretch=1)

        # Detail panel
        self._detail_panel = QTextEdit()
        self._detail_panel.setReadOnly(True)
        self._detail_panel.setMaximumHeight(120)
        self._detail_panel.setPlaceholderText("Select an issue to view details...")
        layout.addWidget(self._detail_panel)

        # Button box
        self._button_box = QDialogButtonBox()
        self._cancel_btn = QPushButton("Cancel")
        self._back_btn = QPushButton("Back to Fix")
        self._start_anyway_btn = QPushButton("Start Anyway")
        self._start_btn = QPushButton("Start Training")

        self._cancel_btn.clicked.connect(self._on_cancel)
        self._back_btn.clicked.connect(self._on_back)
        self._start_anyway_btn.clicked.connect(self._on_start_anyway)
        self._start_btn.clicked.connect(self._on_start)

        self._button_box.addButton(self._cancel_btn, QDialogButtonBox.ButtonRole.RejectRole)
        self._button_box.addButton(self._back_btn, QDialogButtonBox.ButtonRole.ActionRole)
        self._button_box.addButton(self._start_anyway_btn, QDialogButtonBox.ButtonRole.AcceptRole)
        self._button_box.addButton(self._start_btn, QDialogButtonBox.ButtonRole.AcceptRole)
        layout.addWidget(self._button_box)

    def _make_count_label(self, name: str, color: QColor) -> QLabel:
        label = QLabel(f"{name}\n0")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet(
            f"font-size: 12pt; font-weight: bold; color: {color.name()}; "
            f"padding: 8px 16px; border: 1px solid #ddd; border-radius: 4px;"
        )
        label.setObjectName(f"count_{name.lower()}")
        return label

    def _populate(self, result: PreflightResult):
        """Populate the dialog with preflight results."""
        self._update_counts(result)
        self._issue_list.clear()

        for issue in result.issues:
            if issue.severity == PreflightSeverity.PASS:
                continue  # Don't show PASS items in list by default

            item = QListWidgetItem()
            icon = {
                PreflightSeverity.ERROR: "✕",
                PreflightSeverity.WARNING: "⚠",
                PreflightSeverity.PASS: "✓",
            }.get(issue.severity, "?")

            color = {
                PreflightSeverity.ERROR: QColor(244, 67, 54),
                PreflightSeverity.WARNING: QColor(255, 152, 0),
                PreflightSeverity.PASS: QColor(76, 175, 80),
            }.get(issue.severity, QColor(0, 0, 0))

            item.setText(f"{icon} [{issue.severity.value.upper()}] {issue.title}")
            item.setForeground(color)
            item.setData(Qt.ItemDataRole.UserRole, issue)
            self._issue_list.addItem(item)

        # Buttons based on result
        self._update_buttons(result)

    def _update_counts(self, result: PreflightResult):
        self._pass_label.setText(f"Passed\n{result.pass_count}")
        self._warn_label.setText(f"Warnings\n{result.warning_count}")
        self._err_label.setText(f"Errors\n{result.error_count}")

    def _update_buttons(self, result: PreflightResult):
        """Show appropriate buttons based on result."""
        has_errors = result.has_errors
        has_warnings = result.has_warnings

        self._cancel_btn.setVisible(self._checking)
        self._back_btn.setVisible(not self._checking)
        self._start_anyway_btn.setVisible(not self._checking and not has_errors and has_warnings)
        self._start_btn.setVisible(not self._checking and not has_errors and not has_warnings)

        if self._checking:
            self._back_btn.setVisible(False)
            self._start_anyway_btn.setVisible(False)
            self._start_btn.setVisible(False)

    def set_checking(self, checking: bool):
        """Switch to 'checking' mode (shows Cancel, hides result buttons)."""
        self._checking = checking
        self._progress_label.setVisible(checking)
        if checking:
            self._update_buttons_for_checking()
        else:
            self._update_buttons(self._result)

    def set_progress(self, message: str):
        self._progress_label.setText(message)

    def _update_buttons_for_checking(self):
        self._cancel_btn.setVisible(True)
        self._back_btn.setVisible(False)
        self._start_anyway_btn.setVisible(False)
        self._start_btn.setVisible(False)

    def update_result(self, result: PreflightResult):
        """Update the dialog with completed results."""
        self._result = result
        self._checking = False
        self._progress_label.hide()
        self._populate(result)

    def _on_issue_selected(self, row: int):
        if row < 0:
            self._detail_panel.clear()
            return

        item = self._issue_list.item(row)
        issue = item.data(Qt.ItemDataRole.UserRole)
        if not issue:
            return

        lines = [
            f"[{issue.severity.value.upper()}] {issue.title}",
            f"Code: {issue.code}",
            "",
            issue.message,
        ]
        if issue.details:
            lines.append(f"\nDetails: {issue.details}")
        if issue.path:
            lines.append(f"\nPath: {issue.path}")
        if issue.field_name:
            lines.append(f"Field: {issue.field_name}")
        if issue.suggestion:
            lines.append(f"\nSuggestion: {issue.suggestion}")

        self._detail_panel.setPlainText("\n".join(lines))

    # ── Button handlers ─────────────────────────────────────────────

    def _on_back(self):
        self._user_choice = "back"
        self.reject()

    def _on_start(self):
        self._user_choice = "start"
        self.accept()

    def _on_start_anyway(self):
        self._user_choice = "start_anyway"
        self.accept()

    def _on_cancel(self):
        self._user_choice = "cancel"
        self.reject()

    @property
    def user_choice(self) -> str:
        return self._user_choice or "back"

    def closeEvent(self, event):
        """Window close (X, Alt+F4) must NOT start training."""
        self._user_choice = "cancel"
        self.reject()
        event.accept()

    def reject(self):
        """Override to ensure user_choice is set before rejecting."""
        if self._user_choice is None:
            self._user_choice = "cancel"
        super().reject()
