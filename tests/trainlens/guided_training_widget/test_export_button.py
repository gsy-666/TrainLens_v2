"""Regression tests for Export button - QDialog import (NameError fix)."""

import os
import sys
import tempfile
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from PyQt6.QtWidgets import QApplication, QDialog

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from anylabeling.views.training.guided_training_widget import GuidedTrainingWidget


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


class FakeExportDialog(QDialog):
    """A controllable ExportFormatDialog stand-in that never blocks."""

    def __init__(self, parent=None, accept=False):
        super().__init__(parent)
        self._accept = accept
        self._selected_format = "onnx"

    def exec(self):
        return QDialog.DialogCode.Accepted if self._accept else QDialog.DialogCode.Rejected

    def get_selected_format(self):
        return self._selected_format


@pytest.fixture
def widget_with_export_ready(qapp, tmp_path):
    """Create a GuidedTrainingWidget with a valid project dir and best.pt."""
    project_dir = tmp_path / "export_test_project"
    weights_dir = project_dir / "weights"
    weights_dir.mkdir(parents=True)
    best_pt = weights_dir / "best.pt"
    best_pt.write_bytes(b"mock model weights")

    widget = GuidedTrainingWidget(
        parent=None,
        image_list=[],
        output_dir=str(project_dir),
        supported_shape=["rectangle", "polygon"],
    )
    widget.current_project_path = str(project_dir)
    return widget


class TestExportNoNameError:
    """Verify start_export() does not raise NameError for QDialog."""

    def test_rejected_dialog_does_not_raise_nameerror(self, qapp, widget_with_export_ready):
        """When user cancels export (Rejected), no NameError is raised."""
        widget = widget_with_export_ready

        fake_dialog = FakeExportDialog(parent=widget, accept=False)
        mock_export_manager = MagicMock()

        with patch(
            "anylabeling.views.training.guided_training_widget.ExportFormatDialog",
            return_value=fake_dialog,
        ), patch.object(widget, "export_manager", mock_export_manager):
            # This should NOT raise NameError
            widget.start_export()

        # Export should NOT have been called (user cancelled)
        mock_export_manager.start_export.assert_not_called()

    def test_accepted_dialog_does_not_raise_nameerror(self, qapp, widget_with_export_ready):
        """When user accepts export (Accepted), no NameError and export proceeds."""
        widget = widget_with_export_ready

        fake_dialog = FakeExportDialog(parent=widget, accept=True)
        mock_export_manager = MagicMock()
        mock_export_manager.start_export.return_value = (True, "Exported OK")

        with patch(
            "anylabeling.views.training.guided_training_widget.ExportFormatDialog",
            return_value=fake_dialog,
        ), patch.object(widget, "export_manager", mock_export_manager):
            widget.start_export()

        mock_export_manager.start_export.assert_called_once()

    def test_qdialog_dialogcode_accepted_is_accessible(self, qapp):
        """QDialog.DialogCode.Accepted is importable and usable."""
        # This is the exact line that caused NameError
        result = QDialog.DialogCode.Accepted
        assert result == QDialog.DialogCode.Accepted
        assert result == 1  # standard Qt value

    def test_qdialog_dialogcode_rejected_is_accessible(self, qapp):
        """QDialog.DialogCode.Rejected is importable and usable."""
        result = QDialog.DialogCode.Rejected
        assert result == QDialog.DialogCode.Rejected
        assert result == 0  # standard Qt value

    def test_export_manager_not_called_on_rejected(self, qapp, widget_with_export_ready):
        """Double-check: export_manager.start_export never called on Rejected."""
        widget = widget_with_export_ready

        fake_dialog = FakeExportDialog(parent=widget, accept=False)
        mock_export_manager = MagicMock()

        with patch(
            "anylabeling.views.training.guided_training_widget.ExportFormatDialog",
            return_value=fake_dialog,
        ), patch.object(widget, "export_manager", mock_export_manager):
            widget.start_export()

        mock_export_manager.start_export.assert_not_called()

    def test_export_button_triggers_start_export(self, qapp, widget_with_export_ready):
        """Export button click calls start_export without NameError."""
        widget = widget_with_export_ready
        # Train tab is lazily initialized — the export_button only exists after
        widget.ensure_train_tab_initialized()

        fake_dialog = FakeExportDialog(parent=widget, accept=False)
        mock_export_manager = MagicMock()

        with patch(
            "anylabeling.views.training.guided_training_widget.ExportFormatDialog",
            return_value=fake_dialog,
        ), patch.object(widget, "export_manager", mock_export_manager):
            # Simulate clicking the Export button
            widget.export_button.click()

        # Should not have called export (rejected)
        mock_export_manager.start_export.assert_not_called()
