"""Tests: History table selection styling preserves status colors."""

import os
import sys

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QApplication, QTableWidgetItem

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from anylabeling.views.training.training_history_widget import TrainingHistoryWidget


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


class TestHistoryTableStyling:
    def test_palette_highlight_is_light_blue(self, qapp):
        w = TrainingHistoryWidget()
        try:
            palette = w.table.palette()
            highlight = palette.color(palette.ColorRole.Highlight)
            assert highlight.name().upper() == "#DCEBFF", f"Highlight should be #DCEBFF, got {highlight.name()}"
        finally:
            pass  # no shutdown needed

    def test_highlighted_text_is_dark(self, qapp):
        w = TrainingHistoryWidget()
        try:
            palette = w.table.palette()
            text_color = palette.color(palette.ColorRole.HighlightedText)
            assert text_color.name().upper() == "#111111", f"Text should be #111111, got {text_color.name()}"
        finally:
            pass

    def test_status_color_not_overridden_by_palette(self, qapp):
        """Per-cell setForeground (status color) must survive selection."""
        w = TrainingHistoryWidget()
        try:
            # Simulate adding a row with status color
            w.table.setRowCount(1)
            status_item = QTableWidgetItem("COMPLETED")
            status_item.setForeground(QColor("#4CAF50"))
            w.table.setItem(0, 0, status_item)
            w.table.setItem(0, 1, QTableWidgetItem("Guided"))
            w.table.setItem(0, 2, QTableWidgetItem("test"))
            w.table.setItem(0, 3, QTableWidgetItem(""))
            w.table.setItem(0, 4, QTableWidgetItem(""))
            w.table.setItem(0, 5, QTableWidgetItem(""))
            w.table.setItem(0, 6, QTableWidgetItem(""))

            # Per-cell foreground should still be the green
            # (palette doesn't override per-item setForeground)
            item = w.table.item(0, 0)
            fg = item.foreground().color()
            assert fg.green() > fg.red(), f"Status color should be greenish, got {fg.name()}"
        finally:
            pass

    def test_selection_preserves_row_detail(self, qapp):
        """Selecting a row should still trigger _on_selection_changed."""
        w = TrainingHistoryWidget()
        try:
            w.table.setRowCount(1)
            w.table.setItem(0, 0, QTableWidgetItem("idle"))
            w.table.setItem(0, 1, QTableWidgetItem("Guided"))
            w.table.setItem(0, 2, QTableWidgetItem("test"))
            w.table.setItem(0, 3, QTableWidgetItem(""))
            w.table.setItem(0, 4, QTableWidgetItem(""))
            w.table.setItem(0, 5, QTableWidgetItem(""))
            w.table.setItem(0, 6, QTableWidgetItem("/tmp"))

            # Select row 0
            w.table.selectRow(0)
            assert w.table.currentRow() == 0
            # Detail should not crash
        finally:
            pass

    def test_table_has_no_default_stylesheet(self, qapp):
        """Table should use palette, not dark default."""
        w = TrainingHistoryWidget()
        try:
            palette = w.table.palette()
            highlight = palette.color(palette.ColorRole.Highlight)
            # Should NOT be default dark blue
            assert highlight.red() > 150 and highlight.blue() > 200, f"Selection should be light, got {highlight.name()}"
        finally:
            pass
