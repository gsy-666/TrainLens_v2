"""Tests for UltralyticsDialog wrapper

Verifies that UltralyticsDialog is now a thin wrapper around GuidedTrainingWidget.
Uses real PyQt6 with QT_QPA_PLATFORM=offscreen.
"""

import sys
import os
import pytest
from PyQt6.QtWidgets import QApplication, QWidget, QDialog

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from anylabeling.views.training.guided_training_widget import GuidedTrainingWidget
from anylabeling.views.training.ultralytics_dialog import UltralyticsDialog


@pytest.fixture(scope="session")
def qapp():
    """Create QApplication for real Qt tests"""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


class MockParent(QWidget):
    """Mock parent with required attributes"""
    def __init__(self):
        super().__init__()
        self.image_list = ["test1.jpg", "test2.jpg"]
        self.output_dir = "/tmp/test"
        self.supported_shape = ["rectangle", "polygon"]


class TestUltralyticsDialogWrapper:
    """Test UltralyticsDialog wrapper functionality"""

    def test_dialog_can_construct(self, qapp):
        """UltralyticsDialog can be constructed"""
        parent = MockParent()
        dialog = UltralyticsDialog(parent)
        assert dialog is not None

    def test_dialog_is_qdialog(self, qapp):
        """UltralyticsDialog is a QDialog"""
        parent = MockParent()
        dialog = UltralyticsDialog(parent)
        assert isinstance(dialog, QDialog)

    def test_dialog_contains_training_widget(self, qapp):
        """UltralyticsDialog contains GuidedTrainingWidget"""
        parent = MockParent()
        dialog = UltralyticsDialog(parent)

        assert hasattr(dialog, 'training_widget')
        assert isinstance(dialog.training_widget, GuidedTrainingWidget)

    def test_dialog_proxies_training_manager(self, qapp):
        """UltralyticsDialog proxies training_manager"""
        parent = MockParent()
        dialog = UltralyticsDialog(parent)

        assert hasattr(dialog, 'training_manager')
        assert dialog.training_manager is dialog.training_widget.training_manager

    def test_dialog_proxies_export_manager(self, qapp):
        """UltralyticsDialog proxies export_manager"""
        parent = MockParent()
        dialog = UltralyticsDialog(parent)

        assert hasattr(dialog, 'export_manager')
        assert dialog.export_manager is dialog.training_widget.export_manager

    def test_dialog_proxies_image_list_getter(self, qapp):
        """UltralyticsDialog proxies image_list getter"""
        parent = MockParent()
        dialog = UltralyticsDialog(parent)

        assert dialog.image_list == parent.image_list

    def test_dialog_proxies_image_list_setter(self, qapp):
        """UltralyticsDialog proxies image_list setter"""
        parent = MockParent()
        dialog = UltralyticsDialog(parent)

        new_images = ["new1.jpg", "new2.jpg"]
        dialog.image_list = new_images
        assert dialog.training_widget.image_list == new_images

    def test_dialog_proxies_output_dir_getter(self, qapp):
        """UltralyticsDialog proxies output_dir getter"""
        parent = MockParent()
        dialog = UltralyticsDialog(parent)

        assert dialog.output_dir == parent.output_dir

    def test_dialog_proxies_output_dir_setter(self, qapp):
        """UltralyticsDialog proxies output_dir setter"""
        parent = MockParent()
        dialog = UltralyticsDialog(parent)

        new_dir = "/tmp/new_output"
        dialog.output_dir = new_dir
        assert dialog.training_widget.output_dir == new_dir

    def test_dialog_has_window_title(self, qapp):
        """UltralyticsDialog has window title set"""
        parent = MockParent()
        dialog = UltralyticsDialog(parent)

        title = dialog.windowTitle()
        assert title is not None
        assert len(title) > 0

    def test_dialog_layout_has_no_margins(self, qapp):
        """UltralyticsDialog layout has zero margins"""
        parent = MockParent()
        dialog = UltralyticsDialog(parent)

        layout = dialog.layout()
        assert layout is not None
        margins = layout.contentsMargins()
        assert margins.left() == 0
        assert margins.right() == 0
        assert margins.top() == 0
        assert margins.bottom() == 0

    def test_dialog_widget_is_only_child(self, qapp):
        """UltralyticsDialog has only one widget in layout"""
        parent = MockParent()
        dialog = UltralyticsDialog(parent)

        layout = dialog.layout()
        assert layout.count() == 1
        assert layout.itemAt(0).widget() is dialog.training_widget

    def test_dialog_backward_compatible_construction(self, qapp):
        """UltralyticsDialog maintains backward compatible construction"""
        parent = MockParent()

        # Old construction pattern should still work
        dialog = UltralyticsDialog(parent)

        # Should have access to all expected attributes
        assert hasattr(dialog, 'training_manager')
        assert hasattr(dialog, 'export_manager')
        assert hasattr(dialog, 'image_list')
        assert hasattr(dialog, 'output_dir')
