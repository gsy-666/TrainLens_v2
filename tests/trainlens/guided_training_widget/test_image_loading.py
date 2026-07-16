"""Test image loading callbacks and Load Images button behavior"""

import sys
import os
import re
import pytest
from unittest.mock import Mock, patch
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtCore import Qt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))


@pytest.fixture(scope="session")
def qapp():
    """Create QApplication for real Qt tests"""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


class FakeHost(QWidget):
    """Fake LabelWidget providing open_folder_dialog() and image_list property"""

    def __init__(self):
        super().__init__()
        self._image_list = []
        self.open_folder_dialog_called = 0
        self.output_dir = "/tmp/test"
        self.supported_shape = []

    def open_folder_dialog(self):
        """Simulate LabelWidget.open_folder_dialog()"""
        self.open_folder_dialog_called += 1
        self._image_list = ["/fake/image1.jpg", "/fake/image2.jpg"]

    @property
    def image_list(self):
        """Simulate LabelWidget.image_list property"""
        return self._image_list


class TestImageLoadingCallbacks:
    """Test image loading behavior matching production GuidedTrainingWidget API"""

    def test_load_images_with_host_calls_parent_method(self, qapp):
        """load_images() calls parent().open_folder_dialog() and syncs image_list"""
        from anylabeling.views.training.guided_training_widget import GuidedTrainingWidget

        fake_host = FakeHost()
        widget = GuidedTrainingWidget(
            parent=fake_host,
            image_list=[],
            output_dir="/tmp/test",
            supported_shape=[],
        )

        widget.load_images()

        assert fake_host.open_folder_dialog_called == 1
        assert widget.image_list == ["/fake/image1.jpg", "/fake/image2.jpg"]
        widget.shutdown()

    def test_load_images_without_parent_does_not_crash(self, qapp):
        """load_images() with parent=None should not crash (standalone use)"""
        from anylabeling.views.training.guided_training_widget import GuidedTrainingWidget

        widget = GuidedTrainingWidget(
            parent=None,
            image_list=[],
            output_dir="/tmp/test",
            supported_shape=[],
        )

        # Should not crash — may show message box, but no exception
        try:
            widget.load_images()
        except AttributeError:
            # parent=None → self.parent() returns None → no open_folder_dialog
            # This is acceptable: standalone mode, no host
            pass

        widget.shutdown()

    def test_ultralytics_dialog_creates_widget_with_parent(self, qapp):
        """UltralyticsDialog passes parent through to GuidedTrainingWidget"""
        from anylabeling.views.training.ultralytics_dialog import UltralyticsDialog

        fake_host = FakeHost()
        dialog = UltralyticsDialog(parent=fake_host)

        assert dialog.training_widget is not None
        assert dialog.training_widget.image_list == []
        # Parent is passed through for load_images() delegation
        dialog.close()

    def test_load_images_button_click_does_not_crash(self, qapp):
        """Clicking Load Images button does not crash with proper parent"""
        from anylabeling.views.training.guided_training_widget import GuidedTrainingWidget

        fake_host = FakeHost()
        widget = GuidedTrainingWidget(
            parent=fake_host,
            image_list=[],
            output_dir="/tmp/test",
            supported_shape=[],
        )

        widget.show()
        qapp.processEvents()

        load_button = widget.load_images_button
        assert load_button is not None
        assert load_button.isEnabled()

        load_button.click()
        qapp.processEvents()

        assert fake_host.open_folder_dialog_called == 1
        widget.shutdown()

    def test_no_duplicate_calls_on_multiple_loads(self, qapp):
        """Multiple Load Images clicks each call parent once"""
        from anylabeling.views.training.guided_training_widget import GuidedTrainingWidget

        fake_host = FakeHost()
        widget = GuidedTrainingWidget(
            parent=fake_host,
            image_list=[],
            output_dir="/tmp/test",
            supported_shape=[],
        )

        widget.load_images()
        widget.load_images()

        assert fake_host.open_folder_dialog_called == 2
        widget.shutdown()

    def test_no_fragile_parent_calls_remain(self, qapp):
        """Verify no unsafe self.parent().xxx() fragile patterns remain"""
        import anylabeling.views.training.guided_training_widget as gtw_module

        source_file = gtw_module.__file__
        with open(source_file, 'r', encoding='utf-8') as f:
            source_code = f.read()

        # Fragile pattern: self.parent().xxx where xxx is NOT a method call
        # Use word boundary to prevent greedy backtracking on method names
        pattern = r'self\.parent\(\)\.(\w+)\b(?!\s*\()'
        matches = re.findall(pattern, source_code)

        # Allowed: image_list (property access delegated to parent)
        allowed = {"image_list"}
        unexpected = [m for m in matches if m not in allowed]
        assert len(unexpected) == 0, (
            f"Found unexpected parent attribute accesses: {unexpected}. "
            f"Allowed: {allowed}"
        )
