"""Test image loading callbacks and Load Images button behavior.

Updated for explicit dependency injection (open_folder_callback / image_list_getter).
No longer relies on parent() chain traversal.
"""

import sys
import os
import pytest
from unittest.mock import patch
from PyQt6.QtWidgets import QApplication, QWidget, QMessageBox

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


class FakeHost(QWidget):
    """Fake LabelWidget providing image_list and dialog callback."""

    def __init__(self):
        super().__init__()
        self._image_list = []
        self.open_folder_dialog_called = 0
        self.output_dir = "/tmp/test"
        self.supported_shape = []

    def open_folder_dialog(self, _value=False, dirpath=None):
        """Match real LabelWidget.open_folder_dialog signature."""
        self.open_folder_dialog_called += 1
        self._image_list = ["/fake/image1.jpg", "/fake/image2.jpg"]

    @property
    def image_list(self):
        return self._image_list


class TestImageLoadingCallbacks:

    def test_load_images_calls_callback_once(self, qapp):
        from anylabeling.views.training.guided_training_widget import GuidedTrainingWidget

        host = FakeHost()
        widget = GuidedTrainingWidget(
            parent=None,
            image_list=[],
            open_folder_callback=host.open_folder_dialog,
            image_list_getter=lambda: host.image_list,
        )
        widget.load_images()
        assert host.open_folder_dialog_called == 1

    def test_load_images_updates_from_getter(self, qapp):
        from anylabeling.views.training.guided_training_widget import GuidedTrainingWidget

        host = FakeHost()
        widget = GuidedTrainingWidget(
            parent=None, image_list=[],
            open_folder_callback=host.open_folder_dialog,
            image_list_getter=lambda: host.image_list,
        )
        widget.load_images()
        assert widget.image_list == ["/fake/image1.jpg", "/fake/image2.jpg"]

    def test_standalone_shows_info_no_crash(self, qapp):
        from anylabeling.views.training.guided_training_widget import GuidedTrainingWidget

        widget = GuidedTrainingWidget(parent=None, image_list=[])
        with patch.object(QMessageBox, 'information', return_value=None):
            widget.load_images()

    def test_button_click_with_callbacks(self, qapp):
        from anylabeling.views.training.guided_training_widget import GuidedTrainingWidget

        host = FakeHost()
        widget = GuidedTrainingWidget(
            parent=None, image_list=[],
            open_folder_callback=host.open_folder_dialog,
            image_list_getter=lambda: host.image_list,
        )
        widget.show()
        qapp.processEvents()
        widget.load_images_button.click()
        qapp.processEvents()
        assert host.open_folder_dialog_called == 1

    def test_multiple_loads_each_call_callback(self, qapp):
        from anylabeling.views.training.guided_training_widget import GuidedTrainingWidget

        host = FakeHost()
        widget = GuidedTrainingWidget(
            parent=None, image_list=[],
            open_folder_callback=host.open_folder_dialog,
            image_list_getter=lambda: host.image_list,
        )
        widget.load_images()
        widget.load_images()
        assert host.open_folder_dialog_called == 2

    def test_no_parent_dot_open_folder_dialog_in_source(self):
        import inspect
        from anylabeling.views.training.guided_training_widget import GuidedTrainingWidget

        source = inspect.getsource(GuidedTrainingWidget.load_images)
        assert "self.parent().open_folder_dialog" not in source
        assert "self.parent().image_list" not in source
