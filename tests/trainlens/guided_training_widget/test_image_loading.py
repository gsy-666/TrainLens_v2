"""Test image loading callbacks and Load Images button behavior"""

import sys
import os
import re
import pytest
from unittest.mock import Mock
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


@pytest.fixture
def mock_dependencies():
    """Mock JobManager, Adapter, HistoryStore"""
    job_manager = Mock()
    job_manager.get_current_job = Mock(return_value=None)
    job_manager.request_start = Mock(return_value=(True, ""))
    job_manager.request_stop = Mock(return_value=True)
    job_manager.subscribe_events = Mock()
    job_manager.unsubscribe_events = Mock()

    adapter = Mock()
    adapter.can_start = Mock(return_value=(True, ""))

    history_store = Mock()
    history_store.save_record = Mock()

    return job_manager, adapter, history_store


class FakeHost(QWidget):
    """Fake LabelWidget with open_folder_dialog and image_list property"""

    def __init__(self):
        super().__init__()
        self._image_list = []
        self.open_folder_dialog_called = 0
        self.output_dir = "/tmp/test"
        self.supported_shape = []

    def open_folder_dialog(self):
        """Simulate LabelWidget.open_folder_dialog()"""
        self.open_folder_dialog_called += 1
        # Simulate loading some images
        self._image_list = ["/fake/image1.jpg", "/fake/image2.jpg"]

    @property
    def image_list(self):
        """Simulate LabelWidget.image_list property"""
        return self._image_list


class TestImageLoadingCallbacks:
    """Test image loading callbacks and Load Images button"""

    def test_load_images_with_callbacks_calls_host_method(self, qapp, mock_dependencies):
        """Load Images with callbacks calls host's open_folder_dialog exactly once"""
        from anylabeling.views.training.guided_training_widget import GuidedTrainingWidget

        job_manager, adapter, history_store = mock_dependencies
        fake_host = FakeHost()

        widget = GuidedTrainingWidget(
            parent=fake_host,
            image_list=[],
            output_dir="/tmp/test",
            supported_shape=[],
            job_manager=job_manager,
            ultralytics_adapter=adapter,
            history_store=history_store,
            open_folder_callback=fake_host.open_folder_dialog,
            get_image_list_callback=lambda: fake_host.image_list
        )

        # Simulate button click
        widget.load_images()

        assert fake_host.open_folder_dialog_called == 1
        assert widget.image_list == ["/fake/image1.jpg", "/fake/image2.jpg"]

    def test_load_images_without_callbacks_shows_message(self, qapp, mock_dependencies):
        """Load Images without callbacks shows informative message, no crash"""
        from anylabeling.views.training.guided_training_widget import GuidedTrainingWidget

        job_manager, adapter, history_store = mock_dependencies

        widget = GuidedTrainingWidget(
            parent=None,
            image_list=[],
            output_dir="/tmp/test",
            supported_shape=[],
            job_manager=job_manager,
            ultralytics_adapter=adapter,
            history_store=history_store,
            open_folder_callback=None,  # No callback
            get_image_list_callback=None
        )

        # Should not crash
        widget.load_images()
        # Expect QMessageBox.information to have been shown (no assertion, just no crash)

    def test_ultralytics_dialog_extracts_callbacks_from_parent(self, qapp, mock_dependencies):
        """UltralyticsDialog extracts callbacks from parent and passes to widget"""
        from anylabeling.views.training.ultralytics_dialog import UltralyticsDialog

        job_manager, adapter, history_store = mock_dependencies
        fake_host = FakeHost()

        dialog = UltralyticsDialog(
            parent=fake_host,
            job_manager=job_manager,
            ultralytics_adapter=adapter,
            history_store=history_store
        )

        # Verify widget received callbacks
        assert dialog.training_widget._open_folder_callback is not None
        assert dialog.training_widget._get_image_list_callback is not None

        # Simulate Load Images
        dialog.training_widget.load_images()

        assert fake_host.open_folder_dialog_called == 1
        assert dialog.training_widget.image_list == ["/fake/image1.jpg", "/fake/image2.jpg"]

    def test_load_images_button_click_does_not_crash(self, qapp, mock_dependencies):
        """Clicking Load Images button does not crash"""
        from anylabeling.views.training.guided_training_widget import GuidedTrainingWidget

        job_manager, adapter, history_store = mock_dependencies
        fake_host = FakeHost()

        widget = GuidedTrainingWidget(
            parent=fake_host,
            image_list=[],
            output_dir="/tmp/test",
            supported_shape=[],
            job_manager=job_manager,
            ultralytics_adapter=adapter,
            history_store=history_store,
            open_folder_callback=fake_host.open_folder_dialog,
            get_image_list_callback=lambda: fake_host.image_list
        )

        # init_data_tab is called during __init__, button is already created
        widget.show()
        qapp.processEvents()

        # Find and click Load Images button
        load_button = widget.load_images_button
        assert load_button is not None
        assert load_button.isEnabled()

        # Simulate click
        load_button.click()
        qapp.processEvents()

        # No AttributeError
        assert fake_host.open_folder_dialog_called == 1

    def test_no_duplicate_calls_on_multiple_loads(self, qapp, mock_dependencies):
        """Multiple Load Images clicks do not cause duplicate callback calls"""
        from anylabeling.views.training.guided_training_widget import GuidedTrainingWidget

        job_manager, adapter, history_store = mock_dependencies
        fake_host = FakeHost()

        widget = GuidedTrainingWidget(
            parent=fake_host,
            image_list=[],
            output_dir="/tmp/test",
            supported_shape=[],
            job_manager=job_manager,
            ultralytics_adapter=adapter,
            history_store=history_store,
            open_folder_callback=fake_host.open_folder_dialog,
            get_image_list_callback=lambda: fake_host.image_list
        )

        # Load images twice
        widget.load_images()
        widget.load_images()

        # Each call should trigger exactly once
        assert fake_host.open_folder_dialog_called == 2

    def test_no_fragile_parent_calls_remain(self, qapp, mock_dependencies):
        """Verify no unsafe self.parent().xxx calls remain in guided_training_widget.py"""
        import anylabeling.views.training.guided_training_widget as gtw_module

        source_file = gtw_module.__file__
        with open(source_file, 'r', encoding='utf-8') as f:
            source_code = f.read()

        # Find all self.parent().xxx patterns (attribute access, not method calls)
        # Fragile pattern: self.parent().attribute_name (without parentheses after)
        pattern = r'self\.parent\(\)\.\w+(?!\()'
        matches = re.findall(pattern, source_code)

        # Should have no fragile parent attribute dependencies
        assert len(matches) == 0, f"Found unsafe parent attribute calls: {matches}"
