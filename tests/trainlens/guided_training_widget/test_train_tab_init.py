"""Test Train tab initialization in widget tests"""

import sys
import os
import pytest
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from anylabeling.views.training.guided_training_widget import GuidedTrainingWidget


@pytest.fixture(scope="session")
def qapp():
    """Create QApplication for real Qt tests"""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


class TestTrainTabInitialization:
    """Test Train tab lazy initialization"""

    def test_train_tab_not_initialized_by_default(self, qapp):
        """Train tab is not initialized until accessed"""
        widget = GuidedTrainingWidget(
            parent=None,
            image_list=[],
            output_dir="/tmp/test",
            supported_shape=[]
        )

        # Train tab should not be initialized on construction
        assert widget._train_tab_initialized is False

    def test_train_tab_initializes_on_go_to_tab(self, qapp):
        """Train tab initializes when switching to it"""
        widget = GuidedTrainingWidget(
            parent=None,
            image_list=[],
            output_dir="/tmp/test",
            supported_shape=[]
        )

        # Navigate to Train tab
        widget.go_to_specific_tab(2)

        # Should now be initialized
        assert widget._train_tab_initialized is True

    def test_train_tab_has_layout_after_init(self, qapp):
        """Train tab has layout after initialization"""
        widget = GuidedTrainingWidget(
            parent=None,
            image_list=[],
            output_dir="/tmp/test",
            supported_shape=[]
        )

        widget.go_to_specific_tab(2)

        assert widget.train_tab.layout() is not None

    def test_train_tab_has_status_label(self, qapp):
        """Train tab has status label after initialization"""
        widget = GuidedTrainingWidget(
            parent=None,
            image_list=[],
            output_dir="/tmp/test",
            supported_shape=[]
        )

        widget.go_to_specific_tab(2)

        assert hasattr(widget, 'status_label')
        assert widget.status_label is not None

    def test_train_tab_has_progress_bar(self, qapp):
        """Train tab has progress bar after initialization"""
        widget = GuidedTrainingWidget(
            parent=None,
            image_list=[],
            output_dir="/tmp/test",
            supported_shape=[]
        )

        widget.go_to_specific_tab(2)

        assert hasattr(widget, 'progress_bar')
        assert widget.progress_bar is not None

    def test_train_tab_has_log_display(self, qapp):
        """Train tab has log display after initialization"""
        widget = GuidedTrainingWidget(
            parent=None,
            image_list=[],
            output_dir="/tmp/test",
            supported_shape=[]
        )

        widget.go_to_specific_tab(2)

        assert hasattr(widget, 'log_display')
        assert widget.log_display is not None

    def test_train_tab_has_start_button(self, qapp):
        """Train tab has start training button after initialization"""
        widget = GuidedTrainingWidget(
            parent=None,
            image_list=[],
            output_dir="/tmp/test",
            supported_shape=[]
        )

        widget.go_to_specific_tab(2)

        assert hasattr(widget, 'start_training_button')
        assert widget.start_training_button is not None

    def test_train_tab_has_stop_button(self, qapp):
        """Train tab has stop training button after initialization"""
        widget = GuidedTrainingWidget(
            parent=None,
            image_list=[],
            output_dir="/tmp/test",
            supported_shape=[]
        )

        widget.go_to_specific_tab(2)

        assert hasattr(widget, 'stop_training_button')
        assert widget.stop_training_button is not None

    def test_train_tab_has_export_button(self, qapp):
        """Train tab has export button after initialization"""
        widget = GuidedTrainingWidget(
            parent=None,
            image_list=[],
            output_dir="/tmp/test",
            supported_shape=[]
        )

        widget.go_to_specific_tab(2)

        assert hasattr(widget, 'export_button')
        assert widget.export_button is not None

    def test_train_tab_has_image_labels(self, qapp):
        """Train tab has image labels after initialization"""
        widget = GuidedTrainingWidget(
            parent=None,
            image_list=[],
            output_dir="/tmp/test",
            supported_shape=[]
        )

        widget.go_to_specific_tab(2)

        assert hasattr(widget, 'image_labels')
        assert len(widget.image_labels) == 6

    def test_train_tab_initializes_only_once(self, qapp):
        """Train tab initialization is idempotent"""
        widget = GuidedTrainingWidget(
            parent=None,
            image_list=[],
            output_dir="/tmp/test",
            supported_shape=[]
        )

        # Initialize multiple times
        widget.go_to_specific_tab(2)
        widget.go_to_specific_tab(2)
        widget.go_to_specific_tab(2)

        # Should still have exactly one layout and one set of widgets
        assert widget._train_tab_initialized is True
        assert widget.train_tab.layout() is not None
        assert widget.train_tab.layout().count() == 2  # scroll_area + actions_layout

    def test_config_tab_initializes_before_train(self, qapp):
        """Config tab initializes when going to Train tab"""
        widget = GuidedTrainingWidget(
            parent=None,
            image_list=[],
            output_dir="/tmp/test",
            supported_shape=[]
        )

        # Go directly to Train tab
        widget.go_to_specific_tab(2)

        # Config should also be initialized
        assert widget._config_tab_initialized is True
        assert widget._train_tab_initialized is True
