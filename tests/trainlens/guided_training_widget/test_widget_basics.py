"""Basic tests for GuidedTrainingWidget

Tests widget construction, type, and basic properties.
Uses real PyQt6 with QT_QPA_PLATFORM=offscreen.
"""

import sys
import os
import pytest
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QDialog

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


class TestGuidedTrainingWidgetBasics:
    """Test basic GuidedTrainingWidget functionality"""

    def test_widget_can_construct_with_no_parent(self, qapp):
        """GuidedTrainingWidget can be constructed without parent"""
        widget = GuidedTrainingWidget(
            parent=None,
            image_list=[],
            output_dir="/tmp/test",
            supported_shape=["rectangle", "polygon"]
        )
        assert widget is not None

    def test_widget_is_qwidget(self, qapp):
        """GuidedTrainingWidget is a QWidget"""
        widget = GuidedTrainingWidget(
            parent=None,
            image_list=[],
            output_dir="/tmp/test",
            supported_shape=[]
        )
        assert isinstance(widget, QWidget)

    def test_widget_is_not_qdialog(self, qapp):
        """GuidedTrainingWidget is NOT a QDialog"""
        widget = GuidedTrainingWidget(
            parent=None,
            image_list=[],
            output_dir="/tmp/test",
            supported_shape=[]
        )
        assert not isinstance(widget, QDialog)

    def test_widget_has_tab_widget(self, qapp):
        """GuidedTrainingWidget has tab_widget attribute"""
        widget = GuidedTrainingWidget(
            parent=None,
            image_list=[],
            output_dir="/tmp/test",
            supported_shape=[]
        )
        assert hasattr(widget, 'tab_widget')
        assert widget.tab_widget is not None

    def test_widget_has_three_tabs(self, qapp):
        """GuidedTrainingWidget has Data, Config, Train tabs"""
        widget = GuidedTrainingWidget(
            parent=None,
            image_list=[],
            output_dir="/tmp/test",
            supported_shape=[]
        )
        assert widget.tab_widget.count() == 3
        assert widget.tab_widget.tabText(0) in ["Data", "数据"]
        assert widget.tab_widget.tabText(1) in ["Config", "配置"]
        assert widget.tab_widget.tabText(2) in ["Train", "训练"]

    def test_widget_has_training_manager(self, qapp):
        """GuidedTrainingWidget has training_manager"""
        widget = GuidedTrainingWidget(
            parent=None,
            image_list=[],
            output_dir="/tmp/test",
            supported_shape=[]
        )
        assert hasattr(widget, 'training_manager')
        assert widget.training_manager is not None

    def test_widget_has_export_manager(self, qapp):
        """GuidedTrainingWidget has export_manager"""
        widget = GuidedTrainingWidget(
            parent=None,
            image_list=[],
            output_dir="/tmp/test",
            supported_shape=[]
        )
        assert hasattr(widget, 'export_manager')
        assert widget.export_manager is not None

    def test_widget_can_be_embedded_in_layout(self, qapp):
        """GuidedTrainingWidget can be embedded in a layout"""
        container = QWidget()
        layout = QVBoxLayout(container)

        widget = GuidedTrainingWidget(
            parent=None,
            image_list=[],
            output_dir="/tmp/test",
            supported_shape=[]
        )
        layout.addWidget(widget)

        assert widget.parent() is None or widget.parent() == container
        assert layout.count() == 1

    def test_widget_shutdown_is_idempotent(self, qapp):
        """Widget shutdown can be called multiple times"""
        widget = GuidedTrainingWidget(
            parent=None,
            image_list=[],
            output_dir="/tmp/test",
            supported_shape=[]
        )

        # Should not raise exception
        widget.shutdown()
        widget.shutdown()
        widget.shutdown()

    def test_widget_has_active_training_method(self, qapp):
        """Widget has has_active_training method"""
        widget = GuidedTrainingWidget(
            parent=None,
            image_list=[],
            output_dir="/tmp/test",
            supported_shape=[]
        )

        assert hasattr(widget, 'has_active_training')
        assert callable(widget.has_active_training)
        assert widget.has_active_training() is False  # Initially idle

    def test_widget_has_request_stop_method(self, qapp):
        """Widget has request_stop method"""
        widget = GuidedTrainingWidget(
            parent=None,
            image_list=[],
            output_dir="/tmp/test",
            supported_shape=[]
        )

        assert hasattr(widget, 'request_stop')
        assert callable(widget.request_stop)

    def test_widget_accepts_image_list_parameter(self, qapp):
        """Widget accepts image_list parameter"""
        test_images = ["img1.jpg", "img2.jpg"]
        widget = GuidedTrainingWidget(
            parent=None,
            image_list=test_images,
            output_dir="/tmp/test",
            supported_shape=[]
        )

        assert widget.image_list == test_images

    def test_widget_accepts_output_dir_parameter(self, qapp):
        """Widget accepts output_dir parameter"""
        test_dir = "/tmp/output"
        widget = GuidedTrainingWidget(
            parent=None,
            image_list=[],
            output_dir=test_dir,
            supported_shape=[]
        )

        assert widget.output_dir == test_dir

    def test_widget_accepts_supported_shape_parameter(self, qapp):
        """Widget accepts supported_shape parameter"""
        test_shapes = ["rectangle", "polygon", "circle"]
        widget = GuidedTrainingWidget(
            parent=None,
            image_list=[],
            output_dir="/tmp/test",
            supported_shape=test_shapes
        )

        assert widget.supported_shape == test_shapes
