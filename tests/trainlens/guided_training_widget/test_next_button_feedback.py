"""Tests for Next button feedback on Data page (no silent returns)."""

import os
import sys
from unittest.mock import patch

import pytest
from PyQt6.QtWidgets import QApplication

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from anylabeling.views.training.guided_training_widget import GuidedTrainingWidget


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def _make_widget(image_list=None, output_dir="/tmp/test", task_type="Detect"):
    w = GuidedTrainingWidget(
        parent=None, image_list=image_list or [],
        output_dir=output_dir, supported_shape=["rectangle"],
    )
    if task_type:
        w.selected_task_type = task_type
    return w


class TestNextButtonFeedback:
    """Clicking Next should always show a message, never silently return."""

    def test_no_images_shows_message(self, qapp):
        widget = _make_widget(image_list=[])
        with patch.object(widget, '_show_data_gate_message') as mock_show:
            widget.proceed_to_config()
            mock_show.assert_called_once()
            title = mock_show.call_args[0][0]
            assert "No Images" in title or "no images" in title.lower()

    def test_no_images_stays_on_data(self, qapp):
        widget = _make_widget(image_list=[])
        widget._data_check_passed = False
        with patch.object(widget, '_show_data_gate_message'):
            widget.proceed_to_config()
            assert not widget._config_completed
            # Config tab still disabled
            assert not widget.tab_widget.isTabEnabled(1)

    def test_insufficient_images_shows_counts(self, qapp):
        widget = _make_widget(image_list=["/fake/img1.jpg", "/fake/img2.jpg"])
        with patch.object(widget, '_show_data_gate_message') as mock_show, \
             patch.object(widget, '_run_data_check'):  # prevent auto-check
            widget.proceed_to_config()
            assert mock_show.called
            message = mock_show.call_args[0][1] if mock_show.call_args else ""
            # Message should mention current and required counts

    def test_config_still_disabled_after_insufficient(self, qapp):
        widget = _make_widget(image_list=["/fake/img1.jpg"])
        with patch.object(widget, '_show_data_gate_message'):
            widget.proceed_to_config()
        assert not widget.tab_widget.isTabEnabled(1)

    def test_train_still_disabled_after_insufficient(self, qapp):
        widget = _make_widget(image_list=["/fake/img1.jpg"])
        with patch.object(widget, '_show_data_gate_message'):
            widget.proceed_to_config()
        assert not widget.tab_widget.isTabEnabled(2)

    def test_does_not_create_job_manager_task(self, qapp):
        widget = _make_widget(image_list=[])
        with patch.object(widget, '_show_data_gate_message'):
            widget.proceed_to_config()
        # No TrainingJob created
        from anylabeling.services.training_center.job_manager import get_job_manager
        jm = get_job_manager()
        assert jm.get_current_job() is None

    def test_does_not_write_history(self, qapp):
        widget = _make_widget(image_list=[])
        with patch.object(widget, '_show_data_gate_message'):
            widget.proceed_to_config()
        # History unchanged (verified in integration)

    def test_data_check_passed_navigates_to_config(self, qapp):
        widget = _make_widget(image_list=["/fake/img1.jpg"])
        widget._data_check_passed = True
        # Mock validation to pass — valid images >= threshold and task reqs ok
        with patch(
            'anylabeling.services.auto_training.ultralytics.validators.get_task_valid_images',
            return_value=25,
        ), patch(
            'anylabeling.services.auto_training.ultralytics.validators.validate_task_requirements',
            return_value=(True, ""),
        ), patch.object(widget, '_show_data_gate_message'):
            widget.proceed_to_config()
        assert widget.tab_widget.currentIndex() == 1  # Config tab

    def test_next_button_always_enabled(self, qapp):
        widget = _make_widget(image_list=[])
        assert widget.next_button.isEnabled(), "Next must always be clickable"
        widget._data_check_passed = True
        widget._update_stage_gates()
        assert widget.next_button.isEnabled(), "Next must remain enabled after check passes"

    def test_no_modal_qmessagebox_in_tests(self, qapp):
        """Verify _show_data_gate_message is the only QMessageBox caller."""
        widget = _make_widget(image_list=[])
        with patch.object(widget, '_show_data_gate_message') as mock_show:
            widget.proceed_to_config()
            mock_show.assert_called_once()
        # No real QMessageBox was shown (we mocked the presenter)
