"""Tests verifying UI responsiveness during training (tab navigation, config-only disable).

These tests validate that:
- Tab bar is always enabled (user can switch tabs freely)
- Only config controls are disabled during RUNNING
- Console, Stop, Metrics, History remain interactable
- Config controls are re-enabled after terminal events
"""

import os
import sys

import pytest
from PyQt6.QtWidgets import QApplication

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from anylabeling.views.training.guided_training_widget import GuidedTrainingWidget
from anylabeling.views.training.training_center_window import (
    TrainingCenterWindow, TAB_GUIDED, TAB_CUSTOM, TAB_HISTORY,
)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def _simulate_guided_running(widget):
    """Navigate to Train tab and send training_started event (bypass gates for test)."""
    widget._data_check_passed = True
    widget._config_completed = True
    widget.go_to_specific_tab(2)  # init train tab (creates buttons)
    widget.on_training_event("training_started", {
        "total_epochs": 5,
        "save_dir": "/tmp",
    })


# ── GuidedTrainingWidget tab bar ────────────────────────────────────

class TestGuidedTabBarInteractive:
    """The internal tab bar must NOT be disabled."""

    def test_tab_bar_is_enabled_by_default(self, qapp):
        widget = GuidedTrainingWidget(
            parent=None, image_list=[], output_dir="/tmp/test",
            supported_shape=["rectangle"],
        )
        try:
            assert widget.tab_widget.tabBar().isEnabled()
        finally:
            widget.shutdown()

    def test_tab_bar_stays_enabled_after_training_started(self, qapp):
        widget = GuidedTrainingWidget(
            parent=None, image_list=[], output_dir="/tmp/test",
            supported_shape=["rectangle"],
        )
        try:
            _simulate_guided_running(widget)
            assert widget.tab_widget.tabBar().isEnabled()
        finally:
            widget.shutdown()

    def test_metrics_tab_visible_after_training_started(self, qapp):
        widget = GuidedTrainingWidget(
            parent=None, image_list=[], output_dir="/tmp/test",
            supported_shape=["rectangle"],
        )
        try:
            _simulate_guided_running(widget)
            assert widget.tab_widget.isTabVisible(3)
        finally:
            widget.shutdown()


# ── Config control disable scope ────────────────────────────────────

class TestConfigOnlyDisable:
    """Only config-changing controls are disabled during RUNNING."""

    def test_stop_button_enabled_during_running(self, qapp):
        widget = GuidedTrainingWidget(
            parent=None, image_list=[], output_dir="/tmp/test",
            supported_shape=["rectangle"],
        )
        try:
            _simulate_guided_running(widget)
            # Offscreen: cannot use isVisible() (no parent window); check isHidden() and isEnabled()
            assert not widget.stop_training_button.isHidden(), "Stop must not be hidden"
            assert widget.stop_training_button.isEnabled(), "Stop must be enabled"
        finally:
            widget.shutdown()

    def test_start_button_hidden_during_running(self, qapp):
        widget = GuidedTrainingWidget(
            parent=None, image_list=[], output_dir="/tmp/test",
            supported_shape=["rectangle"],
        )
        try:
            _simulate_guided_running(widget)
            assert widget.start_training_button.isHidden(), "Start must be hidden during running"
        finally:
            widget.shutdown()

    def test_console_exists_during_running(self, qapp):
        widget = GuidedTrainingWidget(
            parent=None, image_list=[], output_dir="/tmp/test",
            supported_shape=["rectangle"],
        )
        try:
            _simulate_guided_running(widget)
            assert hasattr(widget, 'log_display')
        finally:
            widget.shutdown()

    def test_config_re_enabled_after_completed(self, qapp):
        widget = GuidedTrainingWidget(
            parent=None, image_list=[], output_dir="/tmp/test",
            supported_shape=["rectangle"],
        )
        try:
            _simulate_guided_running(widget)
            widget.on_training_event("training_completed", {
                "save_dir": "/tmp", "results": "ok",
            })
            # Offscreen: use isHidden() not isVisible()
            assert widget.start_training_button.isHidden()
            assert not widget.previous_button.isHidden()
            assert not widget.export_button.isHidden()
        finally:
            widget.shutdown()

    def test_config_re_enabled_after_stopped(self, qapp):
        widget = GuidedTrainingWidget(
            parent=None, image_list=[], output_dir="/tmp/test",
            supported_shape=["rectangle"],
        )
        try:
            _simulate_guided_running(widget)
            widget.on_training_event("training_stopped", {
                "save_dir": "/tmp",
            })
            assert not widget.start_training_button.isHidden()
            assert widget.start_training_button.isEnabled()
        finally:
            widget.shutdown()

    def test_config_re_enabled_after_error(self, qapp):
        widget = GuidedTrainingWidget(
            parent=None, image_list=[], output_dir="/tmp/test",
            supported_shape=["rectangle"],
        )
        try:
            _simulate_guided_running(widget)
            widget.on_training_event("training_error", {
                "save_dir": "/tmp",
                "error": "test error",
            })
            controls = widget._gather_config_controls()
            enabled_count = sum(
                1 for c in controls
                if hasattr(c, 'isEnabled') and c.isEnabled()
            )
            assert enabled_count > 0
        finally:
            widget.shutdown()

    def test_metrics_dashboard_enabled_during_running(self, qapp):
        widget = GuidedTrainingWidget(
            parent=None, image_list=[], output_dir="/tmp/test",
            supported_shape=["rectangle"],
        )
        try:
            _simulate_guided_running(widget)
            dashboard = widget._metrics_dashboard
            assert dashboard is not None
            assert dashboard.isEnabled()
        finally:
            widget.shutdown()


# ── TrainingCenterWindow tab navigation ─────────────────────────────

class TestTrainingCenterTabNavigation:
    """Top-level tabs (Guided / Custom / History) remain clickable during training."""

    def test_top_tabs_enabled_by_default(self, qapp):
        window = TrainingCenterWindow(parent=None)
        try:
            assert window.tab_widget.tabBar().isEnabled()
        finally:
            window.shutdown()

    def test_custom_tab_accessible_when_guided_training(self, qapp):
        window = TrainingCenterWindow(parent=None)
        try:
            window.tab_widget.setCurrentIndex(TAB_GUIDED)
            _simulate_guided_running(window.guided_widget)
            window.tab_widget.setCurrentIndex(TAB_CUSTOM)
            assert window.tab_widget.currentIndex() == TAB_CUSTOM
        finally:
            window.shutdown()

    def test_history_tab_accessible_when_guided_training(self, qapp):
        window = TrainingCenterWindow(parent=None)
        try:
            window.tab_widget.setCurrentIndex(TAB_GUIDED)
            _simulate_guided_running(window.guided_widget)
            window.tab_widget.setCurrentIndex(TAB_HISTORY)
            assert window.tab_widget.currentIndex() == TAB_HISTORY
        finally:
            window.shutdown()

    def test_switching_tabs_does_not_stop_training(self, qapp):
        window = TrainingCenterWindow(parent=None)
        try:
            window.tab_widget.setCurrentIndex(TAB_GUIDED)
            _simulate_guided_running(window.guided_widget)
            assert window.guided_widget.training_status == "training"
            window.tab_widget.setCurrentIndex(TAB_CUSTOM)
            window.tab_widget.setCurrentIndex(TAB_GUIDED)
            assert window.guided_widget.training_status == "training"
        finally:
            window.shutdown()
