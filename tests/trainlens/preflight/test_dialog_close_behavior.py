"""Tests: closing PreflightDialog must NOT start training."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QDialog, QPushButton

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from anylabeling.services.training_center.preflight.models import (
    PreflightResult, PreflightSeverity, PreflightIssue,
)
from anylabeling.views.training.preflight_dialog import PreflightDialog


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def _make_result(mode="guided", pass_count=15, warn_count=159, error_count=0):
    r = PreflightResult(mode=mode)
    for _ in range(pass_count):
        r.add(PreflightIssue(code="P", severity=PreflightSeverity.PASS, title="p", message="x"))
    for _ in range(warn_count):
        r.add(PreflightIssue(code="W", severity=PreflightSeverity.WARNING, title="w", message="x"))
    for _ in range(error_count):
        r.add(PreflightIssue(code="E", severity=PreflightSeverity.ERROR, title="e", message="x"))
    return r


class TestPreflightDialogClose:
    """Closing the dialog must never start training."""

    def test_start_anyway_returns_accepted(self, qapp):
        r = _make_result(warn_count=5)
        dlg = PreflightDialog(r)
        dlg._on_start_anyway()
        assert dlg.result() == QDialog.DialogCode.Accepted, "Start Anyway → Accepted"
        assert dlg.user_choice == "start_anyway"

    def test_start_training_returns_accepted(self, qapp):
        r = _make_result(warn_count=0)
        dlg = PreflightDialog(r)
        dlg._on_start()
        assert dlg.result() == QDialog.DialogCode.Accepted, "Start Training → Accepted"
        assert dlg.user_choice == "start"

    def test_back_to_fix_returns_rejected(self, qapp):
        r = _make_result(warn_count=5)
        dlg = PreflightDialog(r)
        dlg._on_back()
        assert dlg.result() == QDialog.DialogCode.Rejected, "Back → Rejected"
        assert dlg.user_choice == "back"

    def test_close_window_returns_rejected(self, qapp):
        r = _make_result(warn_count=5)
        dlg = PreflightDialog(r)
        dlg.reject()
        assert dlg.result() == QDialog.DialogCode.Rejected, "Close → Rejected"
        assert dlg.user_choice in ("cancel", "back")

    def test_close_event_sets_cancel(self, qapp):
        r = _make_result(warn_count=5)
        dlg = PreflightDialog(r)
        # Simulate closeEvent
        from PyQt6.QtGui import QCloseEvent
        dlg.close()
        assert dlg.user_choice == "cancel"

    def test_esc_rejected(self, qapp):
        r = _make_result(warn_count=5)
        dlg = PreflightDialog(r)
        dlg.reject()
        assert dlg.result() == QDialog.DialogCode.Rejected

    def test_errors_hide_start_anyway(self, qapp):
        r = _make_result(error_count=1)
        dlg = PreflightDialog(r)
        assert dlg._start_anyway_btn.isHidden(), "Start Anyway hidden when errors exist"
        assert dlg._start_btn.isHidden(), "Start hidden when errors exist"

    def test_warnings_only_show_start_anyway(self, qapp):
        r = _make_result(warn_count=5, error_count=0)
        dlg = PreflightDialog(r)
        assert not dlg._start_anyway_btn.isHidden(), "Start Anyway visible with warnings"
        assert dlg._start_btn.isHidden(), "Start hidden when warnings exist"

    def test_no_issues_show_start_training(self, qapp):
        r = _make_result(warn_count=0, error_count=0)
        dlg = PreflightDialog(r)
        assert not dlg._start_btn.isHidden(), "Start Training visible with no issues"

    def test_user_choice_defaults_to_back(self, qapp):
        r = _make_result()
        dlg = PreflightDialog(r)
        assert dlg.user_choice == "back", "Default user_choice is 'back'"

    def test_back_button_visible_when_errors(self, qapp):
        r = _make_result(error_count=3)
        dlg = PreflightDialog(r)
        assert not dlg._back_btn.isHidden(), "Back to Fix always visible"
