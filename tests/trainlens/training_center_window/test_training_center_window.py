"""Tests for TrainingCenterWindow — unified training UI."""

import pytest
import sys
import os
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, PropertyMock

from PyQt6.QtWidgets import QApplication, QTabWidget, QMainWindow
from PyQt6.QtCore import Qt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from anylabeling.services.training_center.job_manager import get_job_manager, JobManager
from anylabeling.services.training_center.models import (
    TrainingMode, TrainingStatus, TrainingJob,
)
from anylabeling.services.training_center.history import get_history_store, JobHistoryRecord
from anylabeling.views.training.training_center_window import (
    TrainingCenterWindow,
    open_training_center,
    TAB_GUIDED, TAB_CUSTOM, TAB_HISTORY,
)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture(autouse=True)
def reset_center_singleton():
    """Reset the module-level singleton and JobManager state between tests."""
    import anylabeling.views.training.training_center_window as tcw
    tcw._training_center_window = None

    # Also clean JobManager state to prevent cross-test pollution
    jm = get_job_manager()
    with jm._state_lock:
        jm._current_job = None
        jm._current_adapter = None
    jm._status_callbacks.clear()
    jm._event_callbacks.clear()

    yield

    tcw._training_center_window = None
    jm = get_job_manager()
    with jm._state_lock:
        jm._current_job = None
        jm._current_adapter = None
    jm._status_callbacks.clear()
    jm._event_callbacks.clear()


@pytest.fixture
def training_center(qapp, reset_center_singleton):
    """Create a fresh TrainingCenterWindow for each test."""
    window = TrainingCenterWindow(parent=None)
    yield window
    try:
        window.shutdown()
        window.close()
    except Exception:
        pass


# ── Construction tests ─────────────────────────────────────────────────

class TestTrainingCenterWindowConstruction:

    def test_can_construct(self, training_center):
        assert training_center is not None
        assert isinstance(training_center, QMainWindow)

    def test_has_correct_title(self, training_center):
        assert "TrainLens" in training_center.windowTitle()

    def test_has_tab_widget(self, training_center):
        assert hasattr(training_center, 'tab_widget')
        assert isinstance(training_center.tab_widget, QTabWidget)

    def test_has_three_tabs(self, training_center):
        assert training_center.tab_widget.count() >= 3

    def test_tab_labels(self, training_center):
        texts = []
        for i in range(training_center.tab_widget.count()):
            texts.append(training_center.tab_widget.tabText(i))
        assert "Guided" in texts[0] or "Guided" in " ".join(texts)
        assert "Custom" in texts[1] or "Custom" in " ".join(texts)
        assert "History" in texts[2] or "History" in " ".join(texts)


# ── Shared singleton tests ─────────────────────────────────────────────

class TestSharedSingletons:

    def test_window_uses_job_manager_singleton(self, training_center):
        assert training_center.job_manager is get_job_manager()

    def test_guided_widget_shares_job_manager(self, training_center):
        assert training_center.guided_widget.job_manager is training_center.job_manager
        assert training_center.guided_widget.job_manager is get_job_manager()

    def test_run_monitor_shares_job_manager(self, training_center):
        assert training_center.run_monitor_widget.job_manager is training_center.job_manager
        assert training_center.run_monitor_widget.job_manager is get_job_manager()

    def test_all_three_share_same_job_manager(self, training_center):
        jm = get_job_manager()
        assert training_center.job_manager is jm
        assert training_center.guided_widget.job_manager is jm
        assert training_center.run_monitor_widget.job_manager is jm

    def test_history_widget_uses_same_history_store(self, training_center):
        assert training_center.history_widget.history_store is get_history_store()


# ── Non-modal window tests ─────────────────────────────────────────────

class TestNonModal:

    def test_window_is_not_qdialog(self, training_center):
        from PyQt6.QtWidgets import QDialog
        assert not isinstance(training_center, QDialog)

    def test_window_show_does_not_block(self, qapp, training_center):
        """show() returns immediately (non-modal)."""
        training_center.show()
        assert training_center.isVisible()

    def test_open_training_center_returns_window(self, qapp, reset_center_singleton):
        window = open_training_center(parent=None, tab="guided")
        assert isinstance(window, TrainingCenterWindow)
        assert window.isVisible()

    def test_open_training_center_reuses_window(self, qapp, reset_center_singleton):
        w1 = open_training_center(parent=None, tab="guided")
        w2 = open_training_center(parent=None, tab="custom")
        assert w1 is w2

    def test_open_training_center_switches_tab(self, qapp, reset_center_singleton):
        window = open_training_center(parent=None, tab="guided")
        assert window.tab_widget.currentIndex() == TAB_GUIDED

        window = open_training_center(parent=None, tab="custom")
        assert window.tab_widget.currentIndex() == TAB_CUSTOM

        window = open_training_center(parent=None, tab="history")
        assert window.tab_widget.currentIndex() == TAB_HISTORY

    def test_reopen_after_close(self, qapp, reset_center_singleton):
        w1 = open_training_center(parent=None, tab="guided")
        w1.close()
        # The singleton should still be set but window is closed
        w2 = open_training_center(parent=None, tab="custom")
        assert w2.isVisible()


# ── Early mutual exclusion tests ───────────────────────────────────────

class TestEarlyMutualExclusion:

    def test_guided_rejected_when_custom_active(self, qapp, training_center):
        """When a custom job is active, guided start is rejected before data prep."""
        jm = training_center.job_manager

        # Use a real TrainingJob with RUNNING status
        active_job = TrainingJob(
            job_id="custom-active",
            mode=TrainingMode.CUSTOM_SCRIPT,
            status=TrainingStatus.RUNNING,
            created_at=datetime.now(),
            display_name="Custom: train.py",
            workspace=Path("/tmp"),
            command=["python", "train.py"],
        )

        with patch.object(jm, 'get_current_job', return_value=active_job):
            with patch('anylabeling.views.training.guided_training_widget.QMessageBox.critical') as mock_msg:
                training_center.guided_widget.start_training_from_train_tab()
                mock_msg.assert_called_once()

    def test_custom_rejected_when_guided_active(self, qapp, training_center):
        """When a guided job is active, custom start is rejected."""
        jm = training_center.job_manager

        active_job = TrainingJob(
            job_id="guided-active",
            mode=TrainingMode.GUIDED_ULTRALYTICS,
            status=TrainingStatus.RUNNING,
            created_at=datetime.now(),
            display_name="Guided: test",
            workspace=Path("/tmp"),
            command=[],
        )

        with patch.object(jm, 'get_current_job', return_value=active_job):
            with patch('anylabeling.views.run_monitor.run_monitor_widget.QMessageBox.critical') as mock_msg:
                training_center.run_monitor_widget.workspace = MagicMock()
                training_center.run_monitor_widget.script_combo = MagicMock()
                training_center.run_monitor_widget.script_combo.currentIndex.return_value = 0
                training_center.run_monitor_widget._on_start_training()
                mock_msg.assert_called_once()


# ── Tab switching does not stop tasks ──────────────────────────────────

class TestTabSwitchKeepsTasks:

    def test_switch_tab_does_not_stop_job(self, qapp, training_center):
        """Switching tabs should not call request_stop."""
        jm = training_center.job_manager
        with patch.object(jm, 'request_stop') as mock_stop:
            training_center.tab_widget.setCurrentIndex(TAB_CUSTOM)
            training_center.tab_widget.setCurrentIndex(TAB_GUIDED)
            training_center.tab_widget.setCurrentIndex(TAB_HISTORY)
            mock_stop.assert_not_called()


# ── Close window does not stop tasks ───────────────────────────────────

class TestCloseKeepsTasks:

    def test_close_does_not_stop_job(self, qapp, training_center):
        """Closing the window should not call request_stop."""
        jm = training_center.job_manager
        with patch.object(jm, 'request_stop') as mock_stop:
            training_center.close()
            mock_stop.assert_not_called()


# ── Shutdown tests ─────────────────────────────────────────────────────

class TestShutdown:

    def test_shutdown_is_idempotent(self, qapp, training_center):
        """Calling shutdown multiple times does not crash."""
        training_center.shutdown()
        training_center.shutdown()  # second call should be safe

    def test_shutdown_stops_active_job(self, qapp, training_center):
        """Shutdown stops an active job."""
        jm = training_center.job_manager
        active_job = TrainingJob(
            job_id="shutdown-test",
            mode=TrainingMode.GUIDED_ULTRALYTICS,
            status=TrainingStatus.RUNNING,
            created_at=datetime.now(),
            display_name="Test",
            workspace=Path("/tmp"),
            command=[],
        )
        with patch.object(jm, 'get_current_job', return_value=active_job):
            with patch.object(jm, 'request_stop') as mock_stop:
                training_center.shutdown()
                mock_stop.assert_called_once()


# ── Status bar tests ───────────────────────────────────────────────────

class TestStatusBar:

    def test_initial_status_is_idle(self, training_center):
        assert "Idle" in training_center.status_label.text()

    def test_status_updates_when_job_running(self, qapp, training_center):
        """Status bar shows Running when job is active."""
        jm = training_center.job_manager

        # Create a real TrainingJob that is active
        from anylabeling.services.training_center.models import TrainingMode, TrainingStatus, TrainingJob
        job = TrainingJob(
            job_id="test-job",
            mode=TrainingMode.GUIDED_ULTRALYTICS,
            status=TrainingStatus.RUNNING,
            created_at=datetime.now(),
            display_name="Guided: test_run",
            workspace=Path("/tmp"),
            command=[],
        )

        with patch.object(jm, 'get_current_job', return_value=job):
            training_center._update_status_bar_from_current()
            assert "Running" in training_center.status_label.text()


# ── History widget tests ───────────────────────────────────────────────

class TestHistoryWidget:

    def test_history_widget_has_table(self, training_center):
        assert hasattr(training_center.history_widget, 'table')

    def test_history_widget_has_refresh(self, training_center):
        assert hasattr(training_center.history_widget, 'refresh')

    def test_history_refresh_on_tab_switch(self, qapp, training_center):
        with patch.object(training_center.history_widget, 'refresh') as mock_refresh:
            training_center.tab_widget.setCurrentIndex(TAB_HISTORY)
            mock_refresh.assert_called_once()


# ── No duplicate JobManager ────────────────────────────────────────────

class TestSingleJobManager:

    def test_no_duplicate_job_manager_created(self, qapp, reset_center_singleton):
        """Creating TrainingCenterWindow should not create a new JobManager."""
        jm_before = get_job_manager()
        window = TrainingCenterWindow(parent=None)
        jm_after = get_job_manager()
        assert jm_before is jm_after
        assert window.job_manager is jm_before
        window.shutdown()
        window.close()
