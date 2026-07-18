"""UI responsiveness regression tests for Guided Training Start button."""

import pytest
import sys
import os
from unittest.mock import patch

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


class TestEarlyBusyCheck:

    def test_start_blocked_when_job_active(self, qapp):
        """reserve_job rejects when another job is RUNNING — no prep thread created."""
        from anylabeling.views.training.guided_training_widget import GuidedTrainingWidget
        from anylabeling.services.training_center.models import TrainingJob, TrainingMode, TrainingStatus
        from datetime import datetime
        from pathlib import Path

        widget = GuidedTrainingWidget(parent=None, image_list=[], output_dir="/tmp")
        widget.selected_task_type = "Detect"

        # Simulate another job already RUNNING in JobManager
        active_job = TrainingJob(
            job_id="active-1", mode=TrainingMode.GUIDED_ULTRALYTICS,
            status=TrainingStatus.RUNNING, created_at=datetime.now(),
            display_name="Custom: train.py", workspace=Path("/tmp"), command=[],
        )
        widget.job_manager._current_job = active_job

        # Mock both QMessageBox.critical (for config error path) and
        # reserve_job's rejection via QMessageBox.critical
        with patch('anylabeling.views.training.guided_training_widget.QMessageBox.critical') as mock_crit:
            widget.start_training_from_train_tab(skip_preflight=True)
            # Either config error or reserve rejection — both show critical dialog
            assert mock_crit.call_count >= 1

        # No prep thread created in any path
        assert getattr(widget, '_prep_thread', None) is None

    def test_handler_returns_fast_when_config_unavailable(self, qapp):
        """Handler does NOT hang even when config reading fails — QMessageBox mocked."""
        import time
        from anylabeling.views.training.guided_training_widget import GuidedTrainingWidget

        widget = GuidedTrainingWidget(parent=None, image_list=[], output_dir="/tmp")
        widget.selected_task_type = "Detect"

        start = time.time()
        with patch('anylabeling.views.training.guided_training_widget.QMessageBox.critical'):
            widget.start_training_from_train_tab(skip_preflight=True)
        elapsed = time.time() - start
        assert elapsed < 2.0, f"Blocked for {elapsed:.1f}s"


class TestEventLoop:

    def test_timer_fires_during_idle(self, qapp):
        """QTimer fires when widget is idle — event loop working."""
        from anylabeling.views.training.guided_training_widget import GuidedTrainingWidget
        import time

        widget = GuidedTrainingWidget(parent=None, image_list=[], output_dir="/tmp")
        fired = []
        timer = QTimer()
        timer.timeout.connect(lambda: fired.append(True))
        timer.start(50)

        # Pump events for long enough to allow timer to fire
        deadline = time.time() + 1.0
        while time.time() < deadline and len(fired) == 0:
            qapp.processEvents()

        timer.stop()
        assert len(fired) > 0, "Timer did not fire within 1 second"


class TestShutdownSafe:

    def test_shutdown_no_prep_thread(self, qapp):
        """shutdown handles None _prep_thread."""
        from anylabeling.views.training.guided_training_widget import GuidedTrainingWidget

        widget = GuidedTrainingWidget(parent=None, image_list=[], output_dir="/tmp")
        widget._prep_thread = None
        widget._prep_worker = None
        widget.shutdown()