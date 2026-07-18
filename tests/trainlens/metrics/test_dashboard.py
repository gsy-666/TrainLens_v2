"""Tests for TrainingMetricsDashboard widget (offscreen)."""

import sys
import os
import tempfile
from pathlib import Path

import pytest
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from anylabeling.services.training_center.metrics import MetricStore
from anylabeling.views.training.metrics import TrainingMetricsDashboard


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def dashboard(qapp):
    d = TrainingMetricsDashboard()
    yield d
    d.cleanup()


class TestDashboardLifecycle:
    def test_initial_state(self, dashboard):
        assert dashboard._current_job_id is None
        assert not dashboard._poll_timer.isActive()
        assert not dashboard._refresh_timer.isActive()

    def test_bind_job_starts_polling(self, dashboard, tmp_path):
        csv = tmp_path / "results.csv"
        csv.write_text("epoch,train/box_loss\n1,0.5\n", encoding="utf-8")
        dashboard.bind_job("test_job", str(tmp_path))
        assert dashboard._current_job_id == "test_job"
        assert dashboard._poll_timer.isActive()

    def test_bind_job_no_output_dir(self, dashboard):
        dashboard.bind_job("test_job")  # No output_dir
        assert dashboard._current_job_id == "test_job"
        assert not dashboard._poll_timer.isActive()

    def test_clear(self, dashboard, tmp_path):
        csv = tmp_path / "results.csv"
        csv.write_text("epoch,train/box_loss\n1,0.5\n", encoding="utf-8")
        dashboard.bind_job("test_job", str(tmp_path))
        dashboard.clear()
        assert dashboard._current_job_id is None
        assert not dashboard._poll_timer.isActive()

    def test_cleanup_stops_timers(self, dashboard, tmp_path):
        csv = tmp_path / "results.csv"
        csv.write_text("epoch,train/box_loss\n1,0.5\n", encoding="utf-8")
        dashboard.bind_job("test_job", str(tmp_path))
        dashboard.cleanup()
        assert not dashboard._poll_timer.isActive()
        assert not dashboard._refresh_timer.isActive()

    def test_on_run_completed_stops_polling(self, dashboard, tmp_path):
        csv = tmp_path / "results.csv"
        csv.write_text("epoch,train/box_loss\n1,0.5\n", encoding="utf-8")
        dashboard.bind_job("test_job", str(tmp_path))
        dashboard.on_run_completed("test_job")
        assert not dashboard._poll_timer.isActive()

    def test_on_run_completed_wrong_job_ignored(self, dashboard, tmp_path):
        csv = tmp_path / "results.csv"
        csv.write_text("epoch,train/box_loss\n1,0.5\n", encoding="utf-8")
        dashboard.bind_job("test_job", str(tmp_path))
        dashboard.on_run_completed("other_job")
        assert dashboard._poll_timer.isActive()  # Still polling


class TestDashboardMetricsEvent:
    def test_metric_event_adds_sample(self, dashboard):
        dashboard.bind_job("test_job")
        dashboard.on_metric_event("test_job", {
            "epoch": 1,
            "total_epochs": 10,
            "metrics": {"box_loss": 1.5, "mAP50": 0.6},
        })
        dashboard._do_refresh()
        run = dashboard._store.get_run("test_job")
        assert run is not None
        assert len(run.samples) == 1

    def test_metric_event_wrong_job(self, dashboard):
        dashboard.bind_job("job_a")
        dashboard.on_metric_event("job_b", {
            "epoch": 1,
            "metrics": {"loss": 1.0},
        })
        run = dashboard._store.get_run("job_a")
        assert run.samples == []

    def test_metric_event_non_dict_metrics(self, dashboard):
        dashboard.bind_job("test_job")
        dashboard.on_metric_event("test_job", {"epoch": 1, "metrics": "not a dict"})
        run = dashboard._store.get_run("test_job")
        assert run.samples == []


class TestDashboardHistory:
    def test_load_history_from_csv(self, dashboard, tmp_path):
        csv = tmp_path / "results.csv"
        csv.write_text(
            "epoch,train/box_loss,metrics/mAP50(B)\n1,1.5,0.3\n2,1.0,0.6\n3,0.5,0.85\n",
            encoding="utf-8",
        )
        dashboard.load_history("hist_job", str(tmp_path))
        assert dashboard._current_job_id == "hist_job"
        assert dashboard._is_history_mode
        assert not dashboard._poll_timer.isActive()
        run = dashboard._store.get_run("hist_job")
        assert run is not None
        assert len(run.samples) == 3

    def test_load_history_empty_dir(self, dashboard, tmp_path):
        dashboard.load_history("hist_job", str(tmp_path))
        # Should show empty state, not crash
        assert dashboard._is_history_mode


class TestDashboardThrottling:
    def test_schedule_refresh_sets_dirty(self, dashboard):
        dashboard._schedule_refresh()
        assert dashboard._dirty
        assert dashboard._refresh_timer.isActive()

    def test_do_refresh_clears_dirty(self, dashboard):
        dashboard._dirty = True
        dashboard._do_refresh()
        assert not dashboard._dirty

    def test_multiple_schedule_single_timer(self, dashboard):
        dashboard._schedule_refresh()
        dashboard._schedule_refresh()
        dashboard._schedule_refresh()
        # Timer should be single-shot, only one active
        assert dashboard._refresh_timer.isActive()
        assert dashboard._dirty


class TestDashboardExport:
    def test_export_csv_no_data(self, dashboard):
        """Should not crash when export clicked with no data."""
        # Just verify the button exists and is disabled
        assert not dashboard._export_csv_btn.isEnabled()

    def test_export_enabled_with_data(self, dashboard, tmp_path):
        csv = tmp_path / "results.csv"
        csv.write_text("epoch,train/box_loss\n1,0.5\n", encoding="utf-8")
        dashboard.bind_job("test_job", str(tmp_path))
        # Force CSV poll and refresh
        dashboard._store.poll_csv()  # reads CSV into run
        dashboard._do_refresh()
        assert dashboard._export_csv_btn.isEnabled()
        assert dashboard._export_img_btn.isEnabled()
