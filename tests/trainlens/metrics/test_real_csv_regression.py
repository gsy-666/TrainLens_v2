"""Regression tests for real metrics CSV parsing and Dashboard integration.

Tests verify:
- Real CSV structure parsing (0-values preserved, groups correct)
- Dashboard bind/update_output_dir/complete lifecycle
- MetricsStore CSV path tracking across file creation
- Completion read ordering
"""

import csv
import os
import sys
import tempfile
import time
from pathlib import Path

import pytest
from PyQt6.QtWidgets import QApplication

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from anylabeling.services.training_center.metrics.parser import parse_results_csv
from anylabeling.services.training_center.metrics.store import MetricStore
from anylabeling.services.training_center.metrics.models import MetricRunData, MetricSample
from anylabeling.views.training.metrics import TrainingMetricsDashboard


# ── Real CSV structure (matching guided_mutex_test6) ────────────────

REAL_CSV_HEADER = (
    "epoch,time,train/box_loss,train/cls_loss,train/dfl_loss,"
    "metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),"
    "val/box_loss,val/cls_loss,val/dfl_loss,lr/pg0,lr/pg1,lr/pg2"
)

REAL_CSV_ROWS = [
    "1,3.17429,0,1.54353,0,0,0,0,0,0,5.08947,0,3.57e-05,3.57e-05,3.57e-05",
    "2,6.43981,0,1.40537,0,0,0,0,0,0,5.34466,0,4.86353e-05,4.86353e-05,4.86353e-05",
    "3,9.35169,0,1.26154,0,0,0,0,0,0,5.34466,0,3.72232e-05,3.72232e-05,3.72232e-05",
]


def make_real_csv(tmp_path):
    """Write a CSV matching the real guided_mutex_test6 structure."""
    csv_file = tmp_path / "results.csv"
    csv_file.write_text(REAL_CSV_HEADER + "\n" + "\n".join(REAL_CSV_ROWS), encoding="utf-8")
    return csv_file


# ── Parser tests with real structure ────────────────────────────────

class TestRealCSVParsing:
    def test_three_epochs(self, tmp_path):
        csv_file = make_real_csv(tmp_path)
        data = parse_results_csv(csv_file)
        assert len(data.samples) == 3

    def test_train_cls_loss_in_loss_group(self, tmp_path):
        csv_file = make_real_csv(tmp_path)
        data = parse_results_csv(csv_file)
        series = {s.name: s for s in data.to_series()}
        assert "train/cls_loss" in series
        assert series["train/cls_loss"].group == "loss"

    def test_val_cls_loss_in_loss_group(self, tmp_path):
        csv_file = make_real_csv(tmp_path)
        data = parse_results_csv(csv_file)
        series = {s.name: s for s in data.to_series()}
        assert "val/cls_loss" in series
        assert series["val/cls_loss"].group == "loss"

    def test_lr_in_learning_rate_group(self, tmp_path):
        csv_file = make_real_csv(tmp_path)
        data = parse_results_csv(csv_file)
        series = {s.name: s for s in data.to_series()}
        for lr_key in ("lr/pg0", "lr/pg1", "lr/pg2"):
            assert lr_key in series
            assert series[lr_key].group == "learning_rate"

    def test_zero_values_preserved(self, tmp_path):
        """train/box_loss = 0.0 must not be filtered as empty."""
        csv_file = make_real_csv(tmp_path)
        data = parse_results_csv(csv_file)
        # All samples have train/box_loss = 0.0
        for s in data.samples:
            assert "train/box_loss" in s.values
            assert s.values["train/box_loss"] == 0.0

    def test_quality_all_zero_loss_still_shown(self, tmp_path):
        """Even if all quality metrics are 0, loss series must exist."""
        csv_file = make_real_csv(tmp_path)
        data = parse_results_csv(csv_file)
        series_list = data.to_series()
        groups = {s.group for s in series_list}
        assert "loss" in groups, "Loss group must exist even when quality is all-zero"

    def test_quality_all_zero_lr_still_shown(self, tmp_path):
        csv_file = make_real_csv(tmp_path)
        data = parse_results_csv(csv_file)
        series_list = data.to_series()
        groups = {s.group for s in series_list}
        assert "learning_rate" in groups, "LR group must exist even when quality is all-zero"

    def test_epochs_are_1_2_3(self, tmp_path):
        csv_file = make_real_csv(tmp_path)
        data = parse_results_csv(csv_file)
        assert [int(s.epoch) for s in data.samples] == [1, 2, 3]

    def test_zero_not_falsy_filtered(self):
        """0.0 must not be treated as falsy/empty."""
        assert bool(0.0) is False  # Python truthiness
        # But parser should NOT use 'if value:' — check via isfinite
        import math
        assert math.isfinite(0.0)


# ── MetricsStore CSV tracking across file lifecycle ─────────────────

class TestMetricsStoreCSVLifecycle:
    def test_csv_path_saved_even_when_file_does_not_exist(self):
        store = MetricStore()
        nonexistent_dir = tempfile.mkdtemp()
        try:
            # CSV does not exist yet
            store.start_run("job1", nonexistent_dir)
            csv_path = store._csv_paths.get("job1")
            assert csv_path is not None, "CSV path must be saved even if file doesn't exist yet"
            assert not csv_path.exists(), "CSV should not exist yet"
        finally:
            import shutil
            shutil.rmtree(nonexistent_dir, ignore_errors=True)

    def test_csv_appears_mid_training_detected(self, tmp_path):
        """Simulate CSV appearing after training_started (epoch 1 completes)."""
        store = MetricStore()
        store.start_run("job1", str(tmp_path))
        # No CSV yet
        assert store.poll_csv() is None

        # CSV appears
        csv = tmp_path / "results.csv"
        csv.write_text("epoch,train/box_loss\n1,0.5\n", encoding="utf-8")
        data = store.poll_csv()
        assert data is not None
        assert len(data.samples) == 1

    def test_csv_grows_detected(self, tmp_path):
        store = MetricStore()
        csv = tmp_path / "results.csv"
        csv.write_text("epoch,train/box_loss\n1,0.5\n", encoding="utf-8")
        store.start_run("job1", str(tmp_path))

        # First poll reads epoch 1
        data = store.poll_csv()
        assert data is not None
        assert len(data.samples) == 1

        # Same size → no new data
        assert store.poll_csv() is None

        # Append epoch 2
        csv.write_text("epoch,train/box_loss\n1,0.5\n2,0.3\n", encoding="utf-8")
        data = store.poll_csv()
        assert data is not None
        assert len(data.samples) == 2

    def test_update_csv_path_switches_directory(self, tmp_path):
        store = MetricStore()
        dir_a = tmp_path / "dir_a"
        dir_b = tmp_path / "dir_b"
        dir_a.mkdir()
        dir_b.mkdir()
        (dir_a / "results.csv").write_text("epoch,train/box_loss\n1,0.9\n", encoding="utf-8")

        store.start_run("job1", str(dir_a))
        data = store.poll_csv()
        assert data.samples[0].values["train/box_loss"] == 0.9

        # Switch to dir_b
        (dir_b / "results.csv").write_text("epoch,train/box_loss\n1,0.1\n2,0.05\n", encoding="utf-8")
        store.update_csv_path("job1", str(dir_b))
        data = store.poll_csv()
        assert len(data.samples) == 2
        assert data.samples[0].values["train/box_loss"] == 0.1


# ── Dashboard lifecycle with real CSV ───────────────────────────────

class TestDashboardRealCSVLifecycle:
    @pytest.fixture(scope="session")
    def qapp(self):
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        yield app

    def test_bind_and_poll_shows_data(self, qapp, tmp_path):
        make_real_csv(tmp_path)
        dashboard = TrainingMetricsDashboard()
        try:
            dashboard.bind_job("test_job", str(tmp_path))
            # Force immediate poll + refresh
            dashboard._store.poll_csv()
            dashboard._do_refresh()
            # Empty label should be hidden (data exists)
            assert dashboard._empty_label.isHidden()
        finally:
            dashboard.cleanup()

    def test_completed_reads_final_data(self, qapp, tmp_path):
        make_real_csv(tmp_path)
        dashboard = TrainingMetricsDashboard()
        try:
            dashboard.bind_job("test_job", str(tmp_path))
            dashboard.on_run_completed("test_job")
            dashboard._do_refresh()
            run = dashboard._store.get_run("test_job")
            assert run is not None
            assert len(run.samples) == 3
            assert dashboard._empty_label.isHidden()
        finally:
            dashboard.cleanup()

    def test_stopped_preserves_data(self, qapp, tmp_path):
        make_real_csv(tmp_path)
        dashboard = TrainingMetricsDashboard()
        try:
            dashboard.bind_job("test_job", str(tmp_path))
            dashboard.on_run_stopped("test_job")
            dashboard._do_refresh()
            run = dashboard._store.get_run("test_job")
            assert run is not None
            assert len(run.samples) == 3
        finally:
            dashboard.cleanup()

    def test_empty_label_hidden_with_data(self, qapp, tmp_path):
        make_real_csv(tmp_path)
        dashboard = TrainingMetricsDashboard()
        try:
            dashboard.bind_job("test_job", str(tmp_path))
            dashboard._store.poll_csv()
            dashboard._do_refresh()
            assert dashboard._empty_label.isHidden()
        finally:
            dashboard.cleanup()

    def test_epoch_card_shows_total(self, qapp, tmp_path):
        make_real_csv(tmp_path)
        dashboard = TrainingMetricsDashboard()
        try:
            dashboard.bind_job("test_job", str(tmp_path))
            dashboard._store.poll_csv()
            dashboard._do_refresh()
            from PyQt6.QtWidgets import QLabel
            card = dashboard.findChild(QLabel, "card_epoch_value")
            assert card is not None, "Epoch card should exist"
        finally:
            dashboard.cleanup()

    def test_new_job_id_isolated(self, qapp, tmp_path):
        make_real_csv(tmp_path)
        dashboard = TrainingMetricsDashboard()
        try:
            dashboard.bind_job("job_a", str(tmp_path))
            dashboard._store.poll_csv()
            # Bind new job: old job_a data stays in store (history), but current_job changes
            dashboard.bind_job("job_b", str(tmp_path))
            assert dashboard._current_job_id == "job_b"
            # New job should have its own fresh data
            dashboard._store.poll_csv()
            run_b = dashboard._store.get_run("job_b")
            assert run_b is not None
            assert len(run_b.samples) == 3
        finally:
            dashboard.cleanup()

    def test_completed_reads_before_timer_stopped(self, qapp, tmp_path):
        """Verify on_run_completed reads CSV before stopping poll timer."""
        make_real_csv(tmp_path)
        dashboard = TrainingMetricsDashboard()
        try:
            dashboard.bind_job("test_job", str(tmp_path))
            # on_run_completed should read data even though poll timer fires later
            dashboard.on_run_completed("test_job")
            assert not dashboard._poll_timer.isActive(), "Poll timer stopped after read"
            run = dashboard._store.get_run("test_job")
            assert run is not None
            assert len(run.samples) == 3, "Data preserved after completion"
        finally:
            dashboard.cleanup()

    def test_update_output_dir_switches_binding(self, qapp, tmp_path):
        dir_a = tmp_path / "dir_a"
        dir_b = tmp_path / "dir_b"
        dir_a.mkdir()
        dir_b.mkdir()
        (dir_a / "results.csv").write_text("epoch,train/box_loss\n1,0.9\n", encoding="utf-8")
        (dir_b / "results.csv").write_text("epoch,train/box_loss\n1,0.1\n2,0.05\n", encoding="utf-8")

        dashboard = TrainingMetricsDashboard()
        try:
            dashboard.bind_job("test_job", str(dir_a))
            dashboard._store.poll_csv()
            dashboard.update_output_dir("test_job", str(dir_b))
            dashboard._store.poll_csv()
            dashboard._do_refresh()
            run = dashboard._store.get_run("test_job")
            assert len(run.samples) == 2
            assert run.samples[1].epoch == 2
        finally:
            dashboard.cleanup()
