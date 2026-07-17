"""Tests for MetricStore (thread-safe metrics accumulator)."""

import csv
import json
import tempfile
import threading
import time
from pathlib import Path

import pytest

from anylabeling.services.training_center.metrics.models import MetricSample
from anylabeling.services.training_center.metrics.store import MetricStore


class TestMetricStoreBasic:
    def test_is_not_singleton(self):
        """MetricStore does NOT enforce singleton pattern (each call creates new instance)."""
        s1 = MetricStore()
        s2 = MetricStore()
        assert s1 is not s2

    def test_start_run(self):
        store = MetricStore()
        store.start_run("job1", "/tmp/test")
        assert store.get_run("job1") is not None

    def test_get_run_nonexistent(self):
        store = MetricStore()
        assert store.get_run("nonexistent") is None

    def test_add_sample(self):
        store = MetricStore()
        store.start_run("job1")
        sample = MetricSample(job_id="job1", epoch=1, values={"loss": 1.0})
        store.add_sample(sample)
        run = store.get_run("job1")
        assert len(run.samples) == 1
        assert run.samples[0].values["loss"] == 1.0

    def test_add_sample_auto_creates_run(self):
        """add_sample auto-creates a run if job_id is unknown."""
        store = MetricStore()
        sample = MetricSample(job_id="auto_created", epoch=1, values={"loss": 1.0})
        store.add_sample(sample)
        run = store.get_run("auto_created")
        assert run is not None
        assert len(run.samples) == 1

    def test_add_sample_dedup_by_epoch(self):
        store = MetricStore()
        store.start_run("job1")
        store.add_sample(MetricSample(job_id="job1", epoch=1, values={"loss": 2.0}))
        store.add_sample(MetricSample(job_id="job1", epoch=1, values={"loss": 0.5}))  # same epoch
        run = store.get_run("job1")
        assert len(run.samples) == 1  # deduped
        assert run.samples[0].values["loss"] == 0.5  # last write wins

    def test_clear_run(self):
        store = MetricStore()
        store.start_run("job1")
        store.clear_run("job1")
        assert store.get_run("job1") is None


class TestMetricStoreConcurrent:
    def test_concurrent_add_samples(self):
        store = MetricStore()
        store.start_run("job1")
        n_threads = 8
        n_per_thread = 50

        def add_samples(start_epoch):
            for i in range(n_per_thread):
                sample = MetricSample(
                    job_id="job1",
                    epoch=start_epoch + i,
                    values={"loss": 1.0 - i * 0.01},
                )
                store.add_sample(sample)

        threads = []
        for t in range(n_threads):
            t = threading.Thread(target=add_samples, args=(t * n_per_thread,))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        run = store.get_run("job1")
        assert len(run.samples) == n_threads * n_per_thread

    def test_concurrent_same_store(self):
        """Verify no crashes with multiple threads accessing the same store."""
        store = MetricStore()
        errors = []

        def worker(wid):
            try:
                store.start_run(f"job{wid}")
                for i in range(10):
                    store.add_sample(
                        MetricSample(job_id=f"job{wid}", epoch=i, values={"loss": 1.0 / (i + 1)})
                    )
                _ = store.get_run(f"job{wid}")
            except Exception as e:
                errors.append((wid, str(e)))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Errors in concurrent access: {errors}"


class TestMetricStoreCSV:
    def test_poll_csv_reads_new_data(self):
        tmpdir = tempfile.mkdtemp()
        csv_path = Path(tmpdir) / "results.csv"
        try:
            # Write initial CSV
            csv_path.write_text("epoch,train/box_loss\n1,1.5\n", encoding="utf-8")
            store = MetricStore()
            store.start_run("job1", str(tmpdir))
            data = store.poll_csv()
            assert data is not None
            assert len(data.samples) == 1
            assert data.samples[0].values["train/box_loss"] == 1.5

            # Append more rows
            csv_path.write_text("epoch,train/box_loss\n1,1.5\n2,1.2\n3,0.9\n", encoding="utf-8")
            data = store.poll_csv()
            assert len(data.samples) == 3

            # No new data → returns None
            data = store.poll_csv()
            assert data is None
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_poll_csv_no_output_dir(self):
        store = MetricStore()
        store.start_run("job1")  # No output_dir
        assert store.poll_csv() is None

    def test_poll_csv_unknown_job(self):
        store = MetricStore()
        assert store.poll_csv() is None  # No current job

    def test_load_from_output_dir_csv(self):
        tmpdir = tempfile.mkdtemp()
        try:
            csv_path = Path(tmpdir) / "results.csv"
            csv_path.write_text(
                "epoch,train/box_loss,metrics/mAP50(B)\n1,1.5,0.5\n2,1.0,0.7\n",
                encoding="utf-8",
            )
            store = MetricStore()
            data = store.load_from_output_dir("job1", str(tmpdir))
            assert data is not None
            assert len(data.samples) == 2
            assert data.samples[1].epoch == 2
            assert data.samples[1].values["metrics/mAP50(B)"] == 0.7
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_load_from_output_dir_jsonl(self):
        tmpdir = tempfile.mkdtemp()
        try:
            jsonl_path = Path(tmpdir) / "metrics.jsonl"
            lines = [
                json.dumps({"epoch": 1, "metrics": {"loss": 2.0, "acc": 0.4}}),
                json.dumps({"epoch": 2, "metrics": {"loss": 1.5, "acc": 0.6}}),
            ]
            jsonl_path.write_text("\n".join(lines), encoding="utf-8")
            store = MetricStore()
            data = store.load_from_output_dir("job1", str(tmpdir))
            assert data is not None
            assert len(data.samples) == 2
            assert data.samples[0].values["loss"] == 2.0
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_load_from_output_dir_empty(self):
        tmpdir = tempfile.mkdtemp()
        try:
            store = MetricStore()
            data = store.load_from_output_dir("job1", str(tmpdir))
            assert data is None
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
