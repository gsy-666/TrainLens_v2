"""Tests for Job History Store

Verifies JSONL persistence, thread safety, corruption handling, and CRUD operations.
"""

import pytest
import json
import time
import threading
from pathlib import Path
from datetime import datetime
from tempfile import TemporaryDirectory

from anylabeling.services.training_center.models import (
    TrainingMode,
    TrainingStatus,
)
from anylabeling.services.training_center.history import (
    JobHistoryRecord,
    HistoryStore,
    get_history_store,
)


@pytest.fixture
def temp_history_dir(tmp_path):
    """Temporary history directory for testing"""
    return tmp_path / "history"


@pytest.fixture
def store(temp_history_dir):
    """Create fresh HistoryStore for each test"""
    return HistoryStore(history_dir=temp_history_dir)


@pytest.fixture
def sample_record():
    """Create sample job history record"""
    return JobHistoryRecord(
        job_id="test-job-001",
        mode=TrainingMode.GUIDED_ULTRALYTICS.value,
        status=TrainingStatus.IDLE.value,
        created_at=datetime.now().isoformat(),
        display_name="Test Job",
        framework="ultralytics",
        metadata={"test_key": "test_value"},
    )


class TestJobHistoryRecord:
    """Test JobHistoryRecord data structure"""

    def test_record_creation(self):
        """Can create record with required fields"""
        record = JobHistoryRecord(
            job_id="job-001",
            mode="guided_ultralytics",
            status="idle",
            created_at="2026-07-16T10:00:00",
        )

        assert record.job_id == "job-001"
        assert record.mode == "guided_ultralytics"
        assert record.status == "idle"
        assert record.created_at == "2026-07-16T10:00:00"

    def test_record_to_dict(self, sample_record):
        """to_dict serializes all fields"""
        data = sample_record.to_dict()

        assert data["job_id"] == sample_record.job_id
        assert data["mode"] == sample_record.mode
        assert data["status"] == sample_record.status
        assert data["metadata"] == sample_record.metadata

    def test_record_from_dict(self):
        """from_dict deserializes correctly"""
        data = {
            "job_id": "job-002",
            "mode": "custom_script",
            "status": "running",
            "created_at": "2026-07-16T11:00:00",
            "display_name": "Custom Job",
            "total_epochs": 100,
            "metadata": {},
        }

        record = JobHistoryRecord.from_dict(data)

        assert record.job_id == "job-002"
        assert record.mode == "custom_script"
        assert record.total_epochs == 100


class TestHistoryStoreBasics:
    """Test basic store operations"""

    def test_store_initialization_creates_directory(self, temp_history_dir):
        """Store creates history directory on init"""
        store = HistoryStore(history_dir=temp_history_dir)

        assert temp_history_dir.exists()
        assert temp_history_dir.is_dir()

    def test_empty_store_has_no_jobs(self, store):
        """New store returns empty list"""
        jobs = store.list_jobs()

        assert jobs == []

    def test_get_nonexistent_job_returns_none(self, store):
        """Getting nonexistent job returns None"""
        result = store.get_job("nonexistent")

        assert result is None


class TestHistoryStoreCRUD:
    """Test CRUD operations"""

    def test_append_job(self, store, sample_record):
        """Can append new job"""
        store.append_job(sample_record)

        retrieved = store.get_job(sample_record.job_id)
        assert retrieved is not None
        assert retrieved.job_id == sample_record.job_id
        assert retrieved.display_name == sample_record.display_name

    def test_append_multiple_jobs(self, store):
        """Can append multiple jobs"""
        records = [
            JobHistoryRecord(
                job_id=f"job-{i}",
                mode=TrainingMode.GUIDED_ULTRALYTICS.value,
                status=TrainingStatus.IDLE.value,
                created_at=datetime.now().isoformat(),
            )
            for i in range(3)
        ]

        for record in records:
            store.append_job(record)

        jobs = store.list_jobs()
        assert len(jobs) == 3

    def test_update_existing_job(self, store, sample_record):
        """Can update job fields"""
        store.append_job(sample_record)

        store.update_job(
            sample_record.job_id,
            status=TrainingStatus.RUNNING.value,
            started_at=datetime.now().isoformat(),
        )

        updated = store.get_job(sample_record.job_id)
        assert updated.status == TrainingStatus.RUNNING.value
        assert updated.started_at is not None

    def test_update_nonexistent_job_raises(self, store):
        """Updating nonexistent job raises KeyError"""
        with pytest.raises(KeyError, match="not found"):
            store.update_job("nonexistent", status="running")

    def test_finalize_job_with_success(self, store, sample_record):
        """Can finalize job with success status"""
        sample_record.started_at = datetime.now().isoformat()
        store.append_job(sample_record)

        ended_at = datetime.now()
        store.finalize_job(
            sample_record.job_id,
            status=TrainingStatus.COMPLETED,
            ended_at=ended_at,
            final_epoch=100,
            best_map50=0.95,
        )

        finalized = store.get_job(sample_record.job_id)
        assert finalized.status == TrainingStatus.COMPLETED.value
        assert finalized.ended_at is not None
        assert finalized.final_epoch == 100
        assert finalized.best_map50 == 0.95
        assert finalized.duration_seconds is not None

    def test_finalize_job_with_failure(self, store, sample_record):
        """Can finalize job with failure status"""
        store.append_job(sample_record)

        store.finalize_job(
            sample_record.job_id,
            status=TrainingStatus.FAILED,
            ended_at=datetime.now(),
            error_message="CUDA out of memory",
        )

        finalized = store.get_job(sample_record.job_id)
        assert finalized.status == TrainingStatus.FAILED.value
        assert finalized.error_message == "CUDA out of memory"

    def test_finalize_job_is_idempotent(self, store, sample_record):
        """Finalizing already-terminal job preserves state"""
        store.append_job(sample_record)

        # First finalization
        store.finalize_job(
            sample_record.job_id,
            status=TrainingStatus.COMPLETED,
            ended_at=datetime.now(),
            final_epoch=100,
        )

        first_state = store.get_job(sample_record.job_id)

        # Second finalization should be no-op
        store.finalize_job(
            sample_record.job_id,
            status=TrainingStatus.FAILED,
            ended_at=datetime.now(),
            error_message="Should be ignored",
        )

        second_state = store.get_job(sample_record.job_id)

        assert second_state.status == first_state.status
        assert second_state.error_message == first_state.error_message
        assert second_state.final_epoch == first_state.final_epoch

    def test_finalize_with_non_terminal_status_raises(self, store, sample_record):
        """finalize_job rejects non-terminal status"""
        store.append_job(sample_record)

        with pytest.raises(ValueError, match="terminal status"):
            store.finalize_job(
                sample_record.job_id,
                status=TrainingStatus.RUNNING,
            )


class TestHistoryStoreQueries:
    """Test query and filtering operations"""

    def test_list_jobs_returns_most_recent_first(self, store):
        """list_jobs sorts by created_at descending"""
        records = []
        for i in range(3):
            time.sleep(0.01)  # Ensure different timestamps
            record = JobHistoryRecord(
                job_id=f"job-{i}",
                mode=TrainingMode.GUIDED_ULTRALYTICS.value,
                status=TrainingStatus.IDLE.value,
                created_at=datetime.now().isoformat(),
            )
            records.append(record)
            store.append_job(record)

        jobs = store.list_jobs()

        assert len(jobs) == 3
        # Most recent first
        assert jobs[0].job_id == "job-2"
        assert jobs[1].job_id == "job-1"
        assert jobs[2].job_id == "job-0"

    def test_list_jobs_filter_by_mode(self, store):
        """list_jobs can filter by training mode"""
        ultralytics_job = JobHistoryRecord(
            job_id="ultra-001",
            mode=TrainingMode.GUIDED_ULTRALYTICS.value,
            status=TrainingStatus.IDLE.value,
            created_at=datetime.now().isoformat(),
        )
        custom_job = JobHistoryRecord(
            job_id="custom-001",
            mode=TrainingMode.CUSTOM_SCRIPT.value,
            status=TrainingStatus.IDLE.value,
            created_at=datetime.now().isoformat(),
        )

        store.append_job(ultralytics_job)
        store.append_job(custom_job)

        ultra_results = store.list_jobs(mode=TrainingMode.GUIDED_ULTRALYTICS)
        assert len(ultra_results) == 1
        assert ultra_results[0].job_id == "ultra-001"

        custom_results = store.list_jobs(mode=TrainingMode.CUSTOM_SCRIPT)
        assert len(custom_results) == 1
        assert custom_results[0].job_id == "custom-001"

    def test_list_jobs_filter_by_status(self, store):
        """list_jobs can filter by status"""
        completed_job = JobHistoryRecord(
            job_id="completed-001",
            mode=TrainingMode.GUIDED_ULTRALYTICS.value,
            status=TrainingStatus.COMPLETED.value,
            created_at=datetime.now().isoformat(),
        )
        failed_job = JobHistoryRecord(
            job_id="failed-001",
            mode=TrainingMode.GUIDED_ULTRALYTICS.value,
            status=TrainingStatus.FAILED.value,
            created_at=datetime.now().isoformat(),
        )

        store.append_job(completed_job)
        store.append_job(failed_job)

        completed_results = store.list_jobs(status=TrainingStatus.COMPLETED)
        assert len(completed_results) == 1
        assert completed_results[0].job_id == "completed-001"

        failed_results = store.list_jobs(status=TrainingStatus.FAILED)
        assert len(failed_results) == 1
        assert failed_results[0].job_id == "failed-001"

    def test_list_jobs_with_limit(self, store):
        """list_jobs respects limit parameter"""
        for i in range(5):
            record = JobHistoryRecord(
                job_id=f"job-{i}",
                mode=TrainingMode.GUIDED_ULTRALYTICS.value,
                status=TrainingStatus.IDLE.value,
                created_at=datetime.now().isoformat(),
            )
            store.append_job(record)
            time.sleep(0.01)

        jobs = store.list_jobs(limit=3)

        assert len(jobs) == 3


class TestHistoryStorePersistence:
    """Test JSONL file persistence"""

    def test_jobs_persist_across_store_instances(self, temp_history_dir, sample_record):
        """Jobs written by one store can be read by another"""
        store1 = HistoryStore(history_dir=temp_history_dir)
        store1.append_job(sample_record)

        # Create new store instance
        store2 = HistoryStore(history_dir=temp_history_dir)
        retrieved = store2.get_job(sample_record.job_id)

        assert retrieved is not None
        assert retrieved.job_id == sample_record.job_id

    def test_index_file_format_is_jsonl(self, temp_history_dir, sample_record):
        """Index file uses JSONL format"""
        store = HistoryStore(history_dir=temp_history_dir)
        store.append_job(sample_record)

        index_file = temp_history_dir / "index.jsonl"
        assert index_file.exists()

        with open(index_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["job_id"] == sample_record.job_id

    def test_corrupted_line_skipped_with_warning(self, temp_history_dir, sample_record):
        """Corrupted lines are skipped gracefully"""
        store = HistoryStore(history_dir=temp_history_dir)
        store.append_job(sample_record)

        # Corrupt the file by adding invalid JSON
        index_file = temp_history_dir / "index.jsonl"
        with open(index_file, "a", encoding="utf-8") as f:
            f.write("{ invalid json }\n")
            f.write('{"job_id": "job-002", "mode": "custom_script", "status": "idle", "created_at": "2026-07-16T12:00:00", "metadata": {}}\n')

        # Create new store - should load valid lines
        store2 = HistoryStore(history_dir=temp_history_dir)

        jobs = store2.list_jobs()
        job_ids = [j.job_id for j in jobs]

        # Should have original job and job-002, skip corrupted line
        assert sample_record.job_id in job_ids
        assert "job-002" in job_ids


class TestHistoryStoreThreadSafety:
    """Test thread-safe operations"""

    def test_concurrent_appends(self, store):
        """Multiple threads can append concurrently"""
        num_threads = 10
        threads = []

        def append_job(idx):
            record = JobHistoryRecord(
                job_id=f"thread-job-{idx}",
                mode=TrainingMode.GUIDED_ULTRALYTICS.value,
                status=TrainingStatus.IDLE.value,
                created_at=datetime.now().isoformat(),
            )
            store.append_job(record)

        for i in range(num_threads):
            thread = threading.Thread(target=append_job, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        jobs = store.list_jobs()
        assert len(jobs) == num_threads

    def test_concurrent_updates(self, store, sample_record):
        """Multiple threads can update concurrently"""
        store.append_job(sample_record)

        num_threads = 5
        threads = []

        def update_job(idx):
            store.update_job(
                sample_record.job_id,
                metadata={f"thread_{idx}": f"value_{idx}"},
            )

        for i in range(num_threads):
            thread = threading.Thread(target=update_job, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Job should exist and have some thread's update
        updated = store.get_job(sample_record.job_id)
        assert updated is not None


class TestHistoryStoreUnicodePaths:
    """Test Unicode support in paths and content"""

    def test_unicode_in_job_paths(self, store):
        """Can store jobs with Unicode paths"""
        record = JobHistoryRecord(
            job_id="unicode-job-001",
            mode=TrainingMode.GUIDED_ULTRALYTICS.value,
            status=TrainingStatus.IDLE.value,
            created_at=datetime.now().isoformat(),
            workspace="/tmp/训练数据/yolo数据集",
            output_directory="/tmp/输出/runs/train",
        )

        store.append_job(record)

        retrieved = store.get_job(record.job_id)
        assert retrieved.workspace == "/tmp/训练数据/yolo数据集"
        assert retrieved.output_directory == "/tmp/输出/runs/train"

    def test_unicode_in_display_name(self, store):
        """Can store jobs with Unicode display names"""
        record = JobHistoryRecord(
            job_id="unicode-name-001",
            mode=TrainingMode.GUIDED_ULTRALYTICS.value,
            status=TrainingStatus.IDLE.value,
            created_at=datetime.now().isoformat(),
            display_name="目标检测训练任务",
        )

        store.append_job(record)

        retrieved = store.get_job(record.job_id)
        assert retrieved.display_name == "目标检测训练任务"


class TestHistoryStoreSingleton:
    """Test singleton pattern"""

    def test_get_history_store_returns_singleton(self):
        """get_history_store returns same instance"""
        store1 = get_history_store()
        store2 = get_history_store()

        assert store1 is store2

    def test_get_history_store_with_custom_dir_creates_new(self, temp_history_dir):
        """get_history_store with custom dir creates new instance"""
        store1 = get_history_store()
        store2 = get_history_store(history_dir=temp_history_dir)

        assert store1 is not store2


class TestHistoryStoreClear:
    """Test clear operation (for testing)"""

    def test_clear_removes_all_jobs(self, store, sample_record):
        """clear removes all jobs from memory and disk"""
        store.append_job(sample_record)
        assert len(store.list_jobs()) == 1

        store.clear()

        assert len(store.list_jobs()) == 0
        assert not (store.history_dir / "index.jsonl").exists()
