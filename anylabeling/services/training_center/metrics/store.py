"""Training Metrics — in-memory store with job_id isolation."""

import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

from .models import MetricRunData, MetricSample
from .parser import parse_results_csv, parse_metrics_jsonl


class MetricStore:
    """Thread-safe in-memory store for training metrics.

    - Isolates data by job_id.
    - New runStart clears previous live data.
    - Deduplicates samples by (epoch, metric_name).
    - Supports polling results.csv for updates.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._runs: Dict[str, MetricRunData] = {}
        self._active_job_id: Optional[str] = None
        self._csv_paths: Dict[str, Path] = {}
        self._csv_last_sizes: Dict[str, int] = {}

    def start_run(self, job_id: str, output_dir: Optional[str] = None):
        """Begin tracking a new run. Clears previous live data."""
        with self._lock:
            self._active_job_id = job_id
            if job_id in self._runs:
                self._runs[job_id].samples.clear()
            else:
                self._runs[job_id] = MetricRunData(job_id=job_id)
            if output_dir:
                self._runs[job_id].output_dir = output_dir
                # Always save csv_path and set last_size=-1 so first poll picks up existing CSV
                csv_path = Path(output_dir) / "results.csv"
                self._csv_paths[job_id] = csv_path
                self._csv_last_sizes[job_id] = -1

    def add_sample(self, sample: MetricSample):
        """Add a metric sample (from event or parsing)."""
        with self._lock:
            if sample.job_id not in self._runs:
                self._runs[sample.job_id] = MetricRunData(job_id=sample.job_id)
            run = self._runs[sample.job_id]
            # Dedup by epoch
            ep = sample.epoch or sample.step or 0
            for i, existing in enumerate(run.samples):
                existing_ep = existing.epoch or existing.step or 0
                if abs(existing_ep - float(ep)) < 0.001:
                    run.samples[i] = sample
                    break
            else:
                run.samples.append(sample)
            run.last_updated = time.time()

    def poll_csv(self) -> Optional[MetricRunData]:
        """Poll results.csv for changes. Returns updated data or None.

        Only polls when there's an active job with a known output_dir.
        Checks file existence on every poll (CSV may appear mid-training).
        """
        job_id = self._active_job_id
        if not job_id:
            return None
        csv_path = self._csv_paths.get(job_id)
        if not csv_path:
            return None
        if not csv_path.exists():
            return None  # CSV not created yet (epoch 0)

        try:
            current_size = csv_path.stat().st_size
            if current_size <= self._csv_last_sizes.get(job_id, -1):
                return None  # no new data
            self._csv_last_sizes[job_id] = current_size
        except OSError:
            return None

        data = parse_results_csv(csv_path, job_id)
        if data.samples:
            with self._lock:
                self._runs[job_id] = data
            return data
        return None

    def load_from_output_dir(self, job_id: str, output_dir: str) -> Optional[MetricRunData]:
        """Load metrics from files in an output directory.

        Priority: results.csv > metrics.jsonl
        """
        od = Path(output_dir)
        if not od.exists():
            return None

        csv_path = od / "results.csv"
        if csv_path.exists():
            data = parse_results_csv(csv_path, job_id)
        else:
            jsonl_path = od / "metrics.jsonl"
            if jsonl_path.exists():
                data = parse_metrics_jsonl(jsonl_path, job_id)
            else:
                return None

        if data and data.samples:
            with self._lock:
                self._runs[job_id] = data
            return data
        return None

    def get_run(self, job_id: str) -> Optional[MetricRunData]:
        with self._lock:
            return self._runs.get(job_id)

    def clear_run(self, job_id: str):
        with self._lock:
            self._runs.pop(job_id, None)
            self._csv_paths.pop(job_id, None)
            self._csv_last_sizes.pop(job_id, None)
            if self._active_job_id == job_id:
                self._active_job_id = None

    def update_csv_path(self, job_id: str, output_dir: str):
        """Update CSV path for an already-bound job (real save_dir callback).

        Called when the actual training output directory becomes known,
        which may differ from the predicted project/name path.
        Sets last_sizes=-1 to force re-read on next poll.
        """
        csv_path = Path(output_dir) / "results.csv"
        with self._lock:
            self._csv_paths[job_id] = csv_path
            self._csv_last_sizes[job_id] = -1  # force re-read
            if job_id in self._runs:
                self._runs[job_id].output_dir = output_dir

    def finish_run(self, job_id: str):
        """Mark run as finished. Stops live polling. Data preserved for history.

        Does NOT clear csv_paths — caller may need one final poll first.
        """
        with self._lock:
            if self._active_job_id == job_id:
                self._active_job_id = None
