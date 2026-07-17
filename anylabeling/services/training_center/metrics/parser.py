"""Training Metrics — parsers for results.csv and metrics.jsonl."""

import csv
import json
import logging
import math
import time
from pathlib import Path
from typing import Optional

from .models import MetricRunData, MetricSample

_log = logging.getLogger(__name__)
_SKIP_COLS = {"epoch", "time", "remaining", "hours", ""}


def parse_results_csv(path: Path, job_id: str = "") -> MetricRunData:
    """Parse Ultralytics results.csv into MetricRunData.

    Handles: missing epoch, bad cells, duplicate epochs, incomplete last line.
    """
    if not path or not path.exists():
        return MetricRunData(job_id=job_id, source="results.csv")

    data = MetricRunData(job_id=job_id, source="results.csv",
                         output_dir=str(path.parent), last_updated=time.time())
    try:
        text = path.read_text(encoding="utf-8-sig")
    except Exception:
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            return data

    lines = text.splitlines()
    if len(lines) < 2:
        return data

    reader = csv.DictReader(lines)
    if reader.fieldnames is None:
        return data

    seen: dict = {}
    total_rows = 0
    for row_num, row in enumerate(reader):
        if row is None:
            continue
        total_rows += 1
        epoch = _safe_float((row.get("epoch", "") or "").strip())
        if epoch is None:
            epoch = float(row_num + 1)

        values: dict = {}
        for col in reader.fieldnames:
            col_stripped = col.strip()
            if not col_stripped or col_stripped.lower() in _SKIP_COLS:
                continue
            raw = (row.get(col, "") or "").strip()
            if not raw:
                continue
            val = _safe_float(raw)
            if val is not None and not math.isnan(val) and not math.isinf(val):
                values[col_stripped] = val

        if not values:
            continue

        ep_key = round(epoch, 6)
        seen[ep_key] = MetricSample(
            job_id=job_id, epoch=epoch, step=row_num + 1,
            values=values, timestamp=time.time(),
        )

    data.samples = sorted(seen.values(), key=lambda s: s.epoch or 0)
    data.total_epochs = len(data.samples)
    return data


def parse_metrics_jsonl(path: Path, job_id: str = "") -> MetricRunData:
    """Parse metrics.jsonl into MetricRunData.

    One JSON object per line. Skips bad lines, handles NaN/Inf filtering.
    """
    if not path or not path.exists():
        return MetricRunData(job_id=job_id, source="metrics.jsonl")

    data = MetricRunData(job_id=job_id, source="metrics.jsonl",
                         output_dir=str(path.parent), last_updated=time.time())
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return data

    seen: dict = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue

        metrics_raw = obj.get("metrics", {})
        if not isinstance(metrics_raw, dict):
            continue

        values: dict = {}
        for k, v in metrics_raw.items():
            if isinstance(v, (int, float)) and not math.isnan(v) and not math.isinf(v):
                values[str(k)] = float(v)

        if not values:
            continue

        epoch = obj.get("epoch")
        if isinstance(epoch, (int, float)):
            epoch = float(epoch)
        elif isinstance(epoch, str):
            epoch = _safe_float(epoch)
        else:
            step = obj.get("step")
            if isinstance(step, (int, float)):
                epoch = float(step)
            else:
                epoch = float(len(seen) + 1)

        ep_key = round(epoch, 6)
        seen[ep_key] = MetricSample(
            job_id=job_id,
            epoch=epoch,
            step=obj.get("step"),
            total_epochs=obj.get("total_epochs"),
            values=values,
            timestamp=obj.get("timestamp", time.time()),
        )

    data.samples = sorted(seen.values(), key=lambda s: s.epoch or 0)
    data.total_epochs = len(data.samples)
    return data


def _safe_float(raw: str) -> Optional[float]:
    if not raw:
        return None
    try:
        return float(raw)
    except (ValueError, TypeError):
        return None
