"""Training Metrics — lightweight data models."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class MetricSample:
    """A single metrics data point carrying multiple values at one step."""
    job_id: str
    step: Optional[int] = None
    epoch: Optional[float] = None
    timestamp: Optional[float] = None
    total_epochs: Optional[int] = None
    values: Dict[str, float] = field(default_factory=dict)


@dataclass
class MetricSeries:
    """Ordered series of (x, y) points for one named metric."""
    name: str
    display_name: str
    group: str  # "loss", "quality", "learning_rate", "other"
    points: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class MetricRunData:
    """All metrics for one training run."""
    job_id: str
    source: str = ""  # "results.csv", "metrics.jsonl", "event"
    output_dir: Optional[str] = None
    total_epochs: Optional[int] = None
    last_updated: float = 0.0
    samples: List[MetricSample] = field(default_factory=list)

    def to_series(self) -> List[MetricSeries]:
        """Convert samples to grouped MetricSeries, deduplicating by x."""
        groups: Dict[str, Dict[str, Dict[float, float]]] = {}
        names: Dict[str, str] = {}

        for s in self.samples:
            x = s.epoch if s.epoch is not None else (s.step if s.step is not None else 0)
            for key, val in s.values.items():
                group = _classify_group(key)
                display = _display_name(key)
                if group not in groups:
                    groups[group] = {}
                if key not in groups[group]:
                    groups[group][key] = {}
                    names[key] = display
                groups[group][key][float(x)] = val  # last wins for same x

        series = []
        for group, metrics in sorted(groups.items()):
            for key, pts in metrics.items():
                sorted_pts = sorted(pts.items())
                series.append(MetricSeries(
                    name=key,
                    display_name=names.get(key, key),
                    group=group,
                    points=[(x, y) for x, y in sorted_pts],
                ))
        return series


def _classify_group(name: str) -> str:
    low = name.lower()
    if "loss" in low:
        return "loss"
    if any(w in low for w in ("precision", "recall", "map", "accuracy", "top1", "top5", "fitness")):
        return "quality"
    if low.startswith("lr/") or "learning_rate" in low:
        return "learning_rate"
    return "other"


def _display_name(name: str) -> str:
    return name.replace("/", " / ").replace("_", " ").strip()


# ── legacy compatibility ─────────────────────────────────────────────

from dataclasses import dataclass as _dc
from typing import Dict as _Dict, Any as _Any, Optional as _Opt

@_dc
class MetricPoint:
    """Legacy single-observation point (compatibility)."""
    job_id: str = ""
    step: Optional[int] = None
    epoch: Optional[int] = None
    timestamp: float = 0.0
    name: str = ""
    value: float = 0.0
    source: str = ""

@_dc
class MetricsSnapshot:
    """Legacy metrics snapshot (compatibility)."""
    epoch: int = 0
    total_epochs: Optional[int] = None
    train_losses: Dict[str, float] = field(default_factory=dict)
    val_losses: Dict[str, float] = field(default_factory=dict)
    precision: Optional[float] = None
    recall: Optional[float] = None
    map50: Optional[float] = None
    map50_95: Optional[float] = None
    learning_rates: Dict[str, float] = field(default_factory=dict)
    last_updated: float = 0.0

