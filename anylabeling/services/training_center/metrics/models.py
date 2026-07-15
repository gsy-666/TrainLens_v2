"""Training Center - Metrics Models

Unified metric data structures for all training sources.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class MetricPoint:
    """Single metric observation"""
    job_id: str
    step: Optional[int] = None
    epoch: Optional[int] = None
    timestamp: float = 0.0
    name: str = ""
    value: float = 0.0
    source: str = ""


@dataclass
class MetricsSnapshot:
    """Current training metrics snapshot"""
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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "epoch": self.epoch,
            "total_epochs": self.total_epochs,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "precision": self.precision,
            "recall": self.recall,
            "map50": self.map50,
            "map50_95": self.map50_95,
            "learning_rates": self.learning_rates,
            "last_updated": self.last_updated,
        }
