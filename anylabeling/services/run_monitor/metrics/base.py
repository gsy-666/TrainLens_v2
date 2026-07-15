"""Base metric reader interface"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional


class MetricReader(ABC):
    """Abstract base class for metric readers"""

    @abstractmethod
    def can_read(self, workspace_path: Path, run_id: str) -> bool:
        """
        Check if this reader can read metrics from the given run.

        Args:
            workspace_path: Workspace root directory
            run_id: Run identifier

        Returns:
            True if metrics can be read, False otherwise
        """
        pass

    @abstractmethod
    def read_metrics(
        self, workspace_path: Path, run_id: str
    ) -> Optional[List[Dict]]:
        """
        Read training metrics.

        Args:
            workspace_path: Workspace root directory
            run_id: Run identifier

        Returns:
            List of metric dictionaries, or None if no metrics available
        """
        pass
