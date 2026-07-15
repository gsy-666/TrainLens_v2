"""Training Center - Base Adapter Interface

Defines the abstract interface that all training adapters must implement.
Adapters wrap existing training managers without modifying their internal implementation.
"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, Any, Tuple

from ..models import TrainingJob


class TrainingAdapter(ABC):
    """Abstract base class for training adapters

    Adapters use composition to wrap existing training managers.
    They do NOT inherit from or modify existing manager classes.
    """

    @abstractmethod
    def can_start(self) -> Tuple[bool, str]:
        """Check if training can be started

        Returns:
            (can_start, reason) - True if can start, False with reason otherwise
        """
        pass

    @abstractmethod
    def start(self, job: TrainingJob, config: Dict[str, Any]) -> Tuple[bool, str]:
        """Start training with the given job and configuration

        Args:
            job: TrainingJob with metadata
            config: System-specific configuration dictionary

        Returns:
            (success, message) - True if started successfully, False with error otherwise
        """
        pass

    @abstractmethod
    def stop(self) -> bool:
        """Stop the currently running training

        Returns:
            True if stop signal sent successfully, False otherwise
        """
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """Check if training is currently running

        Returns:
            True if training process is active, False otherwise
        """
        pass

    @abstractmethod
    def subscribe(self, callback: Callable) -> None:
        """Subscribe to training events

        Args:
            callback: Function to call with (TrainingEvent) when events occur
        """
        pass

    @abstractmethod
    def unsubscribe(self, callback: Callable) -> None:
        """Unsubscribe from training events

        Args:
            callback: Previously subscribed callback function
        """
        pass
