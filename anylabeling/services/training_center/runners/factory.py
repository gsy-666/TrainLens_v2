"""RunnerFactory — maps execution_mode → TrainingRunner implementation."""

from typing import Dict, Optional

from .base import TrainingRunner


class RunnerFactory:
    """Registry of TrainingRunner implementations.

    Usage:
        factory = RunnerFactory.get_instance()
        factory.register("local", LocalRunner())
        runner = factory.create("local")
        runner2 = factory.create("remote")  # raises NotImplementedError
    """

    _instance: Optional["RunnerFactory"] = None

    def __init__(self):
        self._runners: Dict[str, TrainingRunner] = {}

    @classmethod
    def get_instance(cls) -> "RunnerFactory":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, mode: str, runner: TrainingRunner):
        """Register a runner implementation for an execution mode."""
        self._runners[mode] = runner

    def create(self, execution_mode: str) -> TrainingRunner:
        """Get or create a runner for the given execution mode.

        Raises:
            ValueError: when execution_mode is not registered.
            Never silently falls back to LocalRunner.
        """
        if execution_mode not in self._runners:
            raise ValueError(
                f"Unsupported execution mode: {execution_mode!r}. "
                f"Available: {list(self._runners.keys())}"
            )
        return self._runners[execution_mode]

    def get(self, execution_mode: str) -> Optional[TrainingRunner]:
        """Get runner or None (no exception)."""
        return self._runners.get(execution_mode)


def get_runner(execution_mode: str) -> TrainingRunner:
    """Convenience: get a runner from the global factory.

    Raises ValueError for unimplemented modes.
    Never silently returns LocalRunner.
    """
    return RunnerFactory.get_instance().create(execution_mode)
