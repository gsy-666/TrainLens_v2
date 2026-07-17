"""TrainLens Environment Wizard — stable task controller.

Owns running worker/thread pairs so widget destruction cannot
prematurely collect them. Internal lifecycle signals are never
disconnected by widget cleanup.
"""

import threading
from typing import Dict, Optional, Tuple

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from .worker import EnvironmentWorker


class _TaskHandle:
    """Internal handle for one environment task."""
    __slots__ = ("task_id", "worker", "thread")

    def __init__(self, task_id: str, worker: EnvironmentWorker, thread: QThread):
        self.task_id = task_id
        self.worker = worker
        self.thread = thread


class EnvironmentTaskController(QObject):
    """Stable owner of environment worker/thread pairs.

    - Owns all running threads (QThread, no parent).
    - Internal lifecycle chain (worker.finished→thread.quit→deleteLater)
      is set up here and NEVER disconnected.
    - Widgets subscribe to result/log signals via subscribe/unsubscribe.
    - When a task completes, the handle is removed from the registry.
    """

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._lock = threading.Lock()
        self._tasks: Dict[str, _TaskHandle] = {}
        self._counter: int = 0

    def start_task(self, worker: EnvironmentWorker) -> str:
        """Start an environment task on a new QThread.

        Returns a task_id that can be used to subscribe to results.
        The controller retains strong refs to worker + thread.
        """
        self._counter += 1
        task_id = f"env_task_{self._counter}"
        thread = QThread()  # No parent — controller holds ref

        worker.moveToThread(thread)

        # ── A: internal lifecycle (NEVER disconnected) ──────────────
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda tid=task_id: self._on_task_finished(tid))

        # ── B: UI signals — widgets subscribe via connect_result etc ──

        thread.started.connect(worker.run)

        handle = _TaskHandle(task_id=task_id, worker=worker, thread=thread)
        with self._lock:
            self._tasks[task_id] = handle

        thread.start()
        return task_id

    def connect_result(self, task_id: str, signal_name: str, slot):
        """Subscribe a widget slot to a result signal on the task's worker."""
        handle = self._get_handle(task_id)
        if handle is None:
            return
        signal = getattr(handle.worker, signal_name, None)
        if signal is not None:
            signal.connect(slot)

    def disconnect_result(self, task_id: str, signal_name: str, slot):
        """Unsubscribe a widget slot (safe to call after widget destroyed)."""
        handle = self._get_handle(task_id)
        if handle is None:
            return
        signal = getattr(handle.worker, signal_name, None)
        if signal is not None:
            try:
                signal.disconnect(slot)
            except Exception:
                pass

    def disconnect_all_ui(self, task_id: str):
        """Disconnect all UI-related signals for a task.

        This does NOT touch internal lifecycle signals (finished).
        Safe to call when widget is being destroyed.
        """
        handle = self._get_handle(task_id)
        if handle is None:
            return
        for sig_name in ("detection_done", "venv_created", "requirements_done",
                         "log_message", "status_changed"):
            signal = getattr(handle.worker, sig_name, None)
            if signal is not None:
                try:
                    signal.disconnect()
                except Exception:
                    pass

    def is_running(self, task_id: str) -> bool:
        handle = self._get_handle(task_id)
        if handle is None:
            return False
        return handle.thread.isRunning()

    def active_count(self) -> int:
        with self._lock:
            return len(self._tasks)

    def shutdown(self):
        """Wait for all running tasks (called on app exit)."""
        with self._lock:
            handles = list(self._tasks.values())
        for h in handles:
            if h.thread.isRunning():
                h.thread.quit()
                h.thread.wait(5000)
        with self._lock:
            self._tasks.clear()

    # ── internal ─────────────────────────────────────────────────────

    def _on_task_finished(self, task_id: str):
        """Called via thread.finished — remove handle from registry."""
        with self._lock:
            self._tasks.pop(task_id, None)

    def _get_handle(self, task_id: str) -> Optional[_TaskHandle]:
        with self._lock:
            return self._tasks.get(task_id)
