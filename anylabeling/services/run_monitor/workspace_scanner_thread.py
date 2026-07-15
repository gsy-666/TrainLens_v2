"""Thread-based workspace scanner for Run Monitor

Provides non-blocking workspace scanning using QThread.
"""

from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal

from .models import Workspace
from .workspace_scanner import WorkspaceScanner


class WorkspaceScannerThread(QThread):
    """Thread for non-blocking workspace scanning"""

    # Signals
    progress = pyqtSignal(int, int, str)  # current, total, message
    finished = pyqtSignal(object)  # Workspace result
    error = pyqtSignal(str)  # Error message

    def __init__(self, workspace_path: Path, parent=None):
        super().__init__(parent)
        self.workspace_path = workspace_path
        self.scanner = WorkspaceScanner()

    def run(self):
        """Execute scan in background thread"""
        try:
            workspace = self.scanner.scan(
                self.workspace_path,
                progress_callback=self._progress_callback,
            )
            self.finished.emit(workspace)
        except Exception as e:
            self.error.emit(str(e))

    def cancel(self):
        """Request cancellation of ongoing scan"""
        self.scanner.cancel()
        self.requestInterruption()

    def _progress_callback(self, current: int, total: int, message: str):
        """Forward progress to signal"""
        if not self.isInterruptionRequested():
            self.progress.emit(current, total, message)
