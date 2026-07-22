"""Non-blocking package installer using QProcess.

Key differences from the old install_packages_with_timeout():
1. Runs in a QProcess, never blocks the Qt main thread.
2. No fixed timeout — user can cancel at any time.
3. Real-time output streamed via signals.
4. Large dependencies require explicit user confirmation before installing.
"""

import sys
from PyQt6.QtCore import QObject, QProcess, pyqtSignal, pyqtSlot


class PackageInstaller(QObject):
    """Installs pip packages asynchronously via QProcess."""

    # Signals
    install_started = pyqtSignal(list)  # packages being installed
    install_output = pyqtSignal(str)  # real-time stdout/stderr lines
    install_finished = pyqtSignal(bool, str, str)  # (success, stdout, stderr)
    install_cancelled = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._process: QProcess | None = None
        self._is_running = False
        self._stdout_lines = []
        self._stderr_lines = []

    @property
    def is_running(self) -> bool:
        return self._is_running

    def install(self, packages: list, extra_args: list | None = None):
        """Start installing packages.

        Args:
            packages: List of pip package specs (e.g. ["onnx>=1.15.0"]).
            extra_args: Extra pip arguments (e.g. ["--index-url", "..."]).
        """
        if self._is_running:
            return

        self._is_running = True
        self._stdout_lines = []
        self._stderr_lines = []

        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._on_stdout)
        self._process.finished.connect(self._on_finished)

        cmd = [sys.executable, "-m", "pip", "install", "--no-input"]
        if extra_args:
            cmd.extend(extra_args)
        cmd.extend(packages)

        self.install_started.emit(packages)
        self._process.start(cmd[0], cmd[1:])

    def cancel(self):
        """Cancel the running installation."""
        if self._process and self._is_running:
            self._process.kill()
            self._is_running = False
            self.install_cancelled.emit()

    @pyqtSlot()
    def _on_stdout(self):
        if self._process:
            data = self._process.readAllStandardOutput()
            text = bytes(data).decode("utf-8", errors="replace")
            for line in text.splitlines():
                line = line.strip()
                if line:
                    self._stdout_lines.append(line)
                    self.install_output.emit(line)

    @pyqtSlot(int, QProcess.ExitStatus)
    def _on_finished(self, exit_code, exit_status):
        self._is_running = False
        stdout = "\n".join(self._stdout_lines)
        stderr = ""  # merged channel includes stderr in stdout
        success = exit_code == 0
        self.install_finished.emit(success, stdout, stderr)
