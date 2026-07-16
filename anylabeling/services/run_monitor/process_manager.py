"""Process manager for Run Monitor

Manages training script execution in subprocess.
"""

import os
import platform
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from .models import Run, RunStatus


class ProcessManager(QObject):
    """Manages training subprocess lifecycle"""

    # Signals
    process_started = pyqtSignal(int)  # pid
    process_finished = pyqtSignal(int, int)  # pid, exit_code
    stdout_ready = pyqtSignal(str)  # log line
    stderr_ready = pyqtSignal(str)  # error line

    def __init__(self):
        super().__init__()
        self._process: Optional[subprocess.Popen] = None
        self._run: Optional[Run] = None
        self._stdout_thread: Optional[QThread] = None
        self._stderr_thread: Optional[QThread] = None
        self._stdout_reader: Optional['OutputReader'] = None
        self._stderr_reader: Optional['OutputReader'] = None
        self._finished_streams: set = set()  # Track which streams have finished
        self._stop_requested: bool = False  # Track if user requested stop

    def start(self, run: Run) -> bool:
        """
        Start training subprocess.

        Args:
            run: Run configuration

        Returns:
            True if started successfully, False otherwise
        """
        if self._process is not None:
            return False

        self._run = run

        try:
            # Build command
            cmd = [str(run.python_path), str(run.script_path)] + run.arguments

            # Set working directory to workspace root
            cwd = str(run.workspace_path)

            # Create subprocess
            if platform.system() == "Windows":
                # Windows: no special process group handling needed for taskkill
                self._process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=cwd,
                    text=True,
                    bufsize=1,
                    encoding="utf-8",
                    errors="replace",
                )
            else:
                # Linux/Mac: use process group for clean termination
                self._process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=cwd,
                    text=True,
                    bufsize=1,
                    encoding="utf-8",
                    errors="replace",
                    preexec_fn=os.setsid,
                )

            # Update run metadata
            run.pid = self._process.pid
            run.start_time = datetime.now()
            run.status = RunStatus.RUNNING

            # Reset state for new run
            self._finished_streams.clear()
            self._stop_requested = False

            # Start output readers in threads
            self._start_output_readers()

            # Emit signal
            self.process_started.emit(run.pid)

            return True

        except FileNotFoundError:
            run.status = RunStatus.FAILED
            self.stderr_ready.emit(
                f"Error: Python executable not found: {run.python_path}"
            )
            return False
        except PermissionError:
            run.status = RunStatus.FAILED
            self.stderr_ready.emit(
                f"Error: Permission denied: {run.script_path}"
            )
            return False
        except Exception as e:
            run.status = RunStatus.FAILED
            self.stderr_ready.emit(f"Error starting process: {e}")
            return False

    def stop(self) -> bool:
        """
        Stop training subprocess.

        Returns:
            True if stopped successfully, False otherwise
        """
        if self._process is None:
            return False

        # Idempotent: don't stop twice
        if self._stop_requested:
            return False

        self._stop_requested = True

        if self._run:
            self._run.status = RunStatus.STOPPING

        try:
            pid = self._process.pid

            if platform.system() == "Windows":
                # Windows: use taskkill to kill process tree
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(self._process.pid)],
                    capture_output=True,
                    timeout=5,
                )
            else:
                # Linux/Mac: kill process group
                try:
                    os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
                    # Wait briefly for graceful shutdown
                    try:
                        self._process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        # Force kill if not terminated
                        os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    # Process already terminated
                    pass

            # Start a timer to check if process finished and emit signal if needed
            # This handles the case where force-kill closes pipes before OutputReaders finish
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(100, lambda: self._check_forced_completion(pid))

            return True

        except Exception as e:
            self.stderr_ready.emit(f"Error stopping process: {e}")
            return False

    def _check_forced_completion(self, pid: int):
        """Check if process exited but streams haven't finished, and force emit if needed"""
        # If process has exited but streams haven't both finished, force emit
        if self._process and self._process.poll() is not None:
            if len(self._finished_streams) < 2:
                exit_code = self._process.returncode

                if self._run:
                    self._run.end_time = datetime.now()
                    self._run.exit_code = exit_code
                    if self._run.status == RunStatus.STOPPING:
                        self._run.status = RunStatus.STOPPED
                    else:
                        self._run.status = RunStatus.FAILED

                self.process_finished.emit(pid, exit_code)
                self._cleanup()
            # If still not finished, check again
            elif self._process:
                from PyQt6.QtCore import QTimer
                QTimer.singleShot(100, lambda: self._check_forced_completion(pid))

    def is_running(self) -> bool:
        """Check if process is running"""
        if self._process is None:
            return False
        return self._process.poll() is None

    def wait(self, timeout: Optional[float] = None) -> Optional[int]:
        """
        Wait for process to complete.

        Args:
            timeout: Maximum seconds to wait, None for no timeout

        Returns:
            Exit code, or None if timeout
        """
        if self._process is None:
            return None

        try:
            return self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None

    def _start_output_readers(self):
        """Start threads to read stdout and stderr"""
        # Read stdout in thread
        self._stdout_thread = QThread()
        self._stdout_reader = OutputReader(self._process.stdout, "stdout")
        self._stdout_reader.moveToThread(self._stdout_thread)
        self._stdout_reader.line_ready.connect(self.stdout_ready)
        self._stdout_reader.finished.connect(lambda: self._on_output_finished("stdout"))
        self._stdout_thread.started.connect(self._stdout_reader.run)
        self._stdout_thread.start()

        # Read stderr in thread
        self._stderr_thread = QThread()
        self._stderr_reader = OutputReader(self._process.stderr, "stderr")
        self._stderr_reader.moveToThread(self._stderr_thread)
        self._stderr_reader.line_ready.connect(self.stderr_ready)
        self._stderr_reader.finished.connect(lambda: self._on_output_finished("stderr"))
        self._stderr_thread.started.connect(self._stderr_reader.run)
        self._stderr_thread.start()

    def _on_output_finished(self, stream_name: str):
        """Called when output reader finishes

        Args:
            stream_name: Name of stream that finished ("stdout" or "stderr")
        """
        # Track which stream finished
        self._finished_streams.add(stream_name)

        # Only emit process_finished when BOTH streams are done AND process has exited
        if len(self._finished_streams) >= 2 and self._process and self._process.poll() is not None:
            exit_code = self._process.returncode
            pid = self._process.pid

            if self._run:
                self._run.end_time = datetime.now()
                self._run.exit_code = exit_code
                if exit_code == 0:
                    self._run.status = RunStatus.COMPLETED
                elif self._run.status == RunStatus.STOPPING:
                    self._run.status = RunStatus.STOPPED
                else:
                    self._run.status = RunStatus.FAILED

            self.process_finished.emit(pid, exit_code)

            # Clean up
            self._cleanup()

    def _cleanup(self):
        """Clean up threads and process"""
        if self._stdout_thread:
            self._stdout_thread.quit()
            self._stdout_thread.wait()
            self._stdout_thread = None

        if self._stderr_thread:
            self._stderr_thread.quit()
            self._stderr_thread.wait()
            self._stderr_thread = None

        self._stdout_reader = None
        self._stderr_reader = None
        self._process = None


class OutputReader(QObject):
    """Reads output from subprocess in thread"""

    line_ready = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, stream, name):
        super().__init__()
        self.stream = stream
        self.name = name

    def run(self):
        """Read lines from stream and emit signals"""
        try:
            for line in self.stream:
                line = line.rstrip("\n\r")
                if line:
                    self.line_ready.emit(line)
        except Exception as e:
            self.line_ready.emit(f"Error reading {self.name}: {e}")
        finally:
            self.finished.emit()
