"""Process manager for Run Monitor

Manages training script execution in subprocess.

Architecture:
- ProcessWatcher: Dedicated thread waiting on process.wait() for true exit detection
- OutputReader: Read stdout/stderr streams independently
- Finalizer: Unified, idempotent terminal state handler (_finalize_process)

This design ensures process termination is detected even if OutputReaders fail.
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


class ProcessWatcher(QObject):
    """Watches subprocess and emits signal when it exits

    Runs in dedicated thread, calls process.wait() to get true exit code.
    This is the authoritative source for process termination.
    """

    process_exited = pyqtSignal(int)  # exit_code

    def __init__(self, process: subprocess.Popen):
        super().__init__()
        self.process = process

    def run(self):
        """Wait for process to exit and emit signal with exit code"""
        try:
            exit_code = self.process.wait()
            self.process_exited.emit(exit_code)
        except Exception as e:
            # Process may have been killed or become invalid
            # Try to get return code if available
            if self.process.returncode is not None:
                self.process_exited.emit(self.process.returncode)
            else:
                # Assume failure if we can't determine exit code
                self.process_exited.emit(1)


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

        # Process watcher (authoritative exit detection)
        self._watcher_thread: Optional[QThread] = None
        self._watcher: Optional[ProcessWatcher] = None

        # Output readers
        self._stdout_thread: Optional[QThread] = None
        self._stderr_thread: Optional[QThread] = None
        self._stdout_reader: Optional['OutputReader'] = None
        self._stderr_reader: Optional['OutputReader'] = None

        # State management
        self._stop_requested: bool = False
        self._terminal_emitted: bool = False  # Ensure finalize only once

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
            self._stop_requested = False
            self._terminal_emitted = False

            # Start process watcher (authoritative exit detection)
            self._start_process_watcher()

            # Start output readers
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

            # ProcessWatcher will detect the exit and call _finalize_process
            return True

        except Exception as e:
            self.stderr_ready.emit(f"Error stopping process: {e}")
            return False

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

    def _start_process_watcher(self):
        """Start dedicated thread to watch process exit"""
        self._watcher_thread = QThread()
        self._watcher = ProcessWatcher(self._process)
        self._watcher.moveToThread(self._watcher_thread)
        self._watcher.process_exited.connect(self._on_process_exited)
        self._watcher_thread.started.connect(self._watcher.run)
        self._watcher_thread.start()

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

    def _on_process_exited(self, exit_code: int):
        """Called when ProcessWatcher detects process exit

        This is the authoritative source for process termination.
        """
        self._finalize_process(exit_code)

    def _on_output_finished(self, stream_name: str):
        """Called when output reader finishes

        Args:
            stream_name: Name of stream that finished ("stdout" or "stderr")

        Note: This is no longer used for terminal state detection.
        OutputReaders finishing is informational only.
        """
        # Output reader finished - this is normal but doesn't control termination
        pass

    def _finalize_process(self, exit_code: int):
        """Unified, idempotent terminal state handler

        Args:
            exit_code: Process exit code

        This method can be called by:
        - ProcessWatcher when process exits
        - Timeout handlers
        - Multiple callers in race conditions

        Only the first call takes effect (idempotent).
        """
        # Idempotent: only finalize once
        if self._terminal_emitted:
            return

        self._terminal_emitted = True

        # Save PID before cleanup
        pid = self._process.pid if self._process else 0

        # Determine terminal state based on exit code and stop flag
        if self._run:
            self._run.end_time = datetime.now()
            self._run.exit_code = exit_code

            if self._stop_requested:
                # User explicitly stopped - always STOPPED
                self._run.status = RunStatus.STOPPED
            elif exit_code == 0:
                # Natural successful completion
                self._run.status = RunStatus.COMPLETED
            else:
                # Process failed with non-zero exit
                self._run.status = RunStatus.FAILED

        # Emit terminal signal
        self.process_finished.emit(pid, exit_code)

        # Clean up threads and resources
        self._cleanup()

    def _cleanup(self):
        """Clean up threads and process"""
        # Clean up process watcher
        if self._watcher_thread:
            self._watcher_thread.quit()
            self._watcher_thread.wait(2000)  # Wait up to 2 seconds
            self._watcher_thread = None
        self._watcher = None

        # Clean up output readers
        if self._stdout_thread:
            self._stdout_thread.quit()
            self._stdout_thread.wait(2000)
            self._stdout_thread = None

        if self._stderr_thread:
            self._stderr_thread.quit()
            self._stderr_thread.wait(2000)
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
        """Read lines from stream and emit signals

        Must always emit finished signal, even on error.
        """
        try:
            for line in self.stream:
                line = line.rstrip("\n\r")
                if line:
                    self.line_ready.emit(line)
        except Exception as e:
            self.line_ready.emit(f"Error reading {self.name}: {e}")
        finally:
            # Always emit finished, even if stream was closed abruptly
            self.finished.emit()
