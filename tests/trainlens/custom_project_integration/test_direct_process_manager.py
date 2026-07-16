"""Direct ProcessManager test to diagnose signal issue

Tests ProcessManager directly without JobManager or CustomScriptAdapter layers.
"""

import sys
import os
from pathlib import Path

from PyQt6.QtWidgets import QApplication
from PyQt6.QtTest import QSignalSpy

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from anylabeling.services.run_monitor.process_manager import ProcessManager
from anylabeling.services.run_monitor.models import Run, RunStatus


def test_direct_process_manager():
    """Test ProcessManager directly to see if signals fire"""

    # Create QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    # Create test script
    import tempfile
    temp_dir = tempfile.mkdtemp()
    script_path = Path(temp_dir) / "test.py"
    script_path.write_text("""
import sys
import time

print("START", flush=True)
for i in range(3):
    print(f"STEP {i+1}", flush=True)
    time.sleep(0.1)
print("END", flush=True)
sys.exit(0)
""")

    print(f"\n[TEST] Direct ProcessManager")
    print(f"  Script: {script_path}")
    print(f"  Python: {sys.executable}")

    # Create ProcessManager
    manager = ProcessManager()

    # Setup signal tracking
    started_signals = []
    finished_signals = []
    stdout_lines = []
    stderr_lines = []

    def on_started(pid):
        print(f"  Signal: process_started(pid={pid})")
        started_signals.append(pid)

    def on_finished(pid, exit_code):
        print(f"  Signal: process_finished(pid={pid}, exit_code={exit_code})")
        finished_signals.append((pid, exit_code))

    def on_stdout(line):
        print(f"  stdout: {line}")
        stdout_lines.append(line)

    def on_stderr(line):
        print(f"  stderr: {line}")
        stderr_lines.append(line)

    manager.process_started.connect(on_started)
    manager.process_finished.connect(on_finished)
    manager.stdout_ready.connect(on_stdout)
    manager.stderr_ready.connect(on_stderr)

    # Setup QSignalSpy
    spy_started = QSignalSpy(manager.process_started)
    spy_finished = QSignalSpy(manager.process_finished)

    # Create run
    run = Run(
        run_id="direct-test-001",
        workspace_path=Path(temp_dir),
        python_path=Path(sys.executable),
        script_path=script_path,
        arguments=[],
        status=RunStatus.PREPARING
    )

    # Start process
    print(f"  Starting process...")
    success = manager.start(run)
    print(f"  Start result: {success}")

    if not success:
        print(f"  ERROR: Failed to start")
        return

    # Process events to let started signal propagate
    app.processEvents()

    print(f"  Spy started count: {len(spy_started)}")
    print(f"  Manual started count: {len(started_signals)}")

    # Wait for process_started
    print(f"  Waiting for process_started signal...")
    if spy_started.wait(5000):
        print(f"  process_started received via spy")
    else:
        print(f"  WARNING: process_started NOT received via spy")

    print(f"  After wait - spy started count: {len(spy_started)}")
    print(f"  After wait - manual started count: {len(started_signals)}")

    # Wait for process_finished
    print(f"  Waiting for process_finished signal...")
    if spy_finished.wait(15000):
        print(f"  process_finished received via spy")
    else:
        print(f"  ERROR: process_finished NOT received via spy")

    # Process remaining events
    app.processEvents()
    import time
    time.sleep(0.2)
    app.processEvents()

    print(f"\n[RESULTS]")
    print(f"  spy_started count: {len(spy_started)}")
    print(f"  spy_finished count: {len(spy_finished)}")
    print(f"  manual started count: {len(started_signals)}")
    print(f"  manual finished count: {len(finished_signals)}")
    print(f"  stdout lines: {len(stdout_lines)}")
    print(f"  stderr lines: {len(stderr_lines)}")
    print(f"  manager.is_running(): {manager.is_running()}")

    if stdout_lines:
        print(f"\n  Stdout content:")
        for line in stdout_lines:
            print(f"    {line}")

    if finished_signals:
        print(f"\n  Finished signal: pid={finished_signals[0][0]}, exit_code={finished_signals[0][1]}")
    else:
        print(f"\n  ERROR: No finished signal received")

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_direct_process_manager()
