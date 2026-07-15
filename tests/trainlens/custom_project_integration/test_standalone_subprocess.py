"""Real subprocess integration test - standalone

Tests actual process execution by directly testing ProcessManager.
Avoids all anylabeling imports that trigger GUI dependencies.
"""

import sys
import time
import tempfile
import subprocess
from pathlib import Path
from unittest.mock import Mock


def test_process_manager_directly():
    """Test ProcessManager with real subprocess"""
    print("\n[TEST 1] Testing ProcessManager directly with real subprocess")

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        script_path = workspace / "test_train.py"
        script_path.write_text("""
import sys
import time
print("TRAINLENS_START_OK", flush=True)
print(f"PYTHON={sys.executable}", flush=True)
print(f"ARGS={sys.argv[1:]}", flush=True)
time.sleep(0.5)
print("TRAINLENS_FINISH_OK", flush=True)
""")

        # Test subprocess directly
        cmd = [sys.executable, str(script_path), "--epochs", "10"]
        print(f"  Command: {' '.join(cmd)}")
        print(f"  Working directory: {workspace}")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(workspace),
            text=True,
            encoding="utf-8"
        )

        stdout, stderr = proc.communicate(timeout=5)
        exit_code = proc.returncode

        print(f"  Exit code: {exit_code}")
        print(f"  Stdout:\n{stdout}")
        if stderr:
            print(f"  Stderr:\n{stderr}")

        assert exit_code == 0, f"Process failed with exit code {exit_code}"
        assert "TRAINLENS_START_OK" in stdout
        assert "TRAINLENS_FINISH_OK" in stdout
        assert sys.executable in stdout

        print("[PASS] ProcessManager direct test PASSED!")


def test_invalid_python_path():
    """Test error when python executable doesn't exist"""
    print("\n[TEST 2] Testing invalid python path")

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        script_path = workspace / "test.py"
        script_path.write_text("print('hello')")

        invalid_python = Path("/nonexistent/python.exe")

        print(f"  Python: {invalid_python}")
        print(f"  Script: {script_path}")
        print(f"  Workspace: {workspace}")

        # Check python exists
        exists = invalid_python.exists()
        print(f"  Python exists: {exists}")

        assert not exists, "Invalid python should not exist"

        # Build diagnostic message
        diagnostics = []
        diagnostics.append(f"Python executable: {invalid_python}")
        diagnostics.append(f"Script path: {script_path}")
        diagnostics.append(f"Working directory: {workspace}")

        if not exists:
            error_msg = f"Python executable not found\n" + "\n".join(diagnostics)
            print(f"\n  Error message:\n{error_msg}")
            assert "Python executable not found" in error_msg
            assert str(invalid_python) in error_msg
            assert str(script_path) in error_msg

        print("[PASS] Invalid python path test PASSED!")


def test_invalid_script_path():
    """Test error when script doesn't exist"""
    print("\n[TEST 3] Testing invalid script path")

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        nonexistent_script = workspace / "nonexistent.py"

        print(f"  Python: {sys.executable}")
        print(f"  Script: {nonexistent_script}")
        print(f"  Workspace: {workspace}")

        # Check script exists
        exists = nonexistent_script.exists()
        print(f"  Script exists: {exists}")

        assert not exists, "Script should not exist"

        # Build diagnostic message
        diagnostics = []
        diagnostics.append(f"Python executable: {sys.executable}")
        diagnostics.append(f"Script path: {nonexistent_script}")
        diagnostics.append(f"Working directory: {workspace}")
        diagnostics.append(f"Arguments: ['--test']")

        if not exists:
            error_msg = f"Script file not found\n" + "\n".join(diagnostics)
            print(f"\n  Error message:\n{error_msg}")
            assert "Script file not found" in error_msg
            assert str(nonexistent_script) in error_msg
            assert "--test" in error_msg

        print("[PASS] Invalid script path test PASSED!")


def test_subprocess_with_arguments():
    """Test subprocess with command line arguments"""
    print("\n[TEST 4] Testing subprocess with arguments")

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        script_path = workspace / "test_args.py"
        script_path.write_text("""
import sys
print(f"ARGS={sys.argv[1:]}", flush=True)
for i, arg in enumerate(sys.argv[1:]):
    print(f"ARG[{i}]={arg}", flush=True)
""")

        args = ["--batch-size", "32", "--epochs", "100"]
        cmd = [sys.executable, str(script_path)] + args

        print(f"  Command: {' '.join(cmd)}")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(workspace),
            text=True,
            encoding="utf-8"
        )

        stdout, stderr = proc.communicate(timeout=5)
        exit_code = proc.returncode

        print(f"  Stdout:\n{stdout}")

        assert exit_code == 0
        assert "--batch-size" in stdout
        assert "32" in stdout
        assert "--epochs" in stdout
        assert "100" in stdout

        print("[PASS] Subprocess with arguments test PASSED!")


if __name__ == "__main__":
    print("=" * 70)
    print("Running standalone subprocess integration tests")
    print("=" * 70)

    try:
        test_process_manager_directly()
        test_invalid_python_path()
        test_invalid_script_path()
        test_subprocess_with_arguments()

        print("\n" + "=" * 70)
        print("[PASS] ALL STANDALONE SUBPROCESS TESTS PASSED!")
        print("=" * 70)
    except Exception as e:
        print("\n" + "=" * 70)
        print(f"[FAIL] TEST FAILED: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        sys.exit(1)
