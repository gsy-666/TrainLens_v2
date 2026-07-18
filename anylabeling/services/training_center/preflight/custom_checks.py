"""TrainLens Preflight — Custom Project checks."""

import os
import subprocess
from pathlib import Path

from . import PreflightIssue, PreflightResult, PreflightSeverity


def check_python_executable(result: PreflightResult, python_path: str, check_version: bool = True):
    """Check Python executable exists, is executable, and returns --version."""
    if not python_path or not python_path.strip():
        result.add(PreflightIssue(
            code="PYTHON_EMPTY", severity=PreflightSeverity.ERROR,
            title="Python executable not specified",
            message="No Python executable selected.",
            field_name="python_path",
        ))
        return

    path = Path(python_path)

    if not path.exists():
        result.add(PreflightIssue(
            code="PYTHON_NOT_FOUND", severity=PreflightSeverity.ERROR,
            title="Python executable not found",
            message=f"Python executable does not exist: {python_path}",
            field_name="python_path", path=python_path,
            suggestion="Select a valid Python executable or create a virtual environment.",
        ))
        return

    if not path.is_file():
        result.add(PreflightIssue(
            code="PYTHON_NOT_FILE", severity=PreflightSeverity.ERROR,
            title="Python path is not a file",
            message=f"The Python path is not a regular file: {python_path}",
            field_name="python_path", path=python_path,
        ))
        return

    if not check_version:
        result.add(PreflightIssue(
            code="PYTHON_OK", severity=PreflightSeverity.PASS,
            title="Python executable exists",
            message=f"Python: {python_path}",
            path=python_path,
        ))
        return

    # Test --version
    try:
        proc = subprocess.run(
            [str(path), "--version"],
            capture_output=True, text=True, timeout=10,
        )
        if proc.returncode != 0:
            result.add(PreflightIssue(
                code="PYTHON_VERSION_FAIL", severity=PreflightSeverity.ERROR,
                title="Python executable does not respond",
                message=f"'{python_path} --version' returned exit code {proc.returncode}.",
                field_name="python_path", path=python_path,
                details=proc.stderr.strip(),
            ))
            return
        version = proc.stdout.strip()
        result.add(PreflightIssue(
            code="PYTHON_OK", severity=PreflightSeverity.PASS,
            title=f"Python executable is valid ({version})",
            message=f"Python: {python_path}\nVersion: {version}",
            path=python_path,
        ))
    except subprocess.TimeoutExpired:
        result.add(PreflightIssue(
            code="PYTHON_TIMEOUT", severity=PreflightSeverity.ERROR,
            title="Python check timed out",
            message=f"'{python_path} --version' did not respond within 10 seconds.",
            field_name="python_path", path=python_path,
        ))
    except OSError as e:
        result.add(PreflightIssue(
            code="PYTHON_EXEC_FAIL", severity=PreflightSeverity.ERROR,
            title="Cannot execute Python",
            message=f"Failed to execute '{python_path}': {e}",
            field_name="python_path", path=python_path,
        ))


def check_script(result: PreflightResult, script_path: str, project_dir: str = ""):
    """Check training script exists and is a file."""
    if not script_path or not script_path.strip():
        result.add(PreflightIssue(
            code="SCRIPT_EMPTY", severity=PreflightSeverity.ERROR,
            title="Training script not specified",
            message="No training script selected.",
            field_name="script",
        ))
        return

    path = Path(script_path)

    if not path.exists():
        result.add(PreflightIssue(
            code="SCRIPT_NOT_FOUND", severity=PreflightSeverity.ERROR,
            title="Training script not found",
            message=f"Script does not exist: {script_path}",
            field_name="script", path=script_path,
        ))
        return

    if not path.is_file():
        result.add(PreflightIssue(
            code="SCRIPT_NOT_FILE", severity=PreflightSeverity.ERROR,
            title="Script path is not a file",
            message=f"The script path is not a regular file: {script_path}",
            field_name="script", path=script_path,
        ))
        return

    # Check if script is outside project dir
    if project_dir:
        try:
            proj = Path(project_dir).resolve()
            scr = path.resolve()
            if proj not in scr.parents and proj != scr.parent:
                result.add(PreflightIssue(
                    code="SCRIPT_OUTSIDE_PROJECT", severity=PreflightSeverity.WARNING,
                    title="Training script is outside project directory",
                    message=f"Script: {script_path}\nProject: {project_dir}",
                    field_name="script", path=script_path,
                    suggestion="Script is outside the project. Verify this is intentional.",
                ))
        except (OSError, ValueError):
            pass

    result.add(PreflightIssue(
        code="SCRIPT_OK", severity=PreflightSeverity.PASS,
        title="Training script is valid",
        message=f"Script: {script_path}",
        path=script_path,
    ))


def check_workspace(result: PreflightResult, workspace: str):
    """Check project workspace exists and is a directory."""
    if not workspace or not workspace.strip():
        result.add(PreflightIssue(
            code="WORKSPACE_EMPTY", severity=PreflightSeverity.ERROR,
            title="Project directory not specified",
            message="No project/workspace directory configured.",
            field_name="workspace",
        ))
        return

    path = Path(workspace)

    if not path.exists():
        result.add(PreflightIssue(
            code="WORKSPACE_NOT_FOUND", severity=PreflightSeverity.ERROR,
            title="Project directory does not exist",
            message=f"Workspace directory not found: {workspace}",
            field_name="workspace", path=workspace,
            suggestion="Create the directory or select a different project.",
        ))
        return

    if not path.is_dir():
        result.add(PreflightIssue(
            code="WORKSPACE_NOT_DIR", severity=PreflightSeverity.ERROR,
            title="Project path is not a directory",
            message=f"Workspace path is not a directory: {workspace}",
            field_name="workspace", path=workspace,
        ))
        return

    result.add(PreflightIssue(
        code="WORKSPACE_OK", severity=PreflightSeverity.PASS,
        title="Project directory exists",
        message=f"Workspace: {workspace}",
        path=workspace,
    ))


def check_working_directory(result: PreflightResult, cwd: str):
    """Check working directory exists and is a directory."""
    if not cwd or not cwd.strip():
        # Working directory is optional — default to script's directory
        return

    path = Path(cwd)

    if not path.exists():
        result.add(PreflightIssue(
            code="CWD_NOT_FOUND", severity=PreflightSeverity.ERROR,
            title="Working directory does not exist",
            message=f"Working directory not found: {cwd}",
            field_name="cwd", path=cwd,
        ))
        return

    if not path.is_dir():
        result.add(PreflightIssue(
            code="CWD_NOT_DIR", severity=PreflightSeverity.ERROR,
            title="Working directory is not a directory",
            message=f"Working directory path is not a directory: {cwd}",
            field_name="cwd", path=cwd,
        ))
        return

    result.add(PreflightIssue(
        code="CWD_OK", severity=PreflightSeverity.PASS,
        title="Working directory is valid",
        message=f"Working directory: {cwd}",
        path=cwd,
    ))


def check_environment_busy(result: PreflightResult, busy: bool):
    """Check environment wizard is not running."""
    if busy:
        result.add(PreflightIssue(
            code="ENVIRONMENT_BUSY", severity=PreflightSeverity.ERROR,
            title="Environment operation in progress",
            message="An environment wizard operation (create venv / install requirements) "
                    "is currently running. Wait for it to complete.",
            suggestion="Wait for the environment operation to finish, then retry.",
        ))
    else:
        result.add(PreflightIssue(
            code="ENVIRONMENT_IDLE", severity=PreflightSeverity.PASS,
            title="Environment wizard is idle",
            message="No environment operations in progress.",
        ))
