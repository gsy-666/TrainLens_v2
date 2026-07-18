"""TrainLens Preflight — common checks shared by Guided and Custom."""

import os
import tempfile
from pathlib import Path
from typing import Optional

from . import (
    PreflightIssue, PreflightResult, PreflightSeverity,
    DISK_SPACE_ERROR_BYTES, DISK_SPACE_WARNING_BYTES,
)


def check_active_job(result: PreflightResult, is_active: bool):
    """Check if another training job is already running."""
    if is_active:
        result.add(PreflightIssue(
            code="ACTIVE_JOB",
            severity=PreflightSeverity.ERROR,
            title="Training task already running",
            message="Another training task is currently active. Stop it before starting a new one.",
            suggestion="Stop the running task or wait for it to complete.",
        ))
    else:
        result.add(PreflightIssue(
            code="ACTIVE_JOB", severity=PreflightSeverity.PASS,
            title="No active training task",
            message="No other training task is currently running.",
        ))


def check_output_directory(result: PreflightResult, output_dir: str):
    """Check output directory exists or can be created, and is writable."""
    if not output_dir:
        result.add(PreflightIssue(
            code="OUTPUT_DIR_EMPTY",
            severity=PreflightSeverity.WARNING,
            title="Output directory not set",
            message="No output directory specified. Results may be scattered.",
            field_name="output_dir",
        ))
        return

    path = Path(output_dir)

    # Check path validity
    try:
        path_str = str(path)
    except (ValueError, TypeError):
        result.add(PreflightIssue(
            code="OUTPUT_DIR_INVALID", severity=PreflightSeverity.ERROR,
            title="Invalid output directory path",
            message=f"The output path contains invalid characters: {output_dir!r}",
            field_name="output_dir", path=output_dir,
        ))
        return

    # Check path length
    if len(path_str) > 260:
        result.add(PreflightIssue(
            code="OUTPUT_DIR_LONG_PATH", severity=PreflightSeverity.WARNING,
            title="Output path is very long",
            message=f"Path length ({len(path_str)}) may cause issues on Windows.",
            field_name="output_dir", path=output_dir,
            suggestion="Use a shorter path.",
        ))

    # Check invalid characters (only truly invalid on all platforms)
    invalid_chars = '<>"|?*'
    path_str_check = str(path)
    if os.name == 'nt' and any(c in path_str_check for c in invalid_chars):
        result.add(PreflightIssue(
            code="OUTPUT_DIR_INVALID_CHARS", severity=PreflightSeverity.ERROR,
            title="Output path contains invalid characters",
            message=f"Path contains one of: {invalid_chars}",
            field_name="output_dir", path=output_dir,
        ))
        return

    # Check drive/path validity on Windows
    if os.name == 'nt' and len(path_str_check) >= 2 and path_str_check[1] == ':':
        drive = path_str_check[0]
        if not (('A' <= drive <= 'Z') or ('a' <= drive <= 'z')):
            result.add(PreflightIssue(
                code="OUTPUT_DIR_INVALID_DRIVE", severity=PreflightSeverity.ERROR,
                title="Invalid drive letter",
                message=f"Path has invalid drive letter: {drive}",
                field_name="output_dir", path=output_dir,
            ))
            return

    # Check if exists and is directory
    if path.exists():
        if not path.is_dir():
            result.add(PreflightIssue(
                code="OUTPUT_DIR_NOT_DIR", severity=PreflightSeverity.ERROR,
                title="Output path is not a directory",
                message=f"The path exists but is not a directory: {path}",
                field_name="output_dir", path=output_dir,
            ))
            return
    else:
        # Try to create parent
        parent = path.parent
        if not parent.exists():
            try:
                parent.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                result.add(PreflightIssue(
                    code="OUTPUT_DIR_PARENT_FAIL", severity=PreflightSeverity.ERROR,
                    title="Cannot create output directory",
                    message=f"Cannot create parent directory: {e}",
                    field_name="output_dir", path=output_dir,
                ))
                return

    # Check writability
    try:
        # Create dir if it doesn't exist yet (clean up after)
        existed = path.exists()
        if not existed:
            path.mkdir(parents=True, exist_ok=True)
        probe = path / ".preflight_write_test"
        probe.write_text("test")
        probe.unlink()
        if not existed:
            # Remove dir we just created (only if empty)
            try:
                path.rmdir()
            except OSError:
                pass  # Not empty, leave it
    except (OSError, PermissionError) as e:
        result.add(PreflightIssue(
            code="OUTPUT_DIR_NOT_WRITABLE", severity=PreflightSeverity.ERROR,
            title="Output directory is not writable",
            message=f"Cannot write to output directory: {e}",
            field_name="output_dir", path=output_dir,
            suggestion="Check permissions or choose a different directory.",
        ))
        return

    if path.exists() and path.is_dir():
        result.add(PreflightIssue(
            code="OUTPUT_DIR_OK", severity=PreflightSeverity.PASS,
            title="Output directory is writable",
            message=f"Output directory exists and is writable: {path}",
            path=output_dir,
        ))
    else:
        result.add(PreflightIssue(
            code="OUTPUT_DIR_OK", severity=PreflightSeverity.PASS,
            title="Output directory can be created",
            message=f"Output directory will be created: {path}",
            path=output_dir,
        ))


def check_disk_space(result: PreflightResult, output_dir: str):
    """Check available disk space on the output directory's volume.

    Falls back to nearest existing parent directory.
    On Windows, can fall back to drive root (e.g. D:\\\\).
    """
    if not output_dir:
        return

    path = Path(output_dir)
    # Find the nearest existing ancestor
    checked = path
    while not checked.exists():
        parent = checked.parent
        if parent == checked:
            # Reached root — use drive root on Windows
            if os.name == 'nt' and len(str(path)) >= 2:
                drive = str(path)[:2] + '\\'
                if os.path.exists(drive):
                    checked = Path(drive)
                    break
            return  # Cannot determine
        checked = parent

    try:
        if hasattr(os, 'statvfs'):
            usage = os.statvfs(str(checked))
            free_bytes = usage.f_frsize * usage.f_bavail
        else:
            # Windows: use shutil.disk_usage
            import shutil as _shutil
            free_bytes = _shutil.disk_usage(str(checked)).free
    except (OSError, AttributeError):
        result.add(PreflightIssue(
            code="DISK_SPACE_UNKNOWN", severity=PreflightSeverity.WARNING,
            title="Cannot determine disk space",
            message="Unable to query free disk space on the output volume.",
            path=str(checked),
        ))
        return

    free_gb = free_bytes / (1024 ** 3)

    if free_bytes < DISK_SPACE_ERROR_BYTES:
        result.add(PreflightIssue(
            code="DISK_SPACE_LOW", severity=PreflightSeverity.ERROR,
            title="Insufficient disk space",
            message=f"Only {free_gb:.1f} GB free. At least 1 GB required.",
            path=str(checked),
            suggestion="Free up disk space or choose a different output location.",
        ))
    elif free_bytes < DISK_SPACE_WARNING_BYTES:
        result.add(PreflightIssue(
            code="DISK_SPACE_MODERATE", severity=PreflightSeverity.WARNING,
            title="Disk space is low",
            message=f"Only {free_gb:.1f} GB free. Training may run out of space.",
            path=str(checked),
            suggestion=f"Ensure at least 5 GB free. Currently {free_gb:.1f} GB.",
        ))
    else:
        result.add(PreflightIssue(
            code="DISK_SPACE_OK", severity=PreflightSeverity.PASS,
            title="Sufficient disk space",
            message=f"{free_gb:.1f} GB free on output volume.",
            path=str(checked),
        ))


def check_config_not_empty(result: PreflightResult, field_name: str, value: Optional[str]):
    """Check a config value is not None or empty string."""
    if value is None or (isinstance(value, str) and not value.strip()):
        result.add(PreflightIssue(
            code=f"{field_name.upper()}_EMPTY", severity=PreflightSeverity.ERROR,
            title=f"{field_name} is empty",
            message=f"The {field_name} field must not be empty.",
            field_name=field_name,
        ))
    else:
        result.add(PreflightIssue(
            code=f"{field_name.upper()}_OK", severity=PreflightSeverity.PASS,
            title=f"{field_name} is set",
            message=f"{field_name}: {value}",
            field_name=field_name,
        ))


def check_positive_int(result: PreflightResult, field_name: str, value: int):
    """Check an integer config value is > 0."""
    if value is None:
        result.add(PreflightIssue(
            code=f"{field_name.upper()}_EMPTY", severity=PreflightSeverity.ERROR,
            title=f"{field_name} is not set",
            message=f"The {field_name} field must be set.",
            field_name=field_name,
        ))
    elif value <= 0:
        result.add(PreflightIssue(
            code=f"{field_name.upper()}_INVALID", severity=PreflightSeverity.ERROR,
            title=f"{field_name} must be positive",
            message=f"{field_name} is {value}. Must be > 0.",
            field_name=field_name,
        ))
    else:
        result.add(PreflightIssue(
            code=f"{field_name.upper()}_OK", severity=PreflightSeverity.PASS,
            title=f"{field_name} is valid",
            message=f"{field_name}: {value}",
            field_name=field_name,
        ))
