"""Remote environment diagnostics — run via SSH + remote Python JSON scripts."""

import json
import logging
from typing import List, Optional, Dict, Any

from .models import RemoteProfile, DiagnosticItem, DiagnosticStatus
from .ssh_service import SSHConnectionService

_log = logging.getLogger(__name__)

# Python script that outputs JSON diagnostics on the remote host
DIAGNOSTIC_SCRIPT = r"""
import json, os, sys, platform, shutil

result = {
    "platform": sys.platform,
    "os": platform.platform(),
    "python_version": sys.version.split()[0],
    "python_executable": sys.executable,
}

# Check workspace
workspace = r"{workspace}"
if not workspace:
    result["workspace"] = {"exists": False, "writable": False, "error": "No workspace path"}
else:
    if os.path.isdir(workspace):
        result["workspace"] = {
            "exists": True,
            "writable": os.access(workspace, os.W_OK),
        }
    else:
        try:
            os.makedirs(workspace, exist_ok=True)
            writable = os.access(workspace, os.W_OK)
            result["workspace"] = {"exists": True, "writable": writable, "created": True}
        except Exception as e:
            result["workspace"] = {"exists": False, "writable": False, "error": str(e)}

# Disk space
try:
    usage = shutil.disk_usage(workspace if os.path.isdir(workspace) else "/")
    result["disk"] = {
        "total_gb": round(usage.total / (1024**3), 1),
        "used_gb": round(usage.used / (1024**3), 1),
        "free_gb": round(usage.free / (1024**3), 1),
    }
except Exception as e:
    result["disk"] = {"error": str(e)}

# PyTorch
try:
    import torch
    result["pytorch"] = {
        "installed": True,
        "version": torch.__version__,
        "cuda_version": getattr(torch.version, "cuda", None),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append({
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "total_memory_gb": round(props.total_memory / (1024**3), 1),
                "compute_capability": f"{props.major}.{props.minor}",
            })
        # Quick CUDA tensor test
        try:
            t = torch.ones(100, 100).cuda()
            _ = t @ t
            del t
            result["pytorch"]["cuda_tensor_test"] = True
        except Exception as e:
            result["pytorch"]["cuda_tensor_test"] = False
            result["pytorch"]["cuda_tensor_error"] = str(e)
    result["pytorch"]["gpus"] = gpus
except ImportError:
    result["pytorch"] = {"installed": False}

# Ultralytics
try:
    import ultralytics
    result["ultralytics"] = {
        "installed": True,
        "version": ultralytics.__version__,
    }
except ImportError:
    result["ultralytics"] = {"installed": False}

print(json.dumps(result))
"""


def build_diagnostic_script(workspace: str) -> str:
    """Build the remote diagnostic Python script.

    The workspace is embedded as a Python raw string (r"...") to handle
    backslashes and special characters correctly.
    The script is validated with compile() before sending via stdin.
    """
    # Use repr() for safe Python string literal embedding
    ws_literal = repr(str(workspace))
    script = DIAGNOSTIC_SCRIPT.replace("{workspace}", ws_literal)

    # Validate locally before sending
    try:
        compile(script, "<trainlens_remote_diagnostics>", "exec")
    except SyntaxError as e:
        _log.error("Diagnostic script failed local compile: %s", e)
        raise ValueError(f"Diagnostic script has syntax errors: {e}") from e

    return script


def parse_diagnostic_output(stdout: str) -> Dict[str, Any]:
    """Parse the JSON diagnostic output from remote execution."""
    if not stdout:
        return {}
    try:
        return json.loads(stdout.strip().split("\n")[-1] if "\n" in stdout else stdout)
    except json.JSONDecodeError:
        return {}


def run_remote_diagnostics(
    profile: RemoteProfile,
    ssh: SSHConnectionService,
    password: str = "",
    on_progress: callable = None,
) -> List[DiagnosticItem]:
    """Run the full diagnostic suite on the remote host.

    Args:
        profile: Connection profile
        ssh: Active SSH connection (already connected)
        password: Session password for auth
        on_progress: callback(stage_label: str)

    Returns:
        List of DiagnosticItem results
    """
    results: List[DiagnosticItem] = []

    def add(stage, label, status, message="", details=None):
        item = DiagnosticItem(stage=stage, label=label, status=status, message=message, details=details or {})
        results.append(item)
        if on_progress:
            on_progress(label)
        return item

    def progress(label):
        if on_progress:
            on_progress(label)

    # 1. OS info
    progress("Checking OS info...")
    code, out, err = ssh.execute("uname -a", timeout=10)
    if code == 0:
        add("os", "Operating System", DiagnosticStatus.PASS, out.strip())
    else:
        # Try Windows
        code2, out2, err2 = ssh.execute("ver", timeout=10)
        if code2 == 0:
            add("os", "Operating System", DiagnosticStatus.PASS, out2.strip())
        else:
            add("os", "Operating System", DiagnosticStatus.WARNING, f"uname/ver failed (exit {code})")

    # 2. Remote Python
    progress("Checking Python...")
    py = profile.remote_python or "python3"
    code, out, err = ssh.execute_script(
        "import sys; print(sys.executable); print(sys.version)",
        python_path=py, timeout=15,
    )
    if code == 0:
        lines = out.strip().split("\n")
        py_path = lines[0].strip() if lines else ""
        py_ver = lines[1].strip() if len(lines) > 1 else ""
        add("python", "Remote Python", DiagnosticStatus.PASS,
            f"{py_path}\nPython {py_ver}",
            {"executable": py_path, "version": py_ver})
    else:
        add("python", "Remote Python", DiagnosticStatus.ERROR,
            f"Python not found at '{py}': {err or out}",
            {"exit_code": code, "stderr": err})

    # 3. Full diagnostics via remote Python (stdin piping — no shell quoting)
    progress("Running diagnostics...")
    py = profile.remote_python or "python3"
    script = build_diagnostic_script(profile.remote_workspace or "/tmp")
    code, out, err = ssh.execute_script(script, python_path=py, timeout=30)
    if code != 0:
        add("diag", "Remote Diagnostics", DiagnosticStatus.ERROR,
            f"Script failed (exit {code}): {err or out}")
    else:
        diag = parse_diagnostic_output(out)
        if not diag:
            add("diag", "Remote Diagnostics", DiagnosticStatus.ERROR, "No JSON output from diagnostic script")
        else:
            add("diag", "Remote Diagnostics", DiagnosticStatus.PASS, "Diagnostic script executed successfully")

            # Parse individual results
            _parse_pytorch(results, diag, progress)
            _parse_gpu(results, diag, progress)
            _parse_ultralytics(results, diag, progress)
            _parse_workspace(results, diag, progress)
            _parse_disk(results, diag, progress)

    return results


def _parse_pytorch(results: List[DiagnosticItem], diag: Dict, progress):
    progress("Checking PyTorch...")
    pt = diag.get("pytorch", {})
    if not pt.get("installed"):
        results.append(DiagnosticItem(
            stage="pytorch", label="PyTorch", status=DiagnosticStatus.ERROR,
            message="PyTorch is not installed on the remote host."
        ))
        return
    cuda_ok = pt.get("cuda_available", False)
    status = DiagnosticStatus.PASS if cuda_ok else DiagnosticStatus.WARNING
    msg = f"PyTorch {pt.get('version')}"
    if pt.get("cuda_version"):
        msg += f" · CUDA {pt.get('cuda_version')}"
    if not cuda_ok:
        msg += "\nCUDA is NOT available (CPU-only PyTorch)"
    results.append(DiagnosticItem(
        stage="pytorch", label="PyTorch", status=status, message=msg,
        details=pt,
    ))


def _parse_gpu(results: List[DiagnosticItem], diag: Dict, progress):
    progress("Checking GPU...")
    pt = diag.get("pytorch", {})
    gpus = pt.get("gpus", [])
    if not gpus:
        results.append(DiagnosticItem(
            stage="gpu", label="GPU", status=DiagnosticStatus.WARNING,
            message="No CUDA GPUs detected" if pt.get("cuda_version") else "CUDA not available"
        ))
        return
    for g in gpus:
        results.append(DiagnosticItem(
            stage="gpu", label=f"GPU {g['index']}: {g['name']}",
            status=DiagnosticStatus.PASS,
            message=f"{g['total_memory_gb']} GB · CC {g.get('compute_capability', '?')}",
            details=g,
        ))


def _parse_ultralytics(results: List[DiagnosticItem], diag: Dict, progress):
    progress("Checking Ultralytics...")
    ul = diag.get("ultralytics", {})
    if ul.get("installed"):
        results.append(DiagnosticItem(
            stage="ultralytics", label="Ultralytics", status=DiagnosticStatus.PASS,
            message=f"Version {ul.get('version')}",
            details=ul,
        ))
    else:
        results.append(DiagnosticItem(
            stage="ultralytics", label="Ultralytics", status=DiagnosticStatus.WARNING,
            message="Ultralytics is not installed. Run: pip install ultralytics",
        ))


def _parse_workspace(results: List[DiagnosticItem], diag: Dict, progress):
    progress("Checking workspace...")
    ws = diag.get("workspace", {})
    if ws.get("writable"):
        msg = "Writable"
        if ws.get("created"):
            msg += " (created)"
        results.append(DiagnosticItem(
            stage="workspace", label="Remote Workspace", status=DiagnosticStatus.PASS,
            message=msg, details=ws,
        ))
    else:
        results.append(DiagnosticItem(
            stage="workspace", label="Remote Workspace", status=DiagnosticStatus.ERROR,
            message=ws.get("error", "Not writable"), details=ws,
        ))


def _parse_disk(results: List[DiagnosticItem], diag: Dict, progress):
    progress("Checking disk space...")
    d = diag.get("disk", {})
    if "error" in d:
        results.append(DiagnosticItem(
            stage="disk", label="Disk Space", status=DiagnosticStatus.WARNING,
            message=d.get("error", "Unknown"), details=d,
        ))
    else:
        results.append(DiagnosticItem(
            stage="disk", label="Disk Space", status=DiagnosticStatus.PASS,
            message=f"Free: {d.get('free_gb', '?')} GB / Total: {d.get('total_gb', '?')} GB",
            details=d,
        ))
