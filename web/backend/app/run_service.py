"""Web run-monitor service: run custom training scripts with live output
and resource metrics.

The desktop run_monitor's ProcessManager relies on QThread + Qt signals,
which never fire without a Qt event loop. This service reimplements the
same idea with plain threads, while reusing the Qt-free building blocks
(WorkspaceScanner, ScriptDetector, EnvironmentDetector, EventProtocol,
RunStorage).
"""

import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil

from anylabeling.services.run_monitor.event_protocol import EventProtocol
from anylabeling.services.run_monitor.workspace_scanner import WorkspaceScanner


class WebRunService:
    def __init__(self):
        self.lock = threading.RLock()
        self.proc: Optional[subprocess.Popen] = None
        self.run: Optional[Dict[str, Any]] = None
        self.logs: deque = deque(maxlen=5000)
        self.seq = 0
        self.resources: deque = deque(maxlen=1800)  # ~30 min @1s
        self._sampler_stop = threading.Event()

    # ---- workspace scanning ---------------------------------------------------
    def scan(self, workspace_path: str) -> Dict[str, Any]:
        scanner = WorkspaceScanner()
        ws = scanner.scan(Path(workspace_path))
        return ws.to_dict()

    # ---- log plumbing ----------------------------------------------------------
    def _append_log(self, line: str, stream: str):
        with self.lock:
            self.seq += 1
            self.logs.append({"seq": self.seq, "stream": stream, "line": line})

    def logs_since(self, seq: int) -> Dict[str, Any]:
        with self.lock:
            return {
                "latest": self.seq,
                "lines": [l for l in self.logs if l["seq"] > seq],
            }

    # ---- process lifecycle ------------------------------------------------------
    def start(
        self,
        workspace: str,
        script_path: str,
        python_path: str,
        arguments: str = "",
    ) -> Dict[str, Any]:
        with self.lock:
            if self.is_running():
                raise RuntimeError("已有任务在运行")

            cmd = [python_path or sys.executable, "-u", script_path]
            if arguments.strip():
                cmd.extend(arguments.split())

            env = dict(os.environ)
            env["PYTHONUNBUFFERED"] = "1"
            env["TRAINLENS_RUN_ID"] = uuid.uuid4().hex[:12]

            try:
                self.proc = subprocess.Popen(
                    cmd,
                    cwd=workspace or None,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=env,
                    bufsize=1,
                )
            except Exception as e:  # noqa
                self.proc = None
                raise RuntimeError(f"启动进程失败: {e}")

            self.run = {
                "run_id": env["TRAINLENS_RUN_ID"],
                "workspace": workspace,
                "script_path": script_path,
                "python_path": python_path or sys.executable,
                "arguments": arguments,
                "command": cmd,
                "pid": self.proc.pid,
                "status": "running",
                "started_at": time.time(),
                "ended_at": None,
                "exit_code": None,
            }
            self._append_log(f"── started pid={self.proc.pid}: {' '.join(cmd)} ──", "system")

            threading.Thread(
                target=self._read_stream, args=(self.proc.stdout, "stdout"), daemon=True
            ).start()
            threading.Thread(
                target=self._read_stream, args=(self.proc.stderr, "stderr"), daemon=True
            ).start()
            threading.Thread(target=self._watch, daemon=True).start()

            self._sampler_stop.clear()
            threading.Thread(target=self._sample_resources, daemon=True).start()

            return dict(self.run)

    def _read_stream(self, stream, name: str):
        try:
            for line in iter(stream.readline, ""):
                line = line.rstrip("\n")
                parsed = EventProtocol.parse_line(line)
                if parsed is not None:
                    self._append_log(f"[event] {line}", "event")
                else:
                    self._append_log(line, name)
        except Exception:
            pass
        finally:
            try:
                stream.close()
            except Exception:
                pass

    def _watch(self):
        proc = self.proc
        if proc is None:
            return
        exit_code = proc.wait()
        with self.lock:
            if self.run and self.run["status"] == "running":
                # don't overwrite an explicit "stopped" set by stop()
                self.run["status"] = "completed" if exit_code == 0 else "failed"
                self.run["exit_code"] = exit_code
                self.run["ended_at"] = time.time()
        self._append_log(f"── process exited, code={exit_code} ──", "system")
        self._sampler_stop.set()

    def stop(self) -> Dict[str, Any]:
        with self.lock:
            proc = self.proc
            if not proc or proc.poll() is not None:
                return {"stopped": False, "reason": "not running"}
            pid = proc.pid
        try:
            if sys.platform.startswith("win"):
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    capture_output=True,
                    timeout=15,
                )
            else:
                import signal

                try:
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
                except Exception:
                    proc.terminate()
        except Exception as e:  # noqa
            return {"stopped": False, "reason": str(e)}
        with self.lock:
            if self.run and self.run["status"] == "running":
                self.run["status"] = "stopped"
                self.run["ended_at"] = time.time()
        return {"stopped": True}

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def status(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "running": self.is_running(),
                "run": dict(self.run) if self.run else None,
            }

    # ---- resource sampling -------------------------------------------------------
    def _gpu_sample(self) -> Optional[Dict[str, Any]]:
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if out.returncode != 0:
                return None
            util, mem_used, mem_total = out.stdout.strip().split(", ")
            return {
                "gpu_util": float(util),
                "gpu_mem_used_mb": float(mem_used),
                "gpu_mem_total_mb": float(mem_total),
            }
        except Exception:
            return None

    def _sample_resources(self):
        pid = self.proc.pid if self.proc else None
        proc = None
        if pid:
            try:
                proc = psutil.Process(pid)
            except psutil.NoSuchProcess:
                proc = None
        while not self._sampler_stop.is_set():
            sample: Dict[str, Any] = {
                "ts": time.time(),
                "system_cpu": psutil.cpu_percent(interval=None),
                "system_mem_percent": psutil.virtual_memory().percent,
            }
            if proc is not None:
                try:
                    with proc.oneshot():
                        sample["proc_cpu"] = proc.cpu_percent(interval=None)
                        sample["proc_rss_mb"] = proc.memory_info().rss / 1024 / 1024
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    proc = None
            gpu = self._gpu_sample()
            if gpu:
                sample.update(gpu)
            with self.lock:
                self.resources.append(sample)
            self._sampler_stop.wait(1.0)

    def resource_history(self, limit: int = 300) -> Dict[str, Any]:
        with self.lock:
            items = list(self.resources)[-limit:]
            return {"samples": items}


_service: Optional[WebRunService] = None


def get_run_service() -> WebRunService:
    global _service
    if _service is None:
        _service = WebRunService()
    return _service
