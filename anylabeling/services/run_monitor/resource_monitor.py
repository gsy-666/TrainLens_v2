"""Resource monitoring for Run Monitor

Monitors CPU, memory, and GPU usage of training process.
"""

import time
from typing import Dict, Optional

import psutil
from PyQt6.QtCore import QObject, QThread, pyqtSignal


class ResourceMonitor(QObject):
    """Monitors system and process resources"""

    # Signals
    resource_sample = pyqtSignal(dict)  # Resource snapshot

    def __init__(self):
        super().__init__()
        self._monitoring = False
        self._monitor_thread: Optional[QThread] = None
        self._worker: Optional[ResourceMonitorWorker] = None
        self._pid: Optional[int] = None
        self._interval_ms: int = 1000

    def start_monitoring(self, pid: int, interval_ms: int = 1000):
        """
        Start monitoring process resources.

        Args:
            pid: Process ID to monitor
            interval_ms: Sampling interval in milliseconds
        """
        if self._monitoring:
            return

        self._pid = pid
        self._interval_ms = interval_ms
        self._monitoring = True

        # Create worker and thread
        self._worker = ResourceMonitorWorker(pid, interval_ms)
        self._monitor_thread = QThread()
        self._worker.moveToThread(self._monitor_thread)

        # Connect signals
        self._worker.sample_ready.connect(self.resource_sample)
        self._monitor_thread.started.connect(self._worker.run)

        # Start monitoring
        self._monitor_thread.start()

    def stop_monitoring(self):
        """Stop monitoring"""
        if not self._monitoring:
            return

        self._monitoring = False

        if self._worker:
            self._worker.stop()

        if self._monitor_thread:
            self._monitor_thread.quit()
            self._monitor_thread.wait()
            self._monitor_thread = None

        self._worker = None


class ResourceMonitorWorker(QObject):
    """Worker that performs resource monitoring in thread"""

    sample_ready = pyqtSignal(dict)

    def __init__(self, pid: int, interval_ms: int):
        super().__init__()
        self.pid = pid
        self.interval_ms = interval_ms
        self._running = False
        self._gpu_available = self._check_gpu_availability()

    def stop(self):
        """Stop monitoring loop"""
        self._running = False

    def run(self):
        """Main monitoring loop"""
        self._running = True

        try:
            process = psutil.Process(self.pid)
        except psutil.NoSuchProcess:
            return

        while self._running:
            try:
                sample = self._collect_sample(process)
                self.sample_ready.emit(sample)
                time.sleep(self.interval_ms / 1000.0)
            except psutil.NoSuchProcess:
                # Process terminated
                break
            except Exception as e:
                # Log error but continue
                print(f"Resource monitoring error: {e}")
                time.sleep(self.interval_ms / 1000.0)

    def _collect_sample(self, process: psutil.Process) -> Dict:
        """Collect resource usage snapshot"""
        sample = {
            "timestamp": time.time(),
            "process": {},
            "system": {},
            "gpu": {},
        }

        # Process metrics
        try:
            sample["process"]["cpu_percent"] = process.cpu_percent()
            mem_info = process.memory_info()
            sample["process"]["memory_rss"] = mem_info.rss
            sample["process"]["memory_percent"] = process.memory_percent()
        except Exception:
            pass

        # System metrics
        try:
            sample["system"]["cpu_percent"] = psutil.cpu_percent(percpu=False)
            sample["system"]["cpu_percent_per_core"] = psutil.cpu_percent(
                percpu=True
            )
            vm = psutil.virtual_memory()
            sample["system"]["memory_total"] = vm.total
            sample["system"]["memory_available"] = vm.available
            sample["system"]["memory_used"] = vm.used
            sample["system"]["memory_percent"] = vm.percent
        except Exception:
            pass

        # GPU metrics (if available)
        if self._gpu_available:
            try:
                gpu_stats = self._get_gpu_stats()
                if gpu_stats:
                    sample["gpu"] = gpu_stats
            except Exception:
                pass

        return sample

    def _check_gpu_availability(self) -> bool:
        """Check if NVIDIA GPU monitoring is available"""
        try:
            import subprocess

            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _get_gpu_stats(self) -> Optional[Dict]:
        """Get GPU statistics using nvidia-smi"""
        try:
            import subprocess

            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2,
            )

            if result.returncode != 0:
                return None

            gpus = []
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    gpus.append(
                        {
                            "index": int(parts[0]),
                            "utilization": float(parts[1]),
                            "memory_used": float(parts[2]),
                            "memory_total": float(parts[3]),
                        }
                    )

            return {"gpus": gpus} if gpus else None

        except Exception:
            return None
