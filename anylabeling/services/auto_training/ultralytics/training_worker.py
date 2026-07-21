"""
Standalone Training Worker — zero-Qt, zero-anylabeling imports.

This script is launched by the GUI as a subprocess using the selected
training runtime Python (e.g. D:\\Anaconda\\envs\\pytorch\\python.exe).
It must NOT depend on:
  - PyQt6 / PyQt5                    (not installed in training envs)
  - anylabeling.views / anylabeling.app  (triggers Qt import chain)
  - anylabeling.config               (optional; falls back gracefully)

Usage:
  python training_worker.py --payload <path/to/train_args.json>

The worker:
  1. Emits a structured "worker_ready" event with full runtime diagnostics.
  2. Hard-verifies CUDA for GPU tasks (fails fast on mismatch).
  3. Runs ultralytics.YOLO.train() directly via Python API.
  4. Emits a "training_completed" or "training_error" event on exit.
"""

import argparse
import json
import os
import sys
import traceback
from io import StringIO

TRAINING_WORKER_EVENT_PREFIX = "__XANYLABELING_TRAIN_EVENT__="


class TrainingWorkerLogStream:
    """Buffered line writer that emits structured training_log events."""

    def __init__(self, output_stream):
        self._buffer = ""
        self._output_stream = output_stream

    def write(self, text):
        if not text:
            return
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.strip()
            if line:
                emit_event("training_log", message=line, output_stream=self._output_stream)

    def flush(self):
        line = self._buffer.strip()
        if line:
            emit_event("training_log", message=line, output_stream=self._output_stream)
        self._buffer = ""


def emit_event(event_type: str, output_stream=None, **data):
    """Write a structured JSON event line to stdout."""
    payload = {"event": event_type}
    payload.update(data)
    stream = output_stream or sys.__stdout__ or sys.stdout
    stream.write(
        f"{TRAINING_WORKER_EVENT_PREFIX}"
        f"{json.dumps(payload, ensure_ascii=False)}\n"
    )
    stream.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True, help="Path to training args JSON file")
    args = parser.parse_args()

    # ── Setup ──
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    with open(args.payload, "r", encoding="utf-8") as f:
        train_args = json.load(f)

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    log_stream = TrainingWorkerLogStream(old_stdout)

    # ── Emit worker_ready ──
    requested_device = train_args.get("device", "cpu")
    try:
        import torch as _torch
        _cuda_available = _torch.cuda.is_available()
        _gpu_count = _torch.cuda.device_count() if _cuda_available else 0
        _gpu_names = [_torch.cuda.get_device_name(i) for i in range(_gpu_count)] if _gpu_count > 0 else []
        _torch_version = _torch.__version__
        _torch_cuda_version = getattr(_torch.version, "cuda", None)
    except ImportError:
        _cuda_available = False
        _gpu_count = 0
        _gpu_names = []
        _torch_version = "N/A"
        _torch_cuda_version = None

    emit_event("worker_ready", output_stream=old_stdout,
               sys_executable=sys.executable,
               python_version=sys.version.split()[0],
               torch_version=_torch_version,
               torch_cuda_version=_torch_cuda_version,
               cuda_available=_cuda_available,
               gpu_count=_gpu_count,
               gpu_name=_gpu_names[0] if _gpu_names else "",
               gpu_names=_gpu_names,
               requested_device=requested_device,
               ultralytics_device=str(train_args.get("device", "cpu")))

    # ── CUDA hard verification (GPU tasks only) ──
    requested = str(requested_device)
    if requested not in ("cpu", "auto", ""):
        _is_gpu_requested = requested.startswith("cuda") or requested.isdigit() or (requested != "cpu")
        if _is_gpu_requested and not _cuda_available:
            emit_event("training_error", output_stream=old_stdout,
                       error=f"Runtime CUDA mismatch: requested device={requested_device} but torch.cuda.is_available()=False. "
                             f"sys.executable={sys.executable}, torch={_torch_version}, torch.cuda={_torch_cuda_version}",
                       traceback=f"Runtime: {sys.executable}\nTorch: {_torch_version}\nCUDA available: {_cuda_available}")
            return
        if _is_gpu_requested and _gpu_count == 0:
            emit_event("training_error", output_stream=old_stdout,
                       error=f"GPU not found: requested device={requested_device} but torch.cuda.device_count()=0. "
                             f"Check nvidia-smi and CUDA driver.",
                       traceback=f"Torch CUDA version: {_torch_cuda_version}\nGPU count: {_gpu_count}")
            return

    # ── Run training ──
    save_dir = ""
    epochs_completed = 0
    best_metric = None
    try:
        sys.stdout = log_stream
        sys.stderr = log_stream

        # Normalize model path to OS-native format (Ultralytics may not
        # recognise forward-slash paths on Windows as local files)
        model_path = train_args.pop("model")
        if os.name == "nt":
            model_path = os.path.normpath(model_path)
        emit_event("training_log", output_stream=old_stdout,
                   message=f"Model path (resolved): {model_path}")

        import matplotlib
        matplotlib.use("Agg")

        from ultralytics import YOLO

        model = YOLO(model_path)
        train_args["verbose"] = False
        train_args["show"] = False

        # ── Register epoch_metrics callback ──
        total_epochs = train_args.get("epochs", 1)

        def on_fit_epoch_end(trainer):
            nonlocal epochs_completed, best_metric
            epochs_completed = trainer.epoch + 1
            metrics = {}
            for k, v in trainer.metrics.items():
                try:
                    metrics[str(k)] = round(float(v), 6)
                except (TypeError, ValueError):
                    metrics[str(k)] = str(v)
            best_metric = trainer.best_fitness if hasattr(trainer, 'best_fitness') else None
            try:
                best_metric = round(float(best_metric), 6) if best_metric is not None else None
            except (TypeError, ValueError):
                pass
            emit_event("epoch_metrics", output_stream=old_stdout,
                       epoch=epochs_completed, total_epochs=total_epochs,
                       metrics=metrics, best_metric=best_metric)

        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

        result = model.train(**train_args)
        save_dir = str(result.save_dir) if hasattr(result, "save_dir") else ""
    except Exception as e:
        log_stream.flush()
        emit_event("training_error", output_stream=old_stdout,
                   error=str(e),
                   traceback=traceback.format_exc())
        raise SystemExit(1) from e
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    emit_event("training_completed", output_stream=old_stdout,
               results="Training completed successfully",
               save_dir=save_dir,
               epochs_completed=epochs_completed,
               total_epochs=total_epochs,
               best_metric=best_metric)


if __name__ == "__main__":
    main()
