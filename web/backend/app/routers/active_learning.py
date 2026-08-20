"""Active-learning hard-example scanning.

Runs the currently loaded model over the opened dataset and assigns each
image a difficulty score: the max confidence among its predicted shapes
(no detections -> 0.0; a shape without a score counts as 1.0, i.e. the
model gave no uncertainty signal). The frontend sorts ascending so the
user reviews the images the model is least sure about first.
"""

import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..adapters import label_path_for, load_label_data
from ..deps import session
from ..model_service import get_model_service

router = APIRouter()


class ScanState:
    """Module-level scan state; all mutations go through `lock`."""

    def __init__(self):
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.processed = 0
        self.total = 0
        self.done = False
        self.error: Optional[str] = None
        self.error_count = 0
        self.scores: Dict[str, Dict[str, Any]] = {}
        self.updated_at: Optional[str] = None


_STATE = ScanState()


def _has_label(image_path: Path) -> bool:
    """Labeled = a same-name label json exists and contains shapes."""
    label_path = label_path_for(image_path)
    if not label_path.exists():
        return False
    try:
        data = load_label_data(label_path)
    except Exception:
        return False
    return bool(data.get("shapes"))


def _image_score(shapes: List[Dict[str, Any]]) -> float:
    """Max shape confidence; 0.0 when there are no detections.

    Shape dicts carry `score` (Shape.to_dict). A shape without a score
    (e.g. yolov8 computes one but never attaches it) counts as 1.0.
    """
    best = 0.0
    for s in shapes:
        value = s.get("score")
        if value is None:
            value = s.get("confidence")
        best = max(best, float(value) if value is not None else 1.0)
    return best


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_scan(image_dir: Path, images: List[str], conf: Optional[float]):
    svc = get_model_service()
    try:
        for name in images:
            if _STATE.stop_event.is_set():
                break
            try:
                result = svc.predict(str(image_dir / name), None, conf)
                shapes = result.get("shapes") or []
                entry = {"score": _image_score(shapes), "count": len(shapes)}
            except Exception:  # noqa
                if svc.loaded_info() is None:
                    # Model was unloaded mid-scan: no point continuing.
                    with _STATE.lock:
                        _STATE.error = "Model unloaded during scan"
                    return
                with _STATE.lock:
                    _STATE.error_count += 1
                entry = {"score": None, "count": 0}
            with _STATE.lock:
                _STATE.scores[name] = entry
                _STATE.processed += 1
                _STATE.updated_at = _utc_now_iso()
    except Exception as e:  # noqa
        with _STATE.lock:
            _STATE.error = str(e)
    finally:
        with _STATE.lock:
            _STATE.running = False
            _STATE.done = True


class ScanRequest(BaseModel):
    conf: Optional[float] = None
    scope: str = "unlabeled"  # "unlabeled" | "all"


@router.post("/active_learning/scan")
def start_scan(req: ScanRequest):
    image_dir = session.get_dir()
    if image_dir is None or not session.get_images():
        raise HTTPException(status_code=400, detail="No directory opened")
    svc = get_model_service()
    if not svc.loaded_info():
        raise HTTPException(status_code=400, detail="No model loaded")
    if req.scope not in ("unlabeled", "all"):
        raise HTTPException(status_code=400, detail=f"Unknown scope: {req.scope}")

    with _STATE.lock:
        if _STATE.running:
            raise HTTPException(status_code=409, detail="Scan already running")
        images = session.get_images()
        if req.scope == "unlabeled":
            images = [n for n in images if not _has_label(image_dir / n)]
        _STATE.stop_event.clear()
        _STATE.running = True
        _STATE.processed = 0
        _STATE.total = len(images)
        _STATE.done = False
        _STATE.error = None
        _STATE.error_count = 0
        _STATE.thread = threading.Thread(
            target=_run_scan,
            args=(image_dir, images, req.conf),
            daemon=True,
        )
        _STATE.thread.start()
    return {"started": True, "total": len(images)}


@router.get("/active_learning/scan/status")
def scan_status():
    with _STATE.lock:
        return {
            "running": _STATE.running,
            "processed": _STATE.processed,
            "total": _STATE.total,
            "done": _STATE.done,
            "error": _STATE.error,
            "error_count": _STATE.error_count,
        }


@router.post("/active_learning/scan/stop")
def stop_scan():
    """Ask the running scan to stop after the current image finishes."""
    with _STATE.lock:
        stopping = _STATE.running
        _STATE.stop_event.set()
    return {"stopping": stopping}


@router.get("/active_learning/scores")
def get_scores():
    with _STATE.lock:
        return {"scores": dict(_STATE.scores), "updated_at": _STATE.updated_at}


@router.post("/active_learning/clear")
def clear_scores():
    with _STATE.lock:
        if _STATE.running:
            raise HTTPException(status_code=409, detail="Scan already running")
        _STATE.scores = {}
        _STATE.updated_at = None
    return {"cleared": True}
