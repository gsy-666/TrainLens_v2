"""Video annotation endpoints: open video, serve frames, per-frame labels.

Frames are decoded on demand with OpenCV and served as JPEG. Label files
use the exact same desktop JSON format, stored in
``<video_dir>/<stem>_xlabel/<stem>_frame_000001.json``.
"""

import asyncio
import threading
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from anylabeling.services.auto_labeling.types import AutoLabelingResult
from anylabeling.views.labeling.label_file import LabelFile
from anylabeling.views.labeling.schema import XLABEL_BASIC_FIELDS
from anylabeling.views.labeling.utils.opencv import cv_img_to_qt_img

from .. import backup
from ..adapters import extra_fields, load_label_data
from ..model_service import get_model_service
from ..schemas import SaveLabelsRequest

router = APIRouter()

FRAME_CACHE_SIZE = 8
JPEG_QUALITY = 90


class VideoSession:
    def __init__(self):
        self.lock = threading.RLock()
        self.video_path: Optional[Path] = None
        self.label_dir: Optional[Path] = None
        self.frame_count = 0
        self.fps = 0.0
        self.width = 0
        self.height = 0
        self._capture: Optional[cv2.VideoCapture] = None
        self._capture_index = -1  # frame index the capture will return next
        self._cache: OrderedDict[int, bytes] = OrderedDict()

    def open(self, path: Path):
        with self.lock:
            self.close()
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {path}")
            self._capture = cap
            self.video_path = path
            self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._capture_index = 0
            self.label_dir = path.parent / f"{path.stem}_xlabel"

    def close(self):
        with self.lock:
            if self._capture is not None:
                self._capture.release()
            self._capture = None
            self.video_path = None
            self._cache.clear()
            self._capture_index = -1

    def is_open(self) -> bool:
        return self._capture is not None and self.video_path is not None

    def _read_at(self, index: int) -> Optional[np.ndarray]:
        cap = self._capture
        if cap is None:
            return None
        if index != self._capture_index:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        self._capture_index = index + 1
        return frame if ok else None

    def frame_jpeg(self, index: int) -> Optional[bytes]:
        with self.lock:
            if index in self._cache:
                self._cache.move_to_end(index)
                return self._cache[index]
            frame = self._read_at(index)
            if frame is None:
                return None
            ok, buf = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )
            if not ok:
                return None
            data = buf.tobytes()
            self._cache[index] = data
            if len(self._cache) > FRAME_CACHE_SIZE:
                self._cache.popitem(last=False)
            return data

    def frame_array(self, index: int) -> Optional[np.ndarray]:
        """Raw BGR frame (no caching; sequential access is efficient)."""
        with self.lock:
            return self._read_at(index)

    def label_path(self, index: int) -> Path:
        assert self.video_path is not None and self.label_dir is not None
        return self.label_dir / f"{self.video_path.stem}_frame_{index:06d}.json"


video_session = VideoSession()


class OpenVideoRequest(BaseModel):
    path: str


@router.post("/video/open")
def open_video(req: OpenVideoRequest):
    path = Path(req.path).expanduser()
    if not path.is_file():
        raise HTTPException(status_code=400, detail=f"Not a file: {req.path}")
    try:
        video_session.open(path.resolve())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "video": str(video_session.video_path),
        "frame_count": video_session.frame_count,
        "fps": video_session.fps,
        "width": video_session.width,
        "height": video_session.height,
        "labeled_frames": _labeled_frames(),
    }


@router.post("/video/close")
def close_video():
    video_session.close()
    return {"closed": True}


@router.get("/video/info")
def video_info():
    if not video_session.is_open():
        raise HTTPException(status_code=400, detail="No video opened")
    return {
        "video": str(video_session.video_path),
        "frame_count": video_session.frame_count,
        "fps": video_session.fps,
        "width": video_session.width,
        "height": video_session.height,
        "labeled_frames": _labeled_frames(),
    }


def _labeled_frames():
    if not video_session.is_open():
        return []
    label_dir = video_session.label_dir
    if not label_dir or not label_dir.is_dir():
        return []
    prefix = f"{video_session.video_path.stem}_frame_"
    frames = []
    for p in label_dir.glob(f"{prefix}*.json"):
        try:
            frames.append(int(p.stem[len(prefix):]))
        except ValueError:
            continue
    return sorted(frames)


@router.get("/video/frame")
def get_frame(index: int):
    if not video_session.is_open():
        raise HTTPException(status_code=400, detail="No video opened")
    if index < 0 or index >= video_session.frame_count:
        raise HTTPException(status_code=404, detail=f"Frame out of range: {index}")
    data = video_session.frame_jpeg(index)
    if data is None:
        raise HTTPException(status_code=500, detail=f"Failed to decode frame {index}")
    return Response(content=data, media_type="image/jpeg")


@router.get("/video/labels")
def get_frame_labels(index: int):
    if not video_session.is_open():
        raise HTTPException(status_code=400, detail="No video opened")
    label_path = video_session.label_path(index)
    if not label_path.exists():
        return {"exists": False, "data": None}
    try:
        data = load_label_data(label_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse label file: {e}")
    return {"exists": True, "data": data}


@router.put("/video/labels")
def put_frame_labels(req: SaveLabelsRequest):
    """Save labels for a frame. `req.image` carries the frame index."""
    if not video_session.is_open():
        raise HTTPException(status_code=400, detail="No video opened")
    try:
        index = int(req.image)
    except ValueError:
        raise HTTPException(status_code=400, detail="image must be a frame index in video mode")
    if index < 0 or index >= video_session.frame_count:
        raise HTTPException(status_code=404, detail=f"Frame out of range: {index}")

    video_session.label_dir.mkdir(parents=True, exist_ok=True)
    label_path = video_session.label_path(index)
    frame_name = f"{video_session.video_path.stem}_frame_{index:06d}.jpg"

    shapes = []
    for s in req.shapes:
        d = s.model_dump(exclude_none=True)
        d.setdefault("group_id", None)
        shapes.append(d)

    preserved: dict = {}
    if label_path.exists():
        try:
            preserved = extra_fields(load_label_data(label_path))
        except Exception:
            preserved = {}
    safe_extra = {
        k: v for k, v in (req.other_data or {}).items() if k not in XLABEL_BASIC_FIELDS
    }

    image_height, image_width = req.image_height, req.image_width
    if image_height <= 0 or image_width <= 0:
        image_height, image_width = video_session.height, video_session.width

    try:
        label_file = LabelFile()
        label_file.save(
            filename=str(label_path),
            shapes=shapes,
            image_path=frame_name,
            image_height=image_height,
            image_width=image_width,
            image_data=None,
            other_data={**preserved, **safe_extra},
            flags=req.flags or {},
        )
    except Exception as e:  # noqa
        raise HTTPException(status_code=500, detail=f"Failed to save label file: {e}")

    return {"saved": True, "label_path": str(label_path)}


@router.delete("/video/labels")
def delete_frame_labels(index: int):
    if not video_session.is_open():
        raise HTTPException(status_code=400, detail="No video opened")
    label_path = video_session.label_path(index)
    if label_path.exists():
        label_path.unlink()
    return {"deleted": True}


# ---- MOT tracking ------------------------------------------------------------

track_state = {
    "running": False,
    "current": 0,
    "total": 0,
    "current_frame": None,
    "errors": [],
    "result": None,
}


class TrackRequest(BaseModel):
    start_frame: int = 0
    end_frame: Optional[int] = None  # inclusive, default: last frame
    conf: Optional[float] = None
    iou: Optional[float] = None
    preserve_existing: bool = False  # merge with existing frame labels


def _save_frame_labels(index: int, shapes: list, flags: dict, other: dict):
    video_session.label_dir.mkdir(parents=True, exist_ok=True)
    label_path = video_session.label_path(index)
    frame_name = f"{video_session.video_path.stem}_frame_{index:06d}.jpg"
    existed = label_path.exists()
    original_content = label_path.read_bytes() if existed else None

    preserved = {}
    if existed:
        try:
            preserved = extra_fields(load_label_data(label_path))
        except Exception:
            preserved = {}
    safe_other = {k: v for k, v in other.items() if k not in XLABEL_BASIC_FIELDS}

    label_file = LabelFile()
    label_file.save(
        filename=str(label_path),
        shapes=shapes,
        image_path=frame_name,
        image_height=video_session.height,
        image_width=video_session.width,
        image_data=None,
        other_data={**preserved, **safe_other},
        flags=flags or {},
    )
    backup.add_item(f"frame_{index}", label_path, existed, original_content)


def _run_track(req: TrackRequest):
    svc = get_model_service()
    model = svc.manager.loaded_model_config["model"]
    if req.conf is not None:
        svc.manager.set_auto_labeling_conf(req.conf)
    if req.iou is not None:
        svc.manager.set_auto_labeling_iou(req.iou)
    svc.manager.set_auto_labeling_reset_tracker()

    end = video_session.frame_count - 1
    if req.end_frame is not None:
        end = min(end, req.end_frame)
    frames = list(range(max(0, req.start_frame), end + 1))
    track_state.update(
        running=True,
        current=0,
        total=len(frames),
        current_frame=None,
        errors=[],
        result=None,
    )
    backup.start_backup("video_track", str(video_session.label_dir))

    try:
        for i, idx in enumerate(frames):
            track_state["current"] = i
            track_state["current_frame"] = idx
            try:
                frame = video_session.frame_array(idx)
                if frame is None:
                    raise RuntimeError("failed to decode frame")
                qimg = cv_img_to_qt_img(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result = model.predict_shapes(qimg, None)
                if isinstance(result, AutoLabelingResult):
                    new_shapes = [s.to_dict() for s in result.shapes]
                else:
                    new_shapes = [s.to_dict() for s in (result or [])]

                if req.preserve_existing:
                    label_path = video_session.label_path(idx)
                    if label_path.exists():
                        try:
                            existing = load_label_data(label_path).get("shapes", [])
                            new_shapes = existing + new_shapes
                        except Exception:
                            pass

                _save_frame_labels(idx, new_shapes, {}, {})
            except Exception as e:  # noqa
                track_state["errors"].append({"frame": idx, "error": str(e)})
    finally:
        track_state["current"] = track_state["total"]
        track_state["running"] = False
        track_state["result"] = {
            "frames": track_state["total"],
            "errors": len(track_state["errors"]),
        }


@router.post("/video/track")
async def start_track(req: TrackRequest):
    if not video_session.is_open():
        raise HTTPException(status_code=400, detail="No video opened")
    svc = get_model_service()
    if not svc.loaded_info():
        raise HTTPException(status_code=400, detail="No model loaded")
    model = svc.manager.loaded_model_config["model"]
    if getattr(model, "tracker", None) is None:
        raise HTTPException(
            status_code=400,
            detail="当前模型不含跟踪器，请加载 bytetrack/botsort/tracktrack 类模型",
        )
    if track_state["running"]:
        raise HTTPException(status_code=400, detail="Tracking already running")
    asyncio.create_task(asyncio.to_thread(_run_track, req))
    return {"started": True}


@router.get("/video/track/status")
def get_track_status():
    state = dict(track_state)
    bk = backup.get_backup()
    state["undo_available"] = bool(bk and bk["items"] and bk.get("source") == "video_track")
    return state
