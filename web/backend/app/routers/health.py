"""Dataset health check.

Scans the currently opened dataset for common quality problems:

- duplicate / near-duplicate images (dHash + union-find clustering)
- blurry images (Laplacian variance)
- too dark / too bright images (mean gray value)
- label anomalies in the same-name labelme JSON (out-of-bounds points,
  oversized / tiny boxes, degenerate shapes)

Follows the same background-scan skeleton as active_learning.py: a
module-level state object guarded by an RLock, a daemon thread doing the
work, a stop event, and status polling.
"""

import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException

from ..adapters import label_path_for, load_label_data
from ..deps import session

router = APIRouter()

# ---------------------------------------------------------------------------
# Tunable thresholds (module constants on purpose)
# ---------------------------------------------------------------------------

# Decode cap: metrics are computed on a copy whose longest side is at most
# this many pixels, to bound memory/CPU on large photos. The original width
# and height are still used for the label checks.
MAX_DECODE_DIM = 512

# Blur/brightness are computed on a grayscale thumbnail of this width
# (aspect preserved) so scores are comparable across resolutions.
METRIC_WIDTH = 256

# Laplacian variance below this -> blurry (lower = blurrier).
BLUR_THRESHOLD = 100.0

# Mean gray value (0..255) below this -> too dark, above this -> too bright.
DARK_THRESHOLD = 60.0
BRIGHT_THRESHOLD = 195.0

# dHash: 9x8 grayscale thumbnail -> 64 bits. Two images are similar when the
# Hamming distance of their hashes is at most this value.
DHASH_HAMMING_THRESHOLD = 5

# Pairwise comparison is O(n^2) integer XORs; above this many images we
# fall back to grouping only identical hashes to keep the scan fast.
PAIRWISE_LIMIT = 20000

# A shape point may stick out of the image by at most this many pixels
# before it counts as out_of_bounds.
OOB_TOLERANCE_PX = 2.0

# A rectangle/rotation box covering more than this fraction of the image
# area is flagged oversized.
OVERSIZED_AREA_RATIO = 0.90

# A rectangle covering less than this fraction of the image area is flagged
# tiny (0.01%). Only applied when the image itself is at least
# TINY_MIN_IMAGE_AREA px^2, so small images do not produce false positives.
TINY_AREA_RATIO = 0.0001
TINY_MIN_IMAGE_AREA = 200.0 * 200.0

# Polygon areas below this count as zero (degenerate).
_ZERO_AREA_EPS = 1e-9


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
        self.report: Optional[Dict[str, Any]] = None


_STATE = ScanState()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _imread_unicode(path: Path) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int]]]:
    """cv2.imread fails on non-ASCII (e.g. Chinese) paths on Windows; decode
    from raw bytes instead. Same approach as
    anylabeling.views.labeling.label_converter.imread_unicode.
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None, None
    if data.size == 0:
        return None, None
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        return None, None
    height, width = image.shape[:2]
    return image, (width, height)


def _dhash(gray: np.ndarray) -> int:
    """64-bit difference hash on a 9x8 grayscale thumbnail."""
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    value = 0
    for byte in np.packbits(diff.flatten()):
        value = (value << 8) | int(byte)
    return value


def _polygon_area(points: List[List[float]]) -> float:
    """Shoelace area of a polygon (also used for 4-point boxes)."""
    area = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _check_shapes(
    shapes: List[Dict[str, Any]], width: int, height: int
) -> List[Dict[str, Any]]:
    """Per-shape label anomaly checks. Pixel data is not needed here."""
    issues: List[Dict[str, Any]] = []
    image_area = float(width) * float(height)

    for shape in shapes:
        label = shape.get("label", "")
        shape_type = shape.get("shape_type", "")
        points = shape.get("points") or []

        def add(issue: str, detail: str):
            issues.append({"label": label, "issue": issue, "detail": detail})

        # out_of_bounds applies to every shape type.
        worst: Optional[Tuple[float, str]] = None
        for pt in points:
            if not isinstance(pt, (list, tuple)) or len(pt) < 2:
                continue
            x, y = float(pt[0]), float(pt[1])
            if x < -OOB_TOLERANCE_PX:
                cand = (-x, f"点 ({x:.0f},{y:.0f}) 超出图像左边界 {-x:.0f}px")
            elif x > width + OOB_TOLERANCE_PX:
                cand = (x - width, f"点 ({x:.0f},{y:.0f}) 超出图像右边界 {x - width:.0f}px")
            elif y < -OOB_TOLERANCE_PX:
                cand = (-y, f"点 ({x:.0f},{y:.0f}) 超出图像上边界 {-y:.0f}px")
            elif y > height + OOB_TOLERANCE_PX:
                cand = (y - height, f"点 ({x:.0f},{y:.0f}) 超出图像下边界 {y - height:.0f}px")
            else:
                continue
            if worst is None or cand[0] > worst[0]:
                worst = cand
        if worst is not None:
            add("out_of_bounds", worst[1])

        if shape_type in ("rectangle", "rotation") and len(points) >= 4:
            area = _polygon_area(points)
            ratio = area / image_area if image_area > 0 else 0.0
            if area <= _ZERO_AREA_EPS:
                add("degenerate", "矩形零面积")
            elif ratio > OVERSIZED_AREA_RATIO:
                add("oversized", f"框面积占图像 {ratio * 100:.1f}%")
            elif (
                shape_type == "rectangle"
                and image_area >= TINY_MIN_IMAGE_AREA
                and ratio < TINY_AREA_RATIO
            ):
                add("tiny", f"框面积仅占图像 {ratio * 100:.3f}%")
        elif shape_type == "polygon" and len(points) < 3:
            add("degenerate", f"多边形仅 {len(points)} 个点，不足 3 个")

    return issues


def _analyze_image(image_dir: Path, name: str) -> Dict[str, Any]:
    """Compute all per-image health signals. Raises on decode failure."""
    image_path = image_dir / name
    image, size = _imread_unicode(image_path)
    if image is None or size is None:
        raise ValueError(f"无法解码图像: {name}")
    width, height = size

    # Downscale for metric computation; original size kept for label checks.
    scale = MAX_DECODE_DIM / max(width, height)
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thumb_h = max(1, int(gray.shape[0] * METRIC_WIDTH / gray.shape[1]))
    thumb = cv2.resize(gray, (METRIC_WIDTH, thumb_h), interpolation=cv2.INTER_AREA)
    blur_score = float(cv2.Laplacian(thumb, cv2.CV_64F).var())
    brightness = float(thumb.mean())

    shape_issues: List[Dict[str, Any]] = []
    label_path = label_path_for(image_path)
    if label_path.exists():
        try:
            data = load_label_data(label_path)
        except Exception:  # noqa - corrupted JSON should not kill the scan
            data = None
        if data is not None:
            shape_issues = _check_shapes(data.get("shapes") or [], width, height)

    return {
        "hash": _dhash(gray),
        "blur_score": blur_score,
        "brightness": brightness,
        "shape_issues": shape_issues,
    }


class _UnionFind:
    def __init__(self, items: List[str]):
        self.parent = {x: x for x in items}

    def find(self, x: str) -> str:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _duplicate_groups(hashes: Dict[str, int]) -> List[List[str]]:
    """Cluster images whose dHash Hamming distance is within the threshold."""
    names = sorted(hashes)
    uf = _UnionFind(names)
    if len(names) > PAIRWISE_LIMIT:
        # Too many images for O(n^2): only group identical hashes.
        by_hash: Dict[int, List[str]] = {}
        for n in names:
            by_hash.setdefault(hashes[n], []).append(n)
        return [sorted(g) for g in by_hash.values() if len(g) >= 2]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            dist = bin(hashes[names[i]] ^ hashes[names[j]]).count("1")
            if dist <= DHASH_HAMMING_THRESHOLD:
                uf.union(names[i], names[j])
    clusters: Dict[str, List[str]] = {}
    for n in names:
        clusters.setdefault(uf.find(n), []).append(n)
    return [sorted(g) for g in clusters.values() if len(g) >= 2]


def _build_report(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    hashes = {n: r["hash"] for n, r in results.items()}
    groups = _duplicate_groups(hashes)
    duplicate_images = sum(len(g) for g in groups)

    blurry = sorted(
        (
            {"name": n, "score": round(r["blur_score"], 1)}
            for n, r in results.items()
            if r["blur_score"] < BLUR_THRESHOLD
        ),
        key=lambda e: (e["score"], e["name"]),
    )
    dark = sorted(
        (
            {"name": n, "brightness": round(r["brightness"], 1)}
            for n, r in results.items()
            if r["brightness"] < DARK_THRESHOLD
        ),
        key=lambda e: (e["brightness"], e["name"]),
    )
    bright = sorted(
        (
            {"name": n, "brightness": round(r["brightness"], 1)}
            for n, r in results.items()
            if r["brightness"] > BRIGHT_THRESHOLD
        ),
        key=lambda e: (-e["brightness"], e["name"]),
    )
    shape_issues = [
        {"name": n, "issues": r["shape_issues"]}
        for n, r in sorted(results.items())
        if r["shape_issues"]
    ]

    return {
        "summary": {
            "total_images": len(results),
            "duplicate_groups": len(groups),
            "duplicate_images": duplicate_images,
            "blurry": len(blurry),
            "dark": len(dark),
            "bright": len(bright),
            "shape_issue_images": len(shape_issues),
        },
        "duplicate_groups": groups,
        "blurry": blurry,
        "dark": dark,
        "bright": bright,
        "shape_issues": shape_issues,
        "updated_at": _utc_now_iso(),
    }


def _run_scan(image_dir: Path, images: List[str]):
    results: Dict[str, Dict[str, Any]] = {}
    try:
        for name in images:
            if _STATE.stop_event.is_set():
                break
            try:
                results[name] = _analyze_image(image_dir, name)
            except Exception:  # noqa - skip broken image/JSON, keep going
                with _STATE.lock:
                    _STATE.error_count += 1
            with _STATE.lock:
                _STATE.processed += 1
    except Exception as e:  # noqa
        with _STATE.lock:
            _STATE.error = str(e)
    finally:
        report = _build_report(results)
        with _STATE.lock:
            _STATE.report = report
            _STATE.running = False
            _STATE.done = True


@router.post("/dataset/health/scan")
def start_scan():
    image_dir = session.get_dir()
    if image_dir is None or not session.get_images():
        raise HTTPException(status_code=400, detail="No directory opened")

    with _STATE.lock:
        if _STATE.running:
            raise HTTPException(status_code=409, detail="Scan already running")
        images = session.get_images()
        _STATE.stop_event.clear()
        _STATE.running = True
        _STATE.processed = 0
        _STATE.total = len(images)
        _STATE.done = False
        _STATE.error = None
        _STATE.error_count = 0
        _STATE.thread = threading.Thread(
            target=_run_scan,
            args=(image_dir, images),
            daemon=True,
        )
        _STATE.thread.start()
    return {"started": True, "total": len(images)}


@router.get("/dataset/health/scan/status")
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


@router.post("/dataset/health/scan/stop")
def stop_scan():
    """Ask the running scan to stop after the current image finishes."""
    with _STATE.lock:
        stopping = _STATE.running
        _STATE.stop_event.set()
    return {"stopping": stopping}


@router.get("/dataset/health/report")
def get_report():
    with _STATE.lock:
        if _STATE.report is None:
            return {"updated_at": None}
        return _STATE.report
