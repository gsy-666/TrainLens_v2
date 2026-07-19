"""Local filesystem browsing for the directory picker dialog."""

import os
import string
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..deps import IMAGE_EXTENSIONS

router = APIRouter()

MAX_ENTRIES = 2000
_SCAN_LIMIT = 100  # entries to inspect per dir when checking for images
_CACHE_TTL = 15.0  # seconds

_pool = ThreadPoolExecutor(max_workers=8)

# path -> (timestamp, response dict)
_list_cache: dict[str, tuple[float, dict]] = {}


@router.get("/fs/roots")
def list_roots():
    """Top-level locations: drive letters on Windows, '/' elsewhere."""
    if sys.platform.startswith("win"):
        drives = [
            f"{d}:\\"
            for d in string.ascii_uppercase
            if os.path.exists(f"{d}:\\")
        ]
        return {"roots": drives}
    return {"roots": ["/"]}


def _has_images(directory: Path) -> bool:
    """Quick check: does the directory directly contain any image file?"""
    try:
        with os.scandir(directory) as it:
            for i, entry in enumerate(it):
                if i > _SCAN_LIMIT:  # don't scan huge dirs forever
                    break
                if entry.is_file() and Path(entry.name).suffix.lower() in IMAGE_EXTENSIONS:
                    return True
    except OSError:
        pass
    return False


@router.get("/fs/list")
def list_dir(path: str = "", with_files: str = ""):
    """List a directory.

    ``with_files``: optional comma-separated extensions (e.g. ".mp4,.avi");
    matching files are included in the ``files`` field so the picker can
    select files as well as directories.
    """
    if not path:
        return list_roots()

    directory = Path(path)
    if not directory.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {path}")

    file_exts = {
        e.strip().lower()
        for e in with_files.split(",")
        if e.strip()
    }

    cache_key = f"{directory}|{with_files}"
    cached = _list_cache.get(cache_key)
    if cached and time.monotonic() - cached[0] < _CACHE_TTL:
        return cached[1]

    dirs = []
    files = []
    try:
        with os.scandir(directory) as it:
            for entry in it:
                if len(dirs) >= MAX_ENTRIES:
                    break
                try:
                    if entry.name.startswith(".") or entry.name.startswith("$"):
                        continue
                    if entry.is_dir():
                        dirs.append(
                            {"name": entry.name, "path": entry.path, "has_images": False}
                        )
                    elif (
                        file_exts
                        and entry.is_file()
                        and Path(entry.name).suffix.lower() in file_exts
                    ):
                        files.append({"name": entry.name, "path": entry.path})
                except OSError:
                    continue
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"Permission denied: {path}")

    # Check subdirectories for images concurrently (serial scans are the
    # main cost on dirs with many children / slow disks).
    futures = [_pool.submit(_has_images, Path(d["path"])) for d in dirs]
    for d, fut in zip(dirs, futures):
        try:
            d["has_images"] = fut.result(timeout=5)
        except Exception:
            d["has_images"] = False

    dirs.sort(key=lambda d: d["name"].lower())
    files.sort(key=lambda f: f["name"].lower())
    parent = str(directory.parent) if directory.parent != directory else None
    result = {
        "path": str(directory),
        "parent": parent,
        "dirs": dirs,
        "files": files,
        "has_images": _has_images(directory),
    }
    _list_cache[cache_key] = (time.monotonic(), result)
    if len(_list_cache) > 200:
        oldest = min(_list_cache, key=lambda k: _list_cache[k][0])
        _list_cache.pop(oldest, None)
    return result
