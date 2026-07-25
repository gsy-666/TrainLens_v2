"""Directory browsing and image serving."""

import io
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response
from natsort import natsorted
from PIL import Image

from ..adapters import label_path_for, load_label_data
from ..deps import IMAGE_EXTENSIONS, session
from ..schemas import ImageInfo, OpenDirRequest, OpenDirResponse

router = APIRouter()

_pool = ThreadPoolExecutor(max_workers=16)


def _shape_count(directory: Path, name: str):
    """Shapes in the image's label file, or None when it has none."""
    label_path = label_path_for(directory / name)
    if not label_path.exists():
        return None
    try:
        data = load_label_data(label_path)
        shapes = data.get("shapes", [])
        labels = [s.get("label", "") for s in shapes if s.get("label")]
        return {"count": len(shapes), "labels": labels, "label_counts": dict(Counter(labels))}
    except Exception:
        return 0


def _build_image_infos(directory: Path, images):
    counts = list(_pool.map(lambda n: _shape_count(directory, n), images))
    return [
        ImageInfo(
            filename=name,
            has_label=count is not None,
            shape_count=(count["count"] if isinstance(count, dict) else count),
            labels=(count["labels"] if isinstance(count, dict) else []),
            label_counts=(count["label_counts"] if isinstance(count, dict) else {}),
        )
        for name, count in zip(images, counts)
    ]


@router.post("/dir/open", response_model=OpenDirResponse)
def open_dir(req: OpenDirRequest):
    directory = Path(req.path).expanduser()
    if not directory.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {req.path}")

    images = natsorted(
        p.name
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    session.set_dir(directory.resolve(), images)

    return OpenDirResponse(
        dir=str(directory.resolve()),
        images=_build_image_infos(directory, images),
    )


@router.get("/dir/images", response_model=OpenDirResponse)
def list_images():
    directory = session.get_dir()
    if directory is None:
        raise HTTPException(status_code=400, detail="No directory opened")
    return OpenDirResponse(
        dir=str(directory),
        images=_build_image_infos(directory, session.get_images()),
    )


def _resolve(name: str) -> Path:
    try:
        path = session.resolve_image(name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Image not found: {name}")
    return path


@router.get("/image")
def get_image(path: str):
    image_path = _resolve(path)
    return FileResponse(image_path)


@router.get("/image/thumb")
def get_thumbnail(path: str, max_size: int = 160):
    image_path = _resolve(path)
    try:
        with Image.open(image_path) as img:
            img.thumbnail((max_size, max_size))
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=80)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Thumbnail failed: {e}")
    return Response(content=buf.getvalue(), media_type="image/jpeg")
