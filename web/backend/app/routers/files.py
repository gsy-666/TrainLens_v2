"""Directory browsing and image serving."""

import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response
from natsort import natsorted
from PIL import Image

from ..adapters import label_path_for
from ..deps import IMAGE_EXTENSIONS, session
from ..schemas import ImageInfo, OpenDirRequest, OpenDirResponse

router = APIRouter()

_pool = ThreadPoolExecutor(max_workers=16)


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

    # Check label-file existence concurrently (serial stat calls add up
    # quickly on large datasets).
    flags = list(_pool.map(lambda n: label_path_for(directory / n).exists(), images))

    return OpenDirResponse(
        dir=str(directory.resolve()),
        images=[
            ImageInfo(filename=name, has_label=has_label)
            for name, has_label in zip(images, flags)
        ],
    )


@router.get("/dir/images", response_model=OpenDirResponse)
def list_images():
    directory = session.get_dir()
    if directory is None:
        raise HTTPException(status_code=400, detail="No directory opened")
    return OpenDirResponse(
        dir=str(directory),
        images=[
            ImageInfo(
                filename=name,
                has_label=label_path_for(directory / name).exists(),
            )
            for name in session.get_images()
        ],
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
