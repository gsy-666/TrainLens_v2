"""Upload images / zip archives into the dataset directory."""

import shutil
import zipfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from ..deps import IMAGE_EXTENSIONS, session

router = APIRouter()

ALLOWED_UPLOAD = IMAGE_EXTENSIONS | {".json"}


@router.post("/upload/files")
async def upload_files(files: list[UploadFile] = File(...)):
    """Upload image (and optional label .json) files into the opened dir."""
    image_dir = session.get_dir()
    if image_dir is None:
        raise HTTPException(status_code=400, detail="No directory opened")

    saved, skipped = 0, []
    for uf in files:
        name = Path(uf.filename or "").name  # strip any client path
        if not name:
            continue
        suffix = Path(name).suffix.lower()
        if suffix not in ALLOWED_UPLOAD:
            skipped.append(name)
            continue
        dest = image_dir / name
        with open(dest, "wb") as f:
            shutil.copyfileobj(uf.file, f)
        saved += 1

    # rescan directory so new images appear
    images = sorted(
        p.name
        for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    session.set_dir(image_dir, images)
    return {"saved": saved, "skipped": skipped, "total_images": len(images)}


class UploadZipRequest(BaseModel):
    target_dir: str


@router.post("/upload/zip")
async def upload_zip(file: UploadFile = File(...)):
    """Extract an uploaded zip into the currently opened directory."""
    image_dir = session.get_dir()
    if image_dir is None:
        raise HTTPException(status_code=400, detail="No directory opened")
    if not (file.filename or "").lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip files are accepted")

    extracted, skipped = 0, []
    with zipfile.ZipFile(file.file) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename).name
            if not name or Path(name).suffix.lower() not in ALLOWED_UPLOAD:
                skipped.append(info.filename)
                continue
            dest = image_dir / name
            with zf.open(info) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted += 1

    images = sorted(
        p.name
        for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    session.set_dir(image_dir, images)
    return {"extracted": extracted, "skipped": skipped, "total_images": len(images)}
