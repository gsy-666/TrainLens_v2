"""Dataset export endpoints (YOLO/VOC/COCO/DOTA/Mask/MOT/ODVG)."""

import asyncio
import json
import os
import os.path as osp
import shutil
import time
import zipfile
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from natsort import natsorted
from pydantic import BaseModel

from anylabeling.views.labeling.label_converter import LabelConverter

from ..adapters import label_path_for, load_label_data
from ..deps import session

router = APIRouter()

SUPPORTED_FORMATS = {
    "yolo": {"modes": ["hbb", "obb", "seg"], "default_mode": "hbb"},
    "voc": {"modes": ["rectangle", "polygon"], "default_mode": "rectangle"},
    "coco": {"modes": ["rectangle", "polygon"], "default_mode": "rectangle"},
    "dota": {"modes": [], "default_mode": None},
    "mask": {"modes": [], "default_mode": None},
    "mot": {"modes": [], "default_mode": None},
    "odvg": {"modes": [], "default_mode": None},
}

# in-progress / last export state (single-user tool)
export_state = {
    "running": False,
    "current": 0,
    "total": 0,
    "message": "",
    "result": None,
    "error": None,
}


class ExportRequest(BaseModel):
    format: str
    mode: Optional[str] = None
    output_dir: str
    classes: Optional[List[str]] = None  # default: derived from dataset
    save_images: bool = False
    skip_empty_files: bool = False


def _collect_classes(image_dir: Path, images: List[str]) -> List[str]:
    labels = set()
    for name in images:
        label_path = label_path_for(image_dir / name)
        if not label_path.exists():
            continue
        try:
            for shape in load_label_data(label_path).get("shapes", []):
                if shape.get("label"):
                    labels.add(shape["label"])
        except Exception:
            continue
    return natsorted(labels)


def _run_export(req: ExportRequest):
    export_state.update(
        running=True, current=0, total=0, message="", result=None, error=None
    )
    try:
        image_dir = session.get_dir()
        if image_dir is None:
            raise RuntimeError("No directory opened")
        images = session.get_images()
        if not images:
            raise RuntimeError("Opened directory contains no images")

        out_dir = Path(req.output_dir).expanduser().resolve()
        if out_dir.exists() and any(out_dir.iterdir()):
            raise RuntimeError(f"Output directory is not empty: {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)

        classes = req.classes or _collect_classes(image_dir, images)
        if req.format in ("yolo", "voc", "coco") and not classes:
            raise RuntimeError("No labeled classes found in dataset")

        converter = LabelConverter()
        converter.classes = list(classes)

        mode = req.mode or SUPPORTED_FORMATS[req.format]["default_mode"]
        total = len(images)
        export_state["total"] = total
        written = 0

        abs_images = [str(image_dir / n) for n in images]

        def label_of(name: str) -> Path:
            return label_path_for(image_dir / name)

        if req.format in ("yolo", "voc", "dota", "mask"):
            if req.format == "mask":
                mapping_table = {label: i + 1 for i, label in enumerate(classes)}
                with open(out_dir / "mask_mapping.json", "w", encoding="utf-8") as f:
                    json.dump(mapping_table, f, ensure_ascii=False, indent=2)
            for i, name in enumerate(images):
                export_state["current"] = i
                export_state["message"] = name
                src = label_of(name)
                stem = osp.splitext(name)[0]
                try:
                    if req.format == "yolo":
                        dst = out_dir / f"{stem}.txt"
                        empty = converter.custom_to_yolo(
                            str(src),
                            str(dst),
                            mode,
                            skip_empty_files=req.skip_empty_files,
                        )
                        if req.skip_empty_files and empty and dst.exists():
                            dst.unlink()
                        else:
                            written += 1
                    elif req.format == "voc":
                        dst = out_dir / f"{stem}.xml"
                        empty = converter.custom_to_voc(
                            str(image_dir / name),
                            str(src),
                            str(dst),
                            mode,
                            req.skip_empty_files,
                        )
                        if req.skip_empty_files and empty and dst.exists():
                            dst.unlink()
                        else:
                            written += 1
                    elif req.format == "dota":
                        if src.exists():
                            converter.custom_to_dota(str(src), str(out_dir / f"{stem}.txt"))
                            written += 1
                    elif req.format == "mask":
                        if src.exists():
                            converter.custom_to_mask(
                                str(src), str(out_dir / f"{stem}.png"), mapping_table
                            )
                            written += 1
                    if req.save_images and src.exists():
                        shutil.copy(image_dir / name, out_dir / name)
                except Exception as e:  # noqa
                    raise RuntimeError(f"Export failed on {name}: {e}")

            if req.format == "yolo":
                with open(out_dir / "classes.txt", "w", encoding="utf-8") as f:
                    f.write("\n".join(classes) + "\n")

        elif req.format == "coco":
            export_state["total"] = 1
            export_state["message"] = "building annotations"
            converter.custom_to_coco(abs_images, str(image_dir), str(out_dir), mode)
            written = 1

        elif req.format == "mot":
            export_state["total"] = 1
            converter.custom_to_mot(str(image_dir), str(out_dir))
            written = 1

        elif req.format == "odvg":
            export_state["total"] = 1
            converter.custom_to_odvg(abs_images, str(image_dir), str(out_dir))
            written = 1

        export_state["current"] = export_state["total"]
        export_state["result"] = {
            "output_dir": str(out_dir),
            "files_written": written,
            "format": req.format,
            "mode": mode,
            "classes": classes,
        }
    except Exception as e:  # noqa
        export_state["error"] = str(e)
    finally:
        export_state["running"] = False


@router.get("/export/formats")
def export_formats():
    return {"formats": SUPPORTED_FORMATS}


@router.post("/export")
async def start_export(req: ExportRequest):
    if req.format not in SUPPORTED_FORMATS:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {req.format}")
    if req.mode and req.mode not in SUPPORTED_FORMATS[req.format]["modes"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode '{req.mode}' for {req.format}",
        )
    if export_state["running"]:
        raise HTTPException(status_code=400, detail="Export already running")
    asyncio.create_task(asyncio.to_thread(_run_export, req))
    return {"started": True}


@router.get("/export/status")
def get_export_status():
    return export_state


@router.get("/export/download")
def download_export(path: str):
    """Zip an export output directory for download."""
    out_dir = Path(path).resolve()
    if not out_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {path}")
    zip_path = out_dir.parent / f"{out_dir.name}_export_{int(time.time())}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(out_dir):
            for fn in files:
                fp = Path(root) / fn
                zf.write(fp, fp.relative_to(out_dir))
    return FileResponse(
        zip_path,
        filename=zip_path.name,
        media_type="application/zip",
    )
