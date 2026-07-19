"""Label file read/write, desktop-format compatible."""

from fastapi import APIRouter, HTTPException
from PIL import Image

from ..adapters import label_path_for, load_label_data, save_label_data
from ..deps import session
from ..schemas import LabelFileData, SaveLabelsRequest

router = APIRouter()


def _resolve_image(name: str):
    try:
        path = session.resolve_image(name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Image not found: {name}")
    return path


@router.get("/labels")
def get_labels(image: str):
    image_path = _resolve_image(image)
    label_path = label_path_for(image_path)
    if not label_path.exists():
        return {"exists": False, "data": None}
    try:
        data = load_label_data(label_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse label file: {e}")
    return {"exists": True, "data": data}


@router.put("/labels")
def put_labels(req: SaveLabelsRequest):
    image_path = _resolve_image(req.image)

    image_height, image_width = req.image_height, req.image_width
    if image_height <= 0 or image_width <= 0:
        with Image.open(image_path) as img:
            image_width, image_height = img.size

    shapes = []
    for s in req.shapes:
        d = s.model_dump(exclude_none=True)
        # Desktop Shape.to_dict always emits group_id (null when unset).
        d.setdefault("group_id", None)
        shapes.append(d)
    try:
        label_path = save_label_data(
            image_path=image_path,
            shapes=shapes,
            flags=req.flags,
            other_data=req.other_data,
            image_height=image_height,
            image_width=image_width,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save label file: {e}")

    return {"saved": True, "label_path": str(label_path)}


@router.delete("/labels")
def delete_labels(image: str):
    image_path = _resolve_image(image)
    label_path = label_path_for(image_path)
    if label_path.exists():
        label_path.unlink()
    return {"deleted": True}
