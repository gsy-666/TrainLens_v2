"""Pydantic schemas for the web API."""

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel


class OpenDirRequest(BaseModel):
    path: str


class ImageInfo(BaseModel):
    filename: str
    has_label: bool


class OpenDirResponse(BaseModel):
    dir: str
    images: List[ImageInfo]


class ShapeData(BaseModel):
    label: str
    points: List[Tuple[float, float]]
    shape_type: str = "polygon"
    group_id: Optional[int] = None
    description: str = ""
    difficult: bool = False
    score: Optional[float] = None
    flags: Dict[str, Any] = {}
    attributes: Dict[str, Any] = {}
    kie_linking: List[Any] = []
    direction: Optional[float] = None  # rotation shapes
    locked: Optional[bool] = None

    model_config = {"extra": "allow"}


class LabelFileData(BaseModel):
    """Mirrors the desktop Labelme-style annotation JSON."""

    version: Optional[str] = None
    flags: Dict[str, Any] = {}
    checked: bool = False
    shapes: List[ShapeData] = []
    imagePath: str
    imageData: Optional[str] = None
    imageHeight: int = -1
    imageWidth: int = -1

    model_config = {"extra": "allow"}


class SaveLabelsRequest(BaseModel):
    image: str  # file name relative to the opened dir
    shapes: List[ShapeData]
    flags: Dict[str, Any] = {}
    other_data: Dict[str, Any] = {}
    image_height: int = -1
    image_width: int = -1
