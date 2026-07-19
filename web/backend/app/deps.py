"""Shared session state for the web backend.

The web UI is a single-user local tool, so a simple module-level session
holds the currently opened directory and its image list.
"""

import threading
from pathlib import Path
from typing import List, Optional

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
    ".gif",
}


class SessionState:
    def __init__(self):
        self.lock = threading.RLock()
        self.image_dir: Optional[Path] = None
        self.images: List[str] = []  # file names relative to image_dir

    def set_dir(self, image_dir: Path, images: List[str]):
        with self.lock:
            self.image_dir = image_dir
            self.images = images

    def get_dir(self) -> Optional[Path]:
        with self.lock:
            return self.image_dir

    def get_images(self) -> List[str]:
        with self.lock:
            return list(self.images)

    def resolve_image(self, name: str) -> Path:
        """Resolve an image file name inside the opened directory.

        Raises ValueError when the resolved path escapes the directory.
        """
        with self.lock:
            if self.image_dir is None:
                raise ValueError("No directory opened")
            base = self.image_dir.resolve()
            path = (base / name).resolve()
            if base != path and base not in path.parents:
                raise ValueError(f"Path escapes opened directory: {name}")
            return path


session = SessionState()
