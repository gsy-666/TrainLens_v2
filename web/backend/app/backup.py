"""One-level undo backup shared by batch pre-labeling and video tracking.

Stores pre-write label file contents (absolute paths) so an undo request
can restore them regardless of which feature overwrote them. Files whose
content changed after the write (e.g. manual edits) are skipped on undo.
"""

import hashlib
import time
import uuid
from pathlib import Path
from typing import Optional


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


_backup: Optional[dict] = None


def start_backup(source: str, directory: str) -> dict:
    global _backup
    _backup = {
        "batch_id": uuid.uuid4().hex[:8],
        "source": source,  # "batch" | "video_track"
        "created_at": time.time(),
        "dir": directory,
        "items": [],
    }
    return _backup


def add_item(
    image: str,
    label_path: Path,
    existed: bool,
    original_content: Optional[bytes],
):
    """Record one written label file (call right after writing it).

    ``original_content`` must have been read BEFORE the file was written.
    """
    if _backup is None:
        return
    _backup["items"].append(
        {
            "image": image,
            "label_path": str(label_path),
            "existed": existed,
            "content": original_content,
            "written_hash": sha256(label_path.read_bytes()),
        }
    )


def get_backup() -> Optional[dict]:
    return _backup


def clear_backup():
    global _backup
    _backup = None


def undo() -> dict:
    """Restore all backed-up label files. Returns counters."""
    global _backup
    if not _backup or not _backup["items"]:
        raise ValueError("No batch run to undo")

    restored, deleted, missing, skipped = 0, 0, 0, []
    for item in _backup["items"]:
        label_path = Path(item["label_path"])
        current = label_path.read_bytes() if label_path.exists() else None
        if current is not None and sha256(current) != item["written_hash"]:
            skipped.append(item["image"])
            continue
        if item["existed"]:
            label_path.write_bytes(item["content"])
            restored += 1
        else:
            if current is None:
                missing += 1
            else:
                label_path.unlink()
                deleted += 1

    _backup = None
    return {
        "restored": restored,
        "deleted": deleted,
        "already_missing": missing,
        "skipped_modified": skipped,
    }
