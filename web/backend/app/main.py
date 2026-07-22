"""TrainLens Web UI backend (FastAPI).

Reuses the desktop codebase (anylabeling.*) for label file IO and
auto-labeling model inference. Run from web/backend/:

    uvicorn app.main:app --host 127.0.0.1 --port 8000

When web/frontend/dist exists (npm run build), it is served at / so the
whole app runs from this single process/port.
"""

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Make the repository root importable so `anylabeling.*` can be reused.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from .routers import (  # noqa: E402
    dataset,
    export,
    files,
    fs,
    labels,
    models,
    monitor,
    predict,
    training,
    upload,
    video,
)
from .auth import TokenAuthMiddleware  # noqa: E402

app = FastAPI(title="TrainLens", version="1.0.0")

app.add_middleware(TokenAuthMiddleware)

# The web frontend may be served from any origin (local one-click install
# connecting to a cloud backend). /api/* is guarded by the token middleware
# whenever a token is configured, so an open CORS policy is acceptable here.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(files.router, prefix="/api", tags=["files"])
app.include_router(fs.router, prefix="/api", tags=["fs"])
app.include_router(labels.router, prefix="/api", tags=["labels"])
app.include_router(models.router, prefix="/api", tags=["models"])
app.include_router(predict.router, prefix="/api", tags=["predict"])
app.include_router(export.router, prefix="/api", tags=["export"])
app.include_router(upload.router, prefix="/api", tags=["upload"])
app.include_router(video.router, prefix="/api", tags=["video"])
app.include_router(training.router, prefix="/api", tags=["training"])
app.include_router(monitor.router, prefix="/api", tags=["monitor"])
app.include_router(dataset.router, prefix="/api", tags=["dataset"])


@app.get("/api/health")
def health():
    return {"status": "ok"}


# Serve the built frontend (single-process production mode). Mounted last
# so /api/* routes win. html=True gives SPA-style index.html at /.
_FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if _FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=_FRONTEND_DIST, html=True), name="web")
