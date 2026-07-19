"""Optional bearer-token auth for remote exposure.

Local loopback usage stays token-free. When XANYLABELING_WEB_TOKEN is set
(the start launcher sets it automatically for public binds), every /api/*
route except /api/health requires the token, passed either as an
`Authorization: Bearer` header (XHR) or a `?token=` query param (needed
for <img> tags and download links that cannot set headers).
"""

import os
import secrets

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

TOKEN_ENV = "XANYLABELING_WEB_TOKEN"


def get_configured_token() -> str:
    return os.environ.get(TOKEN_ENV, "")


class TokenAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request, call_next):
        expected = get_configured_token()
        if not expected:
            return await call_next(request)

        path = request.url.path
        if not path.startswith("/api/") or path == "/api/health":
            return await call_next(request)

        provided = ""
        auth = request.headers.get("authorization", "")
        if auth.startswith("Bearer "):
            provided = auth[7:]
        elif "token" in request.query_params:
            provided = request.query_params["token"]

        if provided and secrets.compare_digest(provided, expected):
            return await call_next(request)

        return JSONResponse({"detail": "Unauthorized"}, status_code=401)
