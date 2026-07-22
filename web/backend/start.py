"""Launcher for the TrainLens web backend.

    python start.py [--host 127.0.0.1] [--port 8000] [--token XXX]

Binding to a public interface (anything other than loopback) requires a
token: pass --token, set XANYLABELING_WEB_TOKEN, or let the launcher
generate one and print it.
"""

import argparse
import os
import secrets
import socket
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn  # noqa: E402

LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def _p(msg=""):
    print(msg, flush=True)


def main():
    parser = argparse.ArgumentParser(description="TrainLens Web launcher")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--token", default=None, help="access token for remote use")
    args = parser.parse_args()

    token = args.token or os.environ.get("XANYLABELING_WEB_TOKEN")
    public = args.host not in LOOPBACK
    generated = False
    if public and not token:
        token = secrets.token_urlsafe(16)
        generated = True

    if token:
        os.environ["XANYLABELING_WEB_TOKEN"] = token

    _p("=" * 56)
    _p("  TrainLens Web")
    _p("=" * 56)
    if public:
        try:
            lan_ip = socket.gethostbyname(socket.gethostname())
        except OSError:
            lan_ip = args.host
        _p(f"  本地访问:  http://127.0.0.1:{args.port}")
        suffix = "" if args.host == "0.0.0.0" else f"  (绑定 {args.host})"
        _p(f"  远程访问:  http://{lan_ip}:{args.port}{suffix}")
        note = "  (自动生成，仅本次有效)" if generated else ""
        _p(f"  访问令牌:  {token}{note}")
        _p("  提示: 远程页面打开后输入令牌即可使用")
    else:
        _p(f"  访问地址:  http://127.0.0.1:{args.port}")
        if token:
            _p(f"  访问令牌:  {token}")
    _p("=" * 56)

    uvicorn.run("app.main:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
