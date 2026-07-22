#!/usr/bin/env bash
# ============================================================
#  TrainLens - 一键启动 (Linux / macOS)
# ============================================================
set -e
cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PY="$ROOT/.venv/bin/python"

echo "=========================================="
echo "  TrainLens 一键启动"
echo "=========================================="

# ---- 1. Python 环境 ----
if [ ! -x "$PY" ]; then
  echo "[提示] 未找到项目虚拟环境，尝试使用系统 python3"
  PY=python3
fi
echo "[1/4] Python: $PY"

# ---- 2. 后端依赖 ----
if "$PY" -c "import fastapi, uvicorn" 2>/dev/null; then
  echo "[2/4] 后端依赖已就绪"
else
  echo "[2/4] 安装后端依赖..."
  "$PY" -m pip install -r "$SCRIPT_DIR/backend/requirements.txt"
fi

# ---- 3. 前端构建 ----
if [ -f "$SCRIPT_DIR/frontend/dist/index.html" ]; then
  echo "[3/4] 前端已构建（frontend/dist）"
else
  echo "[3/4] 首次运行，构建前端..."
  if ! command -v npm >/dev/null 2>&1; then
    echo "[错误] 未找到 npm。请先安装 Node.js，或手动执行："
    echo "       cd web/frontend && npm install && npm run build"
    exit 1
  fi
  cd "$SCRIPT_DIR/frontend"
  if [ ! -d node_modules ]; then
    echo "      安装前端依赖（npm install，首次较慢，请耐心等待）..."
    npm install
  fi
  npm run build
  cd "$SCRIPT_DIR"
fi

# ---- 4. 启动（单进程：API + 前端页面）----
echo "[4/4] 启动服务 http://127.0.0.1:8000 （按 Ctrl+C 停止）"
( sleep 3
  if command -v xdg-open >/dev/null 2>&1; then xdg-open http://127.0.0.1:8000
  elif command -v open >/dev/null 2>&1; then open http://127.0.0.1:8000
  fi ) &
cd "$SCRIPT_DIR/backend"
exec "$PY" start.py "$@"
