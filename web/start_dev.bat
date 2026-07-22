@echo off
rem ============================================================
rem  TrainLens - 开发模式（前端热更新）
rem  后端 :8000 + Vite 开发服务器 :5173（自动打开 :5173）
rem ============================================================
setlocal
cd /d %~dp0
set "ROOT=%~dp0.."
set "PY=%ROOT%\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

echo [1/2] 启动后端 http://127.0.0.1:8000
start "xanylabeling-backend" cmd /k "cd /d %~dp0backend && "%PY%" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload"

echo [2/2] 启动前端开发服务器 http://localhost:5173
pushd "%~dp0frontend"
if not exist node_modules call npm install
start "" cmd /c "timeout /t 4 /nobreak >nul & start http://localhost:5173"
npm run dev
popd
