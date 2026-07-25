@echo off
chcp 65001 >nul 2>nul
rem ============================================================
rem  TrainLens - ����ģʽ��ǰ���ȸ��£�
rem  ��� :8000 + Vite ���������� :5173���Զ��� :5173��
rem ============================================================
setlocal
cd /d %~dp0
cd ..
set "ROOT=%CD%"
cd /d %~dp0
set "PY=%ROOT%\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

echo [1/2] ������� http://127.0.0.1:8000
start "xanylabeling-backend" cmd /k "cd /d %~dp0backend && "%PY%" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload"

echo [2/2] ����ǰ�˿��������� http://localhost:5173
pushd "%~dp0frontend"
if not exist node_modules call npm install
start "" cmd /c "timeout /t 4 /nobreak >nul & start http://localhost:5173"
npm run dev
popd
