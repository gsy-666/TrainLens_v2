@echo off
rem ============================================================
rem  TrainLens - 一键启动
rem  双击即可：检查依赖 -> 构建前端 -> 启动服务 -> 打开浏览器
rem
rem  远程访问用法（在云服务器上）：
rem    start_web.bat --host 0.0.0.0 [--port 8000] [--token XXX]
rem  远程模式会自动生成访问令牌并打印在控制台。
rem ============================================================
setlocal
cd /d %~dp0
set "ROOT=%~dp0.."
set "PY=%ROOT%\.venv\Scripts\python.exe"

echo ==========================================
echo   TrainLens 一键启动
echo ==========================================

rem ---- 1. Python 环境 ----
if not exist "%PY%" (
  echo [提示] 未找到项目虚拟环境，尝试使用系统 python
  set "PY=python"
)
echo [1/4] Python: %PY%

rem ---- 2. 后端依赖 ----
"%PY%" -c "import fastapi, uvicorn" >/dev/null 2>&1
if errorlevel 1 (
  echo [2/4] 安装后端依赖...
  "%PY%" -m pip install -r "%~dp0backend\requirements.txt"
  if errorlevel 1 goto :error
) else (
  echo [2/4] 后端依赖已就绪
)

rem ---- 3. 前端构建 ----
if exist "%~dp0frontend\dist\index.html" (
  echo [3/4] 前端已构建（frontend\dist）
) else (
  echo [3/4] 首次运行，构建前端...
  where npm >/dev/null 2>&1
  if errorlevel 1 (
    echo [错误] 未找到 npm。请先安装 Node.js，或手动执行：
    echo        cd web\frontend ^&^& npm install ^&^& npm run build
    goto :error
  )
  pushd "%~dp0frontend"
  if not exist node_modules (
    echo       安装前端依赖（npm install，首次较慢，请耐心等待）...
    call npm install
    if errorlevel 1 ( popd & goto :error )
  )
  call npm run build
  if errorlevel 1 ( popd & goto :error )
  popd
)

rem ---- 4. 启动（单进程：API + 前端页面）----
echo [4/4] 启动服务（按 Ctrl+C 停止）
echo %* | findstr /i "0.0.0.0" >/dev/null || start "" cmd /c "timeout /t 3 /nobreak >/dev/null & start http://127.0.0.1:8000"
cd /d "%~dp0backend"
"%PY%" start.py %*
goto :eof

:error
echo.
echo 启动失败，请检查上方错误信息。
pause
exit /b 1
