@echo off
chcp 65001 >nul 2>nul
rem ============================================================
rem  TrainLens - һ������
rem  ˫�����ɣ�������� -> ����ǰ�� -> �������� -> �������
rem
rem  Զ�̷����÷������Ʒ������ϣ���
rem    start_web.bat --host 0.0.0.0 [--port 8000] [--token XXX]
rem  Զ��ģʽ���Զ����ɷ������Ʋ���ӡ�ڿ���̨��
rem ============================================================
setlocal
cd /d %~dp0
cd ..
set "ROOT=%CD%"
cd /d %~dp0
set "PY=%ROOT%\.venv\Scripts\python.exe"

echo ==========================================
echo   TrainLens һ������
echo ==========================================

rem ---- 1. Python ���� ----
if not exist "%PY%" (
  echo [��ʾ] δ�ҵ���Ŀ���⻷��������ʹ��ϵͳ python
  set "PY=python"
)
echo [1/4] Python: %PY%

rem ---- 2. ������� ----
"%PY%" -c "import fastapi, uvicorn" >nul 2>nul
if errorlevel 1 (
  echo [2/4] ��װ�������...
  "%PY%" -m pip install -r "%~dp0backend\requirements.txt"
  if errorlevel 1 goto :error
) else (
  echo [2/4] ��������Ѿ���
)

rem ---- 3. ǰ�˹��� ----
if exist "%~dp0frontend\dist\index.html" (
  echo [3/4] ǰ���ѹ�����frontend\dist��
) else (
  echo [3/4] �״����У�����ǰ��...
  where npm >nul 2>nul
  if errorlevel 1 (
    echo [����] δ�ҵ� npm�����Ȱ�װ Node.js�����ֶ�ִ�У�
    echo        cd web\frontend ^&^& npm install ^&^& npm run build
    goto :error
  )
  pushd "%~dp0frontend"
  if not exist node_modules (
    echo       ��װǰ��������npm install���״ν����������ĵȴ���...
    call npm install
    if errorlevel 1 ( popd & goto :error )
  )
  call npm run build
  if errorlevel 1 ( popd & goto :error )
  popd
)

rem ---- 4. �����������̣�API + ǰ��ҳ�棩----
echo [4/4] �������񣨰� Ctrl+C ֹͣ��
echo %* | findstr /i "0.0.0.0" >nul 2>nul || start "" cmd /c "timeout /t 3 /nobreak >nul & start http://127.0.0.1:8000"
cd /d "%~dp0backend"

rem Kill any previous process on port 8000
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000.*LISTENING" 2^>nul') do taskkill /f /pid %%a >nul 2>nul

"%PY%" start.py %*
if errorlevel 1 (
  echo.
  echo Server exited unexpectedly. Check errors above.
  pause
  exit /b 1
)
pause
goto :eof

:error
echo.
echo ����ʧ�ܣ������Ϸ�������Ϣ��
pause
exit /b 1
