@echo off
REM X-AnyLabeling / TrainLens Development Launcher for Windows
REM Automatically installs uv, sets up Python environment, and launches the application

setlocal enabledelayedexpansion

echo ========================================
echo X-AnyLabeling Development Launcher
echo ========================================
echo.

REM ============================================
REM Step 1: Check for uv installation
REM ============================================
echo [1/6] Checking for uv package manager...
where uv >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] uv is already installed
    goto :check_venv
)

echo [INFO] uv not found. Installing uv...
echo.

REM Try to install uv using pip
python -m pip install --upgrade uv >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] uv installed successfully via pip
    goto :check_venv
)

REM If pip install failed, try using PowerShell installer
echo [INFO] Attempting to install uv via PowerShell installer...
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install uv
    echo Please install uv manually: https://github.com/astral-sh/uv
    pause
    exit /b 1
)

echo [OK] uv installed successfully
echo.

REM Refresh PATH to include uv
call refreshenv >nul 2>&1

REM ============================================
REM Step 2: Check for .venv directory
REM ============================================
:check_venv
echo [2/6] Checking for virtual environment...
if exist ".venv\" (
    echo [OK] Virtual environment already exists
    goto :detect_gpu
)

echo [INFO] Creating virtual environment...
uv venv .venv
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to create virtual environment
    pause
    exit /b 1
)
echo [OK] Virtual environment created
echo.

REM ============================================
REM Step 3: Detect GPU and determine extra
REM ============================================
:detect_gpu
echo [3/6] Detecting GPU...

REM Check for NVIDIA GPU
nvidia-smi >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] NVIDIA GPU detected
    set EXTRA=gpu
    goto :install_deps
)

echo [INFO] No NVIDIA GPU detected, using CPU mode
set EXTRA=cpu

REM ============================================
REM Step 4: Install dependencies
REM ============================================
:install_deps
echo [4/6] Installing dependencies with extra: %EXTRA%...
echo.

REM Read pyproject.toml to verify extra exists
findstr /C:"[%EXTRA%]" pyproject.toml >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Extra '%EXTRA%' not found in pyproject.toml, falling back to 'cpu'
    set EXTRA=cpu
)

REM Install with uv
uv pip install -e ".[%EXTRA%,dev]"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo [OK] Dependencies installed successfully
echo.

REM ============================================
REM Step 5: Verify installation
REM ============================================
echo [5/6] Verifying installation...

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Check if anylabeling module is importable
python -c "import anylabeling; print('Version:', anylabeling.app_info.__version__)" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to import anylabeling module
    echo Please check the installation
    pause
    exit /b 1
)

echo [OK] Installation verified
echo.

REM ============================================
REM Step 6: Launch application
REM ============================================
echo [6/6] Launching X-AnyLabeling...
echo.
echo ========================================
echo.

REM Launch the application
python -m anylabeling.app %*

REM Check exit code
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ========================================
    echo [ERROR] Application exited with error code: %ERRORLEVEL%
    echo ========================================
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ========================================
echo Application closed successfully
echo ========================================
exit /b 0
