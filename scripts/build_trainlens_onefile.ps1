<#
.SYNOPSIS
    TrainLens ONE-FILE Release Build — produces dist\TrainLens.exe

.DESCRIPTION
    Builds a standalone single-file Windows executable.
    Only cleans build\TrainLens_onefile and dist\TrainLens.exe (never user data).
#>

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot | Split-Path -Parent)

Write-Host "=== TrainLens ONE-FILE Release Build ===" -ForegroundColor Cyan

# 1. Check current directory is project root
if (-not (Test-Path "anylabeling\app.py")) {
    Write-Error "Must run from project root (D:\x-anylabeling). Current: $PWD"
    exit 1
}

# 2. Check .venv
$venvPython = ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Error "Virtual environment not found: $venvPython"
    exit 1
}

# 3. Ensure PyInstaller is installed
Write-Host "Checking PyInstaller..." -ForegroundColor Yellow
& $venvPython -c "import PyInstaller; print(f'PyInstaller {PyInstaller.__version__}')"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing PyInstaller..." -ForegroundColor Yellow
    & $venvPython -m pip install pyinstaller --quiet
}

# 4. Clean only build/dist for this onefile build
Write-Host "Cleaning previous onefile build..." -ForegroundColor Yellow
$buildDir = "build\TrainLens_onefile"
$exePath = "dist\TrainLens.exe"
if (Test-Path $buildDir) { Remove-Item -Recurse -Force $buildDir }
if (Test-Path $exePath) { Remove-Item -Force $exePath }

# 5. Build
$startTime = Get-Date
Write-Host "Building TrainLens ONE-FILE (console=no)..." -ForegroundColor Green
& $venvPython -m PyInstaller packaging\TrainLens_onefile.spec --distpath dist --workpath build\TrainLens_onefile 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "PyInstaller build failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}
$buildTime = (Get-Date) - $startTime

# 6. Verify
if (-not (Test-Path $exePath)) {
    Write-Error "TrainLens.exe not found at $exePath"
    exit 1
}

$exeSize = (Get-Item $exePath).Length / 1MB
Write-Host ""
Write-Host "=== Build SUCCESS ===" -ForegroundColor Green
Write-Host "  File:      $exePath"
Write-Host "  Size:      $([math]::Round($exeSize, 1)) MB"
Write-Host "  Build:     $($buildTime.TotalMinutes.ToString('F1')) min"
Write-Host "  Console:   NO (windowed)"
Write-Host "  Type:      ONE-FILE (standalone)"
