<#
.SYNOPSIS
    TrainLens Debug Build — creates dist\TrainLens\ with console visible.

.DESCRIPTION
    Builds a debug onedir distribution with console=True for troubleshooting.
    Only cleans build\TrainLens and dist\TrainLens (never user data).
#>

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot | Split-Path -Parent)

Write-Host "=== TrainLens Debug Build ===" -ForegroundColor Cyan

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

# 4. Clean only build/dist for TrainLens (NOT user data)
Write-Host "Cleaning previous TrainLens build..." -ForegroundColor Yellow
$buildDir = "build\TrainLens"
$distDir = "dist\TrainLens"
if (Test-Path $buildDir) { Remove-Item -Recurse -Force $buildDir }
if (Test-Path $distDir) { Remove-Item -Recurse -Force $distDir }

# 5. Build
Write-Host "Building TrainLens (debug, console=yes)..." -ForegroundColor Green
$env:TRAINLENS_CONSOLE = "1"
& $venvPython -m PyInstaller packaging\TrainLens.spec --distpath dist --workpath build 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "PyInstaller build failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

# 6. Verify
$exePath = "dist\TrainLens\TrainLens.exe"
if (-not (Test-Path $exePath)) {
    Write-Error "TrainLens.exe not found at $exePath"
    exit 1
}

# 7. Report
$exeSize = (Get-Item $exePath).Length / 1MB
$dirSize = (Get-ChildItem -Recurse $distDir | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host "=== Build SUCCESS ===" -ForegroundColor Green
Write-Host "  EXE:       $exePath"
Write-Host "  EXE size:  $([math]::Round($exeSize, 1)) MB"
Write-Host "  Dir size:  $([math]::Round($dirSize, 1)) MB"
Write-Host "  Console:   YES (debug)"
Write-Host "  Output:    $((Resolve-Path $distDir).Path)"
