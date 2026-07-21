<#
.SYNOPSIS
    TrainLens Smoke Test — validates the packaged distribution.

.DESCRIPTION
    Runs basic validation on dist\TrainLens\TrainLens.exe to ensure
    the frozen build is functional and contains all required dependencies.
#>

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot | Split-Path -Parent)

Write-Host "=== TrainLens Smoke Test ===" -ForegroundColor Cyan

$exePath = "dist\TrainLens\TrainLens.exe"
$internalDir = "dist\TrainLens\_internal"
$failed = 0
$passed = 0

function Test-Check {
    param($name, [ScriptBlock]$test)
    Write-Host -NoNewline "  [$name] "
    try {
        $result = & $test
        if ($result) {
            Write-Host "PASS" -ForegroundColor Green
            $script:passed++
        } else {
            Write-Host "FAIL" -ForegroundColor Red
            $script:failed++
        }
    } catch {
        Write-Host "FAIL ($_)" -ForegroundColor Red
        $script:failed++
    }
}

# 1. EXE existence
Test-Check "EXE exists" { Test-Path $exePath }

# 2. EXE can start and output self-check JSON
Test-Check "Self-check output" {
    $output = & $exePath --packaging-self-check 2>&1
    $json = $output | ConvertFrom-Json
    $json.status -eq "ok"
}

# 3. Self-check: frozen=True
Test-Check "frozen=True" {
    $output = & $exePath --packaging-self-check 2>&1
    $json = $output | ConvertFrom-Json
    $json.frozen -eq $true
}

# 4. Self-check: Qt available
Test-Check "Qt available" {
    $output = & $exePath --packaging-self-check 2>&1
    $json = $output | ConvertFrom-Json
    $json.qt -eq $true
}

# 5. Self-check: Torch available
Test-Check "Torch available" {
    $output = & $exePath --packaging-self-check 2>&1
    $json = $output | ConvertFrom-Json
    $json.torch -ne $null -and $json.torch -ne ""
}

# 6. Self-check: Ultralytics available
Test-Check "Ultralytics available" {
    $output = & $exePath --packaging-self-check 2>&1
    $json = $output | ConvertFrom-Json
    $json.ultralytics -ne $null -and $json.ultralytics -ne ""
}

# 7. Self-check: Paramiko available
Test-Check "Paramiko available" {
    $output = & $exePath --packaging-self-check 2>&1
    $json = $output | ConvertFrom-Json
    $json.paramiko -ne "None" -and $json.paramiko -ne $null
}

# 8. Self-check: OpenCV available
Test-Check "OpenCV available" {
    $output = & $exePath --packaging-self-check 2>&1
    $json = $output | ConvertFrom-Json
    $json.opencv -ne $null -and $json.opencv -ne ""
}

# 9. Self-check: worker_resource=True
Test-Check "Worker resource present" {
    $output = & $exePath --packaging-self-check 2>&1
    $json = $output | ConvertFrom-Json
    $json.worker_resource -eq $true
}

# 10. Self-check: userdata_writable=True
Test-Check "User data writable" {
    $output = & $exePath --packaging-self-check 2>&1
    $json = $output | ConvertFrom-Json
    $json.userdata_writable -eq $true
}

# 11. EXE does not depend on source Python
Test-Check "No source Python dep" {
    $deps = & $exePath --packaging-self-check 2>&1
    -not ($deps -match "x-anylabeling")
}

# 12. No leftover zombie process
Test-Check "Clean exit" {
    $proc = Start-Process -FilePath $exePath -ArgumentList "--packaging-self-check" -NoNewWindow -Wait -PassThru
    Start-Sleep -Seconds 1
    $proc.HasExited -and $proc.ExitCode -eq 0
}

# 13. Log file written
Test-Check "Application log written" {
    $logDir = "$env:LOCALAPPDATA\TrainLens\logs"
    Test-Path $logDir
}

# Summary
Write-Host ""
Write-Host "=== Results: $passed passed, $failed failed ===" -ForegroundColor $(if ($failed -eq 0) { "Green" } else { "Red" })

if ($failed -gt 0) {
    Write-Host ""
    Write-Host "Known limitations:" -ForegroundColor Yellow
    Write-Host "  - Paramiko host key checking requires first GUI connection"
    Write-Host "  - GPU NOT expected in CPU-only frozen build (use external runtime)"
    exit 1
} else {
    Write-Host "All smoke tests passed!" -ForegroundColor Green
    exit 0
}
