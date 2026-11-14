# VisionVoice AI - PowerShell Startup Script
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host " AI Object Detection & Voice Assistant for Visually Impaired People" -ForegroundColor Green
Write-Host " Hackathon Edition - Automated Launch" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""

# Detect Python Command
$pythonCmd = "python"
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $pythonCmd = "py"
    } else {
        $commonPaths = @(
            "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
            "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe",
            "C:\Python311\python.exe",
            "C:\Python310\python.exe"
        )
        $found = $false
        foreach ($p in $commonPaths) {
            if (Test-Path $p) {
                $pythonCmd = "& `"$p`""
                $found = $true
                break
            }
        }
        if (-not $found) {
            Write-Host "[X] Python not found. Please install Python 3.10+ from python.org" -ForegroundColor Red
            Read-Host "Press Enter to exit"
            exit 1
        }
    }
}

Write-Host "[*] Generating synthetic test assets for TC01 - TC05..." -ForegroundColor Yellow
Invoke-Expression "$pythonCmd create_test_assets.py"

Write-Host "`n[*] Launching Streamlit Application on http://localhost:8501..." -ForegroundColor Green
Invoke-Expression "$pythonCmd -m streamlit run app.py --server.port 8501 --browser.gatherUsageStats false"
