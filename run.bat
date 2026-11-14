@echo off
title VisionVoice AI - Object Detection & Voice Assistant
echo =====================================================================
echo  AI Object Detection & Voice Assistant for Visually Impaired People
echo  Hackathon Edition - Automated Launch
echo =====================================================================
echo.

:: Detect Python
set PYTHON_CMD=python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    py --version >nul 2>&1
    if %errorlevel% equ 0 (
        set PYTHON_CMD=py
    ) else (
        echo [!] Python is not in PATH. Checking common Windows installation directories...
        if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
            set PYTHON_CMD="%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
        ) else if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" (
            set PYTHON_CMD="%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
        ) else if exist "C:\Python311\python.exe" (
            set PYTHON_CMD="C:\Python311\python.exe"
        ) else if exist "C:\Python310\python.exe" (
            set PYTHON_CMD="C:\Python310\python.exe"
        ) else (
            echo [X] Python installation not found. Please install Python 3.10+ from python.org
            pause
            exit /b 1
        )
    )
)

echo [*] Using Python: %PYTHON_CMD%
echo.

:: Generate Test Assets
echo [*] Generating synthetic test assets for TC01 - TC05...
%PYTHON_CMD% create_test_assets.py

echo.
echo [*] Starting Streamlit Application...
%PYTHON_CMD% -m streamlit run app.py --server.port 8501 --browser.gatherUsageStats false

pause
