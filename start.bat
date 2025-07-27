@echo off
REM PIPER - AI Classroom Analyzer Windows Startup Script
REM This script provides easy startup for Windows users

echo ======================================================================
echo 🎯 PIPER - AI Classroom Analyzer (Windows)
echo ======================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo ✅ Python found
python --version

REM Check if virtual environment exists
if not exist ".venv" (
    echo 🔄 Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ❌ Failed to create virtual environment
        pause
        exit /b 1
    )
    echo ✅ Virtual environment created
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ Failed to activate virtual environment
    pause
    exit /b 1
)

REM Check if requirements are installed
python -c "import cv2, mediapipe, face_recognition" >nul 2>&1
if errorlevel 1 (
    echo 🔄 Installing requirements...
    python tools\install.py
    if errorlevel 1 (
        echo ❌ Installation failed
        echo Please check the error messages above
        pause
        exit /b 1
    )
)

echo ✅ All requirements satisfied
echo.

REM Ask user which application to run
echo 🎯 Choose Application:
echo 1. Main Application (Recommended)
echo 2. Full AI Suite
echo 3. System Test
echo 4. Requirements Check
echo.

set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo 🚀 Starting Main Application...
    python main_app.py
) else if "%choice%"=="2" (
    echo 🚀 Starting Full AI Suite...
    python run_app.py
) else if "%choice%"=="3" (
    echo 🧪 Running System Test...
    python tests\test_system.py
) else if "%choice%"=="4" (
    echo 🔍 Checking Requirements...
    python tools\check_requirements.py
) else (
    echo 🚀 Starting Main Application (default)...
    python main_app.py
)

echo.
echo 👋 Application finished
pause
