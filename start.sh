#!/bin/bash
# PIPER - AI Classroom Analyzer macOS/Linux Startup Script
# This script provides easy startup for macOS and Linux users

echo "======================================================================"
echo "🎯 PIPER - AI Classroom Analyzer (macOS/Linux)"
echo "======================================================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "❌ Python is not installed"
        echo "Please install Python 3.8+ from https://www.python.org/downloads/"
        echo "Or use your system package manager:"
        echo "  Ubuntu/Debian: sudo apt install python3 python3-pip python3-venv"
        echo "  macOS: brew install python3"
        echo "  CentOS/RHEL: sudo yum install python3 python3-pip"
        read -p "Press Enter to exit..."
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

echo "✅ Python found"
$PYTHON_CMD --version

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Python $PYTHON_VERSION found, but Python 3.8+ is required"
    echo "Please upgrade Python"
    read -p "Press Enter to exit..."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "🔄 Creating virtual environment..."
    $PYTHON_CMD -m venv .venv
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create virtual environment"
        echo "You may need to install python3-venv:"
        echo "  Ubuntu/Debian: sudo apt install python3-venv"
        read -p "Press Enter to exit..."
        exit 1
    fi
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source .venv/bin/activate
if [ $? -ne 0 ]; then
    echo "❌ Failed to activate virtual environment"
    read -p "Press Enter to exit..."
    exit 1
fi

# Check if requirements are installed
python -c "import cv2, mediapipe, face_recognition" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "🔄 Installing requirements..."
    python tools/install.py
    if [ $? -ne 0 ]; then
        echo "❌ Installation failed"
        echo "Please check the error messages above"
        read -p "Press Enter to exit..."
        exit 1
    fi
fi

echo "✅ All requirements satisfied"
echo

# Ask user which application to run
echo "🎯 Choose Application:"
echo "1. Main Application (Recommended)"
echo "2. Full AI Suite"
echo "3. System Test"
echo "4. Requirements Check"
echo

read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo "🚀 Starting Main Application..."
        python main_app.py
        ;;
    2)
        echo "🚀 Starting Full AI Suite..."
        python run_app.py
        ;;
    3)
        echo "🧪 Running System Test..."
        python tests/test_system.py
        ;;
    4)
        echo "🔍 Checking Requirements..."
        python tools/check_requirements.py
        ;;
    *)
        echo "🚀 Starting Main Application (default)..."
        python main_app.py
        ;;
esac

echo
echo "👋 Application finished"
read -p "Press Enter to exit..."
