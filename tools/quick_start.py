#!/usr/bin/env python3
"""
PIPER - AI Classroom Analyzer Quick Start
One-click setup and launch script
"""

import os
import sys
import subprocess
import importlib.util

def check_installation():
    """Check if the system is already installed"""
    print("🔍 Checking installation status...")
    
    required_packages = ["cv2", "mediapipe", "face_recognition", "torch"]
    missing_packages = []
    
    for package in required_packages:
        try:
            spec = importlib.util.find_spec(package)
            if spec is None:
                missing_packages.append(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        return False
    else:
        print("✅ All required packages are installed")
        return True

def run_installation():
    """Run the installation script"""
    print("\n🚀 Running installation...")
    try:
        subprocess.run([sys.executable, "install.py"], check=True)
        return True
    except subprocess.CalledProcessError:
        print("❌ Installation failed")
        return False
    except FileNotFoundError:
        print("❌ install.py not found")
        return False

def choose_application():
    """Let user choose which application to run"""
    print("\n🎯 Choose Application to Run:")
    print("1. Main Application (Recommended) - main_app.py")
    print("2. Full AI Suite - run_app.py")
    print("3. Enhanced Tracking Only - scripts/start_enhanced_tracking.py")
    print("4. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            if choice in ["1", "2", "3", "4"]:
                return int(choice)
            else:
                print("Please enter a valid choice (1-4)")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            sys.exit(0)

def run_application(choice):
    """Run the selected application"""
    applications = {
        1: ("main_app.py", "Main AI Video Tracking Application"),
        2: ("run_app.py", "Full AI Suite with All Models"),
        3: ("scripts/start_enhanced_tracking.py", "Enhanced Tracking System")
    }
    
    if choice == 4:
        print("👋 Goodbye!")
        sys.exit(0)
    
    script_path, description = applications[choice]
    
    print(f"\n🚀 Starting {description}...")
    print("=" * 60)
    print("Controls:")
    print("  - Press 'q' to quit")
    print("  - Press 'r' to reset tracking")
    print("  - Press 's' to save state")
    print("=" * 60)
    
    try:
        if choice == 3:
            # For scripts subdirectory
            subprocess.run([sys.executable, script_path])
        else:
            subprocess.run([sys.executable, script_path])
    except KeyboardInterrupt:
        print("\n\n✅ Application stopped by user")
    except FileNotFoundError:
        print(f"❌ Error: {script_path} not found")
        print("Please make sure all files are in the correct location")
    except Exception as e:
        print(f"❌ Error running application: {e}")

def main():
    """Main quick start function"""
    print("=" * 60)
    print("🎯 PIPER - AI Classroom Analyzer Quick Start")
    print("=" * 60)
    print("Welcome! This script will help you get started quickly.")
    print()
    
    # Check if already installed
    if not check_installation():
        print("\n📦 Installation required...")
        install_success = run_installation()
        if not install_success:
            print("❌ Installation failed. Please check the error messages above.")
            sys.exit(1)
    
    # Choose and run application
    while True:
        choice = choose_application()
        run_application(choice)
        
        # Ask if user wants to run another application
        print("\n" + "=" * 60)
        try:
            again = input("Would you like to run another application? (y/n): ").strip().lower()
            if again not in ["y", "yes"]:
                break
        except KeyboardInterrupt:
            break
    
    print("\n👋 Thank you for using PIPER - AI Classroom Analyzer!")

if __name__ == "__main__":
    main()
