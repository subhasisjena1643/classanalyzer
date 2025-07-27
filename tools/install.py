#!/usr/bin/env python3
"""
PIPER - AI Classroom Analyzer Installation Script
Automated installation and setup for all dependencies
"""

import os
import sys
import subprocess
import platform
import importlib.util

def print_header():
    """Print installation header"""
    print("=" * 70)
    print("🎯 PIPER - AI Classroom Analyzer Installation")
    print("=" * 70)
    print("This script will install all required dependencies for the AI system.")
    print()

def check_python_version():
    """Check if Python version is compatible"""
    print("🔍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Error: Python 3.8+ is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        sys.exit(1)
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")

def check_system():
    """Check system compatibility"""
    print("\n🔍 Checking system compatibility...")
    system = platform.system()
    print(f"✅ Operating System: {system}")
    
    if system == "Windows":
        print("✅ Windows detected - Full compatibility")
    elif system == "Linux":
        print("✅ Linux detected - Full compatibility")
    elif system == "Darwin":
        print("✅ macOS detected - Full compatibility")
    else:
        print(f"⚠️  Unknown system: {system} - May have compatibility issues")

def check_gpu():
    """Check for GPU availability"""
    print("\n🔍 Checking GPU availability...")
    try:
        import subprocess
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected - CUDA acceleration available")
            return True
        else:
            print("ℹ️  No NVIDIA GPU detected - Will use CPU processing")
            return False
    except FileNotFoundError:
        print("ℹ️  nvidia-smi not found - Will use CPU processing")
        return False

def install_requirements_individually():
    """Install requirements one by one if batch installation fails"""
    print("🔄 Installing critical packages individually...")

    critical_packages = [
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "mediapipe>=0.8.0",
        "face-recognition>=1.3.0",
        "dlib>=19.22.0",
        "pillow>=8.0.0",
        "matplotlib>=3.3.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "pyyaml>=5.4.0",
        "loguru>=0.5.0",
    ]

    failed_packages = []
    for package in critical_packages:
        try:
            print(f"  Installing {package}...")
            subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
        except subprocess.CalledProcessError:
            print(f"  ❌ Failed to install {package}")
            failed_packages.append(package)

    if failed_packages:
        print(f"⚠️  Some packages failed to install: {failed_packages}")
        return False
    return True

def install_requirements(gpu_available=False):
    """Install Python requirements"""
    print("\n📦 Installing Python dependencies...")

    # Upgrade pip first
    print("🔄 Upgrading pip...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        print("✅ pip upgraded successfully")
    except subprocess.CalledProcessError as e:
        print(f"⚠️  pip upgrade failed: {e}")

    # Install PyTorch with appropriate backend
    if gpu_available:
        print("🔄 Installing PyTorch with CUDA support...")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install",
                "torch", "torchvision", "torchaudio",
                "--index-url", "https://download.pytorch.org/whl/cu121"
            ], check=True)
            print("✅ PyTorch with CUDA installed successfully")
        except subprocess.CalledProcessError:
            print("⚠️  CUDA PyTorch installation failed, falling back to CPU version...")
            subprocess.run([
                sys.executable, "-m", "pip", "install",
                "torch", "torchvision", "torchaudio"
            ])
    else:
        print("🔄 Installing PyTorch (CPU version)...")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install",
                "torch", "torchvision", "torchaudio"
            ], check=True)
            print("✅ PyTorch (CPU) installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ PyTorch installation failed: {e}")
            return False

    # Install main requirements with error handling
    print("🔄 Installing main requirements...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✅ All requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Requirements installation failed: {e}")
        print("🔄 Trying to install requirements individually...")
        return install_requirements_individually()

def verify_installation():
    """Verify that key packages are installed correctly"""
    print("\n🔍 Verifying installation...")
    
    packages_to_check = [
        ("cv2", "OpenCV"),
        ("mediapipe", "MediaPipe"),
        ("face_recognition", "Face Recognition"),
        ("torch", "PyTorch"),
        ("tensorflow", "TensorFlow"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("sklearn", "Scikit-learn")
    ]
    
    failed_packages = []
    
    for package_name, display_name in packages_to_check:
        try:
            spec = importlib.util.find_spec(package_name)
            if spec is not None:
                print(f"✅ {display_name} - OK")
            else:
                print(f"❌ {display_name} - Not found")
                failed_packages.append(display_name)
        except ImportError:
            print(f"❌ {display_name} - Import error")
            failed_packages.append(display_name)
    
    return len(failed_packages) == 0, failed_packages

def test_camera():
    """Test camera availability"""
    print("\n📹 Testing camera availability...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("✅ Camera test successful")
                cap.release()
                return True
            else:
                print("⚠️  Camera detected but unable to read frames")
                cap.release()
                return False
        else:
            print("⚠️  No camera detected or camera in use")
            return False
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating necessary directories...")
    
    directories = [
        "checkpoints",
        "logs",
        "debug",
        "exports"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")
        else:
            print(f"ℹ️  Directory already exists: {directory}")

def print_success_message():
    """Print success message with next steps"""
    print("\n" + "=" * 70)
    print("🎉 INSTALLATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print()
    print("🚀 Next Steps:")
    print("1. Run the main application:")
    print("   python main_app.py")
    print()
    print("2. Or run the full AI suite:")
    print("   python run_app.py")
    print()
    print("3. For enhanced tracking only:")
    print("   python scripts/start_enhanced_tracking.py")
    print()
    print("📖 For detailed usage instructions, see README.md")
    print()
    print("🎯 The system is ready for AI-powered classroom monitoring!")
    print("=" * 70)

def main():
    """Main installation function"""
    print_header()
    
    try:
        # System checks
        check_python_version()
        check_system()
        gpu_available = check_gpu()
        
        # Installation
        install_requirements(gpu_available)
        
        # Verification
        success, failed_packages = verify_installation()
        
        if not success:
            print(f"\n❌ Installation verification failed for: {', '.join(failed_packages)}")
            print("Please check the error messages above and try running:")
            print("pip install -r requirements.txt")
            sys.exit(1)
        
        # Additional setup
        create_directories()
        camera_ok = test_camera()
        
        if not camera_ok:
            print("\n⚠️  Camera test failed - you may need to:")
            print("   - Connect a camera")
            print("   - Grant camera permissions")
            print("   - Close other applications using the camera")
        
        print_success_message()
        
    except KeyboardInterrupt:
        print("\n\n❌ Installation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Installation failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
