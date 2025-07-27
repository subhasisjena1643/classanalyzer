#!/usr/bin/env python3
"""
System Requirements Checker for PIPER - AI Classroom Analyzer
Comprehensive check for all system requirements and dependencies
"""

import sys
import platform
import subprocess
import importlib.util
import os
from pathlib import Path

def print_header():
    """Print checker header"""
    print("=" * 70)
    print("🔍 PIPER - System Requirements Checker")
    print("=" * 70)
    print("Checking if your system meets all requirements...")
    print()

def check_python_version():
    """Check Python version"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Incompatible")
        print("   Required: Python 3.8 or higher")
        print("   Please upgrade Python: https://www.python.org/downloads/")
        return False

def check_operating_system():
    """Check operating system compatibility"""
    print("\n💻 Checking operating system...")
    
    system = platform.system()
    version = platform.version()
    architecture = platform.architecture()[0]
    
    print(f"✅ OS: {system} {version}")
    print(f"✅ Architecture: {architecture}")
    
    if system in ["Windows", "Linux", "Darwin"]:
        print("✅ Operating system supported")
        return True
    else:
        print(f"⚠️  Operating system '{system}' may have compatibility issues")
        return False

def check_system_resources():
    """Check system resources"""
    print("\n🖥️  Checking system resources...")
    
    try:
        import psutil
        
        # Memory check
        memory_gb = psutil.virtual_memory().total / (1024**3)
        print(f"💾 RAM: {memory_gb:.1f} GB", end="")
        
        if memory_gb >= 8:
            print(" ✅ Sufficient")
        elif memory_gb >= 4:
            print(" ⚠️  Minimum (may be slow)")
        else:
            print(" ❌ Insufficient (4GB+ recommended)")
            return False
        
        # CPU check
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        print(f"🔧 CPU: {cpu_count} cores", end="")
        
        if cpu_freq:
            print(f" @ {cpu_freq.max:.0f}MHz", end="")
        
        if cpu_count >= 4:
            print(" ✅ Good")
        elif cpu_count >= 2:
            print(" ⚠️  Minimum")
        else:
            print(" ❌ Insufficient")
            return False
        
        # Disk space check
        disk_usage = psutil.disk_usage('.')
        free_gb = disk_usage.free / (1024**3)
        print(f"💽 Free disk space: {free_gb:.1f} GB", end="")
        
        if free_gb >= 5:
            print(" ✅ Sufficient")
        else:
            print(" ❌ Insufficient (5GB+ required)")
            return False
        
        return True
        
    except ImportError:
        print("⚠️  psutil not available - cannot check system resources")
        print("   Install with: pip install psutil")
        return False

def check_gpu_support():
    """Check GPU support"""
    print("\n🎮 Checking GPU support...")
    
    # Check NVIDIA GPU
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected")
            # Extract GPU info
            lines = result.stdout.split('\n')
            for line in lines:
                if 'GeForce' in line or 'RTX' in line or 'GTX' in line:
                    gpu_info = line.strip()
                    print(f"   GPU: {gpu_info}")
                    break
            return True
    except FileNotFoundError:
        pass
    
    # Check for other GPU types
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA GPU available: {gpu_name}")
            return True
    except ImportError:
        pass
    
    print("ℹ️  No GPU detected - will use CPU processing")
    print("   For better performance, consider using a CUDA-compatible GPU")
    return False

def check_camera():
    """Check camera availability"""
    print("\n📹 Checking camera...")
    
    try:
        import cv2
        
        # Try to open default camera
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                height, width = frame.shape[:2]
                print(f"✅ Camera available - Resolution: {width}x{height}")
                cap.release()
                return True
            else:
                print("❌ Camera detected but cannot read frames")
                cap.release()
                return False
        else:
            print("❌ No camera detected")
            print("   Please connect a camera or check camera permissions")
            return False
            
    except ImportError:
        print("❌ OpenCV not available - cannot test camera")
        return False

def check_critical_packages():
    """Check critical Python packages"""
    print("\n📦 Checking critical packages...")
    
    critical_packages = [
        ("cv2", "OpenCV", "opencv-python"),
        ("numpy", "NumPy", "numpy"),
        ("mediapipe", "MediaPipe", "mediapipe"),
        ("face_recognition", "Face Recognition", "face-recognition"),
        ("torch", "PyTorch", "torch"),
        ("tensorflow", "TensorFlow", "tensorflow"),
        ("PIL", "Pillow", "pillow"),
        ("pandas", "Pandas", "pandas"),
        ("sklearn", "Scikit-learn", "scikit-learn"),
        ("yaml", "PyYAML", "pyyaml"),
    ]
    
    missing_packages = []
    
    for package_name, display_name, pip_name in critical_packages:
        try:
            spec = importlib.util.find_spec(package_name)
            if spec is not None:
                module = importlib.import_module(package_name)
                version = getattr(module, '__version__', 'Unknown')
                print(f"✅ {display_name} v{version}")
            else:
                print(f"❌ {display_name} - Not installed")
                missing_packages.append(pip_name)
        except Exception as e:
            print(f"❌ {display_name} - Error: {e}")
            missing_packages.append(pip_name)
    
    if missing_packages:
        print(f"\n📋 Missing packages: {', '.join(missing_packages)}")
        print("   Install with: python install.py")
        return False
    
    return True

def generate_system_report():
    """Generate comprehensive system report"""
    print("\n" + "=" * 70)
    print("📊 SYSTEM COMPATIBILITY REPORT")
    print("=" * 70)
    
    checks = [
        ("Python Version", check_python_version),
        ("Operating System", check_operating_system),
        ("System Resources", check_system_resources),
        ("GPU Support", check_gpu_support),
        ("Camera", check_camera),
        ("Critical Packages", check_critical_packages),
    ]
    
    results = {}
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"❌ {check_name} check failed: {e}")
            results[check_name] = False
    
    # Summary
    print("\n📋 Summary:")
    passed = 0
    critical_failed = []
    
    for check_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {check_name:<20} {status}")
        
        if result:
            passed += 1
        elif check_name in ["Python Version", "Critical Packages"]:
            critical_failed.append(check_name)
    
    total = len(results)
    print(f"\n🎯 Score: {passed}/{total} checks passed")
    
    if critical_failed:
        print(f"\n❌ CRITICAL FAILURES: {', '.join(critical_failed)}")
        print("   System is NOT ready. Please fix critical issues first.")
        return False
    elif passed >= total - 1:  # Allow 1 non-critical failure
        print("\n🎉 SYSTEM IS READY!")
        print("✅ You can run the AI Classroom Analyzer")
        print("\n🚀 Next steps:")
        print("   python main_app.py")
        return True
    else:
        print(f"\n⚠️  {total - passed} issue(s) detected")
        print("   System may work but with reduced performance")
        print("   Consider fixing the issues for optimal experience")
        return False

def main():
    """Main requirements checker"""
    print_header()
    success = generate_system_report()
    
    if not success:
        print("\n🔧 Need help? Check the installation guide:")
        print("   python install.py")
        print("   Or see README.md for detailed instructions")
    
    input("\nPress Enter to exit...")
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
