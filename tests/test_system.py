#!/usr/bin/env python3
"""
PIPER - AI Classroom Analyzer System Test
Comprehensive testing script to verify all components
"""

import sys
import importlib.util
import cv2
import time
import traceback

def print_header():
    """Print test header"""
    print("=" * 70)
    print("🧪 PIPER - AI Classroom Analyzer System Test")
    print("=" * 70)
    print("This script will test all system components and dependencies.")
    print()

def test_python_version():
    """Test Python version compatibility"""
    print("🔍 Testing Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Incompatible (requires 3.8+)")
        return False

def test_package_import(package_name, display_name=None):
    """Test if a package can be imported"""
    if display_name is None:
        display_name = package_name
    
    try:
        spec = importlib.util.find_spec(package_name)
        if spec is not None:
            module = importlib.import_module(package_name)
            version = getattr(module, '__version__', 'Unknown')
            print(f"✅ {display_name} v{version} - OK")
            return True
        else:
            print(f"❌ {display_name} - Not found")
            return False
    except Exception as e:
        print(f"❌ {display_name} - Import error: {e}")
        return False

def test_core_packages():
    """Test core Python packages"""
    print("\n📦 Testing core packages...")
    
    packages = [
        ("numpy", "NumPy"),
        ("cv2", "OpenCV"),
        ("PIL", "Pillow"),
        ("pandas", "Pandas"),
        ("matplotlib", "Matplotlib"),
        ("sklearn", "Scikit-learn"),
        ("yaml", "PyYAML"),
    ]
    
    results = []
    for package, name in packages:
        results.append(test_package_import(package, name))
    
    return all(results)

def test_ai_packages():
    """Test AI/ML packages"""
    print("\n🤖 Testing AI/ML packages...")
    
    packages = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("tensorflow", "TensorFlow"),
        ("mediapipe", "MediaPipe"),
        ("face_recognition", "Face Recognition"),
        ("dlib", "dlib"),
        ("transformers", "Transformers"),
        ("stable_baselines3", "Stable Baselines3"),
    ]
    
    results = []
    for package, name in packages:
        results.append(test_package_import(package, name))
    
    return all(results)

def test_gpu_support():
    """Test GPU support"""
    print("\n🎮 Testing GPU support...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA available - {gpu_count} GPU(s) detected")
            print(f"   Primary GPU: {gpu_name}")
            return True
        else:
            print("ℹ️  CUDA not available - Will use CPU processing")
            return False
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False

def test_camera():
    """Test camera functionality"""
    print("\n📹 Testing camera...")
    
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Camera not accessible")
            return False
        
        ret, frame = cap.read()
        if not ret:
            print("❌ Cannot read from camera")
            cap.release()
            return False
        
        height, width = frame.shape[:2]
        print(f"✅ Camera working - Resolution: {width}x{height}")
        cap.release()
        return True
        
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False

def test_face_detection():
    """Test face detection functionality"""
    print("\n👤 Testing face detection...")
    
    try:
        import mediapipe as mp
        mp_face_detection = mp.solutions.face_detection
        mp_drawing = mp.solutions.drawing_utils
        
        with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
            print("✅ MediaPipe face detection initialized")
            return True
            
    except Exception as e:
        print(f"❌ Face detection test failed: {e}")
        return False

def test_face_recognition():
    """Test face recognition functionality"""
    print("\n🔍 Testing face recognition...")
    
    try:
        import face_recognition
        import numpy as np
        
        # Create a dummy image for testing
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        face_locations = face_recognition.face_locations(dummy_image)
        print("✅ Face recognition library working")
        return True
        
    except Exception as e:
        print(f"❌ Face recognition test failed: {e}")
        return False

def test_model_loading():
    """Test loading of AI models"""
    print("\n🧠 Testing model loading...")
    
    try:
        # Test PyTorch model creation
        import torch
        import torch.nn as nn
        
        model = nn.Linear(10, 1)
        print("✅ PyTorch model creation - OK")
        
        # Test TensorFlow model creation
        import tensorflow as tf
        
        tf_model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(10,))])
        print("✅ TensorFlow model creation - OK")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False

def test_file_structure():
    """Test if required files exist"""
    print("\n📁 Testing file structure...")
    
    required_files = [
        "main_app.py",
        "run_app.py",
        "requirements.txt",
        "README.md",
        "config/app_config.yaml"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path} - Found")
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        return False
    
    return True

def run_comprehensive_test():
    """Run all tests and provide summary"""
    print_header()
    
    tests = [
        ("Python Version", test_python_version),
        ("Core Packages", test_core_packages),
        ("AI/ML Packages", test_ai_packages),
        ("GPU Support", test_gpu_support),
        ("Camera", test_camera),
        ("Face Detection", test_face_detection),
        ("Face Recognition", test_face_recognition),
        ("Model Loading", test_model_loading),
        ("File Structure", test_file_structure),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            traceback.print_exc()
            results[test_name] = False
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
    
    print("-" * 70)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is ready for use.")
        print("\n🚀 You can now run:")
        print("   python main_app.py")
        print("   python run_app.py")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        print("   You may need to install missing dependencies or fix configuration.")
        return False

if __name__ == "__main__":
    import os
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
