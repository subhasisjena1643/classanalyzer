#!/usr/bin/env python3
"""
Enhanced Tracking System Startup Script
Bypasses TensorBoard issues and starts the application with enhanced tracking
"""

import os
import sys
import cv2
import numpy as np
import time
import gc
import torch
from loguru import logger

# Disable TensorBoard warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main():
    """Start the enhanced tracking system."""
    print("🎯 STARTING ENHANCED TRACKING SYSTEM")
    print("=" * 60)
    
    try:
        # Import enhanced tracking components
        print("Loading enhanced tracking components...")
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        from utils.enhanced_tracking_overlay import EnhancedTrackingOverlay
        from utils.comprehensive_analyzer import ComprehensiveAnalyzer
        from utils.automatic_cleanup import AutomaticCleanupManager
        
        print("✅ Enhanced tracking components loaded")
        
        # Initialize components
        tracking_overlay = EnhancedTrackingOverlay()
        comprehensive_analyzer = ComprehensiveAnalyzer({})
        cleanup_manager = AutomaticCleanupManager()
        
        print("✅ Components initialized")
        
        # Try to load AI models (with fallbacks)
        models = {}
        
        try:
            from models.face_detection import StateOfTheArtFaceDetector
            models['face_detector'] = StateOfTheArtFaceDetector(ensemble_mode='speed')
            print("✅ Face detector loaded")
        except Exception as e:
            print(f"⚠️  Face detector not available: {e}")
            models['face_detector'] = None

        try:
            from models.face_recognition import StateOfTheArtFaceRecognizer
            models['face_recognizer'] = StateOfTheArtFaceRecognizer()
            print("✅ Face recognizer loaded")
        except Exception as e:
            print(f"⚠️  Face recognizer not available: {e}")
            models['face_recognizer'] = None
        
        # Start video capture
        print("\n🎥 Starting video capture...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Could not open camera")
            return 1
        
        print("✅ Camera opened successfully")
        print("\n🚀 ENHANCED TRACKING SYSTEM RUNNING!")
        print("Press 'q' to quit, 'r' to reset tracking")
        print("=" * 60)
        
        # Main processing loop
        frame_count = 0
        fps_counter = time.time()
        fps = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break

            frame_count += 1

            # Face detection logic: run detection when needed
            detections = []

            # Always try to detect faces for tracking updates
            try:
                # Use OpenCV Haar Cascade for reliable face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))

                for (x, y, w, h) in faces:
                    detections.append({
                        'bbox': [int(x), int(y), int(x + w), int(y + h)],
                        'confidence': 0.85
                    })

                # Only print when detection mode is active
                if tracking_overlay.detection_active and detections:
                    print(f"🎯 Detected {len(detections)} face(s) - switching to tracking mode")

            except Exception as e:
                logger.error(f"Face detection error: {e}")
                # Fallback: try to use MediaPipe if available
                try:
                    import mediapipe as mp
                    mp_face_detection = mp.solutions.face_detection
                    mp_drawing = mp.solutions.drawing_utils

                    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
                        results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                        if results.detections:
                            h, w, _ = frame.shape
                            for detection in results.detections:
                                bbox = detection.location_data.relative_bounding_box
                                x = int(bbox.xmin * w)
                                y = int(bbox.ymin * h)
                                width = int(bbox.width * w)
                                height = int(bbox.height * h)

                                detections.append({
                                    'bbox': [x, y, x + width, y + height],
                                    'confidence': detection.score[0]
                                })
                except:
                    pass  # MediaPipe not available

            # Real-time analysis results (will be calculated by tracking overlay)
            analysis_results = {
                'timestamp': time.time(),
                'frame_number': frame_count
            }
            
            # Update tracking system
            tracking_overlay.update_tracking(detections, analysis_results)
            
            # Create enhanced display
            display_frame = tracking_overlay.draw_tracking_overlay(frame, fps)
            
            # Add system info
            h, w = display_frame.shape[:2]
            
            # Performance info (top right)
            tracking_stats = tracking_overlay.get_tracking_stats()
            info_texts = [
                f"FPS: {fps:.1f}",
                f"Frame: {frame_count}",
                f"Mode: {'Detection' if tracking_overlay.detection_active else 'Tracking'}",
                f"Tracked: {tracking_stats['tracked_objects']}",
                f"Alerts: {tracking_stats['active_alerts']}"
            ]

            for i, text in enumerate(info_texts):
                pos = (w - 280, 30 + (i * 25))
                cv2.rectangle(display_frame, (pos[0] - 2, pos[1] - 20),
                             (pos[0] + 250, pos[1] + 5), (0, 0, 0), -1)
                cv2.putText(display_frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                           0.6, (255, 255, 255), 1)
            
            # Show frame
            cv2.imshow("Enhanced Tracking System", display_frame)
            
            # Calculate FPS
            if frame_count % 10 == 0:
                current_time = time.time()
                fps = 10 / (current_time - fps_counter)
                fps_counter = current_time
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset tracking
                tracking_overlay = EnhancedTrackingOverlay()
                print("🔄 Tracking reset - detection mode re-enabled")
            elif key == ord('d'):
                # Toggle detection mode
                tracking_overlay.detection_active = not tracking_overlay.detection_active
                print(f"🔄 Detection mode: {'ON' if tracking_overlay.detection_active else 'OFF'}")
            
            # Memory management
            if frame_count % 100 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n✅ Enhanced tracking system stopped successfully")
        return 0
        
    except Exception as e:
        print(f"❌ Error starting enhanced tracking system: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
