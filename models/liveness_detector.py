"""
Advanced Liveness Detection System
Prevents spoofing attacks and ensures real human presence detection
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import mediapipe as mp
from loguru import logger


class LivenessDetector:
    """
    Multi-modal liveness detection system combining:
    - Texture analysis for print/screen attack detection
    - Motion analysis for video replay detection
    - 3D face analysis for mask detection
    - Eye blink detection for basic liveness
    """
    
    def __init__(self, device: torch.device = None, config: Any = None):
        """
        Initialize liveness detector.
        
        Args:
            device: PyTorch device for inference
            config: Configuration object
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        
        # Detection parameters
        self.liveness_threshold = config.get("face_detection.liveness_threshold", 0.7) if config else 0.7
        self.anti_spoofing_enabled = config.get("face_detection.anti_spoofing", True) if config else True
        
        # Initialize MediaPipe for face mesh and eye tracking
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Eye landmarks for blink detection
        self.left_eye_landmarks = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.right_eye_landmarks = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Motion tracking for video replay detection
        self.motion_history = []
        self.max_motion_history = 30  # frames
        
        # Texture analysis for print/screen detection
        self.texture_analyzer = TextureAnalyzer()
        
        # Performance tracking
        self.detection_times = []
        
        logger.info("Liveness detector initialized")
    
    def detect_liveness(self, face_crop: np.ndarray, frame_history: List[np.ndarray] = None) -> Dict[str, Any]:
        """
        Comprehensive liveness detection.
        
        Args:
            face_crop: Face region image
            frame_history: Recent frames for motion analysis
            
        Returns:
            Liveness detection results
        """
        start_time = cv2.getTickCount()
        
        try:
            results = {
                'is_live': False,
                'confidence': 0.0,
                'scores': {},
                'details': {}
            }
            
            # 1. Eye blink detection
            blink_score = self._detect_eye_blinks(face_crop)
            results['scores']['blink'] = blink_score
            
            # 2. Texture analysis (anti-spoofing)
            if self.anti_spoofing_enabled:
                texture_score = self._analyze_texture(face_crop)
                results['scores']['texture'] = texture_score
            else:
                texture_score = 1.0
            
            # 3. Motion analysis
            if frame_history and len(frame_history) > 1:
                motion_score = self._analyze_motion(frame_history)
                results['scores']['motion'] = motion_score
            else:
                motion_score = 0.5  # Neutral score if no history
            
            # 4. 3D face analysis
            face_3d_score = self._analyze_3d_face(face_crop)
            results['scores']['face_3d'] = face_3d_score
            
            # 5. Color analysis for screen detection
            color_score = self._analyze_color_distribution(face_crop)
            results['scores']['color'] = color_score
            
            # Combine scores with weights
            weights = {
                'blink': 0.25,
                'texture': 0.30,
                'motion': 0.20,
                'face_3d': 0.15,
                'color': 0.10
            }
            
            combined_score = (
                weights['blink'] * blink_score +
                weights['texture'] * texture_score +
                weights['motion'] * motion_score +
                weights['face_3d'] * face_3d_score +
                weights['color'] * color_score
            )
            
            results['confidence'] = combined_score
            results['is_live'] = combined_score >= self.liveness_threshold
            
            # Add detailed analysis
            results['details'] = {
                'blink_detected': blink_score > 0.5,
                'texture_natural': texture_score > 0.6,
                'motion_natural': motion_score > 0.4,
                'face_3d_valid': face_3d_score > 0.5,
                'color_natural': color_score > 0.5
            }
            
            # Track performance
            end_time = cv2.getTickCount()
            detection_time = (end_time - start_time) / cv2.getTickFrequency() * 1000
            self.detection_times.append(detection_time)
            
            return results
            
        except Exception as e:
            logger.error(f"Liveness detection failed: {e}")
            return {
                'is_live': False,
                'confidence': 0.0,
                'scores': {},
                'details': {'error': str(e)}
            }
    
    def _detect_eye_blinks(self, face_crop: np.ndarray) -> float:
        """Detect eye blinks using facial landmarks."""
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.face_mesh.process(rgb_image)
            
            if not results.multi_face_landmarks:
                return 0.0
            
            face_landmarks = results.multi_face_landmarks[0]
            
            # Calculate Eye Aspect Ratio (EAR) for both eyes
            left_ear = self._calculate_ear(face_landmarks, self.left_eye_landmarks, face_crop.shape)
            right_ear = self._calculate_ear(face_landmarks, self.right_eye_landmarks, face_crop.shape)
            
            # Average EAR
            avg_ear = (left_ear + right_ear) / 2.0
            
            # EAR threshold for blink detection (lower values indicate closed eyes)
            blink_threshold = 0.25
            
            # Score based on EAR variation (indicates natural blinking)
            if avg_ear < blink_threshold:
                return 1.0  # Eyes closed (blink detected)
            elif avg_ear > 0.3:
                return 0.8  # Eyes open
            else:
                return 0.5  # Intermediate state
            
        except Exception as e:
            logger.error(f"Eye blink detection failed: {e}")
            return 0.0
    
    def _calculate_ear(self, landmarks, eye_landmarks: List[int], image_shape: Tuple) -> float:
        """Calculate Eye Aspect Ratio."""
        try:
            h, w = image_shape[:2]
            
            # Get eye landmark coordinates
            eye_points = []
            for idx in eye_landmarks:
                if idx < len(landmarks.landmark):
                    landmark = landmarks.landmark[idx]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    eye_points.append((x, y))
            
            if len(eye_points) < 6:
                return 0.0
            
            # Calculate vertical distances
            vertical_1 = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
            vertical_2 = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
            
            # Calculate horizontal distance
            horizontal = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
            
            # Calculate EAR
            if horizontal > 0:
                ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
                return ear
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"EAR calculation failed: {e}")
            return 0.0
    
    def _analyze_texture(self, face_crop: np.ndarray) -> float:
        """Analyze texture patterns to detect print/screen attacks."""
        return self.texture_analyzer.analyze(face_crop)
    
    def _analyze_motion(self, frame_history: List[np.ndarray]) -> float:
        """Analyze motion patterns to detect video replay attacks."""
        try:
            if len(frame_history) < 2:
                return 0.5
            
            # Calculate optical flow between consecutive frames
            motion_scores = []
            
            for i in range(1, len(frame_history)):
                prev_frame = cv2.cvtColor(frame_history[i-1], cv2.COLOR_BGR2GRAY)
                curr_frame = cv2.cvtColor(frame_history[i], cv2.COLOR_BGR2GRAY)
                
                # Calculate optical flow
                flow = cv2.calcOpticalFlowPyrLK(
                    prev_frame, curr_frame,
                    np.array([[50, 50]], dtype=np.float32),
                    None
                )
                
                if flow[0] is not None:
                    motion_magnitude = np.linalg.norm(flow[0][0] - np.array([50, 50]))
                    motion_scores.append(motion_magnitude)
            
            if motion_scores:
                # Natural motion should have variation
                motion_variance = np.var(motion_scores)
                # Normalize score (higher variance = more natural)
                return min(1.0, motion_variance / 10.0)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Motion analysis failed: {e}")
            return 0.0
    
    def _analyze_3d_face(self, face_crop: np.ndarray) -> float:
        """Analyze 3D face structure to detect mask attacks."""
        try:
            # Convert to RGB
            rgb_image = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.face_mesh.process(rgb_image)
            
            if not results.multi_face_landmarks:
                return 0.0
            
            face_landmarks = results.multi_face_landmarks[0]
            
            # Calculate face depth variation
            z_coords = [landmark.z for landmark in face_landmarks.landmark]
            z_variance = np.var(z_coords)
            
            # Real faces should have more depth variation than flat images
            # Normalize score (higher variance = more 3D = more real)
            depth_score = min(1.0, z_variance * 1000)  # Scale factor
            
            return depth_score
            
        except Exception as e:
            logger.error(f"3D face analysis failed: {e}")
            return 0.0
    
    def _analyze_color_distribution(self, face_crop: np.ndarray) -> float:
        """Analyze color distribution to detect screen/print attacks."""
        try:
            # Convert to different color spaces
            hsv = cv2.cvtColor(face_crop, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(face_crop, cv2.COLOR_BGR2LAB)
            
            # Analyze color distribution in different channels
            # Real faces have specific color characteristics
            
            # HSV analysis
            h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
            
            # LAB analysis
            a_hist = cv2.calcHist([lab], [1], None, [256], [0, 256])
            b_hist = cv2.calcHist([lab], [2], None, [256], [0, 256])
            
            # Calculate distribution entropy (real faces have more varied colors)
            h_entropy = self._calculate_entropy(h_hist)
            s_entropy = self._calculate_entropy(s_hist)
            a_entropy = self._calculate_entropy(a_hist)
            b_entropy = self._calculate_entropy(b_hist)
            
            # Combine entropies
            avg_entropy = (h_entropy + s_entropy + a_entropy + b_entropy) / 4.0
            
            # Normalize score (higher entropy = more natural)
            return min(1.0, avg_entropy / 5.0)
            
        except Exception as e:
            logger.error(f"Color analysis failed: {e}")
            return 0.0
    
    def _calculate_entropy(self, histogram: np.ndarray) -> float:
        """Calculate entropy of histogram."""
        try:
            # Normalize histogram
            hist_norm = histogram / np.sum(histogram)
            
            # Remove zero entries
            hist_norm = hist_norm[hist_norm > 0]
            
            # Calculate entropy
            entropy = -np.sum(hist_norm * np.log2(hist_norm))
            
            return entropy
            
        except Exception as e:
            return 0.0
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if not self.detection_times:
            return {}
        
        return {
            'avg_detection_time_ms': np.mean(self.detection_times),
            'min_detection_time_ms': np.min(self.detection_times),
            'max_detection_time_ms': np.max(self.detection_times),
            'total_detections': len(self.detection_times)
        }


class TextureAnalyzer:
    """Specialized texture analysis for anti-spoofing."""
    
    def __init__(self):
        """Initialize texture analyzer."""
        self.lbp_radius = 3
        self.lbp_n_points = 8 * self.lbp_radius
    
    def analyze(self, face_crop: np.ndarray) -> float:
        """Analyze texture patterns."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            
            # Calculate Local Binary Pattern (LBP)
            lbp = self._calculate_lbp(gray)
            
            # Calculate texture features
            contrast = self._calculate_contrast(gray)
            homogeneity = self._calculate_homogeneity(gray)
            energy = self._calculate_energy(gray)
            
            # Combine features
            texture_score = (contrast * 0.4 + homogeneity * 0.3 + energy * 0.3)
            
            return min(1.0, texture_score)
            
        except Exception as e:
            logger.error(f"Texture analysis failed: {e}")
            return 0.0
    
    def _calculate_lbp(self, image: np.ndarray) -> np.ndarray:
        """Calculate Local Binary Pattern."""
        try:
            # Simple LBP implementation
            rows, cols = image.shape
            lbp = np.zeros_like(image)
            
            for i in range(1, rows - 1):
                for j in range(1, cols - 1):
                    center = image[i, j]
                    code = 0
                    
                    # Compare with 8 neighbors
                    neighbors = [
                        image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                        image[i, j+1], image[i+1, j+1], image[i+1, j],
                        image[i+1, j-1], image[i, j-1]
                    ]
                    
                    for k, neighbor in enumerate(neighbors):
                        if neighbor >= center:
                            code |= (1 << k)
                    
                    lbp[i, j] = code
            
            return lbp
            
        except Exception as e:
            logger.error(f"LBP calculation failed: {e}")
            return np.zeros_like(image)
    
    def _calculate_contrast(self, image: np.ndarray) -> float:
        """Calculate image contrast."""
        return np.std(image) / 255.0
    
    def _calculate_homogeneity(self, image: np.ndarray) -> float:
        """Calculate image homogeneity."""
        # Calculate gradient magnitude
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        return 1.0 - (np.mean(gradient_magnitude) / 255.0)
    
    def _calculate_energy(self, image: np.ndarray) -> float:
        """Calculate image energy."""
        # Normalize image
        normalized = image.astype(np.float32) / 255.0
        
        # Calculate energy as sum of squared pixel values
        energy = np.sum(normalized**2) / normalized.size
        
        return energy
