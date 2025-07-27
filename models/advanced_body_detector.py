"""
Advanced Body Detector - Industry-grade body tracking and posture analysis
Enhanced from previous version with improved accuracy and real-time performance
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Tuple, Optional, Any
from loguru import logger
import time
from sklearn.ensemble import IsolationForest
from collections import deque


class AdvancedBodyDetector:
    """
    Industry-grade body tracking system for comprehensive posture analysis.
    Features:
    - Full body pose estimation with 33 landmarks
    - Movement pattern detection
    - Posture classification (engaged, disengaged, distracted)
    - Real-time performance optimization
    """
    
    def __init__(self, config: Any = None):
        """Initialize advanced body detector."""
        self.config = config
        
        # MediaPipe pose estimation
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # Balance between accuracy and speed
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Movement tracking
        self.movement_history = deque(maxlen=30)  # 1 second at 30 FPS
        self.posture_history = deque(maxlen=60)   # 2 seconds at 30 FPS
        
        # Anomaly detection for unusual movements
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.is_anomaly_detector_trained = False
        
        # Performance tracking
        self.processing_times = []
        self.frame_count = 0
        
        # Posture classification thresholds
        self.engagement_thresholds = {
            'spine_straightness': 0.7,
            'shoulder_alignment': 0.8,
            'head_position': 0.6,
            'movement_stability': 0.5
        }
        
        logger.info("Advanced Body Detector initialized")
    
    def detect_body_pose(self, frame: np.ndarray, person_bbox: List[int] = None) -> Dict[str, Any]:
        """
        Detect and analyze body pose for engagement assessment.
        
        Args:
            frame: Input frame
            person_bbox: Optional bounding box to focus on specific person
            
        Returns:
            Comprehensive body analysis results
        """
        start_time = time.time()
        
        try:
            # Extract person region if bbox provided
            if person_bbox:
                x1, y1, x2, y2 = person_bbox
                person_frame = frame[y1:y2, x1:x2]
                offset = (x1, y1)
            else:
                person_frame = frame
                offset = (0, 0)
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(person_frame, cv2.COLOR_BGR2RGB)
            
            # Process pose
            results = self.pose.process(rgb_frame)
            
            if not results.pose_landmarks:
                return self._create_empty_result()
            
            # Extract landmarks
            landmarks = self._extract_landmarks(results.pose_landmarks, person_frame.shape, offset)
            
            # Analyze posture
            posture_analysis = self._analyze_posture(landmarks)
            
            # Detect movement patterns
            movement_analysis = self._analyze_movement(landmarks)
            
            # Calculate engagement score
            engagement_score = self._calculate_body_engagement_score(posture_analysis, movement_analysis)
            
            # Update history
            self._update_history(landmarks, posture_analysis)
            
            # Track performance
            processing_time = (time.time() - start_time) * 1000
            self.processing_times.append(processing_time)
            self.frame_count += 1
            
            return {
                'landmarks': landmarks,
                'posture_analysis': posture_analysis,
                'movement_analysis': movement_analysis,
                'engagement_score': engagement_score,
                'processing_time_ms': processing_time,
                'confidence': results.pose_landmarks.landmark[0].visibility if results.pose_landmarks else 0.0
            }
            
        except Exception as e:
            logger.error(f"Body pose detection failed: {e}")
            return self._create_empty_result()
    
    def _extract_landmarks(self, pose_landmarks, frame_shape: Tuple, offset: Tuple) -> Dict[str, Dict]:
        """Extract and normalize pose landmarks."""
        h, w = frame_shape[:2]
        offset_x, offset_y = offset
        
        landmarks = {}
        
        # Key body landmarks for engagement analysis
        key_landmarks = {
            'nose': 0,
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26
        }
        
        for name, idx in key_landmarks.items():
            if idx < len(pose_landmarks.landmark):
                landmark = pose_landmarks.landmark[idx]
                landmarks[name] = {
                    'x': landmark.x * w + offset_x,
                    'y': landmark.y * h + offset_y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                }
        
        return landmarks
    
    def _analyze_posture(self, landmarks: Dict) -> Dict[str, Any]:
        """Analyze posture for engagement indicators."""
        try:
            posture_metrics = {}
            
            # 1. Spine straightness
            if all(k in landmarks for k in ['nose', 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']):
                spine_straightness = self._calculate_spine_straightness(landmarks)
                posture_metrics['spine_straightness'] = spine_straightness
            
            # 2. Shoulder alignment
            if 'left_shoulder' in landmarks and 'right_shoulder' in landmarks:
                shoulder_alignment = self._calculate_shoulder_alignment(landmarks)
                posture_metrics['shoulder_alignment'] = shoulder_alignment
            
            # 3. Head position relative to body
            if all(k in landmarks for k in ['nose', 'left_shoulder', 'right_shoulder']):
                head_position = self._calculate_head_position(landmarks)
                posture_metrics['head_position'] = head_position
            
            # 4. Arm position analysis
            arm_analysis = self._analyze_arm_positions(landmarks)
            posture_metrics.update(arm_analysis)
            
            # 5. Overall posture classification
            posture_class = self._classify_posture(posture_metrics)
            posture_metrics['posture_class'] = posture_class
            
            return posture_metrics
            
        except Exception as e:
            logger.error(f"Posture analysis failed: {e}")
            return {}
    
    def _calculate_spine_straightness(self, landmarks: Dict) -> float:
        """Calculate spine straightness metric."""
        try:
            # Get center points
            shoulder_center = (
                (landmarks['left_shoulder']['x'] + landmarks['right_shoulder']['x']) / 2,
                (landmarks['left_shoulder']['y'] + landmarks['right_shoulder']['y']) / 2
            )
            
            hip_center = (
                (landmarks['left_hip']['x'] + landmarks['right_hip']['x']) / 2,
                (landmarks['left_hip']['y'] + landmarks['right_hip']['y']) / 2
            )
            
            nose_pos = (landmarks['nose']['x'], landmarks['nose']['y'])
            
            # Calculate spine angle deviation from vertical
            spine_vector = (hip_center[0] - shoulder_center[0], hip_center[1] - shoulder_center[1])
            spine_angle = np.arctan2(abs(spine_vector[0]), abs(spine_vector[1]))
            
            # Normalize to 0-1 (straighter = higher score)
            straightness = max(0, 1 - (spine_angle / (np.pi / 4)))  # Max deviation 45 degrees
            
            return straightness
            
        except Exception as e:
            logger.error(f"Spine straightness calculation failed: {e}")
            return 0.0
    
    def _calculate_shoulder_alignment(self, landmarks: Dict) -> float:
        """Calculate shoulder alignment metric."""
        try:
            left_shoulder = landmarks['left_shoulder']
            right_shoulder = landmarks['right_shoulder']
            
            # Calculate shoulder tilt
            y_diff = abs(left_shoulder['y'] - right_shoulder['y'])
            x_diff = abs(left_shoulder['x'] - right_shoulder['x'])
            
            if x_diff == 0:
                return 1.0
            
            tilt_angle = np.arctan(y_diff / x_diff)
            
            # Normalize (less tilt = better alignment)
            alignment = max(0, 1 - (tilt_angle / (np.pi / 6)))  # Max acceptable tilt 30 degrees
            
            return alignment
            
        except Exception as e:
            logger.error(f"Shoulder alignment calculation failed: {e}")
            return 0.0
    
    def _calculate_head_position(self, landmarks: Dict) -> float:
        """Calculate head position relative to shoulders."""
        try:
            nose = landmarks['nose']
            shoulder_center_x = (landmarks['left_shoulder']['x'] + landmarks['right_shoulder']['x']) / 2
            
            # Calculate horizontal deviation of head from shoulder center
            head_deviation = abs(nose['x'] - shoulder_center_x)
            shoulder_width = abs(landmarks['left_shoulder']['x'] - landmarks['right_shoulder']['x'])
            
            if shoulder_width == 0:
                return 1.0
            
            # Normalize deviation
            normalized_deviation = head_deviation / shoulder_width
            
            # Good head position = low deviation
            head_position_score = max(0, 1 - normalized_deviation)
            
            return head_position_score
            
        except Exception as e:
            logger.error(f"Head position calculation failed: {e}")
            return 0.0
    
    def _analyze_arm_positions(self, landmarks: Dict) -> Dict[str, Any]:
        """Analyze arm positions for engagement signals."""
        try:
            arm_analysis = {
                'left_arm_raised': False,
                'right_arm_raised': False,
                'both_arms_raised': False,
                'arms_crossed': False,
                'hand_on_face': False
            }
            
            # Check for raised arms (hand raising for participation)
            if all(k in landmarks for k in ['left_shoulder', 'left_elbow', 'left_wrist']):
                left_raised = self._is_arm_raised(landmarks, 'left')
                arm_analysis['left_arm_raised'] = left_raised
            
            if all(k in landmarks for k in ['right_shoulder', 'right_elbow', 'right_wrist']):
                right_raised = self._is_arm_raised(landmarks, 'right')
                arm_analysis['right_arm_raised'] = right_raised
            
            arm_analysis['both_arms_raised'] = arm_analysis['left_arm_raised'] and arm_analysis['right_arm_raised']
            
            # Check for arms crossed (defensive posture)
            if all(k in landmarks for k in ['left_wrist', 'right_wrist', 'left_shoulder', 'right_shoulder']):
                arm_analysis['arms_crossed'] = self._are_arms_crossed(landmarks)
            
            # Check for hand on face (thinking/confusion gesture)
            if all(k in landmarks for k in ['left_wrist', 'right_wrist', 'nose']):
                arm_analysis['hand_on_face'] = self._is_hand_on_face(landmarks)
            
            return arm_analysis
            
        except Exception as e:
            logger.error(f"Arm position analysis failed: {e}")
            return {}
    
    def _is_arm_raised(self, landmarks: Dict, side: str) -> bool:
        """Check if arm is raised (for participation detection)."""
        try:
            shoulder = landmarks[f'{side}_shoulder']
            elbow = landmarks[f'{side}_elbow']
            wrist = landmarks[f'{side}_wrist']
            
            # Arm is raised if wrist is above shoulder
            if wrist['y'] < shoulder['y'] - 20:  # 20 pixel threshold
                return True
            
            # Also check if elbow is significantly raised
            if elbow['y'] < shoulder['y'] - 10:
                return True
            
            return False
            
        except Exception as e:
            return False
    
    def _are_arms_crossed(self, landmarks: Dict) -> bool:
        """Check if arms are crossed (defensive posture)."""
        try:
            left_wrist = landmarks['left_wrist']
            right_wrist = landmarks['right_wrist']
            left_shoulder = landmarks['left_shoulder']
            right_shoulder = landmarks['right_shoulder']
            
            # Arms crossed if wrists are on opposite sides
            left_wrist_on_right = left_wrist['x'] > (left_shoulder['x'] + right_shoulder['x']) / 2
            right_wrist_on_left = right_wrist['x'] < (left_shoulder['x'] + right_shoulder['x']) / 2
            
            return left_wrist_on_right and right_wrist_on_left
            
        except Exception as e:
            return False
    
    def _is_hand_on_face(self, landmarks: Dict) -> bool:
        """Check if hand is on face (thinking/confusion gesture)."""
        try:
            nose = landmarks['nose']
            left_wrist = landmarks['left_wrist']
            right_wrist = landmarks['right_wrist']
            
            # Check distance from wrists to face
            face_threshold = 50  # pixels
            
            left_distance = np.sqrt((left_wrist['x'] - nose['x'])**2 + (left_wrist['y'] - nose['y'])**2)
            right_distance = np.sqrt((right_wrist['x'] - nose['x'])**2 + (right_wrist['y'] - nose['y'])**2)
            
            return left_distance < face_threshold or right_distance < face_threshold
            
        except Exception as e:
            return False
    
    def _analyze_movement(self, landmarks: Dict) -> Dict[str, Any]:
        """Analyze movement patterns for engagement assessment."""
        try:
            movement_analysis = {
                'movement_magnitude': 0.0,
                'movement_stability': 1.0,
                'fidgeting_detected': False,
                'movement_pattern': 'stable'
            }
            
            if len(self.movement_history) < 2:
                self.movement_history.append(landmarks)
                return movement_analysis
            
            # Calculate movement magnitude
            prev_landmarks = self.movement_history[-1]
            movement_magnitude = self._calculate_movement_magnitude(landmarks, prev_landmarks)
            movement_analysis['movement_magnitude'] = movement_magnitude
            
            # Analyze movement stability over time
            if len(self.movement_history) >= 10:
                stability = self._calculate_movement_stability()
                movement_analysis['movement_stability'] = stability
                
                # Detect fidgeting (high frequency, low amplitude movements)
                fidgeting = self._detect_fidgeting()
                movement_analysis['fidgeting_detected'] = fidgeting
                
                # Classify movement pattern
                pattern = self._classify_movement_pattern(movement_magnitude, stability, fidgeting)
                movement_analysis['movement_pattern'] = pattern
            
            # Update movement history
            self.movement_history.append(landmarks)
            
            return movement_analysis
            
        except Exception as e:
            logger.error(f"Movement analysis failed: {e}")
            return {}
    
    def _calculate_movement_magnitude(self, current: Dict, previous: Dict) -> float:
        """Calculate overall movement magnitude between frames."""
        try:
            total_movement = 0.0
            landmark_count = 0
            
            for landmark_name in current:
                if landmark_name in previous:
                    curr = current[landmark_name]
                    prev = previous[landmark_name]
                    
                    # Calculate Euclidean distance
                    movement = np.sqrt((curr['x'] - prev['x'])**2 + (curr['y'] - prev['y'])**2)
                    total_movement += movement
                    landmark_count += 1
            
            return total_movement / max(1, landmark_count)
            
        except Exception as e:
            return 0.0
    
    def _calculate_movement_stability(self) -> float:
        """Calculate movement stability over recent history."""
        try:
            if len(self.movement_history) < 5:
                return 1.0
            
            movements = []
            for i in range(1, len(self.movement_history)):
                movement = self._calculate_movement_magnitude(
                    self.movement_history[i], 
                    self.movement_history[i-1]
                )
                movements.append(movement)
            
            # Stability is inverse of movement variance
            if len(movements) > 1:
                movement_variance = np.var(movements)
                stability = 1.0 / (1.0 + movement_variance)
                return min(1.0, stability)
            
            return 1.0
            
        except Exception as e:
            return 1.0
    
    def _detect_fidgeting(self) -> bool:
        """Detect fidgeting behavior (high frequency, small movements)."""
        try:
            if len(self.movement_history) < 10:
                return False
            
            # Calculate movement frequency and amplitude
            movements = []
            for i in range(1, len(self.movement_history)):
                movement = self._calculate_movement_magnitude(
                    self.movement_history[i], 
                    self.movement_history[i-1]
                )
                movements.append(movement)
            
            # Fidgeting: consistent small movements
            avg_movement = np.mean(movements)
            movement_consistency = 1.0 - np.std(movements) / max(avg_movement, 1.0)
            
            # Thresholds for fidgeting detection
            is_fidgeting = (
                avg_movement > 2.0 and  # Some movement
                avg_movement < 10.0 and  # But not large movements
                movement_consistency > 0.7  # Consistent pattern
            )
            
            return is_fidgeting
            
        except Exception as e:
            return False
    
    def _classify_movement_pattern(self, magnitude: float, stability: float, fidgeting: bool) -> str:
        """Classify overall movement pattern."""
        if fidgeting:
            return 'fidgeting'
        elif magnitude < 2.0 and stability > 0.8:
            return 'stable'
        elif magnitude > 10.0:
            return 'active'
        elif stability < 0.5:
            return 'restless'
        else:
            return 'normal'
    
    def _classify_posture(self, posture_metrics: Dict) -> str:
        """Classify overall posture for engagement assessment."""
        try:
            # Calculate weighted engagement score
            weights = {
                'spine_straightness': 0.3,
                'shoulder_alignment': 0.2,
                'head_position': 0.3,
                'left_arm_raised': 0.1,
                'right_arm_raised': 0.1
            }
            
            score = 0.0
            total_weight = 0.0
            
            for metric, weight in weights.items():
                if metric in posture_metrics:
                    value = posture_metrics[metric]
                    if isinstance(value, bool):
                        value = 1.0 if value else 0.0
                    score += value * weight
                    total_weight += weight
            
            if total_weight > 0:
                normalized_score = score / total_weight
            else:
                normalized_score = 0.5
            
            # Classify based on score
            if normalized_score >= 0.7:
                return 'highly_engaged'
            elif normalized_score >= 0.5:
                return 'engaged'
            elif normalized_score >= 0.3:
                return 'neutral'
            else:
                return 'disengaged'
                
        except Exception as e:
            logger.error(f"Posture classification failed: {e}")
            return 'unknown'
    
    def _calculate_body_engagement_score(self, posture_analysis: Dict, movement_analysis: Dict) -> float:
        """Calculate overall body engagement score."""
        try:
            # Posture component (60% weight)
            posture_score = 0.0
            posture_metrics = ['spine_straightness', 'shoulder_alignment', 'head_position']
            
            for metric in posture_metrics:
                if metric in posture_analysis:
                    posture_score += posture_analysis[metric]
            
            posture_score = posture_score / len(posture_metrics) if posture_metrics else 0.0
            
            # Movement component (40% weight)
            movement_score = movement_analysis.get('movement_stability', 0.5)
            
            # Bonus for participation signals
            participation_bonus = 0.0
            if posture_analysis.get('left_arm_raised', False) or posture_analysis.get('right_arm_raised', False):
                participation_bonus = 0.2
            
            # Penalty for negative indicators
            penalty = 0.0
            if posture_analysis.get('arms_crossed', False):
                penalty += 0.1
            if movement_analysis.get('fidgeting_detected', False):
                penalty += 0.1
            
            # Combine scores
            final_score = (posture_score * 0.6 + movement_score * 0.4 + participation_bonus - penalty)
            
            return max(0.0, min(1.0, final_score))
            
        except Exception as e:
            logger.error(f"Body engagement score calculation failed: {e}")
            return 0.0
    
    def _update_history(self, landmarks: Dict, posture_analysis: Dict):
        """Update historical data for trend analysis."""
        self.posture_history.append({
            'timestamp': time.time(),
            'landmarks': landmarks,
            'posture_analysis': posture_analysis
        })
    
    def _create_empty_result(self) -> Dict[str, Any]:
        """Create empty result when detection fails."""
        return {
            'landmarks': {},
            'posture_analysis': {},
            'movement_analysis': {},
            'engagement_score': 0.0,
            'processing_time_ms': 0.0,
            'confidence': 0.0
        }
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if not self.processing_times:
            return {}
        
        return {
            'avg_processing_time_ms': np.mean(self.processing_times),
            'min_processing_time_ms': np.min(self.processing_times),
            'max_processing_time_ms': np.max(self.processing_times),
            'fps': 1000 / np.mean(self.processing_times) if self.processing_times else 0,
            'total_frames': self.frame_count
        }
    
    def get_engagement_trends(self, window_size: int = 30) -> Dict[str, List[float]]:
        """Get engagement trends over recent history."""
        try:
            if len(self.posture_history) < window_size:
                return {}
            
            recent_history = list(self.posture_history)[-window_size:]
            
            trends = {
                'spine_straightness': [],
                'shoulder_alignment': [],
                'head_position': [],
                'engagement_scores': []
            }
            
            for entry in recent_history:
                posture = entry['posture_analysis']
                for metric in trends:
                    if metric in posture:
                        trends[metric].append(posture[metric])
            
            return trends
            
        except Exception as e:
            logger.error(f"Engagement trends calculation failed: {e}")
            return {}
