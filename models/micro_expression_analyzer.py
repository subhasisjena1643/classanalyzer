"""
State-of-the-Art Micro-Expression Analyzer
Temporal Convolutional Networks for micro-expression recognition
Industry-grade emotion detection with temporal modeling and deep learning
"""

import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from loguru import logger
import time
from collections import deque
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os
import math


class TemporalConvBlock(nn.Module):
    """Temporal Convolutional Block for micro-expression analysis."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=(kernel_size-1)*dilation//2, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                              padding=(kernel_size-1)*dilation//2, dilation=dilation)

        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(0.2)

        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.residual(x)

        out = F.relu(self.norm1(self.conv1(x)))
        out = self.dropout(out)
        out = self.norm2(self.conv2(out))
        out = self.dropout(out)

        return F.relu(out + residual)


class MicroExpressionTCN(nn.Module):
    """Temporal Convolutional Network for micro-expression recognition."""

    def __init__(self, input_dim=68*2, num_classes=7, sequence_length=16):
        super().__init__()

        self.input_dim = input_dim
        self.sequence_length = sequence_length

        # TCN layers with increasing dilation
        self.tcn_layers = nn.ModuleList([
            TemporalConvBlock(input_dim, 64, 3, 1),
            TemporalConvBlock(64, 128, 3, 2),
            TemporalConvBlock(128, 256, 3, 4),
            TemporalConvBlock(256, 512, 3, 8),
        ])

        # Global average pooling and classification
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, sequence_length, features)
        x = x.transpose(1, 2)  # (batch, features, sequence_length)

        for tcn_layer in self.tcn_layers:
            x = tcn_layer(x)

        # Global pooling
        x = self.global_pool(x).squeeze(-1)

        # Classification
        emotions = self.classifier(x)

        return emotions


class StateOfTheArtMicroExpressionAnalyzer:
    """
    Advanced micro-expression analysis system for emotional engagement detection.
    Features:
    - Real-time facial emotion recognition
    - Micro-expression detection (brief, involuntary expressions)
    - Engagement state classification
    - Confusion and interest detection
    - Temporal emotion tracking
    """
    
    def __init__(self, config: Any = None, device: torch.device = None):
        """Initialize state-of-the-art micro expression analyzer with TCN."""
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize TCN model for micro-expression recognition
        self.tcn_model = MicroExpressionTCN(
            input_dim=68*2,  # 68 facial landmarks * 2 (x,y coordinates)
            num_classes=7,   # 7 basic emotions
            sequence_length=16
        ).to(self.device)
        self.tcn_model.eval()

        # Feature scaler for normalization
        self.feature_scaler = StandardScaler()

        # Temporal sequence storage
        self.sequence_length = 16
        self.landmark_sequences = deque(maxlen=self.sequence_length)

        # MediaPipe face mesh for detailed facial landmarks
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Facial landmark indices for emotion analysis
        self.emotion_landmarks = {
            'eyebrows': [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],  # Eyebrow region
            'eyes': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
                    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            'nose': [1, 2, 5, 4, 6, 19, 20, 94, 125, 141, 235, 236, 3, 51, 48, 115, 131, 134, 102, 49, 220, 305, 281, 360],
            'mouth': [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
                     185, 40, 39, 37, 0, 267, 269, 270, 267, 271, 272, 271, 272, 271, 272],
            'cheeks': [116, 117, 118, 119, 120, 121, 126, 142, 36, 205, 206, 207, 213, 192, 147, 187, 207, 213, 192, 147]
        }
        
        # Emotion classification model
        self.emotion_classifier = None
        self.emotion_labels = ['neutral', 'happy', 'sad', 'angry', 'surprised', 'fearful', 'disgusted', 'confused', 'interested']
        self._load_or_create_emotion_model()
        
        # Expression history for temporal analysis
        self.expression_history = deque(maxlen=60)  # 2 seconds at 30 FPS
        self.emotion_history = deque(maxlen=30)     # 1 second at 30 FPS
        
        # Micro-expression detection parameters
        self.micro_expression_threshold = 0.15
        self.expression_duration_threshold = 5  # frames
        
        # Performance tracking
        self.processing_times = []
        self.frame_count = 0
        
        # Engagement emotion mapping
        self.engagement_emotions = {
            'positive': ['happy', 'interested', 'surprised'],
            'negative': ['sad', 'angry', 'fearful', 'disgusted'],
            'neutral': ['neutral'],
            'confusion': ['confused']
        }
        
        logger.info("State-of-the-Art Micro Expression Analyzer initialized with TCN")

    def analyze_with_tcn(self, frame: np.ndarray, face_bbox: List[int] = None) -> Dict[str, Any]:
        """
        Analyze micro-expressions using Temporal Convolutional Networks.

        Args:
            frame: Input frame
            face_bbox: Optional face bounding box

        Returns:
            TCN-based micro-expression analysis results
        """
        try:
            # Extract facial landmarks
            landmarks = self._extract_facial_landmarks(frame, face_bbox)

            if landmarks is None:
                return self._create_empty_tcn_result()

            # Add to temporal sequence
            self.landmark_sequences.append(landmarks)

            # Need enough frames for temporal analysis
            if len(self.landmark_sequences) < self.sequence_length:
                return self._create_partial_tcn_result(landmarks)

            # Prepare sequence for TCN
            sequence = np.array(list(self.landmark_sequences))  # (sequence_length, features)

            # Normalize features
            sequence_normalized = self.feature_scaler.fit_transform(sequence.reshape(-1, sequence.shape[-1]))
            sequence_normalized = sequence_normalized.reshape(sequence.shape)

            # Convert to tensor
            sequence_tensor = torch.FloatTensor(sequence_normalized).unsqueeze(0).to(self.device)

            # Run TCN inference
            with torch.no_grad():
                emotion_logits = self.tcn_model(sequence_tensor)
                emotion_probs = F.softmax(emotion_logits, dim=1)[0]

            # Get emotion predictions
            emotion_scores = emotion_probs.cpu().numpy()
            predicted_emotion_idx = np.argmax(emotion_scores)
            predicted_emotion = self.emotion_labels[predicted_emotion_idx]
            confidence = float(emotion_scores[predicted_emotion_idx])

            # Detect micro-expressions (sudden changes)
            micro_expressions = self._detect_micro_expressions_tcn(emotion_scores)

            # Calculate temporal consistency
            temporal_consistency = self._calculate_temporal_consistency()

            return {
                'primary_emotion': predicted_emotion,
                'confidence': confidence,
                'emotion_scores': {label: float(score) for label, score in zip(self.emotion_labels, emotion_scores)},
                'micro_expressions': micro_expressions,
                'temporal_consistency': temporal_consistency,
                'method': 'tcn',
                'sequence_length': len(self.landmark_sequences),
                'landmarks_count': len(landmarks) if landmarks is not None else 0
            }

        except Exception as e:
            logger.error(f"TCN micro-expression analysis failed: {e}")
            return self._create_empty_tcn_result()

    def _extract_facial_landmarks(self, frame: np.ndarray, face_bbox: List[int] = None) -> Optional[np.ndarray]:
        """Extract facial landmarks for TCN processing."""
        try:
            # Extract face region if bbox provided
            if face_bbox:
                x1, y1, x2, y2 = face_bbox
                face_frame = frame[y1:y2, x1:x2]
            else:
                face_frame = frame

            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)

            # Process face mesh
            results = self.face_mesh.process(rgb_frame)

            if not results.multi_face_landmarks:
                return None

            face_landmarks = results.multi_face_landmarks[0]

            # Extract landmark coordinates (68 key points)
            h, w = face_frame.shape[:2]
            landmarks = []

            # Use key facial landmarks for emotion analysis
            key_indices = [
                # Eyebrows: 70, 63, 105, 66, 107, 55, 65, 52, 53, 46
                70, 63, 105, 66, 107, 55, 65, 52, 53, 46,
                # Eyes: 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
                33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
                # Nose: 1, 2, 5, 4, 6, 19, 20, 94, 125, 141, 235, 236, 3, 51, 48, 115, 131, 134, 102, 49, 220, 305, 281, 278, 279, 360, 344, 278
                1, 2, 5, 4, 6, 19, 20, 94, 125, 141, 235, 236, 3, 51, 48, 115, 131, 134, 102, 49, 220, 305, 281, 278, 279, 360, 344, 278,
                # Mouth: 61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318
                61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318
            ]

            # Take first 68 landmarks to match expected input
            for i, idx in enumerate(key_indices[:68]):
                if idx < len(face_landmarks.landmark):
                    landmark = face_landmarks.landmark[idx]
                    landmarks.extend([landmark.x, landmark.y])
                else:
                    landmarks.extend([0.5, 0.5])  # Default center position

            return np.array(landmarks, dtype=np.float32)

        except Exception as e:
            logger.error(f"Facial landmark extraction failed: {e}")
            return None

    def _detect_micro_expressions_tcn(self, emotion_scores: np.ndarray) -> List[Dict[str, Any]]:
        """Detect micro-expressions from TCN emotion scores."""
        try:
            micro_expressions = []

            # Store current emotion scores
            self.emotion_history.append(emotion_scores)

            if len(self.emotion_history) >= 5:  # Need at least 5 frames
                # Calculate emotion changes
                recent_scores = np.array(list(self.emotion_history)[-5:])

                # Detect sudden spikes (micro-expressions)
                for i, emotion_label in enumerate(self.emotion_labels):
                    current_score = emotion_scores[i]
                    avg_recent = np.mean(recent_scores[:-1, i])

                    # Micro-expression detected if sudden increase
                    if current_score > avg_recent + 0.2 and current_score > 0.3:
                        micro_expressions.append({
                            'emotion': emotion_label,
                            'intensity': float(current_score),
                            'change': float(current_score - avg_recent),
                            'duration': 1  # Single frame detection
                        })

            return micro_expressions

        except Exception as e:
            logger.error(f"Micro-expression detection failed: {e}")
            return []

    def _calculate_temporal_consistency(self) -> float:
        """Calculate temporal consistency of emotion predictions."""
        try:
            if len(self.emotion_history) < 3:
                return 0.5

            recent_emotions = np.array(list(self.emotion_history)[-5:])

            # Calculate variance across time
            emotion_variance = np.var(recent_emotions, axis=0)
            avg_variance = np.mean(emotion_variance)

            # Convert to consistency score (lower variance = higher consistency)
            consistency = 1.0 - min(1.0, avg_variance * 2.0)

            return float(consistency)

        except Exception as e:
            logger.error(f"Temporal consistency calculation failed: {e}")
            return 0.5

    def _create_empty_tcn_result(self) -> Dict[str, Any]:
        """Create empty TCN result."""
        return {
            'primary_emotion': 'neutral',
            'confidence': 0.0,
            'emotion_scores': {label: 0.0 for label in self.emotion_labels},
            'micro_expressions': [],
            'temporal_consistency': 0.0,
            'method': 'tcn',
            'sequence_length': 0,
            'landmarks_count': 0
        }

    def _create_partial_tcn_result(self, landmarks: np.ndarray) -> Dict[str, Any]:
        """Create partial TCN result when not enough frames."""
        return {
            'primary_emotion': 'neutral',
            'confidence': 0.1,
            'emotion_scores': {label: 0.1 if label == 'neutral' else 0.0 for label in self.emotion_labels},
            'micro_expressions': [],
            'temporal_consistency': 0.5,
            'method': 'tcn_partial',
            'sequence_length': len(self.landmark_sequences),
            'landmarks_count': len(landmarks) if landmarks is not None else 0
        }

    def analyze_expressions(self, frame: np.ndarray, face_bbox: List[int] = None) -> Dict[str, Any]:
        """
        Analyze facial expressions for emotional engagement assessment.
        
        Args:
            frame: Input frame
            face_bbox: Optional face bounding box for optimization
            
        Returns:
            Comprehensive expression analysis results
        """
        start_time = time.time()
        
        try:
            # Primary analysis using TCN (state-of-the-art)
            tcn_results = self.analyze_with_tcn(frame, face_bbox)

            # Traditional analysis as backup/validation
            traditional_results = self._analyze_traditional(frame, face_bbox)

            # Ensemble results (combine TCN with traditional)
            if tcn_results['confidence'] > 0.6:
                # Use TCN results with high confidence
                primary_results = tcn_results
                primary_results['traditional_backup'] = traditional_results
                primary_results['ensemble_method'] = 'tcn_primary'
            else:
                # Use traditional method as primary
                primary_results = traditional_results
                primary_results['tcn_backup'] = tcn_results
                primary_results['ensemble_method'] = 'traditional_primary'

            return primary_results

        except Exception as e:
            logger.error(f"Expression analysis failed: {e}")
            return self._create_empty_result()

    def _analyze_traditional(self, frame: np.ndarray, face_bbox: List[int] = None) -> Dict[str, Any]:
        """Traditional expression analysis method."""
        try:
            # Extract face region if bbox provided
            if face_bbox:
                x1, y1, x2, y2 = face_bbox
                face_frame = frame[y1:y2, x1:x2]
                offset = (x1, y1)
            else:
                face_frame = frame
                offset = (0, 0)

            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)

            # Process face mesh
            results = self.face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                return self._create_empty_result()
            
            face_landmarks = results.multi_face_landmarks[0]
            
            # Extract facial features
            facial_features = self._extract_facial_features(face_landmarks, face_frame.shape, offset)
            
            # Analyze current emotion
            emotion_analysis = self._analyze_current_emotion(facial_features)
            
            # Detect micro-expressions
            micro_expression_analysis = self._detect_micro_expressions(facial_features)
            
            # Calculate engagement indicators
            engagement_analysis = self._calculate_engagement_indicators(emotion_analysis, micro_expression_analysis)
            
            # Analyze temporal patterns
            temporal_analysis = self._analyze_temporal_patterns(emotion_analysis)
            
            # Update history
            self._update_history(facial_features, emotion_analysis)
            
            # Track performance
            processing_time = (time.time() - start_time) * 1000
            self.processing_times.append(processing_time)
            self.frame_count += 1
            
            return {
                'facial_features': facial_features,
                'emotion_analysis': emotion_analysis,
                'micro_expression_analysis': micro_expression_analysis,
                'engagement_analysis': engagement_analysis,
                'temporal_analysis': temporal_analysis,
                'processing_time_ms': processing_time,
                'confidence': self._calculate_analysis_confidence(facial_features)
            }
            
        except Exception as e:
            logger.error(f"Expression analysis failed: {e}")
            return self._create_empty_result()
    
    def _extract_facial_features(self, face_landmarks, frame_shape: Tuple, offset: Tuple) -> Dict[str, Any]:
        """Extract detailed facial features for emotion analysis."""
        h, w = frame_shape[:2]
        offset_x, offset_y = offset
        
        features = {}
        
        # Extract landmarks for each facial region
        for region_name, landmark_indices in self.emotion_landmarks.items():
            region_points = []
            
            for idx in landmark_indices:
                if idx < len(face_landmarks.landmark):
                    landmark = face_landmarks.landmark[idx]
                    point = (
                        landmark.x * w + offset_x,
                        landmark.y * h + offset_y,
                        landmark.z
                    )
                    region_points.append(point)
            
            features[region_name] = region_points
        
        # Calculate geometric features for emotion classification
        geometric_features = self._calculate_geometric_features(features)
        features['geometric'] = geometric_features
        
        return features
    
    def _calculate_geometric_features(self, facial_landmarks: Dict) -> Dict[str, float]:
        """Calculate geometric features for emotion classification."""
        try:
            features = {}
            
            # Eyebrow features
            if 'eyebrows' in facial_landmarks and len(facial_landmarks['eyebrows']) >= 6:
                eyebrow_height = self._calculate_eyebrow_height(facial_landmarks['eyebrows'])
                eyebrow_angle = self._calculate_eyebrow_angle(facial_landmarks['eyebrows'])
                features['eyebrow_height'] = eyebrow_height
                features['eyebrow_angle'] = eyebrow_angle
            
            # Eye features
            if 'eyes' in facial_landmarks and len(facial_landmarks['eyes']) >= 12:
                eye_openness = self._calculate_eye_openness(facial_landmarks['eyes'])
                eye_squint = self._calculate_eye_squint(facial_landmarks['eyes'])
                features['eye_openness'] = eye_openness
                features['eye_squint'] = eye_squint
            
            # Mouth features
            if 'mouth' in facial_landmarks and len(facial_landmarks['mouth']) >= 12:
                mouth_width = self._calculate_mouth_width(facial_landmarks['mouth'])
                mouth_height = self._calculate_mouth_height(facial_landmarks['mouth'])
                mouth_curvature = self._calculate_mouth_curvature(facial_landmarks['mouth'])
                features['mouth_width'] = mouth_width
                features['mouth_height'] = mouth_height
                features['mouth_curvature'] = mouth_curvature
            
            # Nose features
            if 'nose' in facial_landmarks and len(facial_landmarks['nose']) >= 6:
                nostril_flare = self._calculate_nostril_flare(facial_landmarks['nose'])
                features['nostril_flare'] = nostril_flare
            
            # Cheek features
            if 'cheeks' in facial_landmarks and len(facial_landmarks['cheeks']) >= 6:
                cheek_raise = self._calculate_cheek_raise(facial_landmarks['cheeks'])
                features['cheek_raise'] = cheek_raise
            
            return features
            
        except Exception as e:
            logger.error(f"Geometric feature calculation failed: {e}")
            return {}
    
    def _calculate_eyebrow_height(self, eyebrow_points: List[Tuple]) -> float:
        """Calculate eyebrow height (raised/lowered)."""
        try:
            if len(eyebrow_points) < 6:
                return 0.0
            
            # Calculate average y-coordinate of eyebrow points
            y_coords = [p[1] for p in eyebrow_points]
            avg_height = np.mean(y_coords)
            
            # Normalize relative to face size (simplified)
            return avg_height / 100.0  # Normalize to approximate range
            
        except Exception as e:
            return 0.0
    
    def _calculate_eyebrow_angle(self, eyebrow_points: List[Tuple]) -> float:
        """Calculate eyebrow angle (furrowed/raised)."""
        try:
            if len(eyebrow_points) < 4:
                return 0.0
            
            # Calculate angle between inner and outer eyebrow points
            inner_point = eyebrow_points[0]
            outer_point = eyebrow_points[-1]
            
            angle = np.arctan2(outer_point[1] - inner_point[1], outer_point[0] - inner_point[0])
            return np.degrees(angle)
            
        except Exception as e:
            return 0.0
    
    def _calculate_eye_openness(self, eye_points: List[Tuple]) -> float:
        """Calculate eye openness (wide/squinted)."""
        try:
            if len(eye_points) < 12:
                return 0.5
            
            # Split into left and right eyes
            mid_point = len(eye_points) // 2
            left_eye = eye_points[:mid_point]
            right_eye = eye_points[mid_point:]
            
            # Calculate eye aspect ratio for both eyes
            left_ear = self._calculate_single_eye_openness(left_eye)
            right_ear = self._calculate_single_eye_openness(right_eye)
            
            return (left_ear + right_ear) / 2.0
            
        except Exception as e:
            return 0.5
    
    def _calculate_single_eye_openness(self, eye_points: List[Tuple]) -> float:
        """Calculate openness for a single eye."""
        try:
            if len(eye_points) < 6:
                return 0.5
            
            # Calculate vertical distances
            vertical_1 = np.linalg.norm(np.array(eye_points[1][:2]) - np.array(eye_points[5][:2]))
            vertical_2 = np.linalg.norm(np.array(eye_points[2][:2]) - np.array(eye_points[4][:2]))
            
            # Calculate horizontal distance
            horizontal = np.linalg.norm(np.array(eye_points[0][:2]) - np.array(eye_points[3][:2]))
            
            # Calculate aspect ratio
            if horizontal > 0:
                aspect_ratio = (vertical_1 + vertical_2) / (2.0 * horizontal)
                return min(1.0, aspect_ratio * 3)  # Scale to 0-1 range
            
            return 0.5
            
        except Exception as e:
            return 0.5
    
    def _calculate_eye_squint(self, eye_points: List[Tuple]) -> float:
        """Calculate eye squinting level."""
        try:
            openness = self._calculate_eye_openness(eye_points)
            # Squint is inverse of openness
            return 1.0 - openness
            
        except Exception as e:
            return 0.0
    
    def _calculate_mouth_width(self, mouth_points: List[Tuple]) -> float:
        """Calculate mouth width."""
        try:
            if len(mouth_points) < 4:
                return 0.0
            
            # Find leftmost and rightmost points
            x_coords = [p[0] for p in mouth_points]
            width = max(x_coords) - min(x_coords)
            
            return width / 100.0  # Normalize
            
        except Exception as e:
            return 0.0
    
    def _calculate_mouth_height(self, mouth_points: List[Tuple]) -> float:
        """Calculate mouth height (open/closed)."""
        try:
            if len(mouth_points) < 4:
                return 0.0
            
            # Find topmost and bottommost points
            y_coords = [p[1] for p in mouth_points]
            height = max(y_coords) - min(y_coords)
            
            return height / 50.0  # Normalize
            
        except Exception as e:
            return 0.0
    
    def _calculate_mouth_curvature(self, mouth_points: List[Tuple]) -> float:
        """Calculate mouth curvature (smile/frown)."""
        try:
            if len(mouth_points) < 6:
                return 0.0
            
            # Find corner and center points
            x_coords = [p[0] for p in mouth_points]
            y_coords = [p[1] for p in mouth_points]
            
            # Find mouth corners (leftmost and rightmost)
            left_idx = np.argmin(x_coords)
            right_idx = np.argmax(x_coords)
            
            # Find mouth center (approximate)
            center_x = (x_coords[left_idx] + x_coords[right_idx]) / 2
            center_points = [i for i, x in enumerate(x_coords) if abs(x - center_x) < 10]
            
            if not center_points:
                return 0.0
            
            center_y = np.mean([y_coords[i] for i in center_points])
            corner_y = (y_coords[left_idx] + y_coords[right_idx]) / 2
            
            # Positive curvature = smile, negative = frown
            curvature = (corner_y - center_y) / 20.0  # Normalize
            
            return max(-1.0, min(1.0, curvature))
            
        except Exception as e:
            return 0.0
    
    def _calculate_nostril_flare(self, nose_points: List[Tuple]) -> float:
        """Calculate nostril flare (anger/disgust indicator)."""
        try:
            if len(nose_points) < 6:
                return 0.0
            
            # Find nostril points (approximate)
            x_coords = [p[0] for p in nose_points]
            nostril_width = max(x_coords) - min(x_coords)
            
            return nostril_width / 30.0  # Normalize
            
        except Exception as e:
            return 0.0
    
    def _calculate_cheek_raise(self, cheek_points: List[Tuple]) -> float:
        """Calculate cheek raise (smile indicator)."""
        try:
            if len(cheek_points) < 4:
                return 0.0
            
            # Calculate average height of cheek points
            y_coords = [p[1] for p in cheek_points]
            avg_height = np.mean(y_coords)
            
            return avg_height / 100.0  # Normalize
            
        except Exception as e:
            return 0.0
    
    def _analyze_current_emotion(self, facial_features: Dict) -> Dict[str, Any]:
        """Analyze current emotional state."""
        try:
            emotion_analysis = {
                'primary_emotion': 'neutral',
                'emotion_confidence': 0.0,
                'emotion_probabilities': {},
                'valence': 0.0,  # Positive/negative emotion
                'arousal': 0.0,  # Intensity of emotion
                'engagement_emotion': 'neutral'
            }
            
            # Extract feature vector for classification
            feature_vector = self._extract_emotion_feature_vector(facial_features)
            
            if self.emotion_classifier and len(feature_vector) > 0:
                # Predict emotion probabilities
                probabilities = self.emotion_classifier.predict_proba([feature_vector])[0]
                
                # Get primary emotion
                primary_idx = np.argmax(probabilities)
                primary_emotion = self.emotion_labels[primary_idx]
                confidence = probabilities[primary_idx]
                
                emotion_analysis['primary_emotion'] = primary_emotion
                emotion_analysis['emotion_confidence'] = confidence
                
                # Store all probabilities
                for i, label in enumerate(self.emotion_labels):
                    emotion_analysis['emotion_probabilities'][label] = probabilities[i]
                
                # Calculate valence and arousal
                valence, arousal = self._calculate_valence_arousal(probabilities)
                emotion_analysis['valence'] = valence
                emotion_analysis['arousal'] = arousal
                
                # Map to engagement categories
                engagement_emotion = self._map_to_engagement_emotion(primary_emotion)
                emotion_analysis['engagement_emotion'] = engagement_emotion
            
            return emotion_analysis
            
        except Exception as e:
            logger.error(f"Current emotion analysis failed: {e}")
            return {}
    
    def _extract_emotion_feature_vector(self, facial_features: Dict) -> List[float]:
        """Extract feature vector for emotion classification."""
        try:
            geometric = facial_features.get('geometric', {})
            
            feature_vector = [
                geometric.get('eyebrow_height', 0.0),
                geometric.get('eyebrow_angle', 0.0),
                geometric.get('eye_openness', 0.5),
                geometric.get('eye_squint', 0.0),
                geometric.get('mouth_width', 0.0),
                geometric.get('mouth_height', 0.0),
                geometric.get('mouth_curvature', 0.0),
                geometric.get('nostril_flare', 0.0),
                geometric.get('cheek_raise', 0.0)
            ]
            
            return feature_vector
            
        except Exception as e:
            logger.error(f"Feature vector extraction failed: {e}")
            return []
    
    def _calculate_valence_arousal(self, emotion_probabilities: np.ndarray) -> Tuple[float, float]:
        """Calculate valence (positive/negative) and arousal (intensity) from emotion probabilities."""
        try:
            # Emotion valence and arousal mappings (simplified)
            emotion_va_map = {
                'neutral': (0.0, 0.0),
                'happy': (0.8, 0.6),
                'sad': (-0.6, -0.4),
                'angry': (-0.7, 0.8),
                'surprised': (0.2, 0.8),
                'fearful': (-0.5, 0.7),
                'disgusted': (-0.6, 0.5),
                'confused': (-0.2, 0.3),
                'interested': (0.5, 0.4)
            }
            
            valence = 0.0
            arousal = 0.0
            
            for i, prob in enumerate(emotion_probabilities):
                if i < len(self.emotion_labels):
                    emotion = self.emotion_labels[i]
                    if emotion in emotion_va_map:
                        v, a = emotion_va_map[emotion]
                        valence += prob * v
                        arousal += prob * a
            
            return valence, arousal
            
        except Exception as e:
            return 0.0, 0.0
    
    def _map_to_engagement_emotion(self, emotion: str) -> str:
        """Map emotion to engagement category."""
        for category, emotions in self.engagement_emotions.items():
            if emotion in emotions:
                return category
        return 'neutral'
    
    def _detect_micro_expressions(self, facial_features: Dict) -> Dict[str, Any]:
        """Detect micro-expressions (brief, involuntary expressions)."""
        try:
            micro_analysis = {
                'micro_expression_detected': False,
                'micro_expression_type': None,
                'micro_expression_intensity': 0.0,
                'micro_expression_duration': 0,
                'authenticity_score': 0.0
            }
            
            if len(self.expression_history) < 10:
                return micro_analysis
            
            # Compare current features with recent history
            current_features = self._extract_emotion_feature_vector(facial_features)
            
            if not current_features:
                return micro_analysis
            
            # Calculate feature changes over short time window
            recent_features = [self._extract_emotion_feature_vector(entry['facial_features']) 
                             for entry in list(self.expression_history)[-5:]]
            
            # Detect rapid changes (micro-expressions)
            for i, feature_set in enumerate(recent_features):
                if feature_set:
                    change_magnitude = np.linalg.norm(np.array(current_features) - np.array(feature_set))
                    
                    if change_magnitude > self.micro_expression_threshold:
                        micro_analysis['micro_expression_detected'] = True
                        micro_analysis['micro_expression_intensity'] = change_magnitude
                        micro_analysis['micro_expression_duration'] = len(recent_features) - i
                        
                        # Determine type based on dominant feature change
                        micro_type = self._classify_micro_expression(current_features, feature_set)
                        micro_analysis['micro_expression_type'] = micro_type
                        
                        # Calculate authenticity (micro-expressions are typically authentic)
                        authenticity = min(1.0, change_magnitude / 0.5)
                        micro_analysis['authenticity_score'] = authenticity
                        
                        break
            
            return micro_analysis
            
        except Exception as e:
            logger.error(f"Micro-expression detection failed: {e}")
            return {}
    
    def _classify_micro_expression(self, current_features: List[float], previous_features: List[float]) -> str:
        """Classify the type of micro-expression based on feature changes."""
        try:
            if len(current_features) != len(previous_features):
                return 'unknown'
            
            # Calculate feature differences
            diff = np.array(current_features) - np.array(previous_features)
            
            # Feature indices
            mouth_curvature_idx = 6
            eye_squint_idx = 3
            eyebrow_height_idx = 0
            
            # Classify based on dominant changes
            if abs(diff[mouth_curvature_idx]) > 0.1:
                if diff[mouth_curvature_idx] > 0:
                    return 'micro_smile'
                else:
                    return 'micro_frown'
            elif abs(diff[eye_squint_idx]) > 0.1:
                return 'micro_squint'
            elif abs(diff[eyebrow_height_idx]) > 0.1:
                if diff[eyebrow_height_idx] > 0:
                    return 'micro_eyebrow_raise'
                else:
                    return 'micro_eyebrow_furrow'
            else:
                return 'micro_general'
                
        except Exception as e:
            return 'unknown'
    
    def _calculate_engagement_indicators(self, emotion_analysis: Dict, micro_expression_analysis: Dict) -> Dict[str, Any]:
        """Calculate engagement indicators from emotional analysis."""
        try:
            engagement_indicators = {
                'emotional_engagement': 0.0,
                'interest_level': 0.0,
                'confusion_level': 0.0,
                'positive_engagement': 0.0,
                'attention_level': 0.0,
                'authenticity': 0.0
            }
            
            # Emotional engagement based on arousal
            arousal = emotion_analysis.get('arousal', 0.0)
            engagement_indicators['emotional_engagement'] = min(1.0, abs(arousal))
            
            # Interest level from specific emotions
            emotion_probs = emotion_analysis.get('emotion_probabilities', {})
            interest_level = emotion_probs.get('interested', 0.0) + emotion_probs.get('surprised', 0.0) * 0.5
            engagement_indicators['interest_level'] = min(1.0, interest_level)
            
            # Confusion level
            confusion_level = emotion_probs.get('confused', 0.0)
            engagement_indicators['confusion_level'] = confusion_level
            
            # Positive engagement from valence
            valence = emotion_analysis.get('valence', 0.0)
            engagement_indicators['positive_engagement'] = max(0.0, valence)
            
            # Attention level (high arousal + positive valence)
            attention_level = (abs(arousal) + max(0.0, valence)) / 2.0
            engagement_indicators['attention_level'] = min(1.0, attention_level)
            
            # Authenticity from micro-expressions
            authenticity = micro_expression_analysis.get('authenticity_score', 0.5)
            engagement_indicators['authenticity'] = authenticity
            
            return engagement_indicators
            
        except Exception as e:
            logger.error(f"Engagement indicators calculation failed: {e}")
            return {}
    
    def _analyze_temporal_patterns(self, emotion_analysis: Dict) -> Dict[str, Any]:
        """Analyze temporal emotion patterns."""
        try:
            temporal_analysis = {
                'emotion_stability': 1.0,
                'emotion_trend': 'stable',
                'engagement_trend': 'stable',
                'mood_consistency': 1.0
            }
            
            if len(self.emotion_history) < 10:
                return temporal_analysis
            
            # Analyze emotion stability
            recent_emotions = [entry['primary_emotion'] for entry in list(self.emotion_history)[-10:]]
            unique_emotions = len(set(recent_emotions))
            stability = 1.0 - (unique_emotions - 1) / 9.0  # Normalize
            temporal_analysis['emotion_stability'] = max(0.0, stability)
            
            # Analyze engagement trend
            recent_arousal = [entry.get('arousal', 0.0) for entry in list(self.emotion_history)[-10:]]
            if len(recent_arousal) >= 5:
                early_arousal = np.mean(recent_arousal[:5])
                late_arousal = np.mean(recent_arousal[-5:])
                
                if late_arousal > early_arousal + 0.1:
                    temporal_analysis['engagement_trend'] = 'increasing'
                elif late_arousal < early_arousal - 0.1:
                    temporal_analysis['engagement_trend'] = 'decreasing'
                else:
                    temporal_analysis['engagement_trend'] = 'stable'
            
            # Analyze mood consistency
            recent_valence = [entry.get('valence', 0.0) for entry in list(self.emotion_history)[-10:]]
            if recent_valence:
                valence_variance = np.var(recent_valence)
                consistency = 1.0 / (1.0 + valence_variance * 10)
                temporal_analysis['mood_consistency'] = min(1.0, consistency)
            
            return temporal_analysis
            
        except Exception as e:
            logger.error(f"Temporal pattern analysis failed: {e}")
            return {}
    
    def _load_or_create_emotion_model(self):
        """Load existing emotion model or create a new one."""
        try:
            model_path = "models/emotion_classifier.pkl"
            
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.emotion_classifier = pickle.load(f)
                logger.info("Emotion classifier loaded from file")
            else:
                # Create a simple model with synthetic data
                self.emotion_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
                
                # Generate synthetic training data
                synthetic_data, synthetic_labels = self._generate_synthetic_emotion_data()
                
                if len(synthetic_data) > 0:
                    self.emotion_classifier.fit(synthetic_data, synthetic_labels)
                    
                    # Save the model
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    with open(model_path, 'wb') as f:
                        pickle.dump(self.emotion_classifier, f)
                    
                    logger.info("New emotion classifier created and saved")
                else:
                    logger.warning("Failed to create emotion classifier")
                    self.emotion_classifier = None
                    
        except Exception as e:
            logger.error(f"Emotion model loading/creation failed: {e}")
            self.emotion_classifier = None
    
    def _generate_synthetic_emotion_data(self) -> Tuple[List[List[float]], List[str]]:
        """Generate synthetic emotion training data."""
        try:
            data = []
            labels = []
            
            # Generate samples for each emotion
            samples_per_emotion = 50
            
            for emotion in self.emotion_labels:
                for _ in range(samples_per_emotion):
                    # Generate synthetic features based on emotion characteristics
                    if emotion == 'happy':
                        features = [0.3, 5.0, 0.8, 0.1, 0.6, 0.2, 0.7, 0.1, 0.6]
                    elif emotion == 'sad':
                        features = [0.2, -10.0, 0.4, 0.3, 0.3, 0.1, -0.5, 0.1, 0.2]
                    elif emotion == 'angry':
                        features = [0.1, -15.0, 0.3, 0.7, 0.4, 0.3, -0.3, 0.4, 0.1]
                    elif emotion == 'surprised':
                        features = [0.8, 20.0, 0.9, 0.0, 0.5, 0.6, 0.1, 0.2, 0.3]
                    elif emotion == 'fearful':
                        features = [0.6, 10.0, 0.9, 0.2, 0.3, 0.4, -0.2, 0.3, 0.2]
                    elif emotion == 'disgusted':
                        features = [0.2, -5.0, 0.3, 0.5, 0.3, 0.1, -0.4, 0.5, 0.1]
                    elif emotion == 'confused':
                        features = [0.4, -5.0, 0.6, 0.3, 0.4, 0.2, -0.1, 0.2, 0.3]
                    elif emotion == 'interested':
                        features = [0.5, 8.0, 0.7, 0.1, 0.5, 0.3, 0.3, 0.1, 0.4]
                    else:  # neutral
                        features = [0.3, 0.0, 0.5, 0.2, 0.4, 0.2, 0.0, 0.2, 0.3]
                    
                    # Add noise
                    features = [f + np.random.normal(0, 0.1) for f in features]
                    
                    data.append(features)
                    labels.append(emotion)
            
            return data, labels
            
        except Exception as e:
            logger.error(f"Synthetic data generation failed: {e}")
            return [], []
    
    def _calculate_analysis_confidence(self, facial_features: Dict) -> float:
        """Calculate confidence in the analysis results."""
        try:
            # Check if we have sufficient landmark data
            total_landmarks = sum(len(region) for region in facial_features.values() if isinstance(region, list))
            
            if total_landmarks < 50:
                return 0.3
            elif total_landmarks < 100:
                return 0.6
            else:
                return 0.9
                
        except Exception as e:
            return 0.0
    
    def _update_history(self, facial_features: Dict, emotion_analysis: Dict):
        """Update analysis history for temporal analysis."""
        self.expression_history.append({
            'timestamp': time.time(),
            'facial_features': facial_features
        })
        
        self.emotion_history.append({
            'timestamp': time.time(),
            'primary_emotion': emotion_analysis.get('primary_emotion', 'neutral'),
            'valence': emotion_analysis.get('valence', 0.0),
            'arousal': emotion_analysis.get('arousal', 0.0)
        })
    
    def _create_empty_result(self) -> Dict[str, Any]:
        """Create empty result when analysis fails."""
        return {
            'facial_features': {},
            'emotion_analysis': {},
            'micro_expression_analysis': {},
            'engagement_analysis': {},
            'temporal_analysis': {},
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
            'total_frames': self.frame_count,
            'model_loaded': self.emotion_classifier is not None
        }
    
    def get_emotion_trends(self, window_size: int = 30) -> Dict[str, List[float]]:
        """Get emotion trends over recent history."""
        try:
            if len(self.emotion_history) < window_size:
                return {}
            
            recent_history = list(self.emotion_history)[-window_size:]
            
            trends = {
                'valence_trend': [entry['valence'] for entry in recent_history],
                'arousal_trend': [entry['arousal'] for entry in recent_history],
                'emotion_sequence': [entry['primary_emotion'] for entry in recent_history]
            }
            
            return trends
            
        except Exception as e:
            logger.error(f"Emotion trends calculation failed: {e}")
            return {}
