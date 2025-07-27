"""
Intelligent Pattern Analyzer - Advanced ML pattern recognition for engagement analysis
Enhanced from previous version with improved accuracy and real-time performance
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from loguru import logger
import time
from collections import deque
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import pickle
import os


class IntelligentPatternAnalyzer:
    """
    Advanced pattern recognition system for behavioral and engagement analysis.
    Features:
    - Multi-modal pattern detection
    - Anomaly detection for unusual behaviors
    - Engagement pattern classification
    - Temporal pattern analysis
    - Predictive engagement modeling
    """
    
    def __init__(self, config: Any = None):
        """Initialize intelligent pattern analyzer."""
        self.config = config
        
        # Pattern detection models
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.pattern_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
        # Pattern history
        self.pattern_history = deque(maxlen=300)  # 10 seconds at 30 FPS
        self.engagement_patterns = deque(maxlen=900)  # 30 seconds at 30 FPS
        
        # Pattern categories
        self.engagement_patterns_types = [
            'highly_engaged', 'engaged', 'neutral', 'disengaged', 'distracted',
            'confused', 'interested', 'bored', 'frustrated', 'focused'
        ]
        
        # Feature weights for different modalities
        self.modality_weights = {
            'face': 0.25,
            'pose': 0.20,
            'gesture': 0.15,
            'eye': 0.20,
            'expression': 0.20
        }
        
        # Temporal analysis parameters
        self.short_term_window = 30   # 1 second
        self.medium_term_window = 150  # 5 seconds
        self.long_term_window = 300   # 10 seconds
        
        # Model training status
        self.models_trained = False
        self.training_data_count = 0
        
        # Performance tracking
        self.processing_times = []
        self.frame_count = 0
        
        # Load or initialize models
        self._load_or_initialize_models()
        
        logger.info("Intelligent Pattern Analyzer initialized")
    
    def analyze_patterns(self, multi_modal_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze patterns from multi-modal engagement data.
        
        Args:
            multi_modal_data: Combined data from all analysis modules
            
        Returns:
            Comprehensive pattern analysis results
        """
        start_time = time.time()
        
        try:
            # Extract unified feature vector
            feature_vector = self._extract_unified_features(multi_modal_data)
            
            if not feature_vector:
                return self._create_empty_result()
            
            # Detect anomalies
            anomaly_analysis = self._detect_anomalies(feature_vector)
            
            # Classify engagement pattern
            pattern_classification = self._classify_engagement_pattern(feature_vector)
            
            # Analyze temporal patterns
            temporal_analysis = self._analyze_temporal_patterns(feature_vector)
            
            # Predict future engagement
            prediction_analysis = self._predict_engagement_trend(feature_vector)
            
            # Detect behavioral clusters
            cluster_analysis = self._analyze_behavioral_clusters()
            
            # Calculate pattern confidence
            pattern_confidence = self._calculate_pattern_confidence(feature_vector, pattern_classification)
            
            # Update history
            self._update_pattern_history(feature_vector, pattern_classification)
            
            # Track performance
            processing_time = (time.time() - start_time) * 1000
            self.processing_times.append(processing_time)
            self.frame_count += 1
            
            return {
                'feature_vector': feature_vector,
                'anomaly_analysis': anomaly_analysis,
                'pattern_classification': pattern_classification,
                'temporal_analysis': temporal_analysis,
                'prediction_analysis': prediction_analysis,
                'cluster_analysis': cluster_analysis,
                'pattern_confidence': pattern_confidence,
                'processing_time_ms': processing_time
            }
            
        except Exception as e:
            logger.error(f"Pattern analysis failed: {e}")
            return self._create_empty_result()
    
    def _extract_unified_features(self, multi_modal_data: Dict[str, Any]) -> List[float]:
        """Extract unified feature vector from multi-modal data."""
        try:
            features = []
            
            # Face detection features
            face_data = multi_modal_data.get('face_detection', {})
            face_features = [
                face_data.get('confidence', 0.0),
                len(face_data.get('faces', [])),
                face_data.get('average_face_size', 0.0) / 1000.0  # Normalize
            ]
            features.extend(face_features)
            
            # Pose/body features
            body_data = multi_modal_data.get('body_analysis', {})
            pose_features = [
                body_data.get('engagement_score', 0.0),
                body_data.get('posture_analysis', {}).get('spine_straightness', 0.0),
                body_data.get('posture_analysis', {}).get('shoulder_alignment', 0.0),
                body_data.get('posture_analysis', {}).get('head_position', 0.0),
                1.0 if body_data.get('posture_analysis', {}).get('left_arm_raised', False) else 0.0,
                1.0 if body_data.get('posture_analysis', {}).get('right_arm_raised', False) else 0.0,
                body_data.get('movement_analysis', {}).get('movement_stability', 0.0),
                1.0 if body_data.get('movement_analysis', {}).get('fidgeting_detected', False) else 0.0
            ]
            features.extend(pose_features)
            
            # Eye tracking features
            eye_data = multi_modal_data.get('eye_tracking', {})
            eye_features = [
                eye_data.get('attention_score', 0.0),
                eye_data.get('gaze_analysis', {}).get('gaze_stability', 0.0),
                1.0 if eye_data.get('blink_analysis', {}).get('blink_detected', False) else 0.0,
                eye_data.get('blink_analysis', {}).get('attention_indicator', 0.0),
                eye_data.get('movement_analysis', {}).get('fixation_stability', 0.0)
            ]
            features.extend(eye_features)
            
            # Expression features
            expression_data = multi_modal_data.get('expression_analysis', {})
            expression_features = [
                expression_data.get('engagement_analysis', {}).get('emotional_engagement', 0.0),
                expression_data.get('engagement_analysis', {}).get('interest_level', 0.0),
                expression_data.get('engagement_analysis', {}).get('confusion_level', 0.0),
                expression_data.get('engagement_analysis', {}).get('positive_engagement', 0.0),
                expression_data.get('engagement_analysis', {}).get('attention_level', 0.0),
                expression_data.get('emotion_analysis', {}).get('valence', 0.0),
                expression_data.get('emotion_analysis', {}).get('arousal', 0.0),
                1.0 if expression_data.get('micro_expression_analysis', {}).get('micro_expression_detected', False) else 0.0
            ]
            features.extend(expression_features)
            
            # Gesture features (if available)
            gesture_data = multi_modal_data.get('gesture_analysis', {})
            gesture_features = [
                1.0 if gesture_data.get('hand_raised', False) else 0.0,
                1.0 if gesture_data.get('pointing', False) else 0.0,
                1.0 if gesture_data.get('thumbs_up', False) else 0.0,
                gesture_data.get('gesture_confidence', 0.0)
            ]
            features.extend(gesture_features)
            
            # Temporal features (if history available)
            if len(self.pattern_history) > 0:
                temporal_features = self._extract_temporal_features()
                features.extend(temporal_features)
            else:
                # Add zeros for temporal features
                features.extend([0.0] * 5)
            
            return features
            
        except Exception as e:
            logger.error(f"Unified feature extraction failed: {e}")
            return []
    
    def _extract_temporal_features(self) -> List[float]:
        """Extract temporal features from pattern history."""
        try:
            if len(self.pattern_history) < 10:
                return [0.0] * 5
            
            # Get recent engagement scores
            recent_scores = [entry.get('engagement_score', 0.0) for entry in list(self.pattern_history)[-30:]]
            
            # Calculate temporal statistics
            mean_engagement = np.mean(recent_scores) if recent_scores else 0.0
            std_engagement = np.std(recent_scores) if len(recent_scores) > 1 else 0.0
            
            # Calculate trend
            if len(recent_scores) >= 10:
                early_mean = np.mean(recent_scores[:10])
                late_mean = np.mean(recent_scores[-10:])
                trend = (late_mean - early_mean) / max(early_mean, 0.1)
            else:
                trend = 0.0
            
            # Calculate stability
            if len(recent_scores) > 1:
                stability = 1.0 / (1.0 + std_engagement)
            else:
                stability = 1.0
            
            # Calculate momentum (rate of change)
            if len(recent_scores) >= 5:
                recent_diff = np.diff(recent_scores[-5:])
                momentum = np.mean(recent_diff) if len(recent_diff) > 0 else 0.0
            else:
                momentum = 0.0
            
            return [mean_engagement, std_engagement, trend, stability, momentum]
            
        except Exception as e:
            logger.error(f"Temporal feature extraction failed: {e}")
            return [0.0] * 5
    
    def _detect_anomalies(self, feature_vector: List[float]) -> Dict[str, Any]:
        """Detect anomalous behavior patterns."""
        try:
            anomaly_analysis = {
                'is_anomaly': False,
                'anomaly_score': 0.0,
                'anomaly_type': 'normal',
                'confidence': 0.0
            }
            
            if not self.models_trained or len(feature_vector) == 0:
                return anomaly_analysis
            
            # Normalize features
            feature_array = np.array(feature_vector).reshape(1, -1)
            
            try:
                normalized_features = self.scaler.transform(feature_array)
            except:
                # If scaler not fitted, return normal
                return anomaly_analysis
            
            # Detect anomaly
            anomaly_prediction = self.anomaly_detector.predict(normalized_features)[0]
            anomaly_score = self.anomaly_detector.decision_function(normalized_features)[0]
            
            anomaly_analysis['is_anomaly'] = anomaly_prediction == -1
            anomaly_analysis['anomaly_score'] = float(anomaly_score)
            
            # Classify anomaly type
            if anomaly_analysis['is_anomaly']:
                anomaly_type = self._classify_anomaly_type(feature_vector)
                anomaly_analysis['anomaly_type'] = anomaly_type
                anomaly_analysis['confidence'] = min(1.0, abs(anomaly_score))
            
            return anomaly_analysis
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return {'is_anomaly': False, 'anomaly_score': 0.0, 'anomaly_type': 'normal', 'confidence': 0.0}
    
    def _classify_anomaly_type(self, feature_vector: List[float]) -> str:
        """Classify the type of detected anomaly."""
        try:
            # Simple heuristic-based anomaly classification
            if len(feature_vector) < 20:
                return 'unknown'
            
            # Check for specific anomaly patterns
            # High movement with low engagement
            movement_features = feature_vector[10:12]  # Movement-related features
            engagement_features = feature_vector[15:20]  # Engagement-related features
            
            high_movement = np.mean(movement_features) > 0.7
            low_engagement = np.mean(engagement_features) < 0.3
            
            if high_movement and low_engagement:
                return 'distracted_movement'
            
            # Very low activity
            all_low = np.mean(feature_vector) < 0.2
            if all_low:
                return 'disengaged'
            
            # Very high activity
            all_high = np.mean(feature_vector) > 0.8
            if all_high:
                return 'hyperactive'
            
            # Inconsistent patterns
            high_variance = np.std(feature_vector) > 0.4
            if high_variance:
                return 'inconsistent_behavior'
            
            return 'unusual_pattern'
            
        except Exception as e:
            return 'unknown'
    
    def _classify_engagement_pattern(self, feature_vector: List[float]) -> Dict[str, Any]:
        """Classify the current engagement pattern."""
        try:
            pattern_classification = {
                'primary_pattern': 'neutral',
                'pattern_confidence': 0.0,
                'pattern_probabilities': {},
                'engagement_level': 0.5
            }
            
            if not self.models_trained or len(feature_vector) == 0:
                # Use heuristic classification
                return self._heuristic_pattern_classification(feature_vector)
            
            # Normalize features
            feature_array = np.array(feature_vector).reshape(1, -1)
            
            try:
                normalized_features = self.scaler.transform(feature_array)
                
                # Predict pattern
                pattern_probabilities = self.pattern_classifier.predict_proba(normalized_features)[0]
                primary_pattern_idx = np.argmax(pattern_probabilities)
                primary_pattern = self.engagement_patterns_types[primary_pattern_idx]
                confidence = pattern_probabilities[primary_pattern_idx]
                
                pattern_classification['primary_pattern'] = primary_pattern
                pattern_classification['pattern_confidence'] = float(confidence)
                
                # Store all probabilities
                for i, pattern_type in enumerate(self.engagement_patterns_types):
                    pattern_classification['pattern_probabilities'][pattern_type] = float(pattern_probabilities[i])
                
                # Calculate overall engagement level
                engagement_level = self._calculate_engagement_level(pattern_probabilities)
                pattern_classification['engagement_level'] = engagement_level
                
            except:
                # Fallback to heuristic
                return self._heuristic_pattern_classification(feature_vector)
            
            return pattern_classification
            
        except Exception as e:
            logger.error(f"Pattern classification failed: {e}")
            return self._heuristic_pattern_classification(feature_vector)
    
    def _heuristic_pattern_classification(self, feature_vector: List[float]) -> Dict[str, Any]:
        """Heuristic-based pattern classification when ML models are not available."""
        try:
            if len(feature_vector) < 10:
                return {
                    'primary_pattern': 'neutral',
                    'pattern_confidence': 0.5,
                    'pattern_probabilities': {},
                    'engagement_level': 0.5
                }
            
            # Calculate engagement indicators
            face_engagement = np.mean(feature_vector[:3]) if len(feature_vector) > 3 else 0.5
            body_engagement = np.mean(feature_vector[3:11]) if len(feature_vector) > 11 else 0.5
            eye_engagement = np.mean(feature_vector[11:16]) if len(feature_vector) > 16 else 0.5
            expression_engagement = np.mean(feature_vector[16:24]) if len(feature_vector) > 24 else 0.5
            
            # Weighted average
            overall_engagement = (
                face_engagement * self.modality_weights['face'] +
                body_engagement * self.modality_weights['pose'] +
                eye_engagement * self.modality_weights['eye'] +
                expression_engagement * self.modality_weights['expression']
            )
            
            # Classify based on engagement level
            if overall_engagement >= 0.8:
                primary_pattern = 'highly_engaged'
            elif overall_engagement >= 0.6:
                primary_pattern = 'engaged'
            elif overall_engagement >= 0.4:
                primary_pattern = 'neutral'
            elif overall_engagement >= 0.2:
                primary_pattern = 'disengaged'
            else:
                primary_pattern = 'distracted'
            
            return {
                'primary_pattern': primary_pattern,
                'pattern_confidence': 0.7,
                'pattern_probabilities': {primary_pattern: 0.7},
                'engagement_level': overall_engagement
            }
            
        except Exception as e:
            logger.error(f"Heuristic pattern classification failed: {e}")
            return {
                'primary_pattern': 'neutral',
                'pattern_confidence': 0.5,
                'pattern_probabilities': {},
                'engagement_level': 0.5
            }
    
    def _calculate_engagement_level(self, pattern_probabilities: np.ndarray) -> float:
        """Calculate overall engagement level from pattern probabilities."""
        try:
            # Engagement weights for each pattern type
            engagement_weights = {
                'highly_engaged': 1.0,
                'engaged': 0.8,
                'focused': 0.9,
                'interested': 0.7,
                'neutral': 0.5,
                'confused': 0.4,
                'disengaged': 0.2,
                'distracted': 0.1,
                'bored': 0.1,
                'frustrated': 0.3
            }
            
            engagement_level = 0.0
            for i, prob in enumerate(pattern_probabilities):
                if i < len(self.engagement_patterns_types):
                    pattern_type = self.engagement_patterns_types[i]
                    weight = engagement_weights.get(pattern_type, 0.5)
                    engagement_level += prob * weight
            
            return min(1.0, max(0.0, engagement_level))
            
        except Exception as e:
            return 0.5
    
    def _analyze_temporal_patterns(self, feature_vector: List[float]) -> Dict[str, Any]:
        """Analyze temporal patterns in engagement."""
        try:
            temporal_analysis = {
                'short_term_trend': 'stable',
                'medium_term_trend': 'stable',
                'long_term_trend': 'stable',
                'pattern_consistency': 1.0,
                'engagement_momentum': 0.0,
                'pattern_transitions': []
            }
            
            if len(self.engagement_patterns) < 30:
                return temporal_analysis
            
            # Analyze different time windows
            short_term_data = list(self.engagement_patterns)[-self.short_term_window:]
            medium_term_data = list(self.engagement_patterns)[-self.medium_term_window:]
            long_term_data = list(self.engagement_patterns)[-self.long_term_window:]
            
            # Calculate trends
            temporal_analysis['short_term_trend'] = self._calculate_trend(short_term_data)
            temporal_analysis['medium_term_trend'] = self._calculate_trend(medium_term_data)
            temporal_analysis['long_term_trend'] = self._calculate_trend(long_term_data)
            
            # Calculate pattern consistency
            consistency = self._calculate_pattern_consistency(medium_term_data)
            temporal_analysis['pattern_consistency'] = consistency
            
            # Calculate engagement momentum
            momentum = self._calculate_engagement_momentum(short_term_data)
            temporal_analysis['engagement_momentum'] = momentum
            
            # Detect pattern transitions
            transitions = self._detect_pattern_transitions(medium_term_data)
            temporal_analysis['pattern_transitions'] = transitions
            
            return temporal_analysis
            
        except Exception as e:
            logger.error(f"Temporal pattern analysis failed: {e}")
            return {}
    
    def _calculate_trend(self, data: List[Dict]) -> str:
        """Calculate trend direction from temporal data."""
        try:
            if len(data) < 10:
                return 'stable'
            
            engagement_scores = [entry.get('engagement_level', 0.5) for entry in data]
            
            # Calculate linear trend
            x = np.arange(len(engagement_scores))
            slope = np.polyfit(x, engagement_scores, 1)[0]
            
            if slope > 0.05:
                return 'increasing'
            elif slope < -0.05:
                return 'decreasing'
            else:
                return 'stable'
                
        except Exception as e:
            return 'stable'
    
    def _calculate_pattern_consistency(self, data: List[Dict]) -> float:
        """Calculate consistency of engagement patterns."""
        try:
            if len(data) < 5:
                return 1.0
            
            patterns = [entry.get('primary_pattern', 'neutral') for entry in data]
            
            # Calculate pattern diversity
            unique_patterns = len(set(patterns))
            max_patterns = min(len(patterns), len(self.engagement_patterns_types))
            
            # Consistency is inverse of diversity
            consistency = 1.0 - (unique_patterns - 1) / max(max_patterns - 1, 1)
            
            return max(0.0, min(1.0, consistency))
            
        except Exception as e:
            return 1.0
    
    def _calculate_engagement_momentum(self, data: List[Dict]) -> float:
        """Calculate engagement momentum (rate of change)."""
        try:
            if len(data) < 5:
                return 0.0
            
            engagement_scores = [entry.get('engagement_level', 0.5) for entry in data]
            
            # Calculate recent changes
            recent_changes = np.diff(engagement_scores[-5:])
            momentum = np.mean(recent_changes) if len(recent_changes) > 0 else 0.0
            
            return max(-1.0, min(1.0, momentum * 10))  # Scale to [-1, 1]
            
        except Exception as e:
            return 0.0
    
    def _detect_pattern_transitions(self, data: List[Dict]) -> List[Dict]:
        """Detect significant pattern transitions."""
        try:
            transitions = []
            
            if len(data) < 10:
                return transitions
            
            patterns = [entry.get('primary_pattern', 'neutral') for entry in data]
            
            # Detect transitions
            for i in range(1, len(patterns)):
                if patterns[i] != patterns[i-1]:
                    transition = {
                        'from_pattern': patterns[i-1],
                        'to_pattern': patterns[i],
                        'frame_index': len(data) - len(patterns) + i,
                        'transition_type': self._classify_transition(patterns[i-1], patterns[i])
                    }
                    transitions.append(transition)
            
            return transitions[-5:]  # Return last 5 transitions
            
        except Exception as e:
            return []
    
    def _classify_transition(self, from_pattern: str, to_pattern: str) -> str:
        """Classify the type of pattern transition."""
        engagement_order = {
            'distracted': 0, 'disengaged': 1, 'bored': 2, 'neutral': 3,
            'confused': 4, 'interested': 5, 'engaged': 6, 'focused': 7, 'highly_engaged': 8
        }
        
        from_level = engagement_order.get(from_pattern, 3)
        to_level = engagement_order.get(to_pattern, 3)
        
        if to_level > from_level + 1:
            return 'positive_jump'
        elif to_level > from_level:
            return 'positive_shift'
        elif to_level < from_level - 1:
            return 'negative_jump'
        elif to_level < from_level:
            return 'negative_shift'
        else:
            return 'lateral_shift'
    
    def _predict_engagement_trend(self, feature_vector: List[float]) -> Dict[str, Any]:
        """Predict future engagement trend."""
        try:
            prediction_analysis = {
                'predicted_trend': 'stable',
                'confidence': 0.5,
                'risk_factors': [],
                'positive_indicators': []
            }
            
            if len(self.engagement_patterns) < 50:
                return prediction_analysis
            
            # Simple trend prediction based on recent patterns
            recent_data = list(self.engagement_patterns)[-30:]
            engagement_scores = [entry.get('engagement_level', 0.5) for entry in recent_data]
            
            # Calculate trend indicators
            recent_trend = np.polyfit(range(len(engagement_scores)), engagement_scores, 1)[0]
            volatility = np.std(engagement_scores)
            current_level = engagement_scores[-1] if engagement_scores else 0.5
            
            # Predict trend
            if recent_trend > 0.02 and volatility < 0.2:
                prediction_analysis['predicted_trend'] = 'improving'
                prediction_analysis['confidence'] = 0.8
            elif recent_trend < -0.02 and volatility < 0.2:
                prediction_analysis['predicted_trend'] = 'declining'
                prediction_analysis['confidence'] = 0.8
            elif volatility > 0.3:
                prediction_analysis['predicted_trend'] = 'unstable'
                prediction_analysis['confidence'] = 0.7
            else:
                prediction_analysis['predicted_trend'] = 'stable'
                prediction_analysis['confidence'] = 0.6
            
            # Identify risk factors and positive indicators
            if current_level < 0.3:
                prediction_analysis['risk_factors'].append('low_current_engagement')
            if volatility > 0.4:
                prediction_analysis['risk_factors'].append('high_volatility')
            if recent_trend < -0.05:
                prediction_analysis['risk_factors'].append('declining_trend')
            
            if current_level > 0.7:
                prediction_analysis['positive_indicators'].append('high_current_engagement')
            if recent_trend > 0.05:
                prediction_analysis['positive_indicators'].append('improving_trend')
            if volatility < 0.1:
                prediction_analysis['positive_indicators'].append('stable_pattern')
            
            return prediction_analysis
            
        except Exception as e:
            logger.error(f"Engagement trend prediction failed: {e}")
            return {}
    
    def _analyze_behavioral_clusters(self) -> Dict[str, Any]:
        """Analyze behavioral clusters in recent data."""
        try:
            cluster_analysis = {
                'num_clusters': 0,
                'cluster_labels': [],
                'cluster_characteristics': {},
                'dominant_cluster': 'unknown'
            }
            
            if len(self.pattern_history) < 50:
                return cluster_analysis
            
            # Extract features from recent history
            recent_features = []
            for entry in list(self.pattern_history)[-50:]:
                if 'feature_vector' in entry:
                    recent_features.append(entry['feature_vector'])
            
            if len(recent_features) < 10:
                return cluster_analysis
            
            # Perform clustering
            features_array = np.array(recent_features)
            
            # Normalize features
            normalized_features = self.scaler.fit_transform(features_array)
            
            # DBSCAN clustering
            clustering = DBSCAN(eps=0.5, min_samples=3)
            cluster_labels = clustering.fit_predict(normalized_features)
            
            cluster_analysis['cluster_labels'] = cluster_labels.tolist()
            cluster_analysis['num_clusters'] = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            
            # Analyze cluster characteristics
            unique_labels = set(cluster_labels)
            for label in unique_labels:
                if label != -1:  # Ignore noise points
                    cluster_mask = cluster_labels == label
                    cluster_features = features_array[cluster_mask]
                    
                    cluster_analysis['cluster_characteristics'][f'cluster_{label}'] = {
                        'size': int(np.sum(cluster_mask)),
                        'mean_engagement': float(np.mean(cluster_features[:, :5])) if cluster_features.shape[1] > 5 else 0.0,
                        'stability': float(1.0 / (1.0 + np.std(cluster_features)))
                    }
            
            # Find dominant cluster
            if cluster_analysis['cluster_characteristics']:
                dominant_cluster = max(
                    cluster_analysis['cluster_characteristics'].keys(),
                    key=lambda x: cluster_analysis['cluster_characteristics'][x]['size']
                )
                cluster_analysis['dominant_cluster'] = dominant_cluster
            
            return cluster_analysis
            
        except Exception as e:
            logger.error(f"Behavioral cluster analysis failed: {e}")
            return {}
    
    def _calculate_pattern_confidence(self, feature_vector: List[float], pattern_classification: Dict) -> float:
        """Calculate confidence in pattern analysis."""
        try:
            # Base confidence from classification
            base_confidence = pattern_classification.get('pattern_confidence', 0.5)
            
            # Adjust based on feature quality
            if len(feature_vector) < 20:
                feature_quality = 0.5
            else:
                # Check for reasonable feature values
                valid_features = sum(1 for f in feature_vector if 0.0 <= f <= 1.0)
                feature_quality = valid_features / len(feature_vector)
            
            # Adjust based on history length
            history_factor = min(1.0, len(self.pattern_history) / 100.0)
            
            # Combine factors
            overall_confidence = base_confidence * feature_quality * (0.5 + 0.5 * history_factor)
            
            return max(0.0, min(1.0, overall_confidence))
            
        except Exception as e:
            return 0.5
    
    def _update_pattern_history(self, feature_vector: List[float], pattern_classification: Dict):
        """Update pattern history for temporal analysis."""
        entry = {
            'timestamp': time.time(),
            'feature_vector': feature_vector,
            'primary_pattern': pattern_classification.get('primary_pattern', 'neutral'),
            'engagement_level': pattern_classification.get('engagement_level', 0.5),
            'pattern_confidence': pattern_classification.get('pattern_confidence', 0.5)
        }
        
        self.pattern_history.append(entry)
        self.engagement_patterns.append(entry)
    
    def _load_or_initialize_models(self):
        """Load existing models or initialize new ones."""
        try:
            models_dir = "models"
            os.makedirs(models_dir, exist_ok=True)
            
            anomaly_path = os.path.join(models_dir, "anomaly_detector.pkl")
            pattern_path = os.path.join(models_dir, "pattern_classifier.pkl")
            scaler_path = os.path.join(models_dir, "feature_scaler.pkl")
            
            # Try to load existing models
            if all(os.path.exists(p) for p in [anomaly_path, pattern_path, scaler_path]):
                with open(anomaly_path, 'rb') as f:
                    self.anomaly_detector = pickle.load(f)
                with open(pattern_path, 'rb') as f:
                    self.pattern_classifier = pickle.load(f)
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                
                self.models_trained = True
                logger.info("Pattern analysis models loaded from files")
            else:
                logger.info("No existing models found, will train with incoming data")
                
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
    
    def train_models_with_data(self, training_data: List[Dict]):
        """Train models with accumulated training data."""
        try:
            if len(training_data) < 100:
                logger.warning("Insufficient training data for model training")
                return False
            
            # Extract features and labels
            features = []
            labels = []
            
            for entry in training_data:
                if 'feature_vector' in entry and 'primary_pattern' in entry:
                    features.append(entry['feature_vector'])
                    labels.append(entry['primary_pattern'])
            
            if len(features) < 50:
                return False
            
            features_array = np.array(features)
            
            # Fit scaler
            self.scaler.fit(features_array)
            normalized_features = self.scaler.transform(features_array)
            
            # Train anomaly detector
            self.anomaly_detector.fit(normalized_features)
            
            # Train pattern classifier
            valid_labels = [label for label in labels if label in self.engagement_patterns_types]
            if len(valid_labels) >= 20:
                valid_indices = [i for i, label in enumerate(labels) if label in self.engagement_patterns_types]
                valid_features = normalized_features[valid_indices]
                
                self.pattern_classifier.fit(valid_features, valid_labels)
                self.models_trained = True
                
                # Save models
                self._save_models()
                
                logger.info(f"Pattern analysis models trained with {len(features)} samples")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return False
    
    def _save_models(self):
        """Save trained models to files."""
        try:
            models_dir = "models"
            os.makedirs(models_dir, exist_ok=True)
            
            with open(os.path.join(models_dir, "anomaly_detector.pkl"), 'wb') as f:
                pickle.dump(self.anomaly_detector, f)
            with open(os.path.join(models_dir, "pattern_classifier.pkl"), 'wb') as f:
                pickle.dump(self.pattern_classifier, f)
            with open(os.path.join(models_dir, "feature_scaler.pkl"), 'wb') as f:
                pickle.dump(self.scaler, f)
            
            logger.info("Pattern analysis models saved")
            
        except Exception as e:
            logger.error(f"Model saving failed: {e}")
    
    def _create_empty_result(self) -> Dict[str, Any]:
        """Create empty result when analysis fails."""
        return {
            'feature_vector': [],
            'anomaly_analysis': {},
            'pattern_classification': {},
            'temporal_analysis': {},
            'prediction_analysis': {},
            'cluster_analysis': {},
            'pattern_confidence': 0.0,
            'processing_time_ms': 0.0
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
            'models_trained': self.models_trained,
            'training_data_count': len(self.pattern_history)
        }
    
    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get summary of detected patterns."""
        try:
            if not self.engagement_patterns:
                return {}
            
            recent_patterns = list(self.engagement_patterns)[-100:]
            
            # Pattern distribution
            pattern_counts = {}
            for entry in recent_patterns:
                pattern = entry.get('primary_pattern', 'neutral')
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
            # Engagement statistics
            engagement_levels = [entry.get('engagement_level', 0.5) for entry in recent_patterns]
            
            return {
                'pattern_distribution': pattern_counts,
                'average_engagement': np.mean(engagement_levels),
                'engagement_std': np.std(engagement_levels),
                'max_engagement': np.max(engagement_levels),
                'min_engagement': np.min(engagement_levels),
                'total_patterns_analyzed': len(recent_patterns)
            }
            
        except Exception as e:
            logger.error(f"Pattern summary generation failed: {e}")
            return {}
