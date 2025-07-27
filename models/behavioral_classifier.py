"""
Behavioral Pattern Classifier - Movement and behavior categorization
Enhanced from previous version with improved classification accuracy
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from loguru import logger
import time
from collections import deque
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os


class BehavioralPatternClassifier:
    """
    Advanced behavioral pattern classification system.
    Features:
    - Movement pattern classification
    - Behavior categorization
    - Engagement behavior analysis
    - Real-time classification
    """
    
    def __init__(self, config: Any = None):
        """Initialize behavioral pattern classifier."""
        self.config = config
        
        # Classification models
        self.movement_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.behavior_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
        # Behavior categories
        self.movement_patterns = [
            'still', 'fidgeting', 'active', 'restless', 'focused_movement', 'distracted_movement'
        ]
        
        self.behavior_categories = [
            'engaged_listening', 'active_participation', 'note_taking', 'thinking',
            'confused', 'distracted', 'bored', 'frustrated', 'interested', 'focused'
        ]
        
        # Pattern history
        self.movement_history = deque(maxlen=150)  # 5 seconds at 30 FPS
        self.behavior_history = deque(maxlen=300)  # 10 seconds at 30 FPS
        
        # Classification thresholds
        self.movement_threshold = 0.6
        self.behavior_threshold = 0.7
        
        # Models trained status
        self.models_trained = False
        
        # Performance tracking
        self.processing_times = []
        self.frame_count = 0
        
        # Load or create models
        self._load_or_create_models()
        
        logger.info("Behavioral Pattern Classifier initialized")
    
    def classify_behavior(self, multi_modal_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify behavioral patterns from multi-modal data.
        
        Args:
            multi_modal_data: Combined data from all analysis modules
            
        Returns:
            Behavioral classification results
        """
        start_time = time.time()
        
        try:
            # Extract behavioral features
            behavioral_features = self._extract_behavioral_features(multi_modal_data)
            
            if not behavioral_features:
                return self._create_empty_result()
            
            # Classify movement patterns
            movement_classification = self._classify_movement_patterns(behavioral_features)
            
            # Classify behavior categories
            behavior_classification = self._classify_behavior_categories(behavioral_features)
            
            # Analyze temporal patterns
            temporal_analysis = self._analyze_temporal_behavior_patterns()
            
            # Calculate engagement behavior score
            engagement_behavior_score = self._calculate_engagement_behavior_score(
                movement_classification, behavior_classification
            )
            
            # Update history
            self._update_behavior_history(movement_classification, behavior_classification)
            
            # Track performance
            processing_time = (time.time() - start_time) * 1000
            self.processing_times.append(processing_time)
            self.frame_count += 1
            
            return {
                'behavioral_features': behavioral_features,
                'movement_classification': movement_classification,
                'behavior_classification': behavior_classification,
                'temporal_analysis': temporal_analysis,
                'engagement_behavior_score': engagement_behavior_score,
                'processing_time_ms': processing_time
            }
            
        except Exception as e:
            logger.error(f"Behavioral classification failed: {e}")
            return self._create_empty_result()
    
    def _extract_behavioral_features(self, multi_modal_data: Dict[str, Any]) -> List[float]:
        """Extract behavioral features from multi-modal data."""
        try:
            features = []
            
            # Body movement features
            body_data = multi_modal_data.get('body_analysis', {})
            movement_data = body_data.get('movement_analysis', {})
            
            movement_features = [
                movement_data.get('movement_magnitude', 0.0),
                movement_data.get('movement_stability', 1.0),
                1.0 if movement_data.get('fidgeting_detected', False) else 0.0,
                self._encode_movement_pattern(movement_data.get('movement_pattern', 'stable'))
            ]
            features.extend(movement_features)
            
            # Posture features
            posture_data = body_data.get('posture_analysis', {})
            posture_features = [
                posture_data.get('spine_straightness', 0.0),
                posture_data.get('shoulder_alignment', 0.0),
                posture_data.get('head_position', 0.0),
                1.0 if posture_data.get('left_arm_raised', False) else 0.0,
                1.0 if posture_data.get('right_arm_raised', False) else 0.0,
                1.0 if posture_data.get('arms_crossed', False) else 0.0,
                1.0 if posture_data.get('hand_on_face', False) else 0.0
            ]
            features.extend(posture_features)
            
            # Eye movement features
            eye_data = multi_modal_data.get('eye_tracking', {})
            eye_movement_data = eye_data.get('movement_analysis', {})
            
            eye_features = [
                eye_movement_data.get('movement_magnitude', 0.0),
                eye_movement_data.get('fixation_stability', 1.0),
                1.0 if eye_movement_data.get('saccade_detected', False) else 0.0,
                self._encode_eye_movement_pattern(eye_movement_data.get('movement_pattern', 'stable'))
            ]
            features.extend(eye_features)
            
            # Expression features
            expression_data = multi_modal_data.get('expression_analysis', {})
            emotion_data = expression_data.get('emotion_analysis', {})
            
            expression_features = [
                emotion_data.get('valence', 0.0),
                emotion_data.get('arousal', 0.0),
                expression_data.get('engagement_analysis', {}).get('confusion_level', 0.0),
                expression_data.get('engagement_analysis', {}).get('interest_level', 0.0),
                1.0 if expression_data.get('micro_expression_analysis', {}).get('micro_expression_detected', False) else 0.0
            ]
            features.extend(expression_features)
            
            # Gesture features
            gesture_data = multi_modal_data.get('gesture_analysis', {})
            gesture_features = [
                1.0 if gesture_data.get('hand_raised', False) else 0.0,
                1.0 if gesture_data.get('pointing', False) else 0.0,
                1.0 if gesture_data.get('thumbs_up', False) else 0.0,
                gesture_data.get('gesture_confidence', 0.0)
            ]
            features.extend(gesture_features)
            
            # Temporal features
            if len(self.behavior_history) > 0:
                temporal_features = self._extract_temporal_behavioral_features()
                features.extend(temporal_features)
            else:
                features.extend([0.0] * 4)  # Placeholder temporal features
            
            return features
            
        except Exception as e:
            logger.error(f"Behavioral feature extraction failed: {e}")
            return []
    
    def _encode_movement_pattern(self, pattern: str) -> float:
        """Encode movement pattern as numerical value."""
        pattern_encoding = {
            'stable': 0.0,
            'fidgeting': 0.2,
            'active': 0.4,
            'restless': 0.6,
            'normal': 0.8,
            'unknown': 1.0
        }
        return pattern_encoding.get(pattern, 0.5)
    
    def _encode_eye_movement_pattern(self, pattern: str) -> float:
        """Encode eye movement pattern as numerical value."""
        pattern_encoding = {
            'fixated': 0.0,
            'stable': 0.2,
            'tracking': 0.4,
            'saccadic': 0.6,
            'wandering': 0.8,
            'unknown': 1.0
        }
        return pattern_encoding.get(pattern, 0.5)
    
    def _extract_temporal_behavioral_features(self) -> List[float]:
        """Extract temporal behavioral features."""
        try:
            if len(self.behavior_history) < 10:
                return [0.0] * 4
            
            recent_behaviors = list(self.behavior_history)[-30:]
            
            # Movement consistency
            movement_patterns = [entry.get('movement_pattern', 'stable') for entry in recent_behaviors]
            movement_consistency = len(set(movement_patterns)) / len(movement_patterns)
            
            # Behavior stability
            behavior_categories = [entry.get('behavior_category', 'neutral') for entry in recent_behaviors]
            behavior_stability = len(set(behavior_categories)) / len(behavior_categories)
            
            # Engagement trend
            engagement_scores = [entry.get('engagement_score', 0.5) for entry in recent_behaviors]
            if len(engagement_scores) >= 10:
                early_engagement = np.mean(engagement_scores[:10])
                late_engagement = np.mean(engagement_scores[-10:])
                engagement_trend = (late_engagement - early_engagement) / max(early_engagement, 0.1)
            else:
                engagement_trend = 0.0
            
            # Activity level
            activity_levels = [entry.get('activity_level', 0.5) for entry in recent_behaviors]
            avg_activity = np.mean(activity_levels)
            
            return [movement_consistency, behavior_stability, engagement_trend, avg_activity]
            
        except Exception as e:
            logger.error(f"Temporal behavioral feature extraction failed: {e}")
            return [0.0] * 4
    
    def _classify_movement_patterns(self, features: List[float]) -> Dict[str, Any]:
        """Classify movement patterns."""
        try:
            movement_classification = {
                'primary_pattern': 'stable',
                'pattern_confidence': 0.0,
                'pattern_probabilities': {},
                'activity_level': 0.5
            }
            
            if not self.models_trained or len(features) < 10:
                # Use heuristic classification
                return self._heuristic_movement_classification(features)
            
            # Use trained model
            feature_array = np.array(features).reshape(1, -1)
            
            try:
                normalized_features = self.scaler.transform(feature_array)
                probabilities = self.movement_classifier.predict_proba(normalized_features)[0]
                
                primary_idx = np.argmax(probabilities)
                primary_pattern = self.movement_patterns[primary_idx]
                confidence = probabilities[primary_idx]
                
                movement_classification['primary_pattern'] = primary_pattern
                movement_classification['pattern_confidence'] = float(confidence)
                
                # Store all probabilities
                for i, pattern in enumerate(self.movement_patterns):
                    movement_classification['pattern_probabilities'][pattern] = float(probabilities[i])
                
                # Calculate activity level
                activity_level = self._calculate_activity_level(probabilities)
                movement_classification['activity_level'] = activity_level
                
            except:
                return self._heuristic_movement_classification(features)
            
            return movement_classification
            
        except Exception as e:
            logger.error(f"Movement pattern classification failed: {e}")
            return self._heuristic_movement_classification(features)
    
    def _heuristic_movement_classification(self, features: List[float]) -> Dict[str, Any]:
        """Heuristic movement pattern classification."""
        try:
            if len(features) < 4:
                return {
                    'primary_pattern': 'stable',
                    'pattern_confidence': 0.5,
                    'pattern_probabilities': {},
                    'activity_level': 0.5
                }
            
            movement_magnitude = features[0]
            movement_stability = features[1]
            fidgeting = features[2] > 0.5
            
            # Classify based on movement characteristics
            if fidgeting:
                primary_pattern = 'fidgeting'
                activity_level = 0.7
            elif movement_magnitude > 0.7:
                primary_pattern = 'active'
                activity_level = 0.8
            elif movement_magnitude > 0.3 and movement_stability < 0.5:
                primary_pattern = 'restless'
                activity_level = 0.6
            elif movement_stability > 0.8:
                primary_pattern = 'still'
                activity_level = 0.2
            else:
                primary_pattern = 'stable'
                activity_level = 0.5
            
            return {
                'primary_pattern': primary_pattern,
                'pattern_confidence': 0.7,
                'pattern_probabilities': {primary_pattern: 0.7},
                'activity_level': activity_level
            }
            
        except Exception as e:
            return {
                'primary_pattern': 'stable',
                'pattern_confidence': 0.5,
                'pattern_probabilities': {},
                'activity_level': 0.5
            }
    
    def _classify_behavior_categories(self, features: List[float]) -> Dict[str, Any]:
        """Classify behavior categories."""
        try:
            behavior_classification = {
                'primary_behavior': 'neutral',
                'behavior_confidence': 0.0,
                'behavior_probabilities': {},
                'engagement_indicator': 0.5
            }
            
            if not self.models_trained or len(features) < 15:
                return self._heuristic_behavior_classification(features)
            
            # Use trained model
            feature_array = np.array(features).reshape(1, -1)
            
            try:
                normalized_features = self.scaler.transform(feature_array)
                probabilities = self.behavior_classifier.predict_proba(normalized_features)[0]
                
                primary_idx = np.argmax(probabilities)
                primary_behavior = self.behavior_categories[primary_idx]
                confidence = probabilities[primary_idx]
                
                behavior_classification['primary_behavior'] = primary_behavior
                behavior_classification['behavior_confidence'] = float(confidence)
                
                # Store all probabilities
                for i, behavior in enumerate(self.behavior_categories):
                    behavior_classification['behavior_probabilities'][behavior] = float(probabilities[i])
                
                # Calculate engagement indicator
                engagement_indicator = self._calculate_behavior_engagement_indicator(probabilities)
                behavior_classification['engagement_indicator'] = engagement_indicator
                
            except:
                return self._heuristic_behavior_classification(features)
            
            return behavior_classification
            
        except Exception as e:
            logger.error(f"Behavior category classification failed: {e}")
            return self._heuristic_behavior_classification(features)
    
    def _heuristic_behavior_classification(self, features: List[float]) -> Dict[str, Any]:
        """Heuristic behavior category classification."""
        try:
            if len(features) < 15:
                return {
                    'primary_behavior': 'neutral',
                    'behavior_confidence': 0.5,
                    'behavior_probabilities': {},
                    'engagement_indicator': 0.5
                }
            
            # Extract key behavioral indicators
            arm_raised = features[7] > 0.5 or features[8] > 0.5  # Left or right arm raised
            hand_on_face = features[10] > 0.5
            arms_crossed = features[9] > 0.5
            valence = features[15]  # Emotional valence
            arousal = features[16]  # Emotional arousal
            confusion = features[17]
            interest = features[18]
            
            # Classify behavior
            if arm_raised:
                primary_behavior = 'active_participation'
                engagement_indicator = 0.9
            elif hand_on_face:
                primary_behavior = 'thinking'
                engagement_indicator = 0.7
            elif arms_crossed and valence < -0.2:
                primary_behavior = 'frustrated'
                engagement_indicator = 0.3
            elif confusion > 0.6:
                primary_behavior = 'confused'
                engagement_indicator = 0.4
            elif interest > 0.7:
                primary_behavior = 'interested'
                engagement_indicator = 0.8
            elif valence > 0.5 and arousal > 0.3:
                primary_behavior = 'engaged_listening'
                engagement_indicator = 0.8
            elif valence < -0.3 and arousal < 0.2:
                primary_behavior = 'bored'
                engagement_indicator = 0.2
            else:
                primary_behavior = 'focused'
                engagement_indicator = 0.6
            
            return {
                'primary_behavior': primary_behavior,
                'behavior_confidence': 0.7,
                'behavior_probabilities': {primary_behavior: 0.7},
                'engagement_indicator': engagement_indicator
            }
            
        except Exception as e:
            return {
                'primary_behavior': 'neutral',
                'behavior_confidence': 0.5,
                'behavior_probabilities': {},
                'engagement_indicator': 0.5
            }
    
    def _calculate_activity_level(self, movement_probabilities: np.ndarray) -> float:
        """Calculate activity level from movement probabilities."""
        try:
            activity_weights = {
                'still': 0.0,
                'fidgeting': 0.6,
                'active': 0.9,
                'restless': 0.8,
                'focused_movement': 0.7,
                'distracted_movement': 0.5
            }
            
            activity_level = 0.0
            for i, prob in enumerate(movement_probabilities):
                if i < len(self.movement_patterns):
                    pattern = self.movement_patterns[i]
                    weight = activity_weights.get(pattern, 0.5)
                    activity_level += prob * weight
            
            return min(1.0, max(0.0, activity_level))
            
        except Exception as e:
            return 0.5
    
    def _calculate_behavior_engagement_indicator(self, behavior_probabilities: np.ndarray) -> float:
        """Calculate engagement indicator from behavior probabilities."""
        try:
            engagement_weights = {
                'engaged_listening': 0.9,
                'active_participation': 1.0,
                'note_taking': 0.8,
                'thinking': 0.7,
                'confused': 0.4,
                'distracted': 0.1,
                'bored': 0.1,
                'frustrated': 0.3,
                'interested': 0.8,
                'focused': 0.9
            }
            
            engagement_indicator = 0.0
            for i, prob in enumerate(behavior_probabilities):
                if i < len(self.behavior_categories):
                    behavior = self.behavior_categories[i]
                    weight = engagement_weights.get(behavior, 0.5)
                    engagement_indicator += prob * weight
            
            return min(1.0, max(0.0, engagement_indicator))
            
        except Exception as e:
            return 0.5
    
    def _analyze_temporal_behavior_patterns(self) -> Dict[str, Any]:
        """Analyze temporal patterns in behavior."""
        try:
            temporal_analysis = {
                'behavior_consistency': 1.0,
                'behavior_trend': 'stable',
                'pattern_transitions': [],
                'dominant_behavior': 'unknown'
            }
            
            if len(self.behavior_history) < 20:
                return temporal_analysis
            
            recent_behaviors = list(self.behavior_history)[-30:]
            
            # Analyze behavior consistency
            behavior_categories = [entry.get('behavior_category', 'neutral') for entry in recent_behaviors]
            unique_behaviors = len(set(behavior_categories))
            consistency = 1.0 - (unique_behaviors - 1) / max(len(self.behavior_categories) - 1, 1)
            temporal_analysis['behavior_consistency'] = max(0.0, consistency)
            
            # Analyze engagement trend
            engagement_scores = [entry.get('engagement_score', 0.5) for entry in recent_behaviors]
            if len(engagement_scores) >= 10:
                early_engagement = np.mean(engagement_scores[:10])
                late_engagement = np.mean(engagement_scores[-10:])
                
                if late_engagement > early_engagement + 0.1:
                    temporal_analysis['behavior_trend'] = 'improving'
                elif late_engagement < early_engagement - 0.1:
                    temporal_analysis['behavior_trend'] = 'declining'
                else:
                    temporal_analysis['behavior_trend'] = 'stable'
            
            # Find dominant behavior
            if behavior_categories:
                from collections import Counter
                behavior_counts = Counter(behavior_categories)
                dominant_behavior = behavior_counts.most_common(1)[0][0]
                temporal_analysis['dominant_behavior'] = dominant_behavior
            
            # Detect pattern transitions
            transitions = []
            for i in range(1, min(len(behavior_categories), 10)):
                if behavior_categories[i] != behavior_categories[i-1]:
                    transitions.append({
                        'from': behavior_categories[i-1],
                        'to': behavior_categories[i],
                        'frame_offset': i
                    })
            
            temporal_analysis['pattern_transitions'] = transitions
            
            return temporal_analysis
            
        except Exception as e:
            logger.error(f"Temporal behavior pattern analysis failed: {e}")
            return {}
    
    def _calculate_engagement_behavior_score(self, movement_classification: Dict, behavior_classification: Dict) -> float:
        """Calculate overall engagement behavior score."""
        try:
            # Movement component (40% weight)
            activity_level = movement_classification.get('activity_level', 0.5)
            movement_score = min(1.0, activity_level * 1.2)  # Slight boost for activity
            
            # Behavior component (60% weight)
            engagement_indicator = behavior_classification.get('engagement_indicator', 0.5)
            
            # Combine scores
            engagement_behavior_score = movement_score * 0.4 + engagement_indicator * 0.6
            
            return max(0.0, min(1.0, engagement_behavior_score))
            
        except Exception as e:
            return 0.5
    
    def _update_behavior_history(self, movement_classification: Dict, behavior_classification: Dict):
        """Update behavior history for temporal analysis."""
        entry = {
            'timestamp': time.time(),
            'movement_pattern': movement_classification.get('primary_pattern', 'stable'),
            'behavior_category': behavior_classification.get('primary_behavior', 'neutral'),
            'activity_level': movement_classification.get('activity_level', 0.5),
            'engagement_score': behavior_classification.get('engagement_indicator', 0.5)
        }
        
        self.behavior_history.append(entry)
    
    def _load_or_create_models(self):
        """Load existing models or create new ones."""
        try:
            models_dir = "models"
            movement_path = os.path.join(models_dir, "movement_classifier.pkl")
            behavior_path = os.path.join(models_dir, "behavior_classifier.pkl")
            scaler_path = os.path.join(models_dir, "behavior_scaler.pkl")
            
            if all(os.path.exists(p) for p in [movement_path, behavior_path, scaler_path]):
                with open(movement_path, 'rb') as f:
                    self.movement_classifier = pickle.load(f)
                with open(behavior_path, 'rb') as f:
                    self.behavior_classifier = pickle.load(f)
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                
                self.models_trained = True
                logger.info("Behavioral classification models loaded")
            else:
                logger.info("No existing behavioral models found")
                
        except Exception as e:
            logger.error(f"Behavioral model loading failed: {e}")
    
    def _create_empty_result(self) -> Dict[str, Any]:
        """Create empty result when classification fails."""
        return {
            'behavioral_features': [],
            'movement_classification': {},
            'behavior_classification': {},
            'temporal_analysis': {},
            'engagement_behavior_score': 0.0,
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
            'models_trained': self.models_trained
        }
    
    def get_behavior_summary(self) -> Dict[str, Any]:
        """Get summary of classified behaviors."""
        try:
            if not self.behavior_history:
                return {}
            
            recent_behaviors = list(self.behavior_history)[-100:]
            
            # Behavior distribution
            movement_patterns = [entry['movement_pattern'] for entry in recent_behaviors]
            behavior_categories = [entry['behavior_category'] for entry in recent_behaviors]
            
            from collections import Counter
            movement_counts = Counter(movement_patterns)
            behavior_counts = Counter(behavior_categories)
            
            # Activity statistics
            activity_levels = [entry['activity_level'] for entry in recent_behaviors]
            engagement_scores = [entry['engagement_score'] for entry in recent_behaviors]
            
            return {
                'movement_distribution': dict(movement_counts),
                'behavior_distribution': dict(behavior_counts),
                'average_activity_level': np.mean(activity_levels),
                'average_engagement_score': np.mean(engagement_scores),
                'activity_std': np.std(activity_levels),
                'engagement_std': np.std(engagement_scores),
                'total_classifications': len(recent_behaviors)
            }
            
        except Exception as e:
            logger.error(f"Behavior summary generation failed: {e}")
            return {}
