"""
AI/ML Models for Classroom Analysis
Only imports existing modules to avoid import errors
"""

from .face_detection import StateOfTheArtFaceDetector as FaceDetector
from .face_recognition import StateOfTheArtFaceRecognizer as FaceRecognizer
from .engagement_analyzer import StateOfTheArtEngagementAnalyzer as EngagementAnalyzer
from .reinforcement_learning import RLAgent
from .liveness_detector import LivenessDetector

# Advanced modules (only existing ones)
from .advanced_body_detector import AdvancedBodyDetector
from .advanced_eye_tracker import StateOfTheArtEyeTracker as AdvancedEyeTracker
from .micro_expression_analyzer import StateOfTheArtMicroExpressionAnalyzer as MicroExpressionAnalyzer
from .intelligent_pattern_analyzer import IntelligentPatternAnalyzer
from .behavioral_classifier import BehavioralPatternClassifier

# Continuous learning modules (only existing ones)
from .continuous_learning_system import ContinuousLearningSystem
from .face_tracking_system import FaceTrackingSystem

__all__ = [
    "FaceDetector",
    "FaceRecognizer",
    "EngagementAnalyzer",
    "RLAgent",
    "LivenessDetector",
    # Advanced modules
    "AdvancedBodyDetector",
    "AdvancedEyeTracker",
    "MicroExpressionAnalyzer",
    "IntelligentPatternAnalyzer",
    "BehavioralPatternClassifier",
    # Continuous learning
    "ContinuousLearningSystem",
    "FaceTrackingSystem"
]
