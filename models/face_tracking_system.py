"""
Face Tracking and Persistence System - Advanced object tracking with alerts
Tracks faces as persistent objects with countdown alerts for missing faces
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from loguru import logger
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import uuid
import threading
from scipy.spatial.distance import euclidean


class TrackingState(Enum):
    """Face tracking states."""
    ACTIVE = "active"
    MISSING = "missing"
    LOST = "lost"
    REIDENTIFIED = "reidentified"


@dataclass
class TrackedFace:
    """Data structure for tracked face objects."""
    face_id: str
    person_id: Optional[str]
    bbox: List[int]  # [x1, y1, x2, y2]
    center: Tuple[float, float]
    embedding: Optional[np.ndarray]
    confidence: float
    state: TrackingState
    
    # Tracking history
    last_seen_frame: int
    last_seen_time: float
    tracking_start_time: float
    total_frames_tracked: int
    
    # Missing alert system
    missing_start_time: Optional[float] = None
    countdown_remaining: float = 30.0
    alert_active: bool = False
    
    # Movement tracking
    position_history: deque = field(default_factory=lambda: deque(maxlen=30))
    velocity: Tuple[float, float] = (0.0, 0.0)
    
    # Visual tracking
    tracking_color: Tuple[int, int, int] = field(default_factory=lambda: (0, 255, 0))
    lock_indicator: bool = True


class FaceTrackingSystem:
    """
    Advanced face tracking and persistence system.
    Features:
    - Persistent face-to-body object locking
    - Unique face ID assignment
    - 30-second countdown alerts for missing faces
    - Re-identification when faces return
    - Visual tracking indicators
    - Movement prediction and tracking
    """
    
    def __init__(self, config: Any = None):
        """Initialize face tracking system."""
        self.config = config
        
        # Tracking parameters
        self.max_distance_threshold = 100.0  # pixels
        self.embedding_similarity_threshold = 0.7
        self.missing_timeout = 30.0  # seconds
        self.reidentification_window = 60.0  # seconds
        
        # Tracked faces storage
        self.tracked_faces: Dict[str, TrackedFace] = {}
        self.face_counter = 0
        
        # Frame tracking
        self.current_frame_number = 0
        self.current_time = time.time()
        
        # Alert system
        self.active_alerts: Dict[str, Dict] = {}
        self.alert_callbacks = []
        
        # Visual settings
        self.tracking_colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
            (0, 128, 128),  # Teal
            (128, 128, 0)   # Olive
        ]
        
        # Performance tracking
        self.processing_times = []
        self.total_faces_tracked = 0
        self.successful_reidentifications = 0
        
        # Threading for alerts
        self.alert_thread = None
        self.alert_thread_running = False
        
        logger.info("Face Tracking System initialized")
    
    def update_tracking(self, frame: np.ndarray, face_detections: List[Dict], 
                       face_embeddings: List[np.ndarray] = None) -> Dict[str, Any]:
        """
        Update face tracking with new detections.
        
        Args:
            frame: Current frame
            face_detections: List of face detection results
            face_embeddings: Optional face embeddings for re-identification
            
        Returns:
            Tracking results with alerts and visual data
        """
        start_time = time.time()
        self.current_frame_number += 1
        self.current_time = time.time()
        
        try:
            # Match detections to existing tracked faces
            matched_faces, new_faces = self._match_detections_to_tracks(
                face_detections, face_embeddings
            )
            
            # Update existing tracked faces
            self._update_existing_tracks(matched_faces)
            
            # Create new tracked faces
            self._create_new_tracks(new_faces, face_embeddings)
            
            # Update missing faces and alerts
            self._update_missing_faces()
            
            # Generate visual tracking data
            visual_data = self._generate_visual_tracking_data(frame)
            
            # Compile tracking results
            tracking_results = self._compile_tracking_results(visual_data)
            
            # Track performance
            processing_time = (time.time() - start_time) * 1000
            self.processing_times.append(processing_time)
            
            return tracking_results
            
        except Exception as e:
            logger.error(f"Face tracking update failed: {e}")
            return self._create_empty_tracking_result()
    
    def _match_detections_to_tracks(self, face_detections: List[Dict], 
                                   face_embeddings: List[np.ndarray] = None) -> Tuple[Dict, List]:
        """Match current detections to existing tracked faces."""
        try:
            matched_faces = {}
            unmatched_detections = []
            used_track_ids = set()
            
            for i, detection in enumerate(face_detections):
                bbox = detection.get('bbox', [])
                if len(bbox) != 4:
                    continue
                
                detection_center = (
                    (bbox[0] + bbox[2]) / 2,
                    (bbox[1] + bbox[3]) / 2
                )
                
                best_match_id = None
                best_match_score = float('inf')
                
                # Try to match with existing tracks
                for face_id, tracked_face in self.tracked_faces.items():
                    if face_id in used_track_ids:
                        continue
                    
                    # Skip faces that have been lost for too long
                    if (tracked_face.state == TrackingState.LOST and 
                        self.current_time - tracked_face.last_seen_time > self.reidentification_window):
                        continue
                    
                    # Calculate distance-based matching
                    distance = euclidean(detection_center, tracked_face.center)
                    
                    # Calculate embedding-based matching if available
                    embedding_similarity = 0.0
                    if (face_embeddings and i < len(face_embeddings) and 
                        tracked_face.embedding is not None):
                        embedding_similarity = self._calculate_embedding_similarity(
                            face_embeddings[i], tracked_face.embedding
                        )
                    
                    # Combined matching score
                    if distance < self.max_distance_threshold:
                        # Prioritize embedding similarity if available
                        if embedding_similarity > self.embedding_similarity_threshold:
                            match_score = distance * (1.0 - embedding_similarity)
                        else:
                            match_score = distance
                        
                        if match_score < best_match_score:
                            best_match_score = match_score
                            best_match_id = face_id
                
                # Assign match or mark as new detection
                if best_match_id:
                    matched_faces[best_match_id] = {
                        'detection': detection,
                        'embedding': face_embeddings[i] if face_embeddings and i < len(face_embeddings) else None,
                        'detection_index': i
                    }
                    used_track_ids.add(best_match_id)
                else:
                    unmatched_detections.append({
                        'detection': detection,
                        'embedding': face_embeddings[i] if face_embeddings and i < len(face_embeddings) else None,
                        'detection_index': i
                    })
            
            return matched_faces, unmatched_detections
            
        except Exception as e:
            logger.error(f"Detection matching failed: {e}")
            return {}, face_detections
    
    def _calculate_embedding_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between face embeddings."""
        try:
            # Normalize embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return max(0.0, similarity)  # Ensure non-negative
            
        except Exception as e:
            logger.error(f"Embedding similarity calculation failed: {e}")
            return 0.0
    
    def _update_existing_tracks(self, matched_faces: Dict):
        """Update existing tracked faces with new detections."""
        try:
            for face_id, match_data in matched_faces.items():
                if face_id not in self.tracked_faces:
                    continue
                
                tracked_face = self.tracked_faces[face_id]
                detection = match_data['detection']
                embedding = match_data.get('embedding')
                
                # Update basic tracking data
                bbox = detection.get('bbox', [])
                if len(bbox) == 4:
                    tracked_face.bbox = bbox
                    new_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                    
                    # Calculate velocity
                    if tracked_face.position_history:
                        prev_center = tracked_face.position_history[-1]
                        tracked_face.velocity = (
                            new_center[0] - prev_center[0],
                            new_center[1] - prev_center[1]
                        )
                    
                    tracked_face.center = new_center
                    tracked_face.position_history.append(new_center)
                
                tracked_face.confidence = detection.get('confidence', 0.0)
                tracked_face.last_seen_frame = self.current_frame_number
                tracked_face.last_seen_time = self.current_time
                tracked_face.total_frames_tracked += 1
                
                # Update embedding if available
                if embedding is not None:
                    tracked_face.embedding = embedding
                
                # Update state
                if tracked_face.state == TrackingState.MISSING:
                    tracked_face.state = TrackingState.REIDENTIFIED
                    self._clear_missing_alert(face_id)
                    self.successful_reidentifications += 1
                    logger.info(f"Face {face_id} re-identified successfully")
                elif tracked_face.state == TrackingState.LOST:
                    tracked_face.state = TrackingState.REIDENTIFIED
                    self.successful_reidentifications += 1
                    logger.info(f"Face {face_id} recovered after being lost")
                else:
                    tracked_face.state = TrackingState.ACTIVE
                
        except Exception as e:
            logger.error(f"Existing tracks update failed: {e}")
    
    def _create_new_tracks(self, new_detections: List[Dict], face_embeddings: List[np.ndarray] = None):
        """Create new tracked faces for unmatched detections."""
        try:
            for detection_data in new_detections:
                detection = detection_data['detection']
                embedding = detection_data.get('embedding')
                
                bbox = detection.get('bbox', [])
                if len(bbox) != 4:
                    continue
                
                # Generate unique face ID
                face_id = f"face_{self.face_counter:04d}_{int(time.time() * 1000) % 10000}"
                self.face_counter += 1
                
                # Calculate center
                center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                
                # Assign tracking color
                color_index = len(self.tracked_faces) % len(self.tracking_colors)
                tracking_color = self.tracking_colors[color_index]
                
                # Create tracked face object
                tracked_face = TrackedFace(
                    face_id=face_id,
                    person_id=detection.get('person_id'),
                    bbox=bbox,
                    center=center,
                    embedding=embedding,
                    confidence=detection.get('confidence', 0.0),
                    state=TrackingState.ACTIVE,
                    last_seen_frame=self.current_frame_number,
                    last_seen_time=self.current_time,
                    tracking_start_time=self.current_time,
                    total_frames_tracked=1,
                    tracking_color=tracking_color
                )
                
                tracked_face.position_history.append(center)
                
                self.tracked_faces[face_id] = tracked_face
                self.total_faces_tracked += 1
                
                logger.info(f"New face tracked: {face_id}")
                
        except Exception as e:
            logger.error(f"New tracks creation failed: {e}")
    
    def _update_missing_faces(self):
        """Update status of faces that are no longer detected."""
        try:
            current_time = self.current_time
            
            for face_id, tracked_face in self.tracked_faces.items():
                # Skip if face was seen in current frame
                if tracked_face.last_seen_frame == self.current_frame_number:
                    continue
                
                time_since_last_seen = current_time - tracked_face.last_seen_time
                
                # Handle missing faces
                if tracked_face.state == TrackingState.ACTIVE:
                    if time_since_last_seen > 1.0:  # 1 second grace period
                        tracked_face.state = TrackingState.MISSING
                        tracked_face.missing_start_time = current_time
                        tracked_face.countdown_remaining = self.missing_timeout
                        self._create_missing_alert(face_id)
                        logger.warning(f"Face {face_id} marked as missing")
                
                elif tracked_face.state == TrackingState.MISSING:
                    # Update countdown
                    tracked_face.countdown_remaining = max(
                        0.0, 
                        self.missing_timeout - (current_time - tracked_face.missing_start_time)
                    )
                    
                    # Check if timeout reached
                    if tracked_face.countdown_remaining <= 0:
                        tracked_face.state = TrackingState.LOST
                        self._clear_missing_alert(face_id)
                        logger.warning(f"Face {face_id} marked as lost after timeout")
                
        except Exception as e:
            logger.error(f"Missing faces update failed: {e}")
    
    def _create_missing_alert(self, face_id: str):
        """Create alert for missing face."""
        try:
            tracked_face = self.tracked_faces.get(face_id)
            if not tracked_face:
                return
            
            alert_data = {
                'face_id': face_id,
                'person_id': tracked_face.person_id,
                'alert_type': 'missing_face',
                'start_time': self.current_time,
                'countdown_duration': self.missing_timeout,
                'message': f"Face {face_id} is missing from camera view",
                'severity': 'warning'
            }
            
            self.active_alerts[face_id] = alert_data
            tracked_face.alert_active = True
            
            # Trigger alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert_data)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")
            
            logger.warning(f"Missing alert created for face {face_id}")
            
        except Exception as e:
            logger.error(f"Missing alert creation failed: {e}")
    
    def _clear_missing_alert(self, face_id: str):
        """Clear alert for re-identified face."""
        try:
            if face_id in self.active_alerts:
                del self.active_alerts[face_id]
            
            tracked_face = self.tracked_faces.get(face_id)
            if tracked_face:
                tracked_face.alert_active = False
                tracked_face.missing_start_time = None
                tracked_face.countdown_remaining = self.missing_timeout
            
            logger.info(f"Missing alert cleared for face {face_id}")
            
        except Exception as e:
            logger.error(f"Missing alert clearing failed: {e}")
    
    def _generate_visual_tracking_data(self, frame: np.ndarray) -> Dict[str, Any]:
        """Generate visual tracking data for overlay."""
        try:
            visual_data = {
                'tracked_faces': [],
                'alerts': [],
                'tracking_stats': {}
            }
            
            for face_id, tracked_face in self.tracked_faces.items():
                # Skip lost faces that are too old
                if (tracked_face.state == TrackingState.LOST and 
                    self.current_time - tracked_face.last_seen_time > 60.0):
                    continue
                
                face_visual = {
                    'face_id': face_id,
                    'person_id': tracked_face.person_id,
                    'bbox': tracked_face.bbox,
                    'center': tracked_face.center,
                    'state': tracked_face.state.value,
                    'tracking_color': tracked_face.tracking_color,
                    'confidence': tracked_face.confidence,
                    'total_frames': tracked_face.total_frames_tracked,
                    'tracking_duration': self.current_time - tracked_face.tracking_start_time,
                    'velocity': tracked_face.velocity,
                    'lock_indicator': tracked_face.lock_indicator
                }
                
                # Add missing alert data
                if tracked_face.state == TrackingState.MISSING:
                    face_visual['countdown_remaining'] = tracked_face.countdown_remaining
                    face_visual['alert_active'] = tracked_face.alert_active
                
                visual_data['tracked_faces'].append(face_visual)
            
            # Add active alerts
            for alert_id, alert_data in self.active_alerts.items():
                alert_visual = {
                    'alert_id': alert_id,
                    'face_id': alert_data['face_id'],
                    'message': alert_data['message'],
                    'countdown': self.missing_timeout - (self.current_time - alert_data['start_time']),
                    'severity': alert_data['severity']
                }
                visual_data['alerts'].append(alert_visual)
            
            # Add tracking statistics
            visual_data['tracking_stats'] = {
                'total_tracked': len(self.tracked_faces),
                'active_faces': len([f for f in self.tracked_faces.values() if f.state == TrackingState.ACTIVE]),
                'missing_faces': len([f for f in self.tracked_faces.values() if f.state == TrackingState.MISSING]),
                'lost_faces': len([f for f in self.tracked_faces.values() if f.state == TrackingState.LOST]),
                'active_alerts': len(self.active_alerts)
            }
            
            return visual_data
            
        except Exception as e:
            logger.error(f"Visual tracking data generation failed: {e}")
            return {}
    
    def _compile_tracking_results(self, visual_data: Dict) -> Dict[str, Any]:
        """Compile comprehensive tracking results."""
        try:
            return {
                'frame_number': self.current_frame_number,
                'timestamp': self.current_time,
                'visual_data': visual_data,
                'tracked_faces': {
                    face_id: {
                        'face_id': face_id,
                        'person_id': face.person_id,
                        'state': face.state.value,
                        'bbox': face.bbox,
                        'confidence': face.confidence,
                        'tracking_duration': self.current_time - face.tracking_start_time,
                        'total_frames': face.total_frames_tracked,
                        'last_seen': face.last_seen_time,
                        'alert_active': face.alert_active,
                        'countdown_remaining': face.countdown_remaining if face.state == TrackingState.MISSING else None
                    }
                    for face_id, face in self.tracked_faces.items()
                },
                'active_alerts': self.active_alerts.copy(),
                'tracking_statistics': {
                    'total_faces_tracked': self.total_faces_tracked,
                    'successful_reidentifications': self.successful_reidentifications,
                    'current_active_tracks': len([f for f in self.tracked_faces.values() if f.state == TrackingState.ACTIVE]),
                    'current_missing_tracks': len([f for f in self.tracked_faces.values() if f.state == TrackingState.MISSING]),
                    'current_lost_tracks': len([f for f in self.tracked_faces.values() if f.state == TrackingState.LOST]),
                    'average_tracking_duration': np.mean([
                        self.current_time - f.tracking_start_time 
                        for f in self.tracked_faces.values()
                    ]) if self.tracked_faces else 0.0
                }
            }
            
        except Exception as e:
            logger.error(f"Tracking results compilation failed: {e}")
            return self._create_empty_tracking_result()
    
    def _create_empty_tracking_result(self) -> Dict[str, Any]:
        """Create empty tracking result."""
        return {
            'frame_number': self.current_frame_number,
            'timestamp': self.current_time,
            'visual_data': {},
            'tracked_faces': {},
            'active_alerts': {},
            'tracking_statistics': {}
        }
    
    def add_alert_callback(self, callback):
        """Add callback function for alert notifications."""
        self.alert_callbacks.append(callback)
    
    def remove_alert_callback(self, callback):
        """Remove alert callback function."""
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)
    
    def get_face_by_id(self, face_id: str) -> Optional[TrackedFace]:
        """Get tracked face by ID."""
        return self.tracked_faces.get(face_id)
    
    def get_active_faces(self) -> List[TrackedFace]:
        """Get all currently active tracked faces."""
        return [f for f in self.tracked_faces.values() if f.state == TrackingState.ACTIVE]
    
    def get_missing_faces(self) -> List[TrackedFace]:
        """Get all currently missing tracked faces."""
        return [f for f in self.tracked_faces.values() if f.state == TrackingState.MISSING]
    
    def cleanup_old_tracks(self, max_age_hours: float = 24.0):
        """Clean up very old lost tracks."""
        try:
            current_time = self.current_time
            max_age_seconds = max_age_hours * 3600
            
            faces_to_remove = []
            for face_id, tracked_face in self.tracked_faces.items():
                if (tracked_face.state == TrackingState.LOST and 
                    current_time - tracked_face.last_seen_time > max_age_seconds):
                    faces_to_remove.append(face_id)
            
            for face_id in faces_to_remove:
                del self.tracked_faces[face_id]
                if face_id in self.active_alerts:
                    del self.active_alerts[face_id]
            
            if faces_to_remove:
                logger.info(f"Cleaned up {len(faces_to_remove)} old tracks")
                
        except Exception as e:
            logger.error(f"Track cleanup failed: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.processing_times:
            return {}
        
        return {
            'avg_processing_time_ms': np.mean(self.processing_times),
            'min_processing_time_ms': np.min(self.processing_times),
            'max_processing_time_ms': np.max(self.processing_times),
            'fps': 1000 / np.mean(self.processing_times) if self.processing_times else 0,
            'total_faces_tracked': self.total_faces_tracked,
            'successful_reidentifications': self.successful_reidentifications,
            'current_tracked_faces': len(self.tracked_faces),
            'active_alerts': len(self.active_alerts)
        }
