"""
Metrics Tracker for Performance Monitoring
Tracks and analyzes system performance metrics
"""

import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from collections import deque, defaultdict
import json
from loguru import logger


class MetricsTracker:
    """
    Comprehensive metrics tracking system for monitoring:
    - Detection accuracy and performance
    - Engagement analysis metrics
    - System performance indicators
    - Real-time analytics
    """
    
    def __init__(self, config: Any = None):
        """
        Initialize metrics tracker.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Target metrics from config
        self.attendance_accuracy_target = config.get("metrics.attendance_accuracy_target", 0.98) if config else 0.98
        self.engagement_precision_target = config.get("metrics.engagement_precision_target", 0.70) if config else 0.70
        self.max_latency_ms = config.get("metrics.max_latency_ms", 5000) if config else 5000
        self.fps_target = config.get("metrics.fps_target", 30) if config else 30
        
        # Metrics storage
        self.frame_metrics = deque(maxlen=1000)  # Last 1000 frames
        self.session_metrics = {}
        self.performance_history = defaultdict(list)
        
        # Real-time counters
        self.total_frames = 0
        self.total_faces_detected = 0
        self.total_faces_recognized = 0
        self.total_live_faces = 0
        self.total_engaged_persons = 0
        
        # Accuracy tracking
        self.attendance_predictions = []
        self.attendance_ground_truth = []
        self.engagement_predictions = []
        self.engagement_ground_truth = []
        
        # Performance tracking
        self.processing_times = deque(maxlen=100)
        self.fps_measurements = deque(maxlen=50)
        
        # Session tracking
        self.session_start_time = None
        self.current_session_id = None
        
        logger.info("Metrics tracker initialized")
    
    def start_session(self, session_id: str):
        """Start tracking a new session."""
        self.current_session_id = session_id
        self.session_start_time = time.time()
        self.session_metrics = {
            'session_id': session_id,
            'start_time': self.session_start_time,
            'frames_processed': 0,
            'total_detections': 0,
            'total_recognitions': 0,
            'total_engagements': 0,
            'performance_stats': {}
        }
        
        logger.info(f"Started metrics tracking for session: {session_id}")
    
    def update_frame_metrics(self, frame_results: Dict[str, Any]):
        """
        Update metrics with frame processing results.
        
        Args:
            frame_results: Results from frame processing
        """
        try:
            frame_number = frame_results.get('frame_number', 0)
            timestamp = frame_results.get('timestamp', time.time())
            processing_time = frame_results.get('processing_time_ms', 0)
            
            # Extract summary data
            summary = frame_results.get('summary', {})
            performance = frame_results.get('performance', {})
            
            # Create frame metrics entry
            frame_metric = {
                'frame_number': frame_number,
                'timestamp': timestamp,
                'processing_time_ms': processing_time,
                'faces_detected': summary.get('total_faces_detected', 0),
                'known_faces': summary.get('known_faces', 0),
                'live_faces': summary.get('live_faces', 0),
                'engaged_persons': summary.get('engaged_persons', 0),
                'attention_rate': summary.get('attention_rate', 0.0),
                'participation_rate': summary.get('participation_rate', 0.0),
                'fps': performance.get('fps', 0),
                'cpu_usage': performance.get('cpu_usage', 0),
                'memory_usage': performance.get('memory_usage', 0)
            }
            
            # Add to frame metrics
            self.frame_metrics.append(frame_metric)
            
            # Update counters
            self.total_frames += 1
            self.total_faces_detected += frame_metric['faces_detected']
            self.total_faces_recognized += frame_metric['known_faces']
            self.total_live_faces += frame_metric['live_faces']
            self.total_engaged_persons += frame_metric['engaged_persons']
            
            # Update performance tracking
            self.processing_times.append(processing_time)
            if frame_metric['fps'] > 0:
                self.fps_measurements.append(frame_metric['fps'])
            
            # Update session metrics
            if self.current_session_id:
                self.session_metrics['frames_processed'] += 1
                self.session_metrics['total_detections'] += frame_metric['faces_detected']
                self.session_metrics['total_recognitions'] += frame_metric['known_faces']
                self.session_metrics['total_engagements'] += frame_metric['engaged_persons']
            
        except Exception as e:
            logger.error(f"Frame metrics update failed: {e}")
    
    def add_attendance_prediction(self, predicted: List[str], ground_truth: List[str]):
        """
        Add attendance prediction for accuracy calculation.
        
        Args:
            predicted: List of predicted person IDs
            ground_truth: List of actual person IDs
        """
        try:
            self.attendance_predictions.append(set(predicted))
            self.attendance_ground_truth.append(set(ground_truth))
            
            # Keep only recent predictions
            max_predictions = 1000
            if len(self.attendance_predictions) > max_predictions:
                self.attendance_predictions = self.attendance_predictions[-max_predictions:]
                self.attendance_ground_truth = self.attendance_ground_truth[-max_predictions:]
            
        except Exception as e:
            logger.error(f"Attendance prediction tracking failed: {e}")
    
    def add_engagement_prediction(self, predicted: List[float], ground_truth: List[float]):
        """
        Add engagement prediction for accuracy calculation.
        
        Args:
            predicted: List of predicted engagement scores
            ground_truth: List of actual engagement scores
        """
        try:
            self.engagement_predictions.extend(predicted)
            self.engagement_ground_truth.extend(ground_truth)
            
            # Keep only recent predictions
            max_predictions = 5000
            if len(self.engagement_predictions) > max_predictions:
                self.engagement_predictions = self.engagement_predictions[-max_predictions:]
                self.engagement_ground_truth = self.engagement_ground_truth[-max_predictions:]
            
        except Exception as e:
            logger.error(f"Engagement prediction tracking failed: {e}")
    
    def calculate_attendance_accuracy(self) -> float:
        """Calculate attendance detection accuracy."""
        try:
            if not self.attendance_predictions or not self.attendance_ground_truth:
                return 0.0
            
            total_accuracy = 0.0
            count = 0
            
            for pred, truth in zip(self.attendance_predictions, self.attendance_ground_truth):
                if len(truth) == 0:
                    # If no ground truth, accuracy is 1.0 if no predictions, 0.0 otherwise
                    accuracy = 1.0 if len(pred) == 0 else 0.0
                else:
                    # Calculate Jaccard similarity (IoU for sets)
                    intersection = len(pred.intersection(truth))
                    union = len(pred.union(truth))
                    accuracy = intersection / union if union > 0 else 0.0
                
                total_accuracy += accuracy
                count += 1
            
            return total_accuracy / count if count > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Attendance accuracy calculation failed: {e}")
            return 0.0
    
    def calculate_engagement_precision(self) -> float:
        """Calculate engagement analysis precision."""
        try:
            if not self.engagement_predictions or not self.engagement_ground_truth:
                return 0.0
            
            # Calculate Mean Absolute Error and convert to precision
            mae = np.mean(np.abs(np.array(self.engagement_predictions) - np.array(self.engagement_ground_truth)))
            
            # Convert MAE to precision (1.0 - normalized_mae)
            precision = max(0.0, 1.0 - mae)
            
            return precision
            
        except Exception as e:
            logger.error(f"Engagement precision calculation failed: {e}")
            return 0.0
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get current real-time metrics."""
        try:
            current_time = time.time()
            
            # Calculate rates
            session_duration = current_time - self.session_start_time if self.session_start_time else 1
            
            # Recent performance metrics
            recent_frames = list(self.frame_metrics)[-10:] if self.frame_metrics else []
            
            avg_processing_time = np.mean([f['processing_time_ms'] for f in recent_frames]) if recent_frames else 0
            avg_fps = np.mean([f['fps'] for f in recent_frames if f['fps'] > 0]) if recent_frames else 0
            avg_attention_rate = np.mean([f['attention_rate'] for f in recent_frames]) if recent_frames else 0
            avg_participation_rate = np.mean([f['participation_rate'] for f in recent_frames]) if recent_frames else 0
            
            metrics = {
                'session_info': {
                    'session_id': self.current_session_id,
                    'duration_seconds': session_duration,
                    'frames_processed': self.total_frames
                },
                'detection_metrics': {
                    'total_faces_detected': self.total_faces_detected,
                    'total_faces_recognized': self.total_faces_recognized,
                    'total_live_faces': self.total_live_faces,
                    'recognition_rate': self.total_faces_recognized / max(1, self.total_faces_detected),
                    'liveness_rate': self.total_live_faces / max(1, self.total_faces_detected)
                },
                'engagement_metrics': {
                    'total_engaged_persons': self.total_engaged_persons,
                    'avg_attention_rate': avg_attention_rate,
                    'avg_participation_rate': avg_participation_rate,
                    'engagement_rate': self.total_engaged_persons / max(1, self.total_faces_detected)
                },
                'performance_metrics': {
                    'avg_processing_time_ms': avg_processing_time,
                    'avg_fps': avg_fps,
                    'target_fps': self.fps_target,
                    'fps_achievement': min(1.0, avg_fps / self.fps_target) if self.fps_target > 0 else 0,
                    'latency_compliance': avg_processing_time <= self.max_latency_ms
                },
                'accuracy_metrics': {
                    'attendance_accuracy': self.calculate_attendance_accuracy(),
                    'engagement_precision': self.calculate_engagement_precision(),
                    'attendance_target': self.attendance_accuracy_target,
                    'engagement_target': self.engagement_precision_target
                },
                'timestamp': current_time
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Real-time metrics calculation failed: {e}")
            return {}
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get comprehensive session summary."""
        try:
            if not self.session_start_time:
                return {}
            
            current_time = time.time()
            session_duration = current_time - self.session_start_time
            
            # Calculate overall statistics
            all_frames = list(self.frame_metrics)
            
            if not all_frames:
                return {'session_id': self.current_session_id, 'duration': session_duration}
            
            # Processing performance
            processing_times = [f['processing_time_ms'] for f in all_frames]
            fps_values = [f['fps'] for f in all_frames if f['fps'] > 0]
            
            # Detection statistics
            faces_per_frame = [f['faces_detected'] for f in all_frames]
            recognition_rates = [f['known_faces'] / max(1, f['faces_detected']) for f in all_frames]
            
            # Engagement statistics
            attention_rates = [f['attention_rate'] for f in all_frames]
            participation_rates = [f['participation_rate'] for f in all_frames]
            
            summary = {
                'session_info': {
                    'session_id': self.current_session_id,
                    'start_time': self.session_start_time,
                    'end_time': current_time,
                    'duration_seconds': session_duration,
                    'total_frames': len(all_frames)
                },
                'performance_summary': {
                    'avg_processing_time_ms': np.mean(processing_times),
                    'min_processing_time_ms': np.min(processing_times),
                    'max_processing_time_ms': np.max(processing_times),
                    'std_processing_time_ms': np.std(processing_times),
                    'avg_fps': np.mean(fps_values) if fps_values else 0,
                    'fps_target_achievement': np.mean([min(1.0, fps / self.fps_target) for fps in fps_values]) if fps_values else 0
                },
                'detection_summary': {
                    'total_faces_detected': sum(faces_per_frame),
                    'avg_faces_per_frame': np.mean(faces_per_frame),
                    'max_faces_per_frame': np.max(faces_per_frame),
                    'avg_recognition_rate': np.mean(recognition_rates),
                    'total_recognitions': self.total_faces_recognized,
                    'total_live_detections': self.total_live_faces
                },
                'engagement_summary': {
                    'avg_attention_rate': np.mean(attention_rates),
                    'avg_participation_rate': np.mean(participation_rates),
                    'total_engaged_detections': self.total_engaged_persons,
                    'attention_rate_std': np.std(attention_rates),
                    'participation_rate_std': np.std(participation_rates)
                },
                'accuracy_summary': {
                    'attendance_accuracy': self.calculate_attendance_accuracy(),
                    'engagement_precision': self.calculate_engagement_precision(),
                    'attendance_target_met': self.calculate_attendance_accuracy() >= self.attendance_accuracy_target,
                    'engagement_target_met': self.calculate_engagement_precision() >= self.engagement_precision_target
                },
                'compliance_summary': {
                    'latency_compliance_rate': np.mean([t <= self.max_latency_ms for t in processing_times]),
                    'fps_compliance_rate': np.mean([fps >= self.fps_target * 0.9 for fps in fps_values]) if fps_values else 0,
                    'overall_performance_score': self._calculate_performance_score()
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Session summary generation failed: {e}")
            return {}
    
    def _calculate_performance_score(self) -> float:
        """Calculate overall performance score (0-1)."""
        try:
            scores = []
            
            # Attendance accuracy score
            attendance_acc = self.calculate_attendance_accuracy()
            scores.append(min(1.0, attendance_acc / self.attendance_accuracy_target))
            
            # Engagement precision score
            engagement_prec = self.calculate_engagement_precision()
            scores.append(min(1.0, engagement_prec / self.engagement_precision_target))
            
            # FPS performance score
            if self.fps_measurements:
                avg_fps = np.mean(self.fps_measurements)
                scores.append(min(1.0, avg_fps / self.fps_target))
            
            # Latency performance score
            if self.processing_times:
                avg_latency = np.mean(self.processing_times)
                latency_score = max(0.0, 1.0 - (avg_latency / self.max_latency_ms))
                scores.append(latency_score)
            
            return np.mean(scores) if scores else 0.0
            
        except Exception as e:
            logger.error(f"Performance score calculation failed: {e}")
            return 0.0
    
    def export_metrics(self, filepath: str, format: str = "json"):
        """
        Export metrics to file.
        
        Args:
            filepath: Output file path
            format: Export format ("json" or "csv")
        """
        try:
            if format.lower() == "json":
                metrics_data = {
                    'session_summary': self.get_session_summary(),
                    'real_time_metrics': self.get_real_time_metrics(),
                    'frame_metrics': list(self.frame_metrics),
                    'export_timestamp': time.time()
                }
                
                with open(filepath, 'w') as f:
                    json.dump(metrics_data, f, indent=2)
            
            elif format.lower() == "csv":
                import pandas as pd
                
                # Convert frame metrics to DataFrame
                df = pd.DataFrame(list(self.frame_metrics))
                df.to_csv(filepath, index=False)
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            logger.info(f"Metrics exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Metrics export failed: {e}")
    
    def reset_session_metrics(self):
        """Reset metrics for new session."""
        self.session_metrics = {}
        self.session_start_time = None
        self.current_session_id = None
        self.total_frames = 0
        self.total_faces_detected = 0
        self.total_faces_recognized = 0
        self.total_live_faces = 0
        self.total_engaged_persons = 0
        
        logger.info("Session metrics reset")
    
    def get_performance_trends(self, window_size: int = 50) -> Dict[str, List[float]]:
        """
        Get performance trends over recent frames.
        
        Args:
            window_size: Number of recent frames to analyze
            
        Returns:
            Dictionary of performance trends
        """
        try:
            recent_frames = list(self.frame_metrics)[-window_size:]
            
            if not recent_frames:
                return {}
            
            trends = {
                'processing_time_trend': [f['processing_time_ms'] for f in recent_frames],
                'fps_trend': [f['fps'] for f in recent_frames],
                'attention_rate_trend': [f['attention_rate'] for f in recent_frames],
                'participation_rate_trend': [f['participation_rate'] for f in recent_frames],
                'faces_detected_trend': [f['faces_detected'] for f in recent_frames],
                'frame_numbers': [f['frame_number'] for f in recent_frames]
            }
            
            return trends
            
        except Exception as e:
            logger.error(f"Performance trends calculation failed: {e}")
            return {}
