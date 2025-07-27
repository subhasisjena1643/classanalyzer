"""
Tracking Overlay System - Visual display for face tracking and alerts
Provides real-time visual feedback for tracked faces and missing alerts
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from loguru import logger
import time
import math


class TrackingOverlay:
    """
    Visual overlay system for face tracking display.
    Features:
    - Face bounding boxes with unique colors
    - Face ID labels and tracking information
    - Missing face countdown alerts
    - Lock indicators and tracking status
    - Performance statistics display
    """
    
    def __init__(self, config: Any = None):
        """Initialize tracking overlay system."""
        self.config = config
        
        # Visual settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2
        self.line_thickness = 2
        
        # Colors
        self.colors = {
            'active': (0, 255, 0),      # Green
            'missing': (0, 165, 255),   # Orange
            'lost': (0, 0, 255),        # Red
            'reidentified': (255, 0, 255), # Magenta
            'alert_bg': (0, 0, 0),      # Black
            'alert_text': (255, 255, 255), # White
            'countdown': (0, 0, 255),   # Red
            'info_bg': (0, 0, 0),       # Black
            'info_text': (255, 255, 255) # White
        }
        
        # Alert animation
        self.alert_blink_interval = 0.5  # seconds
        self.last_blink_time = time.time()
        self.blink_state = True
        
        # Layout settings
        self.margin = 10
        self.alert_panel_height = 150
        self.info_panel_width = 300
        
        logger.info("Tracking Overlay initialized")
    
    def draw_tracking_overlay(self, frame: np.ndarray, tracking_data: Dict[str, Any]) -> np.ndarray:
        """
        Draw complete tracking overlay on frame.
        
        Args:
            frame: Input frame
            tracking_data: Tracking data from FaceTrackingSystem
            
        Returns:
            Frame with tracking overlay
        """
        try:
            overlay_frame = frame.copy()
            
            # Get visual data
            visual_data = tracking_data.get('visual_data', {})
            
            # Draw tracked faces
            self._draw_tracked_faces(overlay_frame, visual_data.get('tracked_faces', []))
            
            # Draw alerts panel
            self._draw_alerts_panel(overlay_frame, visual_data.get('alerts', []))
            
            # Draw tracking statistics
            self._draw_tracking_stats(overlay_frame, visual_data.get('tracking_stats', {}))
            
            # Draw frame information
            self._draw_frame_info(overlay_frame, tracking_data)
            
            return overlay_frame
            
        except Exception as e:
            logger.error(f"Tracking overlay drawing failed: {e}")
            return frame
    
    def _draw_tracked_faces(self, frame: np.ndarray, tracked_faces: List[Dict]):
        """Draw tracked face overlays."""
        try:
            for face_data in tracked_faces:
                face_id = face_data.get('face_id', 'unknown')
                bbox = face_data.get('bbox', [])
                state = face_data.get('state', 'active')
                tracking_color = face_data.get('tracking_color', (0, 255, 0))
                confidence = face_data.get('confidence', 0.0)
                
                if len(bbox) != 4:
                    continue
                
                x1, y1, x2, y2 = map(int, bbox)
                
                # Determine color based on state
                if state == 'active':
                    color = tracking_color
                    thickness = self.line_thickness
                elif state == 'missing':
                    color = self.colors['missing']
                    thickness = self.line_thickness + 1
                    # Blinking effect for missing faces
                    if self._should_blink():
                        color = self.colors['countdown']
                elif state == 'lost':
                    color = self.colors['lost']
                    thickness = 1
                elif state == 'reidentified':
                    color = self.colors['reidentified']
                    thickness = self.line_thickness + 2
                else:
                    color = tracking_color
                    thickness = self.line_thickness
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                
                # Draw lock indicator for active faces
                if state == 'active' and face_data.get('lock_indicator', True):
                    self._draw_lock_indicator(frame, (x1, y1), color)
                
                # Draw face ID label
                label = f"ID: {face_id}"
                if face_data.get('person_id'):
                    label += f" ({face_data['person_id']})"
                
                label_size = cv2.getTextSize(label, self.font, self.font_scale, self.font_thickness)[0]
                label_bg_coords = (x1, y1 - label_size[1] - 10, x1 + label_size[0] + 10, y1)
                
                # Draw label background
                cv2.rectangle(frame, (label_bg_coords[0], label_bg_coords[1]), 
                            (label_bg_coords[2], label_bg_coords[3]), color, -1)
                
                # Draw label text
                cv2.putText(frame, label, (x1 + 5, y1 - 5), self.font, 
                          self.font_scale, (255, 255, 255), self.font_thickness)
                
                # Draw additional info for missing faces
                if state == 'missing':
                    countdown = face_data.get('countdown_remaining', 0)
                    countdown_text = f"Missing: {countdown:.1f}s"
                    
                    # Position countdown below the face
                    countdown_y = y2 + 25
                    countdown_size = cv2.getTextSize(countdown_text, self.font, 
                                                   self.font_scale, self.font_thickness)[0]
                    
                    # Draw countdown background
                    cv2.rectangle(frame, (x1, countdown_y - countdown_size[1] - 5), 
                                (x1 + countdown_size[0] + 10, countdown_y + 5), 
                                self.colors['alert_bg'], -1)
                    
                    # Draw countdown text
                    cv2.putText(frame, countdown_text, (x1 + 5, countdown_y), 
                              self.font, self.font_scale, self.colors['countdown'], 
                              self.font_thickness)
                
                # Draw tracking duration and frame count
                duration = face_data.get('tracking_duration', 0)
                total_frames = face_data.get('total_frames', 0)
                info_text = f"T:{duration:.1f}s F:{total_frames}"
                
                info_y = y2 + 50 if state == 'missing' else y2 + 25
                info_size = cv2.getTextSize(info_text, self.font, 0.4, 1)[0]
                
                cv2.rectangle(frame, (x1, info_y - info_size[1] - 3), 
                            (x1 + info_size[0] + 6, info_y + 3), 
                            (0, 0, 0), -1)
                
                cv2.putText(frame, info_text, (x1 + 3, info_y), 
                          self.font, 0.4, (255, 255, 255), 1)
                
                # Draw velocity indicator for moving faces
                velocity = face_data.get('velocity', (0, 0))
                if abs(velocity[0]) > 1 or abs(velocity[1]) > 1:
                    self._draw_velocity_indicator(frame, face_data.get('center', (x1, y1)), 
                                                velocity, color)
                
        except Exception as e:
            logger.error(f"Tracked faces drawing failed: {e}")
    
    def _draw_lock_indicator(self, frame: np.ndarray, position: Tuple[int, int], color: Tuple[int, int, int]):
        """Draw lock indicator for tracked faces."""
        try:
            x, y = position
            lock_size = 12
            
            # Draw lock body
            cv2.rectangle(frame, (x - lock_size, y - lock_size), 
                         (x - lock_size + 8, y - lock_size + 6), color, -1)
            
            # Draw lock shackle
            cv2.ellipse(frame, (x - lock_size + 4, y - lock_size - 2), 
                       (3, 4), 0, 0, 180, color, 2)
            
        except Exception as e:
            logger.error(f"Lock indicator drawing failed: {e}")
    
    def _draw_velocity_indicator(self, frame: np.ndarray, center: Tuple[float, float], 
                               velocity: Tuple[float, float], color: Tuple[int, int, int]):
        """Draw velocity indicator arrow."""
        try:
            cx, cy = map(int, center)
            vx, vy = velocity
            
            # Scale velocity for visualization
            scale = 5
            end_x = int(cx + vx * scale)
            end_y = int(cy + vy * scale)
            
            # Draw velocity arrow
            cv2.arrowedLine(frame, (cx, cy), (end_x, end_y), color, 2, tipLength=0.3)
            
        except Exception as e:
            logger.error(f"Velocity indicator drawing failed: {e}")
    
    def _draw_alerts_panel(self, frame: np.ndarray, alerts: List[Dict]):
        """Draw alerts panel for missing faces."""
        try:
            if not alerts:
                return
            
            frame_height, frame_width = frame.shape[:2]
            panel_y = frame_height - self.alert_panel_height
            
            # Draw alert panel background
            cv2.rectangle(frame, (0, panel_y), (frame_width, frame_height), 
                         self.colors['alert_bg'], -1)
            
            # Draw alert panel header
            header_text = f"MISSING FACES ALERT ({len(alerts)})"
            header_size = cv2.getTextSize(header_text, self.font, 0.8, 2)[0]
            header_x = (frame_width - header_size[0]) // 2
            
            cv2.putText(frame, header_text, (header_x, panel_y + 30), 
                       self.font, 0.8, self.colors['alert_text'], 2)
            
            # Draw individual alerts
            alert_y = panel_y + 60
            for i, alert in enumerate(alerts[:3]):  # Show max 3 alerts
                face_id = alert.get('face_id', 'unknown')
                countdown = alert.get('countdown', 0)
                message = f"Face {face_id}: {countdown:.1f}s remaining"
                
                # Blinking effect for urgent alerts
                text_color = self.colors['alert_text']
                if countdown < 10 and self._should_blink():
                    text_color = self.colors['countdown']
                
                cv2.putText(frame, message, (self.margin, alert_y + i * 25), 
                          self.font, self.font_scale, text_color, self.font_thickness)
            
            # Show "and more" if there are additional alerts
            if len(alerts) > 3:
                more_text = f"... and {len(alerts) - 3} more"
                cv2.putText(frame, more_text, (self.margin, alert_y + 75), 
                          self.font, 0.5, self.colors['alert_text'], 1)
                
        except Exception as e:
            logger.error(f"Alerts panel drawing failed: {e}")
    
    def _draw_tracking_stats(self, frame: np.ndarray, tracking_stats: Dict):
        """Draw tracking statistics panel."""
        try:
            if not tracking_stats:
                return
            
            frame_height, frame_width = frame.shape[:2]
            panel_x = frame_width - self.info_panel_width
            
            # Draw stats panel background
            cv2.rectangle(frame, (panel_x, 0), (frame_width, 120), 
                         self.colors['info_bg'], -1)
            
            # Draw stats header
            cv2.putText(frame, "TRACKING STATS", (panel_x + 10, 25), 
                       self.font, 0.6, self.colors['info_text'], 2)
            
            # Draw individual stats
            stats_text = [
                f"Total: {tracking_stats.get('total_tracked', 0)}",
                f"Active: {tracking_stats.get('active_faces', 0)}",
                f"Missing: {tracking_stats.get('missing_faces', 0)}",
                f"Lost: {tracking_stats.get('lost_faces', 0)}",
                f"Alerts: {tracking_stats.get('active_alerts', 0)}"
            ]
            
            for i, text in enumerate(stats_text):
                cv2.putText(frame, text, (panel_x + 10, 50 + i * 15), 
                          self.font, 0.4, self.colors['info_text'], 1)
                
        except Exception as e:
            logger.error(f"Tracking stats drawing failed: {e}")
    
    def _draw_frame_info(self, frame: np.ndarray, tracking_data: Dict):
        """Draw frame information."""
        try:
            frame_number = tracking_data.get('frame_number', 0)
            timestamp = tracking_data.get('timestamp', time.time())
            
            # Format timestamp
            time_str = time.strftime("%H:%M:%S", time.localtime(timestamp))
            
            info_text = f"Frame: {frame_number} | Time: {time_str}"
            
            # Draw frame info at top left
            cv2.putText(frame, info_text, (self.margin, 25), 
                       self.font, 0.5, self.colors['info_text'], 1)
            
        except Exception as e:
            logger.error(f"Frame info drawing failed: {e}")
    
    def _should_blink(self) -> bool:
        """Determine if blinking elements should be visible."""
        current_time = time.time()
        if current_time - self.last_blink_time > self.alert_blink_interval:
            self.blink_state = not self.blink_state
            self.last_blink_time = current_time
        return self.blink_state
    
    def draw_tracking_heatmap(self, frame: np.ndarray, tracked_faces: List[Dict], 
                            history_length: int = 100) -> np.ndarray:
        """Draw movement heatmap for tracked faces."""
        try:
            heatmap_frame = frame.copy()
            
            for face_data in tracked_faces:
                position_history = face_data.get('position_history', [])
                color = face_data.get('tracking_color', (0, 255, 0))
                
                if len(position_history) < 2:
                    continue
                
                # Draw movement trail
                for i in range(1, min(len(position_history), history_length)):
                    alpha = i / min(len(position_history), history_length)
                    point1 = tuple(map(int, position_history[i-1]))
                    point2 = tuple(map(int, position_history[i]))
                    
                    # Fade color based on age
                    faded_color = tuple(int(c * alpha) for c in color)
                    
                    cv2.line(heatmap_frame, point1, point2, faded_color, 2)
                    cv2.circle(heatmap_frame, point2, 3, faded_color, -1)
            
            return heatmap_frame
            
        except Exception as e:
            logger.error(f"Tracking heatmap drawing failed: {e}")
            return frame
    
    def create_tracking_dashboard(self, tracking_data: Dict, frame_size: Tuple[int, int]) -> np.ndarray:
        """Create a comprehensive tracking dashboard."""
        try:
            width, height = frame_size
            dashboard = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Title
            title = "FACE TRACKING DASHBOARD"
            title_size = cv2.getTextSize(title, self.font, 1.0, 2)[0]
            title_x = (width - title_size[0]) // 2
            
            cv2.putText(dashboard, title, (title_x, 40), 
                       self.font, 1.0, (255, 255, 255), 2)
            
            # Tracking statistics
            stats = tracking_data.get('tracking_statistics', {})
            
            stats_info = [
                f"Total Faces Tracked: {stats.get('total_faces_tracked', 0)}",
                f"Successful Re-identifications: {stats.get('successful_reidentifications', 0)}",
                f"Currently Active: {stats.get('current_active_tracks', 0)}",
                f"Currently Missing: {stats.get('current_missing_tracks', 0)}",
                f"Currently Lost: {stats.get('current_lost_tracks', 0)}",
                f"Average Tracking Duration: {stats.get('average_tracking_duration', 0):.1f}s"
            ]
            
            for i, info in enumerate(stats_info):
                cv2.putText(dashboard, info, (50, 100 + i * 30), 
                          self.font, 0.6, (255, 255, 255), 1)
            
            # Active alerts summary
            alerts = tracking_data.get('active_alerts', {})
            if alerts:
                alert_title = f"ACTIVE ALERTS ({len(alerts)})"
                cv2.putText(dashboard, alert_title, (50, 300), 
                          self.font, 0.8, (255, 0, 0), 2)
                
                for i, (alert_id, alert_data) in enumerate(list(alerts.items())[:5]):
                    alert_text = f"Face {alert_data.get('face_id', 'unknown')}: Missing"
                    cv2.putText(dashboard, alert_text, (50, 330 + i * 25), 
                              self.font, 0.5, (255, 100, 100), 1)
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Tracking dashboard creation failed: {e}")
            return np.zeros((480, 640, 3), dtype=np.uint8)
    
    def get_overlay_settings(self) -> Dict[str, Any]:
        """Get current overlay settings."""
        return {
            'font': self.font,
            'font_scale': self.font_scale,
            'font_thickness': self.font_thickness,
            'line_thickness': self.line_thickness,
            'colors': self.colors.copy(),
            'alert_blink_interval': self.alert_blink_interval,
            'margin': self.margin,
            'alert_panel_height': self.alert_panel_height,
            'info_panel_width': self.info_panel_width
        }
    
    def update_overlay_settings(self, settings: Dict[str, Any]):
        """Update overlay settings."""
        try:
            for key, value in settings.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            
            logger.info("Overlay settings updated")
            
        except Exception as e:
            logger.error(f"Overlay settings update failed: {e}")
