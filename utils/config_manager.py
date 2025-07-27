"""
Configuration Manager for System Settings
Handles loading, validation, and dynamic updates of configuration
"""

import yaml
import json
import platform
import sys
from typing import Any, Dict, Optional, Union
from pathlib import Path
import os
from loguru import logger

def get_system_config():
    """Get system-specific configuration"""
    system = platform.system().lower()

    # Default camera settings based on system
    camera_configs = {
        'windows': {
            'device_id': 0,
            'preferred_backend': 'dshow',
            'buffer_size': 1
        },
        'linux': {
            'device_id': 0,
            'preferred_backend': 'v4l2',
            'buffer_size': 1
        },
        'darwin': {  # macOS
            'device_id': 0,
            'preferred_backend': 'avfoundation',
            'buffer_size': 1
        }
    }

    return camera_configs.get(system, camera_configs['windows'])

def get_default_paths():
    """Get default paths based on operating system"""
    system = platform.system().lower()
    home = Path.home()

    if system == 'windows':
        return {
            'data_dir': home / 'Documents' / 'PIPER_Data',
            'cache_dir': home / 'AppData' / 'Local' / 'PIPER' / 'Cache',
            'log_dir': home / 'AppData' / 'Local' / 'PIPER' / 'Logs'
        }
    elif system == 'darwin':  # macOS
        return {
            'data_dir': home / 'Documents' / 'PIPER_Data',
            'cache_dir': home / 'Library' / 'Caches' / 'PIPER',
            'log_dir': home / 'Library' / 'Logs' / 'PIPER'
        }
    else:  # Linux
        return {
            'data_dir': home / '.local' / 'share' / 'piper',
            'cache_dir': home / '.cache' / 'piper',
            'log_dir': home / '.local' / 'share' / 'piper' / 'logs'
        }

def detect_optimal_settings():
    """Detect optimal settings based on system capabilities"""
    import psutil

    # Get system specs
    cpu_count = psutil.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)

    # Determine optimal settings
    if memory_gb >= 16 and cpu_count >= 8:
        return {
            'performance_mode': 'high',
            'ai_processing_interval': 3,
            'max_faces': 10,
            'use_gpu': True
        }
    elif memory_gb >= 8 and cpu_count >= 4:
        return {
            'performance_mode': 'medium',
            'ai_processing_interval': 5,
            'max_faces': 5,
            'use_gpu': True
        }
    else:
        return {
            'performance_mode': 'low',
            'ai_processing_interval': 10,
            'max_faces': 3,
            'use_gpu': False
        }


class ConfigManager:
    """
    Manages system configuration with support for:
    - YAML and JSON configuration files
    - Environment variable overrides
    - Dynamic configuration updates
    - Configuration validation
    """
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        self.config_data = {}
        self.env_prefix = "CLASSROOM_ANALYZER_"
        
        # Load configuration
        self._load_config()
        
        # Apply environment variable overrides
        self._apply_env_overrides()
        
        logger.info(f"Configuration loaded from {config_path}")
    
    def _load_config(self):
        """Load configuration from file."""
        try:
            if not self.config_path.exists():
                logger.warning(f"Config file not found: {self.config_path}")
                self.config_data = self._get_default_config()
                return
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                if self.config_path.suffix.lower() == '.yaml' or self.config_path.suffix.lower() == '.yml':
                    self.config_data = yaml.safe_load(f)
                elif self.config_path.suffix.lower() == '.json':
                    self.config_data = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {self.config_path.suffix}")
            
            # Validate configuration
            self._validate_config()
            
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            self.config_data = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "system": {
                "device": "cuda",
                "precision": "fp16",
                "batch_size": 8,
                "max_concurrent_streams": 4,
                "processing_fps": 30,
                "target_latency_ms": 5000
            },
            "camera": {
                "input_source": "phone",
                "resolution": {
                    "width": 1920,
                    "height": 1080
                },
                "phone_connection": {
                    "ip_camera_url": "http://192.168.1.100:8080/video",
                    "droidcam_port": 4747
                },
                "preprocessing": {
                    "normalize": True,
                    "enhance_low_light": True,
                    "denoise": True
                }
            },
            "face_detection": {
                "model_type": "retinaface",
                "confidence_threshold": 0.7,
                "nms_threshold": 0.4,
                "max_faces_per_frame": 50,
                "face_size_min": 30,
                "liveness_detection": True,
                "anti_spoofing": True
            },
            "face_recognition": {
                "embedding_model": "arcface",
                "embedding_dim": 512,
                "similarity_threshold": 0.6,
                "max_gallery_size": 1000,
                "update_embeddings": True
            },
            "engagement": {
                "gesture_recognition": {
                    "model": "mediapipe_holistic",
                    "hand_tracking": True,
                    "pose_estimation": True,
                    "confidence_threshold": 0.5
                },
                "attention_detection": {
                    "head_pose_estimation": True,
                    "eye_gaze_tracking": True,
                    "facial_expression_analysis": True,
                    "attention_threshold": 0.6
                },
                "participation_metrics": {
                    "hand_raise_detection": True,
                    "speaking_detection": False,
                    "interaction_tracking": True
                }
            },
            "reinforcement_learning": {
                "enabled": True,
                "algorithm": "PPO",
                "learning_rate": 0.0003,
                "update_frequency": 100,
                "reward_shaping": {
                    "attendance_accuracy_weight": 0.4,
                    "engagement_precision_weight": 0.3,
                    "latency_penalty_weight": 0.2,
                    "false_positive_penalty": 0.1
                },
                "experience_replay": {
                    "buffer_size": 10000,
                    "batch_size": 64,
                    "min_experiences": 1000
                }
            },
            "privacy": {
                "anonymize_embeddings": True,
                "hash_identifiers": True,
                "local_processing_only": True,
                "data_retention_hours": 24,
                "encryption_enabled": True,
                "consent_required": True
            },
            "optimization": {
                "model_quantization": True,
                "tensorrt_optimization": False,
                "onnx_export": True,
                "multi_threading": True,
                "gpu_memory_fraction": 0.8
            },
            "training": {
                "initial_dataset_path": "data/training",
                "validation_split": 0.2,
                "epochs": 100,
                "early_stopping_patience": 10,
                "learning_rate_scheduler": "cosine",
                "augmentation_enabled": True
            },
            "metrics": {
                "attendance_accuracy_target": 0.98,
                "engagement_precision_target": 0.70,
                "max_latency_ms": 5000,
                "fps_target": 30,
                "memory_usage_limit_gb": 8
            },
            "logging": {
                "level": "INFO",
                "save_predictions": True,
                "save_metrics": True,
                "wandb_project": "classroom-analyzer",
                "tensorboard_enabled": True
            }
        }
    
    def _validate_config(self):
        """Validate configuration values."""
        try:
            # Validate system settings
            system = self.config_data.get("system", {})
            if system.get("batch_size", 1) < 1:
                logger.warning("Invalid batch_size, setting to 1")
                system["batch_size"] = 1
            
            if system.get("processing_fps", 30) < 1:
                logger.warning("Invalid processing_fps, setting to 30")
                system["processing_fps"] = 30
            
            # Validate thresholds
            face_detection = self.config_data.get("face_detection", {})
            confidence = face_detection.get("confidence_threshold", 0.7)
            if not 0.0 <= confidence <= 1.0:
                logger.warning("Invalid confidence_threshold, setting to 0.7")
                face_detection["confidence_threshold"] = 0.7
            
            # Validate privacy settings
            privacy = self.config_data.get("privacy", {})
            retention_hours = privacy.get("data_retention_hours", 24)
            if retention_hours < 1 or retention_hours > 168:  # Max 1 week
                logger.warning("Invalid data_retention_hours, setting to 24")
                privacy["data_retention_hours"] = 24
            
            logger.info("Configuration validation completed")
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides."""
        try:
            env_overrides = {}
            
            for key, value in os.environ.items():
                if key.startswith(self.env_prefix):
                    # Convert environment variable to config key
                    config_key = key[len(self.env_prefix):].lower().replace('_', '.')
                    
                    # Try to convert value to appropriate type
                    converted_value = self._convert_env_value(value)
                    env_overrides[config_key] = converted_value
            
            # Apply overrides
            for key, value in env_overrides.items():
                self._set_nested_value(key, value)
                logger.info(f"Applied environment override: {key} = {value}")
            
        except Exception as e:
            logger.error(f"Environment override application failed: {e}")
    
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type."""
        # Try boolean
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Try integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., "system.batch_size")
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        try:
            keys = key.split('.')
            value = self.config_data
            
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            
            return value
            
        except Exception as e:
            logger.error(f"Failed to get config value for {key}: {e}")
            return default
    
    def set(self, key: str, value: Any):
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., "system.batch_size")
            value: Value to set
        """
        try:
            self._set_nested_value(key, value)
            logger.info(f"Configuration updated: {key} = {value}")
            
        except Exception as e:
            logger.error(f"Failed to set config value for {key}: {e}")
    
    def _set_nested_value(self, key: str, value: Any):
        """Set nested dictionary value using dot notation."""
        keys = key.split('.')
        current = self.config_data
        
        # Navigate to parent of target key
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        # Set the value
        current[keys[-1]] = value
    
    def update(self, updates: Dict[str, Any]):
        """
        Update multiple configuration values.
        
        Args:
            updates: Dictionary of key-value pairs to update
        """
        try:
            for key, value in updates.items():
                self.set(key, value)
            
            logger.info(f"Configuration updated with {len(updates)} changes")
            
        except Exception as e:
            logger.error(f"Configuration update failed: {e}")
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration data."""
        return self.config_data.copy()
    
    def save(self, filepath: str = None):
        """
        Save current configuration to file.
        
        Args:
            filepath: Optional path to save to (defaults to original path)
        """
        try:
            save_path = Path(filepath) if filepath else self.config_path
            
            # Ensure directory exists
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w', encoding='utf-8') as f:
                if save_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(self.config_data, f, default_flow_style=False, indent=2)
                elif save_path.suffix.lower() == '.json':
                    json.dump(self.config_data, f, indent=2)
                else:
                    raise ValueError(f"Unsupported file format: {save_path.suffix}")
            
            logger.info(f"Configuration saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    def reload(self):
        """Reload configuration from file."""
        try:
            self._load_config()
            self._apply_env_overrides()
            logger.info("Configuration reloaded")
            
        except Exception as e:
            logger.error(f"Configuration reload failed: {e}")
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.
        
        Args:
            section: Section name (e.g., "system", "privacy")
            
        Returns:
            Section configuration
        """
        return self.config_data.get(section, {})
    
    def validate_section(self, section: str) -> bool:
        """
        Validate a specific configuration section.
        
        Args:
            section: Section name to validate
            
        Returns:
            True if section is valid
        """
        try:
            section_data = self.get_section(section)
            
            if section == "system":
                return self._validate_system_section(section_data)
            elif section == "privacy":
                return self._validate_privacy_section(section_data)
            elif section == "face_detection":
                return self._validate_face_detection_section(section_data)
            else:
                return True  # No specific validation
                
        except Exception as e:
            logger.error(f"Section validation failed for {section}: {e}")
            return False
    
    def _validate_system_section(self, section: Dict[str, Any]) -> bool:
        """Validate system configuration section."""
        required_fields = ["device", "batch_size", "processing_fps"]
        
        for field in required_fields:
            if field not in section:
                logger.error(f"Missing required system field: {field}")
                return False
        
        if section["batch_size"] < 1:
            logger.error("Invalid batch_size: must be >= 1")
            return False
        
        if section["processing_fps"] < 1:
            logger.error("Invalid processing_fps: must be >= 1")
            return False
        
        return True
    
    def _validate_privacy_section(self, section: Dict[str, Any]) -> bool:
        """Validate privacy configuration section."""
        retention_hours = section.get("data_retention_hours", 24)
        
        if retention_hours < 1 or retention_hours > 168:
            logger.error("Invalid data_retention_hours: must be between 1 and 168")
            return False
        
        return True
    
    def _validate_face_detection_section(self, section: Dict[str, Any]) -> bool:
        """Validate face detection configuration section."""
        confidence = section.get("confidence_threshold", 0.7)
        nms = section.get("nms_threshold", 0.4)
        
        if not 0.0 <= confidence <= 1.0:
            logger.error("Invalid confidence_threshold: must be between 0.0 and 1.0")
            return False
        
        if not 0.0 <= nms <= 1.0:
            logger.error("Invalid nms_threshold: must be between 0.0 and 1.0")
            return False
        
        return True
    
    def export_config(self, format: str = "yaml") -> str:
        """
        Export configuration as string.
        
        Args:
            format: Export format ("yaml" or "json")
            
        Returns:
            Configuration as string
        """
        try:
            if format.lower() == "yaml":
                return yaml.dump(self.config_data, default_flow_style=False, indent=2)
            elif format.lower() == "json":
                return json.dumps(self.config_data, indent=2)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Configuration export failed: {e}")
            return ""
