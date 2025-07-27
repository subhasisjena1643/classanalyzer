#!/usr/bin/env python3
"""
Comprehensive Checkpoint Management System
Handles automatic saving, loading, and recovery of model states during training.

Features:
- Automatic checkpoint saving during RL improvements
- Graceful shutdown handling (Ctrl+C, terminal close)
- Automatic recovery from last checkpoint
- Performance-based checkpoint triggers
- Incremental and full checkpoint support
"""

import os
import json
import pickle
import torch
import time
import signal
import atexit
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from loguru import logger
import threading
import queue


class CheckpointManager:
    """Manages model checkpoints with automatic saving and recovery."""
    
    def __init__(self, checkpoint_dir: str = "outputs/training/checkpoints"):
        """Initialize checkpoint manager."""
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # ENHANCED Checkpoint configuration with cleanup
        self.auto_save_enabled = True
        self.save_interval = 300  # Save every 5 minutes
        self.improvement_threshold = 0.01  # Save when performance improves by 1%
        self.max_checkpoints = 50  # Keep last 50 checkpoints
        self.max_age_days = 7  # Remove checkpoints older than 7 days
        self.cleanup_interval_hours = 6  # Cleanup every 6 hours
        self.keep_best_n = 5  # Always keep 5 best performing checkpoints
        self.max_storage_mb = 1000  # Maximum storage for checkpoints (1GB)
        
        # State tracking
        self.last_save_time = time.time()
        self.last_cleanup_time = time.time()  # Track cleanup timing
        self.last_performance = {}
        self.checkpoint_performance = {}  # Track performance scores
        self.checkpoint_queue = queue.Queue()
        self.shutdown_requested = False
        
        # Current session info
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.checkpoint_counter = 0
        
        # Setup graceful shutdown handlers
        self._setup_shutdown_handlers()
        
        # Start background checkpoint thread
        self.checkpoint_thread = threading.Thread(target=self._checkpoint_worker, daemon=True)
        self.checkpoint_thread.start()
        
        logger.info(f"🔄 Checkpoint Manager initialized")
        logger.info(f"   📁 Directory: {self.checkpoint_dir}")
        logger.info(f"   🆔 Session: {self.session_id}")
        logger.info(f"   ⏱️  Auto-save interval: {self.save_interval}s")
        logger.info(f"   📈 Improvement threshold: {self.improvement_threshold:.1%}")
    
    def _setup_shutdown_handlers(self):
        """Setup handlers for graceful shutdown."""
        # Handle Ctrl+C (SIGINT)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # Handle terminal close (SIGTERM)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Handle program exit
        atexit.register(self._cleanup_on_exit)
        
        logger.info("🛡️  Graceful shutdown handlers registered")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"🛑 Shutdown signal received: {signum}")
        self.shutdown_requested = True
        self._save_emergency_checkpoint()
        logger.info("💾 Emergency checkpoint saved")
        exit(0)
    
    def _cleanup_on_exit(self):
        """Cleanup function called on program exit."""
        if not self.shutdown_requested:
            logger.info("🔄 Program exit detected - saving final checkpoint")
            self._save_emergency_checkpoint()
    
    def _checkpoint_worker(self):
        """Background worker for processing checkpoint saves."""
        while not self.shutdown_requested:
            try:
                # Check for queued checkpoints
                try:
                    checkpoint_data = self.checkpoint_queue.get(timeout=1.0)
                    self._save_checkpoint_to_disk(checkpoint_data)
                    self.checkpoint_queue.task_done()
                except queue.Empty:
                    pass
                
                # Check for time-based auto-save
                if self._should_auto_save():
                    logger.info("⏰ Time-based auto-save triggered")
                    # This will be handled by the main thread calling save_checkpoint
                
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ Checkpoint worker error: {e}")
    
    def save_checkpoint(self, models: Dict[str, Any], performance_metrics: Dict[str, float], 
                       force: bool = False, checkpoint_type: str = "auto") -> bool:
        """
        Save model checkpoint with performance tracking.
        
        Args:
            models: Dictionary of model states to save
            performance_metrics: Current performance metrics
            force: Force save regardless of improvement
            checkpoint_type: Type of checkpoint (auto, manual, emergency, improvement)
        """
        try:
            # Check if we should save based on improvement
            should_save = force or self._should_save_checkpoint(performance_metrics)
            
            if not should_save and checkpoint_type == "auto":
                return False
            
            # Prepare checkpoint data
            checkpoint_data = {
                'session_id': self.session_id,
                'checkpoint_id': f"{self.session_id}_{self.checkpoint_counter:04d}",
                'timestamp': datetime.now().isoformat(),
                'checkpoint_type': checkpoint_type,
                'performance_metrics': performance_metrics,
                'models': {},
                'metadata': {
                    'checkpoint_counter': self.checkpoint_counter,
                    'improvement_detected': self._is_improvement(performance_metrics),
                    'save_reason': self._get_save_reason(performance_metrics, force, checkpoint_type)
                }
            }
            
            # Save model states
            for model_name, model in models.items():
                if hasattr(model, 'state_dict'):
                    checkpoint_data['models'][model_name] = model.state_dict()
                elif isinstance(model, dict):
                    checkpoint_data['models'][model_name] = model
                else:
                    logger.warning(f"⚠️  Cannot save model {model_name}: unsupported type")
            
            # Queue checkpoint for background saving
            self.checkpoint_queue.put(checkpoint_data)
            
            # Update tracking
            self.last_save_time = time.time()
            self.last_performance = performance_metrics.copy()
            self.checkpoint_counter += 1
            
            logger.info(f"💾 Checkpoint queued: {checkpoint_data['checkpoint_id']}")
            logger.info(f"   📊 Performance: {performance_metrics}")
            logger.info(f"   🔄 Type: {checkpoint_type}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save checkpoint: {e}")
            return False
    
    def _save_checkpoint_to_disk(self, checkpoint_data: Dict[str, Any]):
        """Save checkpoint data to disk."""
        try:
            checkpoint_id = checkpoint_data['checkpoint_id']
            checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.pkl"
            
            # Save checkpoint
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            # Save metadata separately for quick access
            metadata_file = self.checkpoint_dir / f"{checkpoint_id}_metadata.json"
            metadata = {
                'checkpoint_id': checkpoint_id,
                'timestamp': checkpoint_data['timestamp'],
                'checkpoint_type': checkpoint_data['checkpoint_type'],
                'performance_metrics': checkpoint_data['performance_metrics'],
                'metadata': checkpoint_data['metadata']
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Update latest checkpoint reference
            self._update_latest_checkpoint(checkpoint_id)
            
            # Clean up old checkpoints
            self._cleanup_old_checkpoints()
            
            logger.info(f"✅ Checkpoint saved: {checkpoint_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save checkpoint to disk: {e}")
    
    def _should_save_checkpoint(self, performance_metrics: Dict[str, float]) -> bool:
        """Determine if checkpoint should be saved based on performance."""
        # Always save if no previous performance data
        if not self.last_performance:
            return True
        
        # Check for improvement
        if self._is_improvement(performance_metrics):
            return True
        
        # Check time-based auto-save
        if self._should_auto_save():
            return True
        
        return False
    
    def _is_improvement(self, performance_metrics: Dict[str, float]) -> bool:
        """Check if current performance is an improvement."""
        if not self.last_performance:
            return True
        
        # Check key metrics for improvement
        key_metrics = ['accuracy', 'engagement_precision', 'attendance_accuracy']
        
        for metric in key_metrics:
            if metric in performance_metrics and metric in self.last_performance:
                current = performance_metrics[metric]
                previous = self.last_performance[metric]
                
                # Check if improvement exceeds threshold
                if current > previous + self.improvement_threshold:
                    logger.info(f"📈 Improvement detected in {metric}: {previous:.3f} → {current:.3f}")
                    return True
        
        return False
    
    def _should_auto_save(self) -> bool:
        """Check if auto-save should trigger based on time."""
        return time.time() - self.last_save_time >= self.save_interval
    
    def _get_save_reason(self, performance_metrics: Dict[str, float], 
                        force: bool, checkpoint_type: str) -> str:
        """Get reason for saving checkpoint."""
        if force:
            return "forced_save"
        elif checkpoint_type == "emergency":
            return "emergency_shutdown"
        elif checkpoint_type == "manual":
            return "manual_save"
        elif self._is_improvement(performance_metrics):
            return "performance_improvement"
        elif self._should_auto_save():
            return "time_based_auto_save"
        else:
            return "unknown"
    
    def _save_emergency_checkpoint(self):
        """Save emergency checkpoint during shutdown."""
        try:
            # Create minimal emergency checkpoint
            emergency_data = {
                'session_id': self.session_id,
                'checkpoint_id': f"{self.session_id}_emergency",
                'timestamp': datetime.now().isoformat(),
                'checkpoint_type': "emergency",
                'performance_metrics': self.last_performance,
                'metadata': {
                    'checkpoint_counter': self.checkpoint_counter,
                    'save_reason': "emergency_shutdown"
                }
            }
            
            emergency_file = self.checkpoint_dir / f"{emergency_data['checkpoint_id']}.pkl"
            with open(emergency_file, 'wb') as f:
                pickle.dump(emergency_data, f)
            
            self._update_latest_checkpoint(emergency_data['checkpoint_id'])
            
            logger.info(f"🚨 Emergency checkpoint saved: {emergency_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save emergency checkpoint: {e}")
    
    def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load the most recent checkpoint."""
        try:
            latest_file = self.checkpoint_dir / "latest_checkpoint.txt"
            
            if not latest_file.exists():
                logger.info("📂 No previous checkpoint found")
                return None
            
            # Read latest checkpoint ID
            with open(latest_file, 'r') as f:
                checkpoint_id = f.read().strip()
            
            # Load checkpoint
            checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.pkl"
            
            if not checkpoint_file.exists():
                logger.warning(f"⚠️  Latest checkpoint file not found: {checkpoint_file}")
                return None
            
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            logger.info(f"📂 Loaded checkpoint: {checkpoint_id}")
            logger.info(f"   📅 Timestamp: {checkpoint_data.get('timestamp', 'Unknown')}")
            logger.info(f"   📊 Performance: {checkpoint_data.get('performance_metrics', {})}")
            
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"❌ Failed to load latest checkpoint: {e}")
            return None
    
    def _update_latest_checkpoint(self, checkpoint_id: str):
        """Update reference to latest checkpoint."""
        try:
            latest_file = self.checkpoint_dir / "latest_checkpoint.txt"
            with open(latest_file, 'w') as f:
                f.write(checkpoint_id)
        except Exception as e:
            logger.error(f"❌ Failed to update latest checkpoint reference: {e}")
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to save space."""
        try:
            # Get all checkpoint files
            checkpoint_files = list(self.checkpoint_dir.glob("session_*.pkl"))
            
            if len(checkpoint_files) <= self.max_checkpoints:
                return
            
            # Sort by modification time (oldest first)
            checkpoint_files.sort(key=lambda x: x.stat().st_mtime)
            
            # Remove oldest checkpoints
            files_to_remove = checkpoint_files[:-self.max_checkpoints]
            
            for file_path in files_to_remove:
                try:
                    file_path.unlink()
                    # Also remove corresponding metadata file
                    metadata_file = file_path.with_suffix('').with_suffix('_metadata.json')
                    if metadata_file.exists():
                        metadata_file.unlink()
                    
                    logger.info(f"🗑️  Removed old checkpoint: {file_path.name}")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to remove old checkpoint {file_path}: {e}")
                    
        except Exception as e:
            logger.error(f"❌ Failed to cleanup old checkpoints: {e}")
    
    def get_checkpoint_history(self) -> List[Dict[str, Any]]:
        """Get history of all available checkpoints."""
        try:
            history = []
            metadata_files = list(self.checkpoint_dir.glob("*_metadata.json"))
            
            for metadata_file in metadata_files:
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    history.append(metadata)
                except Exception as e:
                    logger.warning(f"⚠️  Failed to read metadata {metadata_file}: {e}")
            
            # Sort by timestamp
            history.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            
            return history
            
        except Exception as e:
            logger.error(f"❌ Failed to get checkpoint history: {e}")
            return []
    
    def cleanup_old_checkpoints(self) -> Dict[str, int]:
        """
        ENHANCED CLEANUP: Remove old checkpoints based on age, count, and performance.

        Returns:
            Dictionary with cleanup statistics
        """
        try:
            logger.info("🧹 Starting enhanced checkpoint cleanup...")

            # Get all checkpoint files
            checkpoint_files = list(self.checkpoint_dir.glob("*.pth"))

            if not checkpoint_files:
                logger.info("📁 No checkpoints found for cleanup")
                return {'removed': 0, 'kept': 0}

            removed_count = 0

            # 1. Clean up by age (remove files older than max_age_days)
            removed_count += self._cleanup_by_age(checkpoint_files)

            # 2. Clean up by count (keep only max_checkpoints recent files)
            removed_count += self._cleanup_by_count()

            # 3. Clean up by storage limit
            removed_count += self._cleanup_by_storage()

            # 4. Clean up temporary and corrupted files
            removed_count += self._cleanup_temp_files()

            kept_count = len(list(self.checkpoint_dir.glob("*.pth")))

            logger.info(f"✅ Checkpoint cleanup completed")
            logger.info(f"   🗑️ Removed: {removed_count} files")
            logger.info(f"   📁 Kept: {kept_count} files")

            # Update last cleanup time
            self.last_cleanup_time = time.time()

            return {'removed': removed_count, 'kept': kept_count}

        except Exception as e:
            logger.error(f"❌ Checkpoint cleanup failed: {e}")
            return {'removed': 0, 'kept': 0}

    def _cleanup_by_age(self, checkpoint_files: List[Path]) -> int:
        """Remove checkpoints older than max_age_days."""
        removed = 0
        cutoff_time = time.time() - (self.max_age_days * 24 * 3600)

        for checkpoint_file in checkpoint_files:
            try:
                # Skip best checkpoints from age cleanup
                if 'best' in checkpoint_file.name:
                    continue

                file_age = checkpoint_file.stat().st_mtime
                if file_age < cutoff_time:
                    checkpoint_file.unlink()
                    removed += 1
                    logger.debug(f"🗑️ Removed old checkpoint: {checkpoint_file.name}")

            except Exception as e:
                logger.warning(f"⚠️ Failed to remove old checkpoint {checkpoint_file.name}: {e}")

        return removed

    def _cleanup_by_count(self) -> int:
        """Keep only the most recent checkpoints up to max_checkpoints."""
        checkpoint_files = list(self.checkpoint_dir.glob("*episode*.pth"))

        if len(checkpoint_files) <= self.max_checkpoints:
            return 0

        # Sort by modification time (newest first)
        checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        # Remove oldest checkpoints
        removed = 0
        for checkpoint_file in checkpoint_files[self.max_checkpoints:]:
            try:
                checkpoint_file.unlink()
                removed += 1
                logger.debug(f"🗑️ Removed excess checkpoint: {checkpoint_file.name}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to remove checkpoint {checkpoint_file.name}: {e}")

        return removed

    def _cleanup_by_storage(self) -> int:
        """Remove checkpoints if storage exceeds max_storage_mb."""
        total_size = 0
        checkpoint_files = []

        # Calculate total size and collect files
        for checkpoint_file in self.checkpoint_dir.glob("*.pth"):
            size = checkpoint_file.stat().st_size
            total_size += size
            checkpoint_files.append((checkpoint_file, size))

        total_size_mb = total_size / (1024 * 1024)

        if total_size_mb <= self.max_storage_mb:
            return 0

        logger.info(f"📊 Storage limit exceeded: {total_size_mb:.1f}MB > {self.max_storage_mb}MB")

        # Sort by modification time (oldest first for removal)
        checkpoint_files.sort(key=lambda x: x[0].stat().st_mtime)

        removed = 0
        current_size_mb = total_size_mb

        for checkpoint_file, file_size in checkpoint_files:
            # Skip best checkpoints
            if 'best' in checkpoint_file.name:
                continue

            if current_size_mb <= self.max_storage_mb:
                break

            try:
                checkpoint_file.unlink()
                current_size_mb -= file_size / (1024 * 1024)
                removed += 1
                logger.debug(f"🗑️ Removed for storage: {checkpoint_file.name}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to remove checkpoint {checkpoint_file.name}: {e}")

        return removed

    def _cleanup_temp_files(self) -> int:
        """Remove temporary files and corrupted checkpoints."""
        removed = 0

        # Remove .tmp files
        for temp_file in self.checkpoint_dir.glob("*.tmp"):
            try:
                temp_file.unlink()
                removed += 1
                logger.debug(f"🗑️ Removed temp file: {temp_file.name}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to remove temp file {temp_file.name}: {e}")

        # Check for corrupted checkpoints
        for checkpoint_file in self.checkpoint_dir.glob("*.pth"):
            try:
                # Try to load checkpoint to verify it's not corrupted
                torch.load(checkpoint_file, map_location='cpu')
            except Exception:
                # Corrupted checkpoint, remove it
                try:
                    checkpoint_file.unlink()
                    removed += 1
                    logger.warning(f"🗑️ Removed corrupted checkpoint: {checkpoint_file.name}")
                except Exception as e:
                    logger.error(f"❌ Failed to remove corrupted checkpoint {checkpoint_file.name}: {e}")

        return removed

    def get_storage_info(self) -> Dict[str, float]:
        """Get storage information for checkpoint directory."""
        try:
            total_size = 0
            file_count = 0

            for checkpoint_file in self.checkpoint_dir.glob("*"):
                if checkpoint_file.is_file():
                    total_size += checkpoint_file.stat().st_size
                    file_count += 1

            # Convert to MB
            total_size_mb = total_size / (1024 * 1024)

            return {
                'total_size_mb': total_size_mb,
                'file_count': file_count,
                'directory': str(self.checkpoint_dir),
                'storage_limit_mb': self.max_storage_mb,
                'usage_percentage': (total_size_mb / self.max_storage_mb) * 100 if self.max_storage_mb > 0 else 0
            }

        except Exception as e:
            logger.error(f"❌ Failed to get storage info: {e}")
            return {'total_size_mb': 0.0, 'file_count': 0, 'directory': str(self.checkpoint_dir)}

    def force_cleanup(self) -> Dict[str, int]:
        """Force immediate cleanup regardless of time interval."""
        logger.info("🧹 Forcing immediate checkpoint cleanup...")
        return self.cleanup_old_checkpoints()

    def auto_cleanup_check(self):
        """Check if automatic cleanup should be performed."""
        current_time = time.time()

        # Check if enough time has passed since last cleanup
        if current_time - self.last_cleanup_time > (self.cleanup_interval_hours * 3600):
            logger.info("⏰ Automatic cleanup interval reached")
            self.cleanup_old_checkpoints()

    def shutdown(self):
        """Gracefully shutdown checkpoint manager."""
        logger.info("🔄 Shutting down checkpoint manager...")
        self.shutdown_requested = True

        # Perform final cleanup
        logger.info("🧹 Performing final cleanup before shutdown...")
        self.cleanup_old_checkpoints()

        # Wait for pending checkpoints to save
        if not self.checkpoint_queue.empty():
            logger.info("⏳ Waiting for pending checkpoints to save...")
            self.checkpoint_queue.join()

        logger.info("✅ Checkpoint manager shutdown complete")
