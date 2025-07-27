"""
Automatic Cleanup System for Memory Management
Handles checkpoint cleanup, log rotation, and cache management
"""

import os
import shutil
import time
import gc
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger
import torch


class AutomaticCleanupManager:
    """
    Comprehensive cleanup manager for optimal memory and disk usage.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Cleanup configuration
        self.max_episode_checkpoints = self.config.get("max_episode_checkpoints", 5)
        self.max_log_age_days = self.config.get("max_log_age_days", 7)
        self.max_checkpoint_age_days = self.config.get("max_checkpoint_age_days", 30)
        self.cleanup_interval_minutes = self.config.get("cleanup_interval_minutes", 30)
        
        # Directories to manage
        self.checkpoint_dir = Path("checkpoints")
        self.log_dir = Path("logs")
        self.cache_dirs = [Path("."), Path("models"), Path("training"), Path("utils")]
        
        # Important files to preserve
        self.preserve_patterns = [
            "*_best*",
            "*_final*", 
            "latest_checkpoint.txt",
            "*revolutionary*",
            "*hybrid*"
        ]
        
        logger.info("Automatic Cleanup Manager initialized")
    
    def perform_full_cleanup(self) -> Dict[str, any]:
        """Perform comprehensive cleanup and return statistics."""
        logger.info("🧹 Starting comprehensive cleanup...")
        
        stats = {
            "checkpoints_removed": 0,
            "logs_removed": 0,
            "cache_cleared": 0,
            "space_freed_mb": 0,
            "memory_freed_mb": 0
        }
        
        # 1. Clean old episode checkpoints
        stats.update(self._cleanup_episode_checkpoints())
        
        # 2. Clean old logs
        stats.update(self._cleanup_old_logs())
        
        # 3. Clear Python cache
        stats.update(self._clear_python_cache())
        
        # 4. Memory cleanup
        stats.update(self._cleanup_memory())
        
        # 5. Clean temporary files
        stats.update(self._cleanup_temp_files())
        
        logger.info(f"✅ Cleanup completed: {stats}")
        return stats
    
    def _cleanup_episode_checkpoints(self) -> Dict[str, int]:
        """Clean old episode checkpoints, keeping only the latest ones."""
        stats = {"checkpoints_removed": 0, "space_freed_mb": 0}
        
        if not self.checkpoint_dir.exists():
            return stats
        
        # Find all episode checkpoints
        episode_files = list(self.checkpoint_dir.glob("*episode_*.pth"))
        
        if len(episode_files) <= self.max_episode_checkpoints:
            return stats
        
        # Sort by modification time (newest first)
        episode_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Remove old episode checkpoints
        files_to_remove = episode_files[self.max_episode_checkpoints:]
        
        for file_path in files_to_remove:
            try:
                size_mb = file_path.stat().st_size / 1024 / 1024
                file_path.unlink()
                stats["checkpoints_removed"] += 1
                stats["space_freed_mb"] += size_mb
                logger.debug(f"Removed old checkpoint: {file_path.name}")
            except Exception as e:
                logger.warning(f"Failed to remove {file_path}: {e}")
        
        logger.info(f"Cleaned {stats['checkpoints_removed']} old episode checkpoints")
        return stats
    
    def _cleanup_old_logs(self) -> Dict[str, int]:
        """Remove log files older than specified days."""
        stats = {"logs_removed": 0, "space_freed_mb": 0}
        
        if not self.log_dir.exists():
            return stats
        
        cutoff_time = time.time() - (self.max_log_age_days * 86400)
        
        for log_file in self.log_dir.glob("*.log"):
            try:
                if log_file.stat().st_mtime < cutoff_time:
                    size_mb = log_file.stat().st_size / 1024 / 1024
                    log_file.unlink()
                    stats["logs_removed"] += 1
                    stats["space_freed_mb"] += size_mb
                    logger.debug(f"Removed old log: {log_file.name}")
            except Exception as e:
                logger.warning(f"Failed to remove log {log_file}: {e}")
        
        logger.info(f"Cleaned {stats['logs_removed']} old log files")
        return stats
    
    def _clear_python_cache(self) -> Dict[str, int]:
        """Clear Python __pycache__ directories."""
        stats = {"cache_cleared": 0, "space_freed_mb": 0}
        
        for base_dir in self.cache_dirs:
            if not base_dir.exists():
                continue
                
            try:
                # Get all __pycache__ directories safely
                pycache_dirs = list(base_dir.rglob("__pycache__"))

                for pycache_dir in pycache_dirs:
                    try:
                        # Skip if directory doesn't exist or is not accessible
                        if not pycache_dir.exists() or not pycache_dir.is_dir():
                            continue

                        # Calculate size before removal (with error handling)
                        size_mb = 0
                        try:
                            size_mb = sum(f.stat().st_size for f in pycache_dir.rglob("*") if f.is_file()) / 1024 / 1024
                        except (OSError, PermissionError):
                            size_mb = 0  # Can't calculate size, continue anyway

                        # Remove cache directory with ignore_errors
                        shutil.rmtree(pycache_dir, ignore_errors=True)

                        # Only count as success if directory was actually removed
                        if not pycache_dir.exists():
                            stats["cache_cleared"] += 1
                            stats["space_freed_mb"] += size_mb
                            logger.debug(f"Cleared cache: {pycache_dir}")

                    except (OSError, PermissionError, FileNotFoundError):
                        # Skip files that are in use or inaccessible
                        continue
                    except Exception as e:
                        logger.debug(f"Skipped cache {pycache_dir}: {e}")
                        continue

            except Exception as e:
                logger.debug(f"Error scanning {base_dir} for cache: {e}")
                continue
        
        logger.info(f"Cleared {stats['cache_cleared']} Python cache directories")
        return stats
    
    def _cleanup_memory(self) -> Dict[str, int]:
        """Perform memory cleanup and garbage collection."""
        stats = {"memory_freed_mb": 0}
        
        # Get memory before cleanup
        import psutil
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024
        
        # Force garbage collection
        gc.collect()
        
        # Clear PyTorch cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Get memory after cleanup
        memory_after = process.memory_info().rss / 1024 / 1024
        stats["memory_freed_mb"] = max(0, memory_before - memory_after)
        
        logger.info(f"Memory cleanup freed {stats['memory_freed_mb']:.1f} MB")
        return stats

    def cleanup_old_checkpoints(self, keep_best: bool = True, keep_latest: int = 5) -> Dict[str, int]:
        """Clean up old checkpoint files, keeping only the best and latest ones."""
        stats = {"checkpoints_removed": 0, "space_freed_mb": 0}

        checkpoint_dir = Path("checkpoints")
        if not checkpoint_dir.exists():
            return stats

        try:
            # Get all checkpoint files
            checkpoint_files = list(checkpoint_dir.glob("*.pth")) + list(checkpoint_dir.glob("*.pt"))

            if len(checkpoint_files) <= keep_latest:
                return stats  # Not enough files to clean

            # Sort by modification time (newest first)
            checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            # Keep the latest N files
            files_to_keep = set(checkpoint_files[:keep_latest])

            # Keep best files if requested
            if keep_best:
                best_files = [f for f in checkpoint_files if "best" in f.name.lower()]
                files_to_keep.update(best_files)

            # Remove old files
            for file_path in checkpoint_files:
                if file_path not in files_to_keep:
                    try:
                        size_mb = file_path.stat().st_size / 1024 / 1024
                        file_path.unlink()
                        stats["checkpoints_removed"] += 1
                        stats["space_freed_mb"] += size_mb
                    except Exception as e:
                        logger.debug(f"Failed to remove checkpoint {file_path}: {e}")

            if stats["checkpoints_removed"] > 0:
                logger.info(f"Cleaned {stats['checkpoints_removed']} old checkpoints, freed {stats['space_freed_mb']:.1f} MB")

        except Exception as e:
            logger.error(f"Checkpoint cleanup failed: {e}")

        return stats

    def _cleanup_temp_files(self) -> Dict[str, int]:
        """Clean temporary files and emergency saves."""
        stats = {"temp_files_removed": 0, "space_freed_mb": 0}
        
        # Clean emergency session files older than 1 day
        if self.checkpoint_dir.exists():
            cutoff_time = time.time() - 86400  # 1 day
            
            for temp_file in self.checkpoint_dir.glob("session_*_emergency.*"):
                try:
                    if temp_file.stat().st_mtime < cutoff_time:
                        size_mb = temp_file.stat().st_size / 1024 / 1024
                        temp_file.unlink()
                        stats["temp_files_removed"] += 1
                        stats["space_freed_mb"] += size_mb
                        logger.debug(f"Removed temp file: {temp_file.name}")
                except Exception as e:
                    logger.warning(f"Failed to remove temp file {temp_file}: {e}")
        
        logger.info(f"Cleaned {stats['temp_files_removed']} temporary files")
        return stats
    
    def should_preserve_file(self, file_path: Path) -> bool:
        """Check if a file should be preserved based on patterns."""
        for pattern in self.preserve_patterns:
            if file_path.match(pattern):
                return True
        return False
    
    def get_disk_usage_stats(self) -> Dict[str, float]:
        """Get current disk usage statistics."""
        stats = {}
        
        # Checkpoint directory size
        if self.checkpoint_dir.exists():
            checkpoint_size = sum(f.stat().st_size for f in self.checkpoint_dir.rglob("*") if f.is_file())
            stats["checkpoints_mb"] = checkpoint_size / 1024 / 1024
        
        # Log directory size
        if self.log_dir.exists():
            log_size = sum(f.stat().st_size for f in self.log_dir.rglob("*") if f.is_file())
            stats["logs_mb"] = log_size / 1024 / 1024
        
        # Cache size
        cache_size = 0
        for base_dir in self.cache_dirs:
            if base_dir.exists():
                for pycache_dir in base_dir.rglob("__pycache__"):
                    cache_size += sum(f.stat().st_size for f in pycache_dir.rglob("*") if f.is_file())
        stats["cache_mb"] = cache_size / 1024 / 1024
        
        return stats
    
    def emergency_cleanup(self) -> Dict[str, any]:
        """Perform emergency cleanup when disk space is critically low."""
        logger.warning("🚨 Performing emergency cleanup!")
        
        stats = self.perform_full_cleanup()
        
        # Additional emergency measures
        # Remove all but the 2 most recent episode checkpoints
        if self.checkpoint_dir.exists():
            episode_files = list(self.checkpoint_dir.glob("*episode_*.pth"))
            if len(episode_files) > 2:
                episode_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                for file_path in episode_files[2:]:
                    try:
                        size_mb = file_path.stat().st_size / 1024 / 1024
                        file_path.unlink()
                        stats["checkpoints_removed"] += 1
                        stats["space_freed_mb"] += size_mb
                    except Exception as e:
                        logger.error(f"Emergency cleanup failed for {file_path}: {e}")
        
        logger.warning(f"🚨 Emergency cleanup completed: {stats}")
        return stats


# Global cleanup manager instance
cleanup_manager = AutomaticCleanupManager()


def perform_cleanup():
    """Convenience function to perform cleanup."""
    return cleanup_manager.perform_full_cleanup()


def emergency_cleanup():
    """Convenience function for emergency cleanup."""
    return cleanup_manager.emergency_cleanup()
