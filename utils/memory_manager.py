#!/usr/bin/env python3
"""
Comprehensive Memory Management System
Prevents memory bloat and ensures robust performance during training and inference.

Features:
- Real-time memory monitoring
- Automatic garbage collection
- GPU memory optimization
- Memory leak detection
- Resource cleanup
- Memory-aware batch sizing
"""

import gc
import psutil
import torch
import threading
import time
import warnings
from typing import Dict, List, Optional, Callable, Any
from loguru import logger
from pathlib import Path
import numpy as np
from collections import deque
import weakref


class MemoryManager:
    """Comprehensive memory management for robust AI/ML operations."""
    
    def __init__(self, 
                 max_cpu_memory_gb: float = 12.0,
                 max_gpu_memory_gb: float = 5.0,
                 monitoring_interval: float = 30.0,
                 cleanup_threshold: float = 0.85):
        """
        Initialize memory manager.
        
        Args:
            max_cpu_memory_gb: Maximum CPU memory usage in GB
            max_gpu_memory_gb: Maximum GPU memory usage in GB  
            monitoring_interval: Memory monitoring interval in seconds
            cleanup_threshold: Memory usage threshold to trigger cleanup (0.0-1.0)
        """
        self.max_cpu_memory = max_cpu_memory_gb * 1024 * 1024 * 1024  # Convert to bytes
        self.max_gpu_memory = max_gpu_memory_gb * 1024 * 1024 * 1024
        self.monitoring_interval = monitoring_interval
        self.cleanup_threshold = cleanup_threshold
        
        # Memory tracking
        self.memory_history = deque(maxlen=100)  # Last 100 measurements
        self.gpu_memory_history = deque(maxlen=100)
        self.peak_memory_usage = 0
        self.peak_gpu_usage = 0
        
        # Monitoring state
        self.monitoring_active = False
        self.monitor_thread = None
        self.cleanup_callbacks = []
        self.managed_objects = weakref.WeakSet()
        
        # Performance optimization
        self.auto_gc_enabled = True
        self.gpu_cache_cleanup_enabled = True
        self.memory_efficient_mode = False
        
        # Statistics
        self.cleanup_count = 0
        self.memory_warnings = 0
        self.oom_preventions = 0
        
        # Initialize GPU monitoring if available
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.gpu_device = torch.cuda.current_device()
            torch.cuda.empty_cache()
        
        logger.info("🧠 Memory Manager initialized")
        logger.info(f"   💾 Max CPU Memory: {max_cpu_memory_gb:.1f} GB")
        logger.info(f"   🎮 Max GPU Memory: {max_gpu_memory_gb:.1f} GB")
        logger.info(f"   🔄 Monitoring interval: {monitoring_interval}s")
        logger.info(f"   ⚠️  Cleanup threshold: {cleanup_threshold:.1%}")
    
    def start_monitoring(self):
        """Start continuous memory monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("🔄 Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop memory monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        logger.info("⏹️  Memory monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self._check_memory_usage()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"❌ Memory monitoring error: {e}")
                time.sleep(self.monitoring_interval)
    
    def _check_memory_usage(self):
        """Check current memory usage and trigger cleanup if needed."""
        try:
            # CPU memory check
            cpu_memory = self.get_cpu_memory_usage()
            self.memory_history.append(cpu_memory)
            self.peak_memory_usage = max(self.peak_memory_usage, cpu_memory['used_gb'])
            
            # GPU memory check
            if self.gpu_available:
                gpu_memory = self.get_gpu_memory_usage()
                self.gpu_memory_history.append(gpu_memory)
                self.peak_gpu_usage = max(self.peak_gpu_usage, gpu_memory['used_gb'])
            
            # Check for cleanup triggers
            cpu_usage_ratio = cpu_memory['used_gb'] / (self.max_cpu_memory / (1024**3))
            
            if cpu_usage_ratio >= self.cleanup_threshold:
                logger.warning(f"⚠️  High CPU memory usage: {cpu_usage_ratio:.1%}")
                self.memory_warnings += 1
                self._trigger_cleanup("high_cpu_memory")
            
            if self.gpu_available:
                gpu_usage_ratio = gpu_memory['used_gb'] / (self.max_gpu_memory / (1024**3))
                if gpu_usage_ratio >= self.cleanup_threshold:
                    logger.warning(f"⚠️  High GPU memory usage: {gpu_usage_ratio:.1%}")
                    self.memory_warnings += 1
                    self._trigger_cleanup("high_gpu_memory")
            
            # Log periodic status
            if len(self.memory_history) % 10 == 0:  # Every 10 checks
                self._log_memory_status()
                
        except Exception as e:
            logger.error(f"❌ Memory check failed: {e}")
    
    def get_cpu_memory_usage(self) -> Dict[str, float]:
        """Get current CPU memory usage."""
        try:
            memory = psutil.virtual_memory()
            process = psutil.Process()
            process_memory = process.memory_info()
            
            return {
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'used_gb': memory.used / (1024**3),
                'percent': memory.percent,
                'process_rss_gb': process_memory.rss / (1024**3),
                'process_vms_gb': process_memory.vms / (1024**3)
            }
        except Exception as e:
            logger.error(f"❌ Failed to get CPU memory usage: {e}")
            return {'total_gb': 0, 'available_gb': 0, 'used_gb': 0, 'percent': 0, 
                   'process_rss_gb': 0, 'process_vms_gb': 0}
    
    def get_gpu_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage."""
        if not self.gpu_available:
            return {'total_gb': 0, 'used_gb': 0, 'free_gb': 0, 'percent': 0}
        
        try:
            memory_stats = torch.cuda.memory_stats(self.gpu_device)
            allocated = torch.cuda.memory_allocated(self.gpu_device)
            reserved = torch.cuda.memory_reserved(self.gpu_device)
            
            # Get total GPU memory
            total_memory = torch.cuda.get_device_properties(self.gpu_device).total_memory
            
            return {
                'total_gb': total_memory / (1024**3),
                'allocated_gb': allocated / (1024**3),
                'reserved_gb': reserved / (1024**3),
                'used_gb': reserved / (1024**3),  # Use reserved as "used"
                'free_gb': (total_memory - reserved) / (1024**3),
                'percent': (reserved / total_memory) * 100,
                'peak_allocated_gb': memory_stats.get('allocated_bytes.all.peak', 0) / (1024**3)
            }
        except Exception as e:
            logger.error(f"❌ Failed to get GPU memory usage: {e}")
            return {'total_gb': 0, 'used_gb': 0, 'free_gb': 0, 'percent': 0}
    
    def _trigger_cleanup(self, reason: str):
        """Trigger memory cleanup."""
        logger.info(f"🧹 Triggering memory cleanup: {reason}")
        
        try:
            cleanup_start = time.time()
            
            # 1. Run custom cleanup callbacks
            for callback in self.cleanup_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.warning(f"⚠️  Cleanup callback failed: {e}")
            
            # 2. Clear GPU cache if available
            if self.gpu_available and self.gpu_cache_cleanup_enabled:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # 3. Force garbage collection
            if self.auto_gc_enabled:
                collected = gc.collect()
                logger.info(f"🗑️  Garbage collected {collected} objects")
            
            # 4. Clear managed objects that support cleanup
            self._cleanup_managed_objects()
            
            cleanup_time = time.time() - cleanup_start
            self.cleanup_count += 1
            
            logger.info(f"✅ Memory cleanup completed in {cleanup_time:.2f}s")
            
            # Check if cleanup was effective
            self._verify_cleanup_effectiveness()
            
        except Exception as e:
            logger.error(f"❌ Memory cleanup failed: {e}")
    
    def _cleanup_managed_objects(self):
        """Cleanup managed objects that support it."""
        cleaned_count = 0
        
        for obj in list(self.managed_objects):
            try:
                if hasattr(obj, 'cleanup_memory'):
                    obj.cleanup_memory()
                    cleaned_count += 1
                elif hasattr(obj, 'clear_cache'):
                    obj.clear_cache()
                    cleaned_count += 1
            except Exception as e:
                logger.warning(f"⚠️  Failed to cleanup managed object: {e}")
        
        if cleaned_count > 0:
            logger.info(f"🧹 Cleaned up {cleaned_count} managed objects")
    
    def _verify_cleanup_effectiveness(self):
        """Verify that cleanup was effective."""
        try:
            # Wait a moment for cleanup to take effect
            time.sleep(1.0)
            
            # Check memory usage again
            cpu_memory = self.get_cpu_memory_usage()
            cpu_usage_ratio = cpu_memory['used_gb'] / (self.max_cpu_memory / (1024**3))
            
            if cpu_usage_ratio >= self.cleanup_threshold:
                logger.warning(f"⚠️  Cleanup ineffective - CPU memory still high: {cpu_usage_ratio:.1%}")
                self._emergency_memory_management()
            else:
                logger.info(f"✅ Cleanup effective - CPU memory: {cpu_usage_ratio:.1%}")
                
        except Exception as e:
            logger.error(f"❌ Cleanup verification failed: {e}")
    
    def _emergency_memory_management(self):
        """Emergency memory management when normal cleanup fails."""
        logger.warning("🚨 Emergency memory management activated")
        
        try:
            # Enable memory efficient mode
            self.memory_efficient_mode = True
            
            # More aggressive cleanup
            for i in range(3):  # Multiple GC passes
                collected = gc.collect()
                logger.info(f"🗑️  Emergency GC pass {i+1}: {collected} objects")
            
            # Clear all possible caches
            if self.gpu_available:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            
            # Warn about potential OOM
            self.oom_preventions += 1
            logger.warning("⚠️  Potential OOM prevented - consider reducing batch size")
            
        except Exception as e:
            logger.error(f"❌ Emergency memory management failed: {e}")
    
    def register_cleanup_callback(self, callback: Callable):
        """Register a cleanup callback function."""
        self.cleanup_callbacks.append(callback)
        logger.info(f"📝 Registered cleanup callback: {callback.__name__}")
    
    def register_managed_object(self, obj: Any):
        """Register an object for memory management."""
        self.managed_objects.add(obj)
    
    def optimize_batch_size(self, base_batch_size: int, memory_factor: float = 0.8) -> int:
        """Optimize batch size based on available memory."""
        try:
            cpu_memory = self.get_cpu_memory_usage()
            available_ratio = cpu_memory['available_gb'] / cpu_memory['total_gb']
            
            if self.gpu_available:
                gpu_memory = self.get_gpu_memory_usage()
                gpu_available_ratio = gpu_memory['free_gb'] / gpu_memory['total_gb']
                available_ratio = min(available_ratio, gpu_available_ratio)
            
            # Adjust batch size based on available memory
            if available_ratio < 0.2:  # Less than 20% available
                optimized_size = max(1, base_batch_size // 4)
            elif available_ratio < 0.4:  # Less than 40% available
                optimized_size = max(1, base_batch_size // 2)
            elif available_ratio < 0.6:  # Less than 60% available
                optimized_size = int(base_batch_size * 0.75)
            else:
                optimized_size = base_batch_size
            
            if optimized_size != base_batch_size:
                logger.info(f"📊 Optimized batch size: {base_batch_size} → {optimized_size}")
            
            return optimized_size
            
        except Exception as e:
            logger.error(f"❌ Batch size optimization failed: {e}")
            return base_batch_size
    
    def _log_memory_status(self):
        """Log current memory status."""
        try:
            cpu_memory = self.get_cpu_memory_usage()
            
            logger.info("📊 MEMORY STATUS:")
            logger.info(f"   💾 CPU: {cpu_memory['used_gb']:.1f}/{cpu_memory['total_gb']:.1f} GB ({cpu_memory['percent']:.1f}%)")
            logger.info(f"   🔄 Process: {cpu_memory['process_rss_gb']:.1f} GB RSS")
            logger.info(f"   📈 Peak: {self.peak_memory_usage:.1f} GB")
            
            if self.gpu_available:
                gpu_memory = self.get_gpu_memory_usage()
                logger.info(f"   🎮 GPU: {gpu_memory['used_gb']:.1f}/{gpu_memory['total_gb']:.1f} GB ({gpu_memory['percent']:.1f}%)")
                logger.info(f"   📈 GPU Peak: {self.peak_gpu_usage:.1f} GB")
            
            logger.info(f"   🧹 Cleanups: {self.cleanup_count}")
            logger.info(f"   ⚠️  Warnings: {self.memory_warnings}")
            logger.info(f"   🚨 OOM Preventions: {self.oom_preventions}")
            
        except Exception as e:
            logger.error(f"❌ Memory status logging failed: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        try:
            cpu_memory = self.get_cpu_memory_usage()
            stats = {
                'cpu_memory': cpu_memory,
                'peak_memory_gb': self.peak_memory_usage,
                'cleanup_count': self.cleanup_count,
                'memory_warnings': self.memory_warnings,
                'oom_preventions': self.oom_preventions,
                'memory_efficient_mode': self.memory_efficient_mode,
                'monitoring_active': self.monitoring_active
            }
            
            if self.gpu_available:
                stats['gpu_memory'] = self.get_gpu_memory_usage()
                stats['peak_gpu_usage_gb'] = self.peak_gpu_usage
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to get memory stats: {e}")
            return {}
    
    def cleanup(self):
        """Final cleanup and shutdown."""
        try:
            logger.info("🧹 Memory Manager cleanup...")
            
            # Stop monitoring
            self.stop_monitoring()
            
            # Final cleanup
            self._trigger_cleanup("shutdown")
            
            # Clear callbacks and managed objects
            self.cleanup_callbacks.clear()
            self.managed_objects.clear()
            
            logger.info("✅ Memory Manager cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Memory Manager cleanup failed: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


class MemoryAwareDataLoader:
    """Memory-aware data loader that adjusts batch size dynamically."""

    def __init__(self, dataset, base_batch_size: int, memory_manager: MemoryManager):
        self.dataset = dataset
        self.base_batch_size = base_batch_size
        self.memory_manager = memory_manager
        self.current_batch_size = base_batch_size

    def __iter__(self):
        """Iterate with dynamic batch sizing."""
        while True:
            # Optimize batch size based on current memory
            self.current_batch_size = self.memory_manager.optimize_batch_size(
                self.base_batch_size
            )

            # Yield batches
            for i in range(0, len(self.dataset), self.current_batch_size):
                batch = self.dataset[i:i + self.current_batch_size]
                yield batch

                # Check memory after each batch
                if i % 10 == 0:  # Every 10 batches
                    self.current_batch_size = self.memory_manager.optimize_batch_size(
                        self.base_batch_size
                    )


class MemoryEfficientCache:
    """Memory-efficient cache with automatic cleanup."""

    def __init__(self, max_size: int = 1000, memory_manager: Optional[MemoryManager] = None):
        self.max_size = max_size
        self.memory_manager = memory_manager
        self.cache = {}
        self.access_order = deque()

        if memory_manager:
            memory_manager.register_managed_object(self)

    def get(self, key: str) -> Any:
        """Get item from cache."""
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None

    def put(self, key: str, value: Any):
        """Put item in cache with memory management."""
        # Remove oldest items if at capacity
        while len(self.cache) >= self.max_size:
            oldest_key = self.access_order.popleft()
            del self.cache[oldest_key]

        self.cache[key] = value
        self.access_order.append(key)

    def cleanup_memory(self):
        """Cleanup cache for memory management."""
        # Remove half the cache, starting with least recently used
        items_to_remove = len(self.cache) // 2

        for _ in range(items_to_remove):
            if self.access_order:
                oldest_key = self.access_order.popleft()
                if oldest_key in self.cache:
                    del self.cache[oldest_key]

    def clear_cache(self):
        """Clear entire cache."""
        self.cache.clear()
        self.access_order.clear()


def memory_efficient_decorator(memory_manager: MemoryManager):
    """Decorator for memory-efficient function execution."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Check memory before execution
            cpu_memory = memory_manager.get_cpu_memory_usage()
            if cpu_memory['percent'] > 85:
                logger.warning(f"⚠️  High memory before {func.__name__}: {cpu_memory['percent']:.1f}%")
                memory_manager._trigger_cleanup(f"before_{func.__name__}")

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # Cleanup after execution
                if memory_manager.auto_gc_enabled:
                    gc.collect()

                if memory_manager.gpu_available and memory_manager.gpu_cache_cleanup_enabled:
                    torch.cuda.empty_cache()

        return wrapper
    return decorator
