"""
Performance Monitor for System Resource Tracking
Monitors CPU, GPU, memory usage and system performance
"""

import psutil
import time
import threading
from typing import Dict, List, Optional, Any
from collections import deque
import numpy as np
from loguru import logger

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    logger.warning("GPUtil not available, GPU monitoring disabled")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class PerformanceMonitor:
    """
    System performance monitoring for resource usage tracking.
    Monitors CPU, GPU, memory, and provides performance analytics.
    """
    
    def __init__(self, monitoring_interval: float = 1.0, history_size: int = 300):
        """
        Initialize performance monitor.
        
        Args:
            monitoring_interval: Seconds between monitoring updates
            history_size: Number of historical measurements to keep
        """
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        
        # Performance history
        self.cpu_history = deque(maxlen=history_size)
        self.memory_history = deque(maxlen=history_size)
        self.gpu_history = deque(maxlen=history_size)
        self.timestamp_history = deque(maxlen=history_size)
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        
        # System info
        self.cpu_count = psutil.cpu_count()
        self.memory_total = psutil.virtual_memory().total
        self.gpu_info = self._get_gpu_info()
        
        # Performance thresholds
        self.cpu_warning_threshold = 80.0  # %
        self.memory_warning_threshold = 85.0  # %
        self.gpu_warning_threshold = 90.0  # %
        
        # Alert tracking
        self.alerts = deque(maxlen=100)
        
        logger.info("Performance monitor initialized")
    
    def _get_gpu_info(self) -> List[Dict[str, Any]]:
        """Get GPU information."""
        gpu_info = []
        
        try:
            if GPU_AVAILABLE:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    gpu_info.append({
                        'id': gpu.id,
                        'name': gpu.name,
                        'memory_total': gpu.memoryTotal,
                        'driver': gpu.driver
                    })
            
            elif TORCH_AVAILABLE and torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    gpu_info.append({
                        'id': i,
                        'name': props.name,
                        'memory_total': props.total_memory // (1024**2),  # MB
                        'driver': 'CUDA'
                    })
            
        except Exception as e:
            logger.error(f"GPU info retrieval failed: {e}")
        
        return gpu_info
    
    def start_monitoring(self):
        """Start continuous performance monitoring."""
        if self.is_monitoring:
            logger.warning("Performance monitoring already running")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.is_monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                timestamp = time.time()
                
                # Get current measurements
                cpu_usage = self.get_cpu_usage()
                memory_usage = self.get_memory_usage()
                gpu_usage = self.get_gpu_usage()
                
                # Store in history
                self.cpu_history.append(cpu_usage)
                self.memory_history.append(memory_usage)
                self.gpu_history.append(gpu_usage)
                self.timestamp_history.append(timestamp)
                
                # Check for alerts
                self._check_performance_alerts(cpu_usage, memory_usage, gpu_usage)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(self.monitoring_interval)
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            return psutil.cpu_percent(interval=None)
        except Exception as e:
            logger.error(f"CPU usage retrieval failed: {e}")
            return 0.0
    
    def get_memory_usage(self) -> float:
        """Get current memory usage percentage."""
        try:
            memory = psutil.virtual_memory()
            return memory.percent
        except Exception as e:
            logger.error(f"Memory usage retrieval failed: {e}")
            return 0.0
    
    def get_gpu_usage(self) -> List[Dict[str, float]]:
        """Get current GPU usage for all GPUs."""
        gpu_usage = []
        
        try:
            if GPU_AVAILABLE:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    gpu_usage.append({
                        'id': gpu.id,
                        'utilization': gpu.load * 100,
                        'memory_used': gpu.memoryUsed,
                        'memory_total': gpu.memoryTotal,
                        'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                        'temperature': gpu.temperature
                    })
            
            elif TORCH_AVAILABLE and torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    memory_allocated = torch.cuda.memory_allocated(i) // (1024**2)  # MB
                    memory_reserved = torch.cuda.memory_reserved(i) // (1024**2)  # MB
                    memory_total = torch.cuda.get_device_properties(i).total_memory // (1024**2)  # MB
                    
                    gpu_usage.append({
                        'id': i,
                        'utilization': 0.0,  # Not available through PyTorch
                        'memory_used': memory_allocated,
                        'memory_total': memory_total,
                        'memory_percent': (memory_allocated / memory_total) * 100,
                        'temperature': 0.0  # Not available through PyTorch
                    })
            
        except Exception as e:
            logger.error(f"GPU usage retrieval failed: {e}")
        
        return gpu_usage
    
    def get_disk_usage(self) -> Dict[str, float]:
        """Get disk usage information."""
        try:
            disk = psutil.disk_usage('/')
            return {
                'total_gb': disk.total / (1024**3),
                'used_gb': disk.used / (1024**3),
                'free_gb': disk.free / (1024**3),
                'percent': (disk.used / disk.total) * 100
            }
        except Exception as e:
            logger.error(f"Disk usage retrieval failed: {e}")
            return {}
    
    def get_network_stats(self) -> Dict[str, float]:
        """Get network I/O statistics."""
        try:
            net_io = psutil.net_io_counters()
            return {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv
            }
        except Exception as e:
            logger.error(f"Network stats retrieval failed: {e}")
            return {}
    
    def get_process_info(self) -> Dict[str, Any]:
        """Get current process information."""
        try:
            process = psutil.Process()
            
            return {
                'pid': process.pid,
                'cpu_percent': process.cpu_percent(),
                'memory_percent': process.memory_percent(),
                'memory_info': process.memory_info()._asdict(),
                'num_threads': process.num_threads(),
                'create_time': process.create_time(),
                'status': process.status()
            }
        except Exception as e:
            logger.error(f"Process info retrieval failed: {e}")
            return {}
    
    def _check_performance_alerts(self, cpu_usage: float, memory_usage: float, gpu_usage: List[Dict]):
        """Check for performance alerts and warnings."""
        try:
            current_time = time.time()
            
            # CPU alert
            if cpu_usage > self.cpu_warning_threshold:
                self._add_alert('cpu_high', f"High CPU usage: {cpu_usage:.1f}%", current_time)
            
            # Memory alert
            if memory_usage > self.memory_warning_threshold:
                self._add_alert('memory_high', f"High memory usage: {memory_usage:.1f}%", current_time)
            
            # GPU alerts
            for gpu in gpu_usage:
                if gpu.get('memory_percent', 0) > self.gpu_warning_threshold:
                    self._add_alert(
                        'gpu_memory_high',
                        f"High GPU {gpu['id']} memory usage: {gpu['memory_percent']:.1f}%",
                        current_time
                    )
                
                if gpu.get('temperature', 0) > 80:  # 80°C threshold
                    self._add_alert(
                        'gpu_temperature_high',
                        f"High GPU {gpu['id']} temperature: {gpu['temperature']:.1f}°C",
                        current_time
                    )
            
        except Exception as e:
            logger.error(f"Performance alert check failed: {e}")
    
    def _add_alert(self, alert_type: str, message: str, timestamp: float):
        """Add performance alert."""
        alert = {
            'type': alert_type,
            'message': message,
            'timestamp': timestamp
        }
        
        self.alerts.append(alert)
        logger.warning(f"Performance alert: {message}")
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current comprehensive performance statistics."""
        try:
            return {
                'cpu': {
                    'usage_percent': self.get_cpu_usage(),
                    'count': self.cpu_count,
                    'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
                },
                'memory': {
                    'usage_percent': self.get_memory_usage(),
                    'total_gb': self.memory_total / (1024**3),
                    'available_gb': psutil.virtual_memory().available / (1024**3)
                },
                'gpu': self.get_gpu_usage(),
                'disk': self.get_disk_usage(),
                'network': self.get_network_stats(),
                'process': self.get_process_info(),
                'timestamp': time.time()
            }
        except Exception as e:
            logger.error(f"Current stats retrieval failed: {e}")
            return {}
    
    def get_performance_summary(self, window_minutes: int = 5) -> Dict[str, Any]:
        """
        Get performance summary over specified time window.
        
        Args:
            window_minutes: Time window in minutes
            
        Returns:
            Performance summary statistics
        """
        try:
            if not self.timestamp_history:
                return {}
            
            current_time = time.time()
            window_seconds = window_minutes * 60
            cutoff_time = current_time - window_seconds
            
            # Filter data within time window
            indices = [i for i, ts in enumerate(self.timestamp_history) if ts >= cutoff_time]
            
            if not indices:
                return {}
            
            # Extract data for window
            cpu_data = [self.cpu_history[i] for i in indices]
            memory_data = [self.memory_history[i] for i in indices]
            
            # Calculate GPU averages
            gpu_summary = {}
            if self.gpu_history and indices:
                gpu_data = [self.gpu_history[i] for i in indices]
                if gpu_data and gpu_data[0]:  # Check if GPU data exists
                    for gpu_id in range(len(gpu_data[0])):
                        utilizations = [frame[gpu_id]['utilization'] for frame in gpu_data if gpu_id < len(frame)]
                        memory_percents = [frame[gpu_id]['memory_percent'] for frame in gpu_data if gpu_id < len(frame)]
                        
                        if utilizations:
                            gpu_summary[f'gpu_{gpu_id}'] = {
                                'avg_utilization': np.mean(utilizations),
                                'max_utilization': np.max(utilizations),
                                'avg_memory_percent': np.mean(memory_percents),
                                'max_memory_percent': np.max(memory_percents)
                            }
            
            summary = {
                'time_window_minutes': window_minutes,
                'data_points': len(indices),
                'cpu': {
                    'avg_usage': np.mean(cpu_data),
                    'max_usage': np.max(cpu_data),
                    'min_usage': np.min(cpu_data),
                    'std_usage': np.std(cpu_data)
                },
                'memory': {
                    'avg_usage': np.mean(memory_data),
                    'max_usage': np.max(memory_data),
                    'min_usage': np.min(memory_data),
                    'std_usage': np.std(memory_data)
                },
                'gpu': gpu_summary,
                'alerts_count': len([a for a in self.alerts if a['timestamp'] >= cutoff_time])
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Performance summary calculation failed: {e}")
            return {}
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        try:
            return {
                'platform': {
                    'system': psutil.WINDOWS if psutil.WINDOWS else 'unix',
                    'architecture': psutil.cpu_count(logical=False),
                    'logical_cpus': psutil.cpu_count(logical=True)
                },
                'cpu': {
                    'physical_cores': psutil.cpu_count(logical=False),
                    'logical_cores': psutil.cpu_count(logical=True),
                    'max_frequency': psutil.cpu_freq().max if psutil.cpu_freq() else None,
                    'current_frequency': psutil.cpu_freq().current if psutil.cpu_freq() else None
                },
                'memory': {
                    'total_gb': self.memory_total / (1024**3),
                    'available_gb': psutil.virtual_memory().available / (1024**3)
                },
                'gpu': self.gpu_info,
                'disk': self.get_disk_usage(),
                'boot_time': psutil.boot_time()
            }
        except Exception as e:
            logger.error(f"System info retrieval failed: {e}")
            return {}
    
    def get_recent_alerts(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent performance alerts."""
        try:
            return list(self.alerts)[-count:]
        except Exception as e:
            logger.error(f"Recent alerts retrieval failed: {e}")
            return []
    
    def clear_alerts(self):
        """Clear all performance alerts."""
        self.alerts.clear()
        logger.info("Performance alerts cleared")
    
    def export_performance_data(self, filepath: str):
        """Export performance history to file."""
        try:
            import json
            
            data = {
                'system_info': self.get_system_info(),
                'performance_history': {
                    'timestamps': list(self.timestamp_history),
                    'cpu_usage': list(self.cpu_history),
                    'memory_usage': list(self.memory_history),
                    'gpu_usage': list(self.gpu_history)
                },
                'alerts': list(self.alerts),
                'export_timestamp': time.time()
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Performance data exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Performance data export failed: {e}")
    
    def optimize_performance(self) -> Dict[str, str]:
        """Provide performance optimization recommendations."""
        try:
            recommendations = {}
            
            # CPU recommendations
            if self.cpu_history:
                avg_cpu = np.mean(list(self.cpu_history)[-10:])  # Last 10 measurements
                if avg_cpu > 80:
                    recommendations['cpu'] = "High CPU usage detected. Consider reducing batch size or processing resolution."
                elif avg_cpu < 30:
                    recommendations['cpu'] = "Low CPU usage. Consider increasing batch size for better throughput."
            
            # Memory recommendations
            if self.memory_history:
                avg_memory = np.mean(list(self.memory_history)[-10:])
                if avg_memory > 85:
                    recommendations['memory'] = "High memory usage detected. Consider reducing model complexity or batch size."
            
            # GPU recommendations
            if self.gpu_history and self.gpu_history[-1]:
                for gpu in self.gpu_history[-1]:
                    if gpu['memory_percent'] > 90:
                        recommendations[f'gpu_{gpu["id"]}'] = "High GPU memory usage. Consider reducing batch size or model precision."
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Performance optimization analysis failed: {e}")
            return {}
