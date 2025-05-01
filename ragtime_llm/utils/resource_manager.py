"""
Resource management and monitoring system.
"""

import os
import psutil
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
from datetime import datetime
import threading
import time

logger = logging.getLogger(__name__)

@dataclass
class ResourceMetrics:
    """System resource metrics."""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    gpu_memory_percent: Optional[float] = None
    timestamp: float = time.time()

class ResourceManager:
    """Manages system resources and monitoring."""
    
    def __init__(self, 
                 base_dir: str,
                 max_disk_usage: float = 0.9,  # 90% max disk usage
                 check_interval: int = 60,     # Check every 60 seconds
                 metrics_history: int = 1000): # Keep last 1000 metrics
        """Initialize resource manager.
        
        Args:
            base_dir: Base directory for resource monitoring
            max_disk_usage: Maximum allowed disk usage (0.0 to 1.0)
            check_interval: Interval between resource checks in seconds
            metrics_history: Number of historical metrics to keep
        """
        self.base_dir = Path(base_dir)
        self.max_disk_usage = max_disk_usage
        self.check_interval = check_interval
        self.metrics_history = metrics_history
        self.metrics: List[ResourceMetrics] = []
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        # Create metrics directory
        self.metrics_dir = self.base_dir / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize resource tracking
        self._start_monitoring()
    
    def _start_monitoring(self):
        """Start resource monitoring in background thread."""
        if not self._monitoring:
            self._monitoring = True
            self._monitor_thread = threading.Thread(
                target=self._monitor_resources,
                daemon=True
            )
            self._monitor_thread.start()
    
    def _monitor_resources(self):
        """Monitor system resources in background."""
        while self._monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics.append(metrics)
                
                # Trim history if needed
                if len(self.metrics) > self.metrics_history:
                    self.metrics = self.metrics[-self.metrics_history:]
                
                # Save metrics periodically
                if len(self.metrics) % 10 == 0:  # Save every 10 metrics
                    self._save_metrics()
                
                # Check for resource warnings
                self._check_resource_warnings(metrics)
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                time.sleep(self.check_interval)
    
    def _collect_metrics(self) -> ResourceMetrics:
        """Collect current system metrics."""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            # Disk usage
            disk_usage = shutil.disk_usage(self.base_dir)
            disk_percent = disk_usage.used / disk_usage.total
            
            # GPU memory if available
            gpu_memory_percent = None
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory_percent = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
            except ImportError:
                pass
            
            return ResourceMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_percent=disk_percent,
                gpu_memory_percent=gpu_memory_percent
            )
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return ResourceMetrics(0, 0, 0)
    
    def _check_resource_warnings(self, metrics: ResourceMetrics):
        """Check for resource usage warnings."""
        # Disk usage warning
        if metrics.disk_percent > self.max_disk_usage:
            logger.warning(
                f"High disk usage detected: {metrics.disk_percent:.1%}. "
                f"Consider cleaning up unused resources."
            )
        
        # Memory warning
        if metrics.memory_percent > 0.9:  # 90% memory usage
            logger.warning(
                f"High memory usage detected: {metrics.memory_percent:.1%}. "
                f"Consider reducing batch sizes or model cache."
            )
        
        # GPU memory warning
        if metrics.gpu_memory_percent and metrics.gpu_memory_percent > 0.9:
            logger.warning(
                f"High GPU memory usage detected: {metrics.gpu_memory_percent:.1%}. "
                f"Consider reducing model batch size or using CPU fallback."
            )
    
    def _save_metrics(self):
        """Save metrics to disk."""
        try:
            metrics_file = self.metrics_dir / f"metrics_{datetime.now().strftime('%Y%m%d')}.json"
            with open(metrics_file, 'w') as f:
                json.dump([m.__dict__ for m in self.metrics], f)
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    def get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage."""
        metrics = self._collect_metrics()
        return {
            "cpu_percent": metrics.cpu_percent,
            "memory_percent": metrics.memory_percent,
            "disk_percent": metrics.disk_percent,
            "gpu_memory_percent": metrics.gpu_memory_percent
        }
    
    def get_metrics_history(self, 
                          start_time: Optional[float] = None,
                          end_time: Optional[float] = None) -> List[ResourceMetrics]:
        """Get historical metrics within time range."""
        if not start_time:
            start_time = 0
        if not end_time:
            end_time = time.time()
            
        return [
            m for m in self.metrics
            if start_time <= m.timestamp <= end_time
        ]
    
    def cleanup(self, min_free_space: float = 0.2) -> bool:
        """Clean up resources to maintain minimum free space.
        
        Args:
            min_free_space: Minimum required free space (0.0 to 1.0)
            
        Returns:
            bool: True if cleanup was successful
        """
        try:
            # Get current disk usage
            disk_usage = shutil.disk_usage(self.base_dir)
            free_space = disk_usage.free / disk_usage.total
            
            if free_space >= min_free_space:
                return True
            
            # Clean up old metrics files
            metrics_files = sorted(
                self.metrics_dir.glob("metrics_*.json"),
                key=lambda x: x.stat().st_mtime
            )
            
            # Remove oldest metrics files until we have enough space
            while free_space < min_free_space and metrics_files:
                oldest_file = metrics_files.pop(0)
                oldest_file.unlink()
                
                # Recheck disk usage
                disk_usage = shutil.disk_usage(self.base_dir)
                free_space = disk_usage.free / disk_usage.total
            
            return free_space >= min_free_space
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return False
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()
            self._monitor_thread = None 