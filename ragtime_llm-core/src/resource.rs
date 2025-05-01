//! Resource management functionality
//! 
//! This module provides functionality for monitoring and controlling system resources,
//! including CPU, memory, and GPU usage.

use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, error, warn};
use thiserror::Error;
use serde::{Serialize, Deserialize};
use sysinfo::{System, SystemExt, ProcessExt, CpuExt};

/// Resource limits configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum memory usage in MB
    pub max_memory_mb: u64,
    /// Maximum CPU usage percentage
    pub max_cpu_percent: f32,
    /// Maximum GPU memory usage in MB
    pub max_gpu_memory_mb: Option<u64>,
    /// Maximum number of concurrent operations
    pub max_concurrent_ops: usize,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_memory_mb: 1024 * 8,  // 8GB
            max_cpu_percent: 80.0,
            max_gpu_memory_mb: None,
            max_concurrent_ops: 4,
        }
    }
}

/// Current resource usage
#[derive(Debug, Clone, Serialize)]
pub struct ResourceUsage {
    /// Current memory usage in MB
    pub memory_mb: u64,
    /// Current CPU usage percentage
    pub cpu_percent: f32,
    /// Current GPU memory usage in MB
    pub gpu_memory_mb: Option<u64>,
    /// Number of active operations
    pub active_ops: usize,
}

/// Resource manager for monitoring and controlling system resources
pub struct ResourceManager {
    limits: ResourceLimits,
    system: System,
    active_ops: usize,
}

impl ResourceManager {
    /// Create a new resource manager
    pub fn new(limits: ResourceLimits) -> Result<Self, RagtimeError> {
        info!("Initializing resource manager");
        
        let mut system = System::new_all();
        system.refresh_all();
        
        Ok(Self {
            limits,
            system,
            active_ops: 0,
        })
    }
    
    /// Check if current resource usage is within limits
    pub fn check_resources(&mut self) -> Result<(), RagtimeError> {
        self.system.refresh_all();
        
        // Check memory usage
        let memory_mb = self.system.used_memory() / 1024 / 1024;
        if memory_mb > self.limits.max_memory_mb {
            return Err(RagtimeError::ResourceExhausted);
        }
        
        // Check CPU usage
        let cpu_percent = self.system.global_cpu_info().cpu_usage();
        if cpu_percent > self.limits.max_cpu_percent {
            return Err(RagtimeError::ResourceExhausted);
        }
        
        // Check GPU memory if available
        if let Some(max_gpu_mb) = self.limits.max_gpu_memory_mb {
            if let Some(gpu_mb) = self.get_gpu_memory() {
                if gpu_mb > max_gpu_mb {
                    return Err(RagtimeError::ResourceExhausted);
                }
            }
        }
        
        // Check concurrent operations
        if self.active_ops >= self.limits.max_concurrent_ops {
            return Err(RagtimeError::ResourceExhausted);
        }
        
        Ok(())
    }
    
    /// Update resource usage statistics
    pub fn update_usage(&mut self) -> Result<(), RagtimeError> {
        self.system.refresh_all();
        Ok(())
    }
    
    /// Get current resource usage
    pub fn get_usage(&self) -> Result<ResourceUsage, RagtimeError> {
        Ok(ResourceUsage {
            memory_mb: self.system.used_memory() / 1024 / 1024,
            cpu_percent: self.system.global_cpu_info().cpu_usage(),
            gpu_memory_mb: self.get_gpu_memory(),
            active_ops: self.active_ops,
        })
    }
    
    /// Get GPU memory usage if available
    fn get_gpu_memory(&self) -> Option<u64> {
        // TODO: Implement GPU memory monitoring
        None
    }
    
    /// Increment active operations count
    pub fn increment_ops(&mut self) {
        self.active_ops += 1;
    }
    
    /// Decrement active operations count
    pub fn decrement_ops(&mut self) {
        if self.active_ops > 0 {
            self.active_ops -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_resource_limits() {
        let limits = ResourceLimits::default();
        assert_eq!(limits.max_memory_mb, 1024 * 8);
        assert_eq!(limits.max_cpu_percent, 80.0);
        assert_eq!(limits.max_concurrent_ops, 4);
    }
    
    #[test]
    fn test_resource_manager() {
        let limits = ResourceLimits {
            max_memory_mb: 1024,
            max_cpu_percent: 50.0,
            max_gpu_memory_mb: Some(1024),
            max_concurrent_ops: 2,
        };
        
        let mut manager = ResourceManager::new(limits).unwrap();
        
        // Check resources
        assert!(manager.check_resources().is_ok());
        
        // Update usage
        assert!(manager.update_usage().is_ok());
        
        // Get usage
        let usage = manager.get_usage().unwrap();
        assert!(usage.memory_mb <= 1024);
        assert!(usage.cpu_percent <= 50.0);
    }
    
    #[test]
    fn test_concurrent_ops() {
        let limits = ResourceLimits {
            max_concurrent_ops: 2,
            ..Default::default()
        };
        
        let mut manager = ResourceManager::new(limits).unwrap();
        
        // Increment operations
        manager.increment_ops();
        manager.increment_ops();
        assert!(manager.check_resources().is_err());
        
        // Decrement operations
        manager.decrement_ops();
        assert!(manager.check_resources().is_ok());
    }
} 