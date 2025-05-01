//! Distributed processing functionality
//! 
//! This module provides functionality for distributed processing of tasks
//! across multiple worker threads.

use std::sync::Arc;
use tokio::sync::{RwLock, Semaphore};
use rayon::prelude::*;
use tracing::{info, error, warn};
use thiserror::Error;

use crate::{RagtimeError, ResourceManager};

/// Trait for items that can be processed
pub trait Processable: Send + Sync {
    type Output: Send + Sync;
    
    fn process(&self) -> Result<Self::Output, RagtimeError>;
}

/// Distributed processor for parallel task execution
pub struct DistributedProcessor {
    num_workers: usize,
    resource_manager: Arc<RwLock<ResourceManager>>,
    semaphore: Arc<Semaphore>,
}

impl DistributedProcessor {
    /// Create a new distributed processor
    pub fn new(
        num_workers: usize,
        resource_manager: Arc<RwLock<ResourceManager>>,
    ) -> Result<Self, RagtimeError> {
        info!("Initializing distributed processor with {} workers", num_workers);
        
        Ok(Self {
            num_workers,
            resource_manager,
            semaphore: Arc::new(Semaphore::new(num_workers)),
        })
    }
    
    /// Process a batch of items in parallel
    pub async fn process_batch<T: Processable>(
        &self,
        items: Vec<T>,
        batch_size: usize,
    ) -> Result<Vec<T::Output>, RagtimeError> {
        info!("Processing batch of {} items", items.len());
        
        // Split items into batches
        let batches: Vec<Vec<T>> = items
            .chunks(batch_size)
            .map(|chunk| chunk.to_vec())
            .collect();
        
        // Process batches in parallel
        let results: Vec<Result<Vec<T::Output>, RagtimeError>> = batches
            .into_par_iter()
            .map(|batch| {
                let _permit = self.semaphore.clone().try_acquire_owned()
                    .map_err(|_| RagtimeError::ResourceExhausted)?;
                
                // Process items in batch
                let results: Vec<Result<T::Output, RagtimeError>> = batch
                    .into_par_iter()
                    .map(|item| item.process())
                    .collect();
                
                // Collect results
                results.into_iter().collect()
            })
            .collect();
        
        // Flatten results
        let results: Vec<T::Output> = results
            .into_iter()
            .collect::<Result<Vec<Vec<T::Output>>, RagtimeError>>()?
            .into_iter()
            .flatten()
            .collect();
        
        Ok(results)
    }
    
    /// Get the number of worker threads
    pub fn num_workers(&self) -> usize {
        self.num_workers
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mockall::predicate::*;
    
    struct TestItem {
        value: i32,
    }
    
    impl Processable for TestItem {
        type Output = i32;
        
        fn process(&self) -> Result<Self::Output, RagtimeError> {
            Ok(self.value * 2)
        }
    }
    
    #[tokio::test]
    async fn test_batch_processing() {
        let resource_manager = Arc::new(RwLock::new(ResourceManager::new(
            ResourceLimits::default()
        ).unwrap()));
        
        let processor = DistributedProcessor::new(4, resource_manager).unwrap();
        
        let items: Vec<TestItem> = (1..=10)
            .map(|i| TestItem { value: i })
            .collect();
        
        let results = processor.process_batch(items, 3).await.unwrap();
        
        assert_eq!(results.len(), 10);
        assert_eq!(results, vec![2, 4, 6, 8, 10, 12, 14, 16, 18, 20]);
    }
    
    #[tokio::test]
    async fn test_resource_exhaustion() {
        let resource_manager = Arc::new(RwLock::new(ResourceManager::new(
            ResourceLimits {
                max_memory_mb: 1,
                max_cpu_percent: 10,
                ..Default::default()
            }
        ).unwrap()));
        
        let processor = DistributedProcessor::new(1, resource_manager).unwrap();
        
        let items: Vec<TestItem> = (1..=100)
            .map(|i| TestItem { value: i })
            .collect();
        
        let result = processor.process_batch(items, 10).await;
        assert!(result.is_err());
    }
} 