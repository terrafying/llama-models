//! Core functionality for the Ragtime LLM toolkit
//! 
//! This module provides the core functionality for the Ragtime LLM toolkit,
//! including distributed processing, resource management, and vector operations.

use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, error, warn};
use thiserror::Error;
use serde::{Serialize, Deserialize};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::PathBuf;
use std::collections::HashMap;

mod distributed;
mod resource;
mod vector;
mod error;
mod discovery;

pub use distributed::*;
pub use resource::*;
pub use vector::*;
pub use error::*;
use discovery::{ResourceDiscovery, DiscoveryConfig, ModelInfo};

/// Core configuration for the Ragtime LLM toolkit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Number of worker threads
    pub num_workers: usize,
    /// Maximum batch size for processing
    pub max_batch_size: usize,
    /// Resource limits
    pub resource_limits: ResourceLimits,
    /// Vector store configuration
    pub vector_store: VectorStoreConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            num_workers: num_cpus::get(),
            max_batch_size: 1024,
            resource_limits: ResourceLimits::default(),
            vector_store: VectorStoreConfig::default(),
        }
    }
}

/// Main Ragtime LLM engine
pub struct RagtimeEngine {
    config: Arc<Config>,
    resource_manager: Arc<RwLock<ResourceManager>>,
    vector_store: Arc<RwLock<VectorStore>>,
    distributed_processor: Arc<DistributedProcessor>,
}

impl RagtimeEngine {
    /// Create a new Ragtime LLM engine
    pub fn new(config: Config) -> Result<Self, RagtimeError> {
        info!("Initializing Ragtime LLM engine");
        
        let config = Arc::new(config);
        let resource_manager = Arc::new(RwLock::new(ResourceManager::new(
            config.resource_limits.clone()
        )?));
        
        let vector_store = Arc::new(RwLock::new(VectorStore::new(
            config.vector_store.clone()
        )?));
        
        let distributed_processor = Arc::new(DistributedProcessor::new(
            config.num_workers,
            resource_manager.clone(),
        )?);
        
        Ok(Self {
            config,
            resource_manager,
            vector_store,
            distributed_processor,
        })
    }
    
    /// Process a batch of items
    pub async fn process_batch<T: Processable>(
        &self,
        items: Vec<T>,
    ) -> Result<Vec<T::Output>, RagtimeError> {
        info!("Processing batch of {} items", items.len());
        
        // Check resource availability
        let mut resource_manager = self.resource_manager.write().await;
        resource_manager.check_resources()?;
        
        // Process items in parallel
        let results = self.distributed_processor
            .process_batch(items, self.config.max_batch_size)
            .await?;
        
        // Update resource usage
        resource_manager.update_usage()?;
        
        Ok(results)
    }
    
    /// Add vectors to the vector store
    pub async fn add_vectors(
        &self,
        vectors: Vec<Vector>,
        metadata: Vec<Metadata>,
    ) -> Result<(), RagtimeError> {
        info!("Adding {} vectors to store", vectors.len());
        
        let mut vector_store = self.vector_store.write().await;
        vector_store.add_vectors(vectors, metadata).await?;
        
        Ok(())
    }
    
    /// Search for similar vectors
    pub async fn search_vectors(
        &self,
        query: Vector,
        k: usize,
    ) -> Result<Vec<SearchResult>, RagtimeError> {
        info!("Searching for {} nearest neighbors", k);
        
        let vector_store = self.vector_store.read().await;
        let results = vector_store.search(query, k).await?;
        
        Ok(results)
    }
    
    /// Get engine statistics
    pub async fn get_stats(&self) -> Result<EngineStats, RagtimeError> {
        let resource_manager = self.resource_manager.read().await;
        let vector_store = self.vector_store.read().await;
        
        Ok(EngineStats {
            resource_usage: resource_manager.get_usage()?,
            vector_store_stats: vector_store.get_stats().await?,
            num_workers: self.config.num_workers,
        })
    }
}

/// Engine statistics
#[derive(Debug, Clone, Serialize)]
pub struct EngineStats {
    /// Current resource usage
    pub resource_usage: ResourceUsage,
    /// Vector store statistics
    pub vector_store_stats: VectorStoreStats,
    /// Number of worker threads
    pub num_workers: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use mockall::predicate::*;
    
    #[tokio::test]
    async fn test_engine_initialization() {
        let config = Config::default();
        let engine = RagtimeEngine::new(config).unwrap();
        
        let stats = engine.get_stats().await.unwrap();
        assert_eq!(stats.num_workers, num_cpus::get());
    }
    
    #[tokio::test]
    async fn test_batch_processing() {
        let config = Config::default();
        let engine = RagtimeEngine::new(config).unwrap();
        
        let items = vec![1, 2, 3, 4, 5];
        let results = engine.process_batch(items).await.unwrap();
        
        assert_eq!(results.len(), 5);
    }
}

#[pymodule]
fn ragtime_llm(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyResourceDiscovery>()?;
    m.add_class::<PyModelInfo>()?;
    Ok(())
}

#[pyclass]
struct PyModelInfo {
    inner: ModelInfo,
}

#[pymethods]
impl PyModelInfo {
    #[new]
    fn new(
        name: String,
        path: String,
        model_type: String,
        size_bytes: u64,
        last_modified: String,
        metadata: HashMap<String, String>,
    ) -> PyResult<Self> {
        let path = PathBuf::from(path);
        let last_modified = chrono::DateTime::parse_from_rfc3339(&last_modified)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
            .with_timezone(&chrono::Utc);

        Ok(Self {
            inner: ModelInfo {
                name,
                path,
                model_type,
                size_bytes,
                last_modified,
                metadata,
            },
        })
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    #[getter]
    fn path(&self) -> String {
        self.inner.path.to_string_lossy().into_owned()
    }

    #[getter]
    fn model_type(&self) -> String {
        self.inner.model_type.clone()
    }

    #[getter]
    fn size_bytes(&self) -> u64 {
        self.inner.size_bytes
    }

    #[getter]
    fn last_modified(&self) -> String {
        self.inner.last_modified.to_rfc3339()
    }

    #[getter]
    fn metadata(&self) -> HashMap<String, String> {
        self.inner.metadata.clone()
    }
}

#[pyclass]
struct PyResourceDiscovery {
    inner: ResourceDiscovery,
}

#[pymethods]
impl PyResourceDiscovery {
    #[new]
    fn new(
        search_paths: Vec<String>,
        model_patterns: Vec<String>,
        cache_ttl_seconds: u64,
        max_parallel_searches: usize,
    ) -> PyResult<Self> {
        let config = DiscoveryConfig {
            search_paths: search_paths.into_iter().map(PathBuf::from).collect(),
            model_patterns,
            cache_ttl_seconds,
            max_parallel_searches,
        };

        Ok(Self {
            inner: ResourceDiscovery::new(config),
        })
    }

    fn discover_models(&self, py: Python<'_>) -> PyResult<Vec<PyModelInfo>> {
        py.allow_threads(|| {
            let runtime = tokio::runtime::Runtime::new()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            
            runtime.block_on(async {
                let models = self.inner.discover_models().await
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                
                Ok(models.into_iter()
                    .map(|m| PyModelInfo { inner: m })
                    .collect())
            })
        })
    }

    fn get_model_info(&self, py: Python<'_>, name: &str) -> PyResult<Option<PyModelInfo>> {
        py.allow_threads(|| {
            let runtime = tokio::runtime::Runtime::new()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            
            runtime.block_on(async {
                let model = self.inner.get_model_info(name).await
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                
                Ok(model.map(|m| PyModelInfo { inner: m }))
            })
        })
    }

    fn clear_cache(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| {
            let runtime = tokio::runtime::Runtime::new()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            
            runtime.block_on(async {
                self.inner.clear_cache().await;
                Ok(())
            })
        })
    }
} 