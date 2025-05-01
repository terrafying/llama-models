//! Vector store functionality
//! 
//! This module provides functionality for efficient vector operations and similarity search,
//! using FAISS for high-performance vector indexing and retrieval.

use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, error, warn};
use thiserror::Error;
use serde::{Serialize, Deserialize};
use faiss::{Index, IndexImpl, MetricType};
use ndarray::{Array1, Array2};

/// Vector store configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorStoreConfig {
    /// Vector dimension
    pub dimension: usize,
    /// Number of clusters for IVF index
    pub num_clusters: usize,
    /// Number of probes for search
    pub num_probes: usize,
    /// Metric type for similarity search
    pub metric_type: MetricType,
}

impl Default for VectorStoreConfig {
    fn default() -> Self {
        Self {
            dimension: 768,
            num_clusters: 100,
            num_probes: 10,
            metric_type: MetricType::L2,
        }
    }
}

/// Vector store statistics
#[derive(Debug, Clone, Serialize)]
pub struct VectorStoreStats {
    /// Number of vectors in store
    pub num_vectors: usize,
    /// Vector dimension
    pub dimension: usize,
    /// Index type
    pub index_type: String,
    /// Memory usage in MB
    pub memory_mb: u64,
}

/// Vector type
pub type Vector = Array1<f32>;

/// Vector metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metadata {
    /// Unique identifier
    pub id: String,
    /// Additional metadata
    pub data: serde_json::Value,
}

/// Search result
#[derive(Debug, Clone, Serialize)]
pub struct SearchResult {
    /// Vector identifier
    pub id: String,
    /// Similarity score
    pub score: f32,
    /// Vector metadata
    pub metadata: Metadata,
}

/// Vector store for efficient vector operations
pub struct VectorStore {
    config: VectorStoreConfig,
    index: IndexImpl,
    metadata: Vec<Metadata>,
}

impl VectorStore {
    /// Create a new vector store
    pub fn new(config: VectorStoreConfig) -> Result<Self, RagtimeError> {
        info!("Initializing vector store");
        
        // Create FAISS index
        let index = faiss::index_factory(
            config.dimension as i64,
            &format!("IVF{},Flat", config.num_clusters),
            config.metric_type,
        )?;
        
        Ok(Self {
            config,
            index,
            metadata: Vec::new(),
        })
    }
    
    /// Add vectors to the store
    pub async fn add_vectors(
        &mut self,
        vectors: Vec<Vector>,
        metadata: Vec<Metadata>,
    ) -> Result<(), RagtimeError> {
        info!("Adding {} vectors to store", vectors.len());
        
        // Convert vectors to FAISS format
        let mut data = Vec::with_capacity(vectors.len() * self.config.dimension);
        for vector in vectors {
            data.extend_from_slice(vector.as_slice().unwrap());
        }
        
        // Add vectors to index
        self.index.add(&data)?;
        
        // Store metadata
        self.metadata.extend(metadata);
        
        Ok(())
    }
    
    /// Search for similar vectors
    pub async fn search(
        &self,
        query: Vector,
        k: usize,
    ) -> Result<Vec<SearchResult>, RagtimeError> {
        info!("Searching for {} nearest neighbors", k);
        
        // Set number of probes
        if let Some(index) = self.index.as_ref().as_ivf() {
            index.set_nprobe(self.config.num_probes as i32)?;
        }
        
        // Search index
        let (distances, labels) = self.index.search(
            query.as_slice().unwrap(),
            k as i64,
        )?;
        
        // Convert results
        let results: Vec<SearchResult> = labels
            .iter()
            .zip(distances.iter())
            .filter_map(|(&label, &distance)| {
                if label >= 0 {
                    let idx = label as usize;
                    Some(SearchResult {
                        id: self.metadata[idx].id.clone(),
                        score: distance,
                        metadata: self.metadata[idx].clone(),
                    })
                } else {
                    None
                }
            })
            .collect();
        
        Ok(results)
    }
    
    /// Get vector store statistics
    pub async fn get_stats(&self) -> Result<VectorStoreStats, RagtimeError> {
        Ok(VectorStoreStats {
            num_vectors: self.metadata.len(),
            dimension: self.config.dimension,
            index_type: self.index.description()?,
            memory_mb: self.index.ntotal()? as u64 * self.config.dimension as u64 * 4 / 1024 / 1024,
        })
    }
    
    /// Save vector store to disk
    pub async fn save(&self, path: &str) -> Result<(), RagtimeError> {
        info!("Saving vector store to {}", path);
        
        // Save index
        self.index.write_file(path)?;
        
        // Save metadata
        let metadata_path = format!("{}.meta", path);
        let metadata_json = serde_json::to_string(&self.metadata)?;
        tokio::fs::write(metadata_path, metadata_json).await?;
        
        Ok(())
    }
    
    /// Load vector store from disk
    pub async fn load(path: &str, config: VectorStoreConfig) -> Result<Self, RagtimeError> {
        info!("Loading vector store from {}", path);
        
        // Load index
        let index = faiss::read_index(path)?;
        
        // Load metadata
        let metadata_path = format!("{}.meta", path);
        let metadata_json = tokio::fs::read_to_string(metadata_path).await?;
        let metadata: Vec<Metadata> = serde_json::from_str(&metadata_json)?;
        
        Ok(Self {
            config,
            index,
            metadata,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    
    #[tokio::test]
    async fn test_vector_store() {
        let config = VectorStoreConfig::default();
        let mut store = VectorStore::new(config).unwrap();
        
        // Create test vectors
        let vectors = vec![
            array![1.0, 0.0, 0.0],
            array![0.0, 1.0, 0.0],
            array![0.0, 0.0, 1.0],
        ];
        
        let metadata = vec![
            Metadata {
                id: "1".to_string(),
                data: serde_json::json!({"label": "x"}),
            },
            Metadata {
                id: "2".to_string(),
                data: serde_json::json!({"label": "y"}),
            },
            Metadata {
                id: "3".to_string(),
                data: serde_json::json!({"label": "z"}),
            },
        ];
        
        // Add vectors
        store.add_vectors(vectors, metadata).await.unwrap();
        
        // Search
        let query = array![1.0, 0.0, 0.0];
        let results = store.search(query, 2).await.unwrap();
        
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "1");
    }
    
    #[tokio::test]
    async fn test_save_load() {
        let config = VectorStoreConfig::default();
        let mut store = VectorStore::new(config).unwrap();
        
        // Add test data
        let vectors = vec![array![1.0, 0.0, 0.0]];
        let metadata = vec![Metadata {
            id: "1".to_string(),
            data: serde_json::json!({}),
        }];
        
        store.add_vectors(vectors, metadata).await.unwrap();
        
        // Save and load
        store.save("test_index").await.unwrap();
        let loaded = VectorStore::load("test_index", config).await.unwrap();
        
        assert_eq!(loaded.metadata.len(), 1);
        assert_eq!(loaded.metadata[0].id, "1");
    }
} 