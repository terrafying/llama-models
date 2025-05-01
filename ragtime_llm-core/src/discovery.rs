use std::path::{Path, PathBuf};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use anyhow::{Result, Context};
use walkdir::WalkDir;
use glob::Pattern;
use tracing::{info, warn, error};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub path: PathBuf,
    pub model_type: String,
    pub size_bytes: u64,
    pub last_modified: chrono::DateTime<chrono::Utc>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryConfig {
    pub search_paths: Vec<PathBuf>,
    pub model_patterns: Vec<String>,
    pub cache_ttl_seconds: u64,
    pub max_parallel_searches: usize,
}

pub struct ResourceDiscovery {
    config: DiscoveryConfig,
    cache: Arc<RwLock<HashMap<String, ModelInfo>>>,
    last_scan: Arc<RwLock<chrono::DateTime<chrono::Utc>>>,
}

impl ResourceDiscovery {
    pub fn new(config: DiscoveryConfig) -> Self {
        Self {
            config,
            cache: Arc::new(RwLock::new(HashMap::new())),
            last_scan: Arc::new(RwLock::new(chrono::Utc::now())),
        }
    }

    pub async fn discover_models(&self) -> Result<Vec<ModelInfo>> {
        let now = chrono::Utc::now();
        let last_scan = *self.last_scan.read().await;
        
        // Check if cache is still valid
        if (now - last_scan).num_seconds() < self.config.cache_ttl_seconds as i64 {
            return Ok(self.cache.read().await.values().cloned().collect());
        }

        // Perform new scan
        let mut discovered_models = Vec::new();
        let mut tasks = Vec::new();

        for path in &self.config.search_paths {
            for pattern in &self.config.model_patterns {
                let pattern = Pattern::new(pattern)
                    .context("Invalid glob pattern")?;
                
                let task = tokio::spawn({
                    let path = path.clone();
                    let pattern = pattern.clone();
                    async move {
                        Self::scan_directory(&path, &pattern).await
                    }
                });
                
                tasks.push(task);
            }
        }

        // Collect results from parallel scans
        for task in tasks {
            match task.await {
                Ok(Ok(models)) => discovered_models.extend(models),
                Ok(Err(e)) => error!("Error scanning directory: {}", e),
                Err(e) => error!("Task error: {}", e),
            }
        }

        // Update cache
        let mut cache = self.cache.write().await;
        cache.clear();
        for model in &discovered_models {
            cache.insert(model.name.clone(), model.clone());
        }

        // Update last scan time
        *self.last_scan.write().await = now;

        Ok(discovered_models)
    }

    async fn scan_directory(path: &Path, pattern: &Pattern) -> Result<Vec<ModelInfo>> {
        let mut models = Vec::new();

        for entry in WalkDir::new(path)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }

            let path_str = path.to_string_lossy();
            if !pattern.matches(&path_str) {
                continue;
            }

            match Self::extract_model_info(path).await {
                Ok(Some(info)) => models.push(info),
                Ok(None) => warn!("Skipping non-model file: {}", path_str),
                Err(e) => error!("Error processing file {}: {}", path_str, e),
            }
        }

        Ok(models)
    }

    async fn extract_model_info(path: &Path) -> Result<Option<ModelInfo>> {
        let metadata = tokio::fs::metadata(path).await?;
        let name = path.file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| anyhow::anyhow!("Invalid filename"))?;

        // Basic model type detection
        let model_type = if name.ends_with(".gguf") {
            "gguf"
        } else if name.ends_with(".bin") {
            "bin"
        } else if name.ends_with(".pt") {
            "pytorch"
        } else {
            return Ok(None);
        };

        let mut metadata_map = HashMap::new();
        metadata_map.insert("format".to_string(), model_type.to_string());
        metadata_map.insert("size".to_string(), metadata.len().to_string());

        Ok(Some(ModelInfo {
            name: name.to_string(),
            path: path.to_path_buf(),
            model_type: model_type.to_string(),
            size_bytes: metadata.len(),
            last_modified: chrono::DateTime::from(metadata.modified()?),
            metadata: metadata_map,
        }))
    }

    pub async fn get_model_info(&self, name: &str) -> Result<Option<ModelInfo>> {
        // Check cache first
        if let Some(model) = self.cache.read().await.get(name) {
            return Ok(Some(model.clone()));
        }

        // If not in cache, perform a fresh scan
        self.discover_models().await?;
        
        // Check cache again after scan
        Ok(self.cache.read().await.get(name).cloned())
    }

    pub async fn clear_cache(&self) {
        self.cache.write().await.clear();
        *self.last_scan.write().await = chrono::Utc::now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use std::fs::File;
    use std::io::Write;

    #[tokio::test]
    async fn test_model_discovery() -> Result<()> {
        let temp_dir = tempdir()?;
        let model_path = temp_dir.path().join("test_model.gguf");
        
        // Create a dummy model file
        let mut file = File::create(&model_path)?;
        file.write_all(b"dummy model data")?;

        let config = DiscoveryConfig {
            search_paths: vec![temp_dir.path().to_path_buf()],
            model_patterns: vec!["*.gguf".to_string()],
            cache_ttl_seconds: 3600,
            max_parallel_searches: 4,
        };

        let discovery = ResourceDiscovery::new(config);
        let models = discovery.discover_models().await?;

        assert!(!models.is_empty());
        assert_eq!(models[0].name, "test_model.gguf");
        assert_eq!(models[0].model_type, "gguf");

        Ok(())
    }

    #[tokio::test]
    async fn test_cache_behavior() -> Result<()> {
        let temp_dir = tempdir()?;
        let model_path = temp_dir.path().join("test_model.gguf");
        
        // Create a dummy model file
        let mut file = File::create(&model_path)?;
        file.write_all(b"dummy model data")?;

        let config = DiscoveryConfig {
            search_paths: vec![temp_dir.path().to_path_buf()],
            model_patterns: vec!["*.gguf".to_string()],
            cache_ttl_seconds: 1,
            max_parallel_searches: 4,
        };

        let discovery = ResourceDiscovery::new(config);
        
        // First scan
        let models1 = discovery.discover_models().await?;
        assert!(!models1.is_empty());

        // Immediate second scan should use cache
        let models2 = discovery.discover_models().await?;
        assert_eq!(models1, models2);

        // Wait for cache to expire
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
        
        // Should perform new scan
        let models3 = discovery.discover_models().await?;
        assert_eq!(models1, models3);

        Ok(())
    }
} 