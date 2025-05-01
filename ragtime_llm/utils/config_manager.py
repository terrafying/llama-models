"""
Configuration management system.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import yaml
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)

class ModelConfig(BaseModel):
    """Model configuration."""
    model_id: str
    provider: str = "local"
    quantization: Optional[str] = None
    max_memory: Optional[Dict[int, str]] = None
    device_map: Optional[str] = None
    use_metal: bool = False
    auto_download: bool = True

class StorageConfig(BaseModel):
    """Storage configuration."""
    cache_dir: str = ".cache"
    ipfs_api: Optional[str] = None
    local_volumes: List[str] = Field(default_factory=list)
    max_cache_size: int = 10 * 1024 * 1024 * 1024  # 10GB
    cleanup_threshold: float = 0.9  # 90% usage

class ResourceConfig(BaseModel):
    """Resource configuration."""
    max_disk_usage: float = 0.9
    check_interval: int = 60
    metrics_history: int = 1000
    min_free_space: float = 0.2

class SystemConfig(BaseModel):
    """System-wide configuration."""
    models: Dict[str, ModelConfig] = Field(default_factory=dict)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    resources: ResourceConfig = Field(default_factory=ResourceConfig)
    environment: str = "development"
    
    @validator('environment')
    def validate_environment(cls, v):
        """Validate environment name."""
        valid_envs = ['development', 'staging', 'production']
        if v not in valid_envs:
            raise ValueError(f"Environment must be one of {valid_envs}")
        return v

class ConfigManager:
    """Manages system configuration."""
    
    def __init__(self, config_dir: str = "config"):
        """Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config: Optional[SystemConfig] = None
        self._load_config()
    
    def _load_config(self):
        """Load configuration from files."""
        try:
            # Load base config
            base_config = self._load_yaml("base.yaml")
            
            # Load environment-specific config
            env = os.getenv("RAGTIME_ENV", "development")
            env_config = self._load_yaml(f"{env}.yaml")
            
            # Merge configurations
            merged_config = self._merge_configs(base_config, env_config)
            
            # Validate and create config object
            self.config = SystemConfig(**merged_config)
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            # Use default configuration
            self.config = SystemConfig()
    
    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load YAML configuration file."""
        config_file = self.config_dir / filename
        if not config_file.exists():
            return {}
            
        try:
            with open(config_file) as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return {}
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configurations, with override taking precedence."""
        merged = base.copy()
        
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
                
        return merged
    
    def get_config(self) -> SystemConfig:
        """Get current configuration."""
        if not self.config:
            self._load_config()
        return self.config
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        if not self.config:
            self._load_config()
            
        # Convert updates to config object
        current_dict = self.config.dict()
        updated_dict = self._merge_configs(current_dict, updates)
        
        # Validate and update
        self.config = SystemConfig(**updated_dict)
        
        # Save to file
        self._save_config()
    
    def _save_config(self):
        """Save current configuration to file."""
        if not self.config:
            return
            
        try:
            # Save environment-specific config
            env = self.config.environment
            config_file = self.config_dir / f"{env}.yaml"
            
            with open(config_file, 'w') as f:
                yaml.dump(self.config.dict(), f)
                
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def get_model_config(self, model_type: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model type."""
        if not self.config:
            self._load_config()
        return self.config.models.get(model_type)
    
    def get_storage_config(self) -> StorageConfig:
        """Get storage configuration."""
        if not self.config:
            self._load_config()
        return self.config.storage
    
    def get_resource_config(self) -> ResourceConfig:
        """Get resource configuration."""
        if not self.config:
            self._load_config()
        return self.config.resources
    
    def validate_config(self) -> bool:
        """Validate current configuration."""
        if not self.config:
            return False
            
        try:
            # Validate all model configurations
            for model_type, model_config in self.config.models.items():
                ModelConfig(**model_config.dict())
            
            # Validate storage configuration
            StorageConfig(**self.config.storage.dict())
            
            # Validate resource configuration
            ResourceConfig(**self.config.resources.dict())
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False 