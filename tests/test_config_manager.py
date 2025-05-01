"""Tests for configuration management system."""

import os
import pytest
from pathlib import Path
from ragtime_llm.utils.config_manager import (
    ConfigManager,
    SystemConfig,
    ModelConfig,
    StorageConfig,
    ResourceConfig
)

@pytest.fixture
def temp_config_dir(tmp_path):
    """Create temporary config directory."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    return config_dir

@pytest.fixture
def config_manager(temp_config_dir):
    """Create ConfigManager instance."""
    return ConfigManager(str(temp_config_dir))

def test_config_manager_initialization(config_manager):
    """Test ConfigManager initialization."""
    assert config_manager.config_dir.exists()
    assert isinstance(config_manager.get_config(), SystemConfig)

def test_default_config(config_manager):
    """Test default configuration values."""
    config = config_manager.get_config()
    
    # Check environment
    assert config.environment == "development"
    
    # Check storage config
    assert config.storage.cache_dir == ".cache"
    assert config.storage.max_cache_size == 10 * 1024 * 1024 * 1024
    assert config.storage.cleanup_threshold == 0.9
    
    # Check resource config
    assert config.resources.max_disk_usage == 0.9
    assert config.resources.check_interval == 60
    assert config.resources.metrics_history == 1000
    assert config.resources.min_free_space == 0.2

def test_config_update(config_manager):
    """Test configuration updates."""
    # Update storage config
    updates = {
        "storage": {
            "cache_dir": "/tmp/cache",
            "max_cache_size": 20 * 1024 * 1024 * 1024
        }
    }
    config_manager.update_config(updates)
    
    config = config_manager.get_config()
    assert config.storage.cache_dir == "/tmp/cache"
    assert config.storage.max_cache_size == 20 * 1024 * 1024 * 1024

def test_model_config(config_manager):
    """Test model configuration."""
    # Add model config
    updates = {
        "models": {
            "test_model": {
                "model_id": "test/model",
                "provider": "huggingface",
                "quantization": "8bit",
                "auto_download": True
            }
        }
    }
    config_manager.update_config(updates)
    
    # Get model config
    model_config = config_manager.get_model_config("test_model")
    assert model_config is not None
    assert model_config.model_id == "test/model"
    assert model_config.provider == "huggingface"
    assert model_config.quantization == "8bit"
    assert model_config.auto_download is True

def test_config_validation(config_manager):
    """Test configuration validation."""
    # Valid config
    assert config_manager.validate_config()
    
    # Invalid config - invalid environment
    updates = {"environment": "invalid"}
    with pytest.raises(ValueError):
        config_manager.update_config(updates)
    
    # Invalid config - invalid model config
    updates = {
        "models": {
            "test_model": {
                "model_id": "test/model",
                "provider": "invalid_provider"
            }
        }
    }
    with pytest.raises(ValueError):
        config_manager.update_config(updates)

def test_config_file_loading(temp_config_dir):
    """Test loading configuration from files."""
    # Create base config
    base_config = """
    environment: development
    storage:
      cache_dir: /tmp/base
    """
    (temp_config_dir / "base.yaml").write_text(base_config)
    
    # Create environment config
    env_config = """
    storage:
      cache_dir: /tmp/env
    """
    (temp_config_dir / "development.yaml").write_text(env_config)
    
    # Create config manager
    manager = ConfigManager(str(temp_config_dir))
    
    # Check merged config
    config = manager.get_config()
    assert config.storage.cache_dir == "/tmp/env"  # Environment config overrides base

def test_config_persistence(config_manager):
    """Test configuration persistence."""
    # Update config
    updates = {
        "storage": {
            "cache_dir": "/tmp/persist"
        }
    }
    config_manager.update_config(updates)
    
    # Create new manager instance
    new_manager = ConfigManager(str(config_manager.config_dir))
    
    # Check config was persisted
    config = new_manager.get_config()
    assert config.storage.cache_dir == "/tmp/persist" 