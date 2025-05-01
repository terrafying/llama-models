"""
Tests for the model discovery functionality.
"""

import pytest
from pathlib import Path
import tempfile
import shutil
from ragtime_llm.utils.discovery import ResourceDiscovery, ModelInfo

@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture
def discovery(temp_dir):
    """Create a ResourceDiscovery instance with a temporary directory."""
    return ResourceDiscovery(str(temp_dir))

def test_model_info_creation():
    """Test ModelInfo class creation and serialization."""
    model_info = ModelInfo(
        name="test-model",
        path=Path("/path/to/model.gguf"),
        source="local",
        size=1000,
        quantization="Q4_K_M",
        format="gguf",
        metadata={"test": "value"}
    )
    
    assert model_info.name == "test-model"
    assert model_info.source == "local"
    assert model_info.size == 1000
    assert model_info.quantization == "Q4_K_M"
    assert model_info.format == "gguf"
    assert model_info.metadata["test"] == "value"
    
    # Test serialization
    model_dict = model_info.to_dict()
    assert model_dict["name"] == "test-model"
    assert model_dict["source"] == "local"
    assert model_dict["size"] == 1000
    assert model_dict["quantization"] == "Q4_K_M"
    assert model_dict["format"] == "gguf"
    assert model_dict["metadata"]["test"] == "value"

def test_local_model_discovery(temp_dir, discovery):
    """Test discovery of local models."""
    # Create test model directories
    model_dirs = {
        'llm': ['models/llm', 'llama_models'],
        'embedding': ['models/embedding'],
        'audio': ['models/audio'],
        'video': ['models/video']
    }
    
    for model_type, dirs in model_dirs.items():
        for dir_path in dirs:
            full_path = temp_dir / dir_path
            full_path.mkdir(parents=True)
            
            # Create some test model files
            (full_path / "test_model.gguf").touch()
            (full_path / "test_model.pt").touch()
    
    # Discover models
    models = discovery.discover_models()
    
    # Verify results
    for model_type in ['llm', 'embedding', 'audio', 'video']:
        assert len(models[model_type]) > 0
        for model in models[model_type]:
            assert isinstance(model, ModelInfo)
            assert model.source == "local"
            assert model.format in ['gguf', 'pt']

def test_huggingface_model_discovery(discovery):
    """Test discovery of Hugging Face models."""
    models = discovery.discover_models()
    
    # Verify that we get some Hugging Face models
    hf_models = [m for m in models['llm'] if m.source == "huggingface"]
    assert len(hf_models) > 0
    
    # Verify model info structure
    for model in hf_models:
        assert isinstance(model, ModelInfo)
        assert model.source == "huggingface"
        assert "tags" in model.metadata
        assert "downloads" in model.metadata
        assert "likes" in model.metadata

def test_model_download(temp_dir, discovery):
    """Test downloading a model from Hugging Face."""
    # Try to download a small model
    model_info = discovery.download_model("distilbert-base-uncased", model_type="embedding")
    
    assert model_info is not None
    assert model_info.source == "local"
    assert model_info.path.exists()
    assert model_info.format == "transformers"
    assert model_info.metadata["original_source"] == "huggingface"
    
    # Verify the model files exist
    assert (model_info.path / "config.json").exists()
    assert (model_info.path / "pytorch_model.bin").exists()
    assert (model_info.path / "tokenizer.json").exists()

def test_get_model_info(temp_dir, discovery):
    """Test getting information about a specific model."""
    # Create a test model
    model_path = temp_dir / "models" / "llm" / "test_model.gguf"
    model_path.parent.mkdir(parents=True)
    model_path.touch()
    
    # Discover models
    discovery.discover_all()
    
    # Get model info
    model_info = discovery.get_model_info("test_model")
    assert model_info is not None
    assert model_info.name == "test_model"
    assert model_info.source == "local"
    assert model_info.format == "gguf"
    
    # Test non-existent model
    assert discovery.get_model_info("non_existent_model") is None 