"""
Central discovery module for auto-detecting resources across the project.
"""

from pathlib import Path
import importlib
import inspect
import json
from typing import Dict, List, Any, Optional, Union
import glob
import os
from huggingface_hub import HfApi, ModelFilter
import torch

class ModelInfo:
    """Class to store model information."""
    def __init__(
        self,
        name: str,
        path: Optional[Path] = None,
        source: str = "local",
        size: Optional[int] = None,
        quantization: Optional[str] = None,
        format: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.path = path
        self.source = source  # "local" or "huggingface"
        self.size = size
        self.quantization = quantization
        self.format = format
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "path": str(self.path) if self.path else None,
            "source": self.source,
            "size": self.size,
            "quantization": self.quantization,
            "format": self.format,
            "metadata": self.metadata
        }

class ResourceDiscovery:
    """Central class for discovering resources across the project."""
    
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.resources = {}
        self.hf_api = HfApi()
        self.discover_all()
    
    def discover_all(self) -> Dict[str, Any]:
        """Discover all available resources."""
        self.resources = {
            'models': self.discover_models(),
            'notebooks': self.discover_notebooks(),
            'videos': self.discover_videos(),
            'playlists': self.discover_playlists(),
            'modules': self.discover_modules(),
            'examples': self.discover_examples(),
            'configs': self.discover_configs()
        }
        return self.resources
    
    def discover_models(self) -> Dict[str, List[ModelInfo]]:
        """Discover available models both locally and from Hugging Face."""
        models = {
            'llm': [],
            'embedding': [],
            'audio': [],
            'video': []
        }
        
        # Discover local models
        local_models = self._discover_local_models()
        for model_type, model_list in local_models.items():
            models[model_type].extend(model_list)
        
        # Discover Hugging Face models
        hf_models = self._discover_huggingface_models()
        for model_type, model_list in hf_models.items():
            models[model_type].extend(model_list)
        
        return models

    def _discover_local_models(self) -> Dict[str, List[ModelInfo]]:
        """Discover models available locally."""
        models = {
            'llm': [],
            'embedding': [],
            'audio': [],
            'video': []
        }
        
        # Check model directories
        model_dirs = {
            'llm': ['models/llm', 'llama_models'],
            'embedding': ['models/embedding'],
            'audio': ['models/audio'],
            'video': ['models/video']
        }
        
        for model_type, dirs in model_dirs.items():
            for dir_path in dirs:
                full_path = self.base_dir / dir_path
                if full_path.exists():
                    # Look for various model formats
                    for ext in ['*.pt', '*.gguf', '*.bin', '*.safetensors']:
                        for model_path in full_path.glob(ext):
                            model_info = self._analyze_local_model(model_path)
                            if model_info:
                                models[model_type].append(model_info)
        
        return models

    def _analyze_local_model(self, model_path: Path) -> Optional[ModelInfo]:
        """Analyze a local model file to extract metadata."""
        try:
            size = model_path.stat().st_size
            name = model_path.stem
            
            # Try to determine quantization and format
            quantization = None
            format = model_path.suffix[1:]  # Remove the dot
            
            if format == 'gguf':
                # Parse quantization from filename
                parts = name.split('.')
                if len(parts) > 1:
                    quantization = parts[-1]
            
            return ModelInfo(
                name=name,
                path=model_path,
                source="local",
                size=size,
                quantization=quantization,
                format=format
            )
        except Exception as e:
            print(f"Error analyzing model {model_path}: {e}")
            return None

    def _discover_huggingface_models(self) -> Dict[str, List[ModelInfo]]:
        """Discover available models from Hugging Face."""
        models = {
            'llm': [],
            'embedding': [],
            'audio': [],
            'video': []
        }
        
        try:
            # Search for LLM models
            llm_models = self.hf_api.list_models(
                filter=ModelFilter(
                    task="text-generation",
                    library="transformers"
                ),
                limit=50
            )
            
            for model in llm_models:
                model_info = ModelInfo(
                    name=model.id,
                    source="huggingface",
                    metadata={
                        "tags": model.tags,
                        "downloads": model.downloads,
                        "likes": model.likes
                    }
                )
                models['llm'].append(model_info)
            
            # Search for embedding models
            embedding_models = self.hf_api.list_models(
                filter=ModelFilter(
                    task="sentence-similarity",
                    library="transformers"
                ),
                limit=20
            )
            
            for model in embedding_models:
                model_info = ModelInfo(
                    name=model.id,
                    source="huggingface",
                    metadata={
                        "tags": model.tags,
                        "downloads": model.downloads,
                        "likes": model.likes
                    }
                )
                models['embedding'].append(model_info)
                
        except Exception as e:
            print(f"Error discovering Hugging Face models: {e}")
        
        return models

    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get detailed information about a specific model."""
        for model_type, models in self.resources['models'].items():
            for model in models:
                if model.name == model_name:
                    return model
        return None

    def download_model(self, model_name: str, model_type: str = 'llm') -> Optional[ModelInfo]:
        """Download a model from Hugging Face."""
        try:
            # Check if model exists locally first
            local_model = self.get_model_info(model_name)
            if local_model and local_model.source == "local":
                return local_model
            
            # Download from Hugging Face
            model_path = self.base_dir / "models" / model_type / model_name
            model_path.mkdir(parents=True, exist_ok=True)
            
            # Use transformers to download
            from transformers import AutoModel, AutoTokenizer
            
            model = AutoModel.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Save locally
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            
            # Create model info
            model_info = ModelInfo(
                name=model_name,
                path=model_path,
                source="local",
                format="transformers",
                metadata={
                    "original_source": "huggingface",
                    "model_type": model_type
                }
            )
            
            # Update resources
            self.resources['models'][model_type].append(model_info)
            
            return model_info
            
        except Exception as e:
            print(f"Error downloading model {model_name}: {e}")
            return None

    def discover_notebooks(self) -> Dict[str, List[str]]:
        """Discover available Jupyter notebooks."""
        notebooks = {
            'analysis': [],
            'examples': [],
            'tutorials': []
        }
        
        # Find all .ipynb files
        for nb_path in self.base_dir.rglob('*.ipynb'):
            rel_path = nb_path.relative_to(self.base_dir)
            if 'analysis' in str(rel_path):
                notebooks['analysis'].append(str(rel_path))
            elif 'examples' in str(rel_path):
                notebooks['examples'].append(str(rel_path))
            elif 'tutorials' in str(rel_path):
                notebooks['tutorials'].append(str(rel_path))
        
        return notebooks
    
    def discover_videos(self) -> Dict[str, List[str]]:
        """Discover processed videos."""
        videos = {
            'processed': [],
            'raw': [],
            'generated': []
        }
        
        video_dirs = {
            'processed': 'output/videos',
            'raw': 'downloads/videos',
            'generated': 'output/generated'
        }
        
        for video_type, dir_path in video_dirs.items():
            full_path = self.base_dir / dir_path
            if full_path.exists():
                videos[video_type].extend([
                    f.name for f in full_path.glob('*.mp4')
                ])
        
        return videos
    
    def discover_playlists(self) -> Dict[str, List[str]]:
        """Discover processed playlists."""
        playlists = {
            'processed': [],
            'raw': []
        }
        
        playlist_dirs = {
            'processed': 'output/playlists',
            'raw': 'downloads/playlists'
        }
        
        for playlist_type, dir_path in playlist_dirs.items():
            full_path = self.base_dir / dir_path
            if full_path.exists():
                playlists[playlist_type].extend([
                    f.name for f in full_path.glob('*.json')
                ])
        
        return playlists
    
    def discover_modules(self) -> Dict[str, List[str]]:
        """Discover available Python modules and their functions."""
        modules = {}
        package_dir = self.base_dir / 'ragtime_llm'
        
        if package_dir.exists():
            for py_file in package_dir.rglob('*.py'):
                if py_file.name != '__init__.py':
                    module_path = str(py_file.relative_to(self.base_dir))[:-3].replace('/', '.')
                    try:
                        module = importlib.import_module(module_path)
                        functions = [
                            name for name, obj in inspect.getmembers(module)
                            if inspect.isfunction(obj) and obj.__module__ == module_path
                        ]
                        if functions:
                            modules[module_path] = functions
                    except ImportError:
                        continue
        
        return modules
    
    def discover_examples(self) -> Dict[str, List[str]]:
        """Discover example scripts and their usage."""
        examples = {
            'scripts': [],
            'configs': []
        }
        
        # Find example scripts
        for script in self.base_dir.glob('examples/*.py'):
            examples['scripts'].append(script.name)
        
        # Find example configs
        for config in self.base_dir.glob('examples/configs/*.json'):
            examples['configs'].append(config.name)
        
        return examples
    
    def discover_configs(self) -> Dict[str, List[str]]:
        """Discover configuration files."""
        configs = {
            'app': [],
            'model': [],
            'processing': []
        }
        
        config_dirs = {
            'app': 'config/app',
            'model': 'config/model',
            'processing': 'config/processing'
        }
        
        for config_type, dir_path in config_dirs.items():
            full_path = self.base_dir / dir_path
            if full_path.exists():
                configs[config_type].extend([
                    f.name for f in full_path.glob('*.json')
                ])
        
        return configs
    
    def get_resource(self, resource_type: str, sub_type: Optional[str] = None) -> Any:
        """Get specific resource type."""
        if resource_type in self.resources:
            if sub_type and sub_type in self.resources[resource_type]:
                return self.resources[resource_type][sub_type]
            return self.resources[resource_type]
        return None
    
    def to_json(self) -> str:
        """Convert resources to JSON string."""
        return json.dumps(self.resources, indent=2)
    
    def save_resources(self, file_path: str) -> None:
        """Save discovered resources to a JSON file."""
        with open(file_path, 'w') as f:
            json.dump(self.resources, f, indent=2)

# Create a singleton instance
discovery = ResourceDiscovery() 