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
import asyncio
from datetime import datetime
import logging
from ragtime_llm import PyResourceDiscovery, PyModelInfo

logger = logging.getLogger(__name__)

class ModelInfo:
    """Class to store model information."""
    def __init__(self, py_model_info: PyModelInfo):
        self.name = py_model_info.name
        self.path = Path(py_model_info.path)
        self.model_type = py_model_info.model_type
        self.size_bytes = py_model_info.size_bytes
        self.last_modified = datetime.fromisoformat(py_model_info.last_modified.replace('Z', '+00:00'))
        self.metadata = py_model_info.metadata

    def __str__(self) -> str:
        return f"ModelInfo(name='{self.name}', type='{self.model_type}', size={self.size_bytes})"

    def __repr__(self) -> str:
        return self.__str__()

class ResourceDiscovery:
    """Central class for discovering resources across the project."""
    
    def __init__(
        self,
        search_paths: List[str],
        model_patterns: Optional[List[str]] = None,
        cache_ttl_seconds: int = 3600,
        max_parallel_searches: int = 4
    ):
        """
        Initialize the resource discovery system.

        Args:
            search_paths: List of paths to search for models
            model_patterns: List of glob patterns to match model files
            cache_ttl_seconds: Time-to-live for the model cache in seconds
            max_parallel_searches: Maximum number of parallel directory searches
        """
        if model_patterns is None:
            model_patterns = ["*.gguf", "*.bin", "*.pt"]

        self._discovery = PyResourceDiscovery(
            search_paths=search_paths,
            model_patterns=model_patterns,
            cache_ttl_seconds=cache_ttl_seconds,
            max_parallel_searches=max_parallel_searches
        )

    async def discover_models(self) -> List[ModelInfo]:
        """
        Discover models in the configured search paths.

        Returns:
            List of discovered ModelInfo objects
        """
        try:
            py_models = await asyncio.to_thread(self._discovery.discover_models)
            return [ModelInfo(m) for m in py_models]
        except Exception as e:
            logger.error(f"Error discovering models: {e}")
            raise

    async def get_model_info(self, name: str) -> Optional[ModelInfo]:
        """
        Get information about a specific model.

        Args:
            name: Name of the model to look up

        Returns:
            ModelInfo object if found, None otherwise
        """
        try:
            py_model = await asyncio.to_thread(self._discovery.get_model_info, name)
            return ModelInfo(py_model) if py_model else None
        except Exception as e:
            logger.error(f"Error getting model info for {name}: {e}")
            raise

    async def clear_cache(self) -> None:
        """Clear the model discovery cache."""
        try:
            await asyncio.to_thread(self._discovery.clear_cache)
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            raise

    def __str__(self) -> str:
        return f"ResourceDiscovery(search_paths={self._discovery.search_paths})"

    def __repr__(self) -> str:
        return self.__str__()

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