"""
Unified LLM engine with shared resource management and efficient inference.
"""

import os
import torch
import logging
import requests
from typing import Dict, Optional, List, Union, Any
from dataclasses import dataclass
from pathlib import Path
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_cpp import Llama
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Model download URLs
MODEL_URLS = {
    # Llama 3 models
    "llama-3.2-3b.Q4_K_S.gguf": "https://huggingface.co/TheBloke/Llama-3.2-3B-GGUF/resolve/main/llama-3.2-3b.Q4_K_S.gguf",
    "llama-3.2-3b.Q4_K_M.gguf": "https://huggingface.co/TheBloke/Llama-3.2-3B-GGUF/resolve/main/llama-3.2-3b.Q4_K_M.gguf",
    "llama-3.2-3b.Q4_0.gguf": "https://huggingface.co/TheBloke/Llama-3.2-3B-GGUF/resolve/main/llama-3.2-3b.Q4_0.gguf",
    "llama-3.2-7b.Q4_K_S.gguf": "https://huggingface.co/TheBloke/Llama-3.2-7B-GGUF/resolve/main/llama-3.2-7b.Q4_K_S.gguf",
    "llama-3.2-7b.Q4_K_M.gguf": "https://huggingface.co/TheBloke/Llama-3.2-7B-GGUF/resolve/main/llama-3.2-7b.Q4_K_M.gguf",
    "llama-3.2-7b.Q4_0.gguf": "https://huggingface.co/TheBloke/Llama-3.2-7B-GGUF/resolve/main/llama-3.2-7b.Q4_0.gguf",
    
    # Llama 2 models (kept for backward compatibility)
    "llama-2-7b-chat.Q4_K_S.gguf": "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_S.gguf",
    "llama-2-7b-chat.Q4_K_M.gguf": "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf",
    "llama-2-7b-chat.Q4_0.gguf": "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_0.gguf",
    "llama-2-13b-chat.Q4_K_S.gguf": "https://huggingface.co/TheBloke/Llama-2-13B-Chat-GGUF/resolve/main/llama-2-13b-chat.Q4_K_S.gguf",
    "llama-2-13b-chat.Q4_K_M.gguf": "https://huggingface.co/TheBloke/Llama-2-13B-Chat-GGUF/resolve/main/llama-2-13b-chat.Q4_K_M.gguf",
    "llama-2-13b-chat.Q4_0.gguf": "https://huggingface.co/TheBloke/Llama-2-13B-Chat-GGUF/resolve/main/llama-2-13b-chat.Q4_0.gguf"
}

@dataclass
class ModelConfig:
    """Configuration for model loading and inference."""
    model_id: str
    quantization: str = "q4_0"
    max_memory: Optional[Dict[int, str]] = None
    device_map: Optional[str] = None
    torch_dtype: Optional[torch.dtype] = None
    use_cache: bool = True
    trust_remote_code: bool = False
    use_metal: bool = False
    auto_download: bool = True

class GPUMemoryManager:
    """Manages GPU memory allocation and cleanup."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.allocated_memory = 0
        self.max_memory = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
    
    def allocate(self, size_bytes: int) -> bool:
        """Try to allocate GPU memory."""
        if self.device.type == "cpu":
            return True
            
        if self.allocated_memory + size_bytes > self.max_memory:
            return False
            
        self.allocated_memory += size_bytes
        return True
    
    def free(self, size_bytes: int):
        """Free allocated GPU memory."""
        self.allocated_memory = max(0, self.allocated_memory - size_bytes)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    def get_available_memory(self) -> int:
        """Get available GPU memory in bytes."""
        if self.device.type == "cpu":
            return float("inf")
        return self.max_memory - self.allocated_memory

class ModelCache:
    """Manages model and tokenizer caching."""
    
    def __init__(self, max_size: int = 2):
        self.max_size = max_size
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.lru_order: List[str] = []
    
    def get_model(self, model_id: str) -> Optional[Any]:
        """Get cached model if available."""
        if model_id in self.models:
            # Update LRU order
            self.lru_order.remove(model_id)
            self.lru_order.append(model_id)
            return self.models[model_id]
        return None
    
    def get_tokenizer(self, model_id: str) -> Optional[Any]:
        """Get cached tokenizer if available."""
        if model_id in self.tokenizers:
            return self.tokenizers[model_id]
        return None
    
    def add_model(self, model_id: str, model: Any):
        """Add model to cache with LRU eviction."""
        if len(self.models) >= self.max_size:
            # Evict least recently used model
            evict_id = self.lru_order.pop(0)
            del self.models[evict_id]
            if evict_id in self.tokenizers:
                del self.tokenizers[evict_id]
        
        self.models[model_id] = model
        self.lru_order.append(model_id)
    
    def add_tokenizer(self, model_id: str, tokenizer: Any):
        """Add tokenizer to cache."""
        self.tokenizers[model_id] = tokenizer

class UnifiedLLMEngine:
    """Unified LLM engine with shared resource management."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the LLM engine.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory_manager = GPUMemoryManager()
        self.model_cache = ModelCache(max_size=self.config.get("max_cached_models", 2))
        self.default_model_path = Path(self.config.get("model_path", "models"))
        self.default_model_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized UnifiedLLMEngine on {self.device}")
    
    def _load_model(self, config: ModelConfig) -> Any:
        """Load a model based on its type."""
        try:
            if config.model_id.startswith("llama-"):
                return self._load_llama_model(config)
            else:
                return self._load_hf_model(config)
        except Exception as e:
            logger.error(f"Error loading model {config.model_id}: {e}")
            raise
    
    def _load_hf_model(self, config: ModelConfig) -> Any:
        """Load a HuggingFace model."""
        try:
            model = AutoModelForCausalLM.from_pretrained(
                config.model_id,
                device_map=config.device_map,
                torch_dtype=config.torch_dtype,
                trust_remote_code=config.trust_remote_code
            )
            return model
        except Exception as e:
            logger.error(f"Error loading HuggingFace model {config.model_id}: {e}")
            raise
    
    def _load_llama_model(self, config: ModelConfig) -> Any:
        """Load a llama.cpp model."""
        try:
            model_path = self._get_model_path(config.model_id)
            if not model_path.exists():
                if config.auto_download:
                    model_path = self._download_model(config.model_id)
                else:
                    raise FileNotFoundError(f"Model not found at {model_path}")
            
            model = Llama(
                model_path=str(model_path),
                n_ctx=4096,
                n_gpu_layers=-1 if self.device.type == "cuda" else 0,
                n_batch=512,
                use_mlock=True,
                use_mmap=True,
                use_metal=config.use_metal
            )
            return model
        except Exception as e:
            logger.error(f"Error loading llama.cpp model {config.model_id}: {e}")
            raise
    
    def _get_model_path(self, model_id: str) -> Path:
        """Get the path for a model."""
        return self.default_model_path / model_id
    
    def _download_model(self, model_id: str) -> Path:
        """Download a model from HuggingFace."""
        model_path = self._get_model_path(model_id)
        if model_path.exists():
            logger.info(f"Model {model_id} already exists at {model_path}")
            return model_path
        
        # Use HuggingFace's built-in download
        try:
            AutoModelForCausalLM.from_pretrained(
                model_id,
                cache_dir=str(self.default_model_path),
                local_files_only=False
            )
            return model_path
        except Exception as e:
            logger.error(f"Error downloading model {model_id}: {e}")
            raise
    
    def generate(self, 
                prompt: str, 
                model_id: str,
                max_tokens: int = 500,
                temperature: float = 0.7,
                top_p: float = 0.9,
                stop: Optional[List[str]] = None,
                stream: bool = False) -> Union[str, List[str]]:
        """Generate text using the specified model.
        
        Args:
            prompt: Input prompt
            model_id: Model identifier
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stop: List of stop sequences
            stream: Whether to stream the response
            
        Returns:
            Generated text or list of text chunks if streaming
        """
        try:
            config = ModelConfig(model_id=model_id)
            model = self._load_model(config)
            
            if isinstance(model, Llama):
                # llama.cpp generation
                response = model(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop,
                    stream=stream
                )
                
                if stream:
                    return [chunk["text"] for chunk in response]
                return response["choices"][0]["text"]
            else:
                # HuggingFace generation
                inputs = model.tokenizer(prompt, return_tensors="pt").to(self.device)
                
                if stream:
                    streamer = TextStreamer(model.tokenizer)
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=True,
                        pad_token_id=model.tokenizer.eos_token_id,
                        streamer=streamer
                    )
                    return streamer.get_generated_text()
                else:
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=True,
                        pad_token_id=model.tokenizer.eos_token_id
                    )
                    return model.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise
    
    def cleanup(self):
        """Clean up resources."""
        self.model_cache = ModelCache()  # Clear cache
        self.memory_manager.free(self.memory_manager.allocated_memory)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

class TextStreamer:
    """Helper class for streaming text generation."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.generated_text = []
    
    def __call__(self, text: str):
        self.generated_text.append(text)
    
    def get_generated_text(self) -> List[str]:
        return self.generated_text 