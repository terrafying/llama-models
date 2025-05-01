"""
LLM utilities for text generation and inference.
"""

import os
import logging
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_cpp import Llama
from ragtime_llm.core.llm_engine import UnifiedLLMEngine, ModelConfig

logger = logging.getLogger(__name__)

@dataclass
class LLMConfig:
    """Configuration for LLM initialization."""
    model_id: str
    provider: str = "local"  # local, openai, or exo
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    max_tokens: int = 500
    temperature: float = 0.7
    top_p: float = 0.9
    stop: Optional[List[str]] = None
    stream: bool = False
    use_metal: bool = False  # Enable Metal acceleration for Apple Silicon
    auto_download: bool = True  # Automatically download models if not found
    model_path: Optional[str] = None  # Custom model path

class LLMProvider:
    """Base class for LLM providers."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.engine = UnifiedLLMEngine({"model_path": config.model_path})
    
    def generate(self, 
                prompt: str,
                max_tokens: Optional[int] = None,
                temperature: Optional[float] = None,
                top_p: Optional[float] = None,
                stop: Optional[List[str]] = None,
                stream: bool = False) -> Union[str, List[str]]:
        """Generate text using the LLM.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate (uses config if None)
            temperature: Sampling temperature (uses config if None)
            top_p: Top-p sampling parameter (uses config if None)
            stop: List of stop sequences
            stream: Whether to stream the response
            
        Returns:
            Generated text or list of text chunks if streaming
        """
        return self.engine.generate(
            prompt=prompt,
            model_id=self.config.model_id,
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=temperature or self.config.temperature,
            top_p=top_p or self.config.top_p,
            stop=stop or self.config.stop,
            stream=stream or self.config.stream
        )

class LocalLLM(LLMProvider):
    """Local LLM using llama.cpp or HuggingFace."""
    pass

class OpenAILLM(LLMProvider):
    """OpenAI API LLM provider."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        if not config.api_key:
            raise ValueError("OpenAI API key is required")
        os.environ["OPENAI_API_KEY"] = config.api_key
        if config.api_base:
            os.environ["OPENAI_API_BASE"] = config.api_base

class ExoLLM(LLMProvider):
    """Exo Labs LLM provider."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        if not config.api_key:
            raise ValueError("Exo Labs API key is required")
        if not config.api_base:
            raise ValueError("Exo Labs API base URL is required")

def get_llm_provider(config: LLMConfig) -> LLMProvider:
    """Get the appropriate LLM provider based on configuration."""
    providers = {
        "local": LocalLLM,
        "openai": OpenAILLM,
        "exo": ExoLLM
    }
    
    provider_class = providers.get(config.provider.lower())
    if not provider_class:
        raise ValueError(f"Unsupported LLM provider: {config.provider}")
    
    return provider_class(config) 