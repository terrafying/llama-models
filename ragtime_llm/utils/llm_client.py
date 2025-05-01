"""
LLM client interface for error analysis and other tasks.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import json

class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def analyze(self, prompt: str) -> str:
        """Analyze a prompt and return a response."""
        pass

class OpenAIClient(LLMClient):
    """OpenAI API client for error analysis."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """Initialize the OpenAI client.
        
        Args:
            api_key: OpenAI API key. If None, will try to get from environment.
            model: Model to use for analysis.
        """
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key)
            self.model = model
        except ImportError:
            raise ImportError("Please install openai package: pip install openai")

    def analyze(self, prompt: str) -> str:
        """Analyze a prompt using OpenAI API.
        
        Args:
            prompt: The prompt to analyze.
            
        Returns:
            JSON string containing the analysis.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert Python developer and test engineer. Analyze the error and provide a structured response in JSON format with the following fields: analysis (string), priority (high/medium/low), suggested_fixes (list of strings), prevention_strategies (list of strings)."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
        except Exception as e:
            return json.dumps({
                "analysis": f"Error analyzing with OpenAI: {str(e)}",
                "priority": "medium",
                "suggested_fixes": ["Check API key and internet connection"],
                "prevention_strategies": ["Implement fallback analysis"]
            })

class AnthropicClient(LLMClient):
    """Anthropic Claude API client for error analysis."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-sonnet-20240229"):
        """Initialize the Anthropic client.
        
        Args:
            api_key: Anthropic API key. If None, will try to get from environment.
            model: Model to use for analysis.
        """
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
            self.model = model
        except ImportError:
            raise ImportError("Please install anthropic package: pip install anthropic")

    def analyze(self, prompt: str) -> str:
        """Analyze a prompt using Anthropic API.
        
        Args:
            prompt: The prompt to analyze.
            
        Returns:
            JSON string containing the analysis.
        """
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=0.3,
                system="You are an expert Python developer and test engineer. Analyze the error and provide a structured response in JSON format with the following fields: analysis (string), priority (high/medium/low), suggested_fixes (list of strings), prevention_strategies (list of strings).",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            return json.dumps({
                "analysis": f"Error analyzing with Anthropic: {str(e)}",
                "priority": "medium",
                "suggested_fixes": ["Check API key and internet connection"],
                "prevention_strategies": ["Implement fallback analysis"]
            })

def get_llm_client(provider: str = "openai", **kwargs) -> LLMClient:
    """Get an LLM client instance.
    
    Args:
        provider: LLM provider to use ("openai" or "anthropic")
        **kwargs: Additional arguments to pass to the client constructor
        
    Returns:
        LLMClient instance
    """
    if provider.lower() == "openai":
        return OpenAIClient(**kwargs)
    elif provider.lower() == "anthropic":
        return AnthropicClient(**kwargs)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}") 