"""
Unified RAG System for YouTube Videos with GPU Acceleration and Multiple LLM Options.

This module provides a comprehensive RAG (Retrieval-Augmented Generation) system for:
1. Processing and indexing YouTube videos
2. Generating embeddings and managing vector storage
3. Querying video content using natural language
4. Generating video responses

Key Components:
- YouTubeRAG: Main RAG system implementation
- Hardware detection and model selection
- Distributed processing integration
- Web interface for user interaction

System Context:
- Integrates with multiple LLM providers
- Uses distributed processing for video handling
- Supports both local and cloud-based models
- Implements efficient resource management

Performance Context:
- Automatic hardware detection and optimization
- GPU acceleration when available
- Distributed processing for scalability
- Efficient memory management
"""

import os
import json
import numpy as np
import torch
import platform
import psutil
from pytube import YouTube, Playlist
from youtube_transcript_api import YouTubeTranscriptApi
import whisper
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional, Any, Union
import gradio as gr
import openai
import requests
import cv2
from PIL import Image
import tiktoken
from moviepy import VideoFileClip
import tempfile
import subprocess
import re
from datetime import datetime
import torch.nn.functional as F
import shutil
import gc

from ragtime_llm.video.video_generator import VideoGenerator
from ragtime_llm.video.video_composer import VideoComposer, VideoSegment
from ragtime_llm.utils.logger import logger
from ragtime_llm.core.llm_utils import LLMConfig, get_llm_provider
from ragtime_llm.core.distributed_processor import DistributedProcessor

# Default configuration
DEFAULT_EXO_ENDPOINT = "https://api.exo.ai/v1"

def detect_hardware_capabilities() -> Dict[str, Any]:
    """Detect hardware capabilities and return appropriate model configuration.
    
    Analyzes system resources and returns optimal model configuration:
    - CPU/GPU availability and memory
    - Platform-specific optimizations
    - Model size recommendations
    
    Returns:
        Dictionary containing hardware configuration
    """
    system = platform.system()
    is_mac = system == "Darwin"
    is_arm = platform.machine() == "arm64"
    
    # Get available memory
    total_memory = psutil.virtual_memory().total / (1024 * 1024 * 1024)  # Convert to GB
    
    # Check for GPU
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024 * 1024)  # Convert to GB
    else:
        gpu_memory = 0
    
    # Model selection logic
    if is_mac and is_arm:
        # Mac with Apple Silicon
        if total_memory >= 32:
            # Mac Studio or high-end Mac
            return {
                "model": "llama-3.2-7b.Q4_K_M.gguf",
                "provider": "local",
                "use_metal": True
            }
        elif total_memory >= 16:
            # Mac Mini M2/M3 or MacBook Pro
            return {
                "model": "llama-3.2-3b.Q4_K_M.gguf",
                "provider": "local",
                "use_metal": True
            }
        else:
            # Lower-end Mac
            return {
                "model": "llama-3.2-3b.Q4_K_S.gguf",
                "provider": "local",
                "use_metal": True
            }
    elif has_cuda:
        # NVIDIA GPU
        if gpu_memory >= 16:
            return {
                "model": "llama-3.2-7b.Q4_K_M.gguf",
                "provider": "local",
                "use_metal": False
            }
        elif gpu_memory >= 8:
            return {
                "model": "llama-3.2-3b.Q4_K_M.gguf",
                "provider": "local",
                "use_metal": False
            }
        else:
            return {
                "model": "llama-3.2-3b.Q4_K_S.gguf",
                "provider": "local",
                "use_metal": False
            }
    else:
        # CPU-only system
        if total_memory >= 32:
            return {
                "model": "llama-3.2-3b.Q4_K_M.gguf",
                "provider": "local",
                "use_metal": False
            }
        else:
            return {
                "model": "llama-3.2-3b.Q4_K_S.gguf",
                "provider": "local",
                "use_metal": False
            }

# Get default model based on hardware
HARDWARE_CONFIG = detect_hardware_capabilities()
DEFAULT_MODEL = HARDWARE_CONFIG["model"]

class YouTubeRAG:
    """Unified RAG system for YouTube videos with GPU acceleration.
    
    Provides a comprehensive system for:
    - Video processing and indexing
    - Natural language querying
    - Response generation
    - Video response creation
    
    System Context:
    - Integrates with multiple LLM providers
    - Uses distributed processing for video handling
    - Supports both local and cloud-based models
    - Implements efficient resource management
    
    Performance Context:
    - Automatic hardware detection and optimization
    - GPU acceleration when available
    - Distributed processing for scalability
    - Efficient memory management
    """
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 llm_config: Optional[LLMConfig] = None,
                 cache_dir: str = ".cache",
                 num_workers: Optional[int] = None):
        """Initialize the RAG system.
        
        Args:
            embedding_model: Name of the sentence transformer model to use
            llm_config: Configuration for the LLM provider
            cache_dir: Directory to store cached models and data
            num_workers: Number of Ray workers for distributed processing
        """
        try:
            self.cache_dir = cache_dir
            os.makedirs(cache_dir, exist_ok=True)
            
            # Initialize distributed processor
            self.distributed_processor = DistributedProcessor(num_workers=num_workers)
            
            # Initialize embedding model (only used for single-item processing)
            logger.info(f"Initializing embedding model: {embedding_model}")
            self.embedding_model = SentenceTransformer(embedding_model)
            if torch.cuda.is_available():
                self.embedding_model = self.embedding_model.to("cuda")
                logger.info("Using CUDA for embeddings")
            elif torch.backends.mps.is_available():
                self.embedding_model = self.embedding_model.to("mps")
                logger.info("Using MPS for embeddings")
            
            # Initialize LLM provider with hardware-appropriate model
            if llm_config is None:
                hardware_config = detect_hardware_capabilities()
                logger.info(f"Detected hardware configuration: {hardware_config}")
                llm_config = LLMConfig(
                    model_id=hardware_config["model"],
                    provider=hardware_config["provider"],
                    use_metal=hardware_config["use_metal"],
                    auto_download=True
                )
            
            logger.info(f"Initializing LLM provider: {llm_config.provider} with model {llm_config.model_id}")
            self.llm = get_llm_provider(llm_config)
            
            # Initialize video components
            self.video_generator = VideoGenerator()
            self.video_composer = VideoComposer()
            
            # Initialize storage
            self.vectors = []
            self.documents = []
            self.metadata = []
            
            logger.info("RAG system initialized successfully")
            
        except Exception as e:
            error_msg = f"Failed to initialize RAG system: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def add_video(self, video_path_or_url: str) -> None:
        """Add a video to the RAG system.
        
        Processes and indexes a video for later querying:
        - Downloads video if URL provided
        - Transcribes video content
        - Generates embeddings
        - Stores metadata
        
        Args:
            video_path_or_url: Path to local video file or YouTube URL
        """
        try:
            # Download video if URL
            if video_path_or_url.startswith(("http://", "https://")):
                yt = YouTube(video_path_or_url)
                video_id = yt.video_id
                video_path = os.path.join(self.cache_dir, f"{video_id}.mp4")
                
                if not os.path.exists(video_path):
                    logger.info(f"Downloading video: {yt.title}")
                    yt.streams.filter(progressive=True, file_extension="mp4").first().download(
                        output_path=self.cache_dir,
                        filename=f"{video_id}.mp4"
                    )
                
                metadata = {
                    "video_id": video_id,
                    "title": yt.title,
                    "author": yt.author,
                    "length": yt.length,
                    "views": yt.views,
                    "publish_date": yt.publish_date.isoformat() if yt.publish_date else None
                }
            else:
                video_path = video_path_or_url
                video_id = os.path.splitext(os.path.basename(video_path))[0]
                metadata = {
                    "video_id": video_id,
                    "title": os.path.basename(video_path),
                    "length": VideoFileClip(video_path).duration,
                    "video_path": video_path
                }
            
            # Split video into segments
            video = VideoFileClip(video_path)
            segment_duration = 60  # 1-minute segments
            segments = []
            for start in range(0, int(video.duration), segment_duration):
                end = min(start + segment_duration, video.duration)
                segments.append({
                    "start_time": start,
                    "end_time": end
                })
            video.close()
            
            # Transcribe segments in parallel
            transcriptions = self.distributed_processor.transcribe_video_segments(
                video_path,
                segments
            )
            
            # Combine transcriptions
            text = " ".join(t["text"] for t in transcriptions if t["text"])
            
            # Generate embeddings in parallel
            embeddings = self.distributed_processor.generate_embeddings_batch([text])
            
            # Store data
            self.vectors.append(embeddings[0])
            self.documents.append(text)
            self.metadata.append(metadata)
            
            logger.info(f"Successfully added video: {metadata['title']}")
            return len(text.split())  # Return number of chunks processed
            
        except Exception as e:
            error_msg = f"Error adding video {video_path_or_url}: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def add_playlist(self, playlist_url: str) -> None:
        """Add all videos from a YouTube playlist.
        
        Processes and indexes all videos in a playlist:
        - Downloads videos in parallel
        - Transcribes video content
        - Generates embeddings
        - Stores metadata
        
        Args:
            playlist_url: URL of the YouTube playlist
        """
        try:
            playlist = Playlist(playlist_url)
            for video_url in playlist.video_urls:
                self.add_video(video_url)
        except Exception as e:
            logger.error(f"Error adding playlist {playlist_url}: {e}")
            raise
    
    def generate_response(self, 
                         query: str,
                         max_tokens: int = 500,
                         temperature: float = 0.7,
                         top_p: float = 0.9,
                         stream: bool = False,
                         llm_provider: Optional[str] = None,
                         model: Optional[str] = None,
                         api_key: Optional[str] = None,
                         api_base: Optional[str] = None) -> Union[str, List[str]]:
        """Generate a response to a query using the RAG system.
        
        Process:
        1. Generate query embedding
        2. Find most similar documents
        3. Generate response using LLM
        
        Args:
            query: The query to generate a response for
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation
            top_p: Top-p sampling parameter
            stream: Whether to stream the response
            llm_provider: Optional provider to use for this request
            model: Optional model to use for this request
            api_key: Optional API key for the provider
            api_base: Optional API base URL for the provider
            
        Returns:
            Generated response text or list of response chunks if streaming
        """
        try:
            # Update LLM provider if specified
            if llm_provider or model or api_key or api_base:
                config = LLMConfig(
                    model_id=model or self.llm.config.model_id,
                    provider=llm_provider or self.llm.config.provider,
                    api_key=api_key,
                    api_base=api_base,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    use_metal=self.llm.config.use_metal,
                    auto_download=self.llm.config.auto_download
                )
                self.llm = get_llm_provider(config)
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query)
            
            # Find most similar documents
            similarities = []
            for vec in self.vectors:
                similarity = np.dot(query_embedding, vec) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(vec)
                )
                similarities.append(similarity)
            
            # Get top 3 most similar documents
            top_indices = np.argsort(similarities)[-3:][::-1]
            context = "\n\n".join([self.documents[i] for i in top_indices])
            
            # Generate prompt
            prompt = f"""Based on the following context from YouTube videos, answer the question.
            
Context:
{context}

Question: {query}

Answer:"""
            
            # Generate response
            return self.llm.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=stream
            )
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def generate_video_response(self,
                              query: str,
                              output_path: str,
                              max_tokens: int = 500,
                              temperature: float = 0.7,
                              top_p: float = 0.9) -> str:
        """Generate a video response to a query.
        
        Process:
        1. Generate text response
        2. Find relevant video segments
        3. Generate video response
        
        Args:
            query: The query to generate a response for
            output_path: Path to save the output video
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation
            top_p: Top-p sampling parameter
            
        Returns:
            Path to the generated video
        """
        try:
            # Generate text response
            response = self.generate_response(
                query=query,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            
            # Get relevant video segments
            query_embedding = self.embedding_model.encode(query)
            similarities = []
            for vec in self.vectors:
                similarity = np.dot(query_embedding, vec) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(vec)
                )
                similarities.append(similarity)
            
            top_index = np.argmax(similarities)
            video_path = self.metadata[top_index]["video_path"]
            
            # Generate video
            video = self.video_generator.generate(
                text=response,
                video_path=video_path,
                output_path=output_path
            )
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating video response: {e}")
            raise
    
    def cleanup(self):
        """Clean up resources.
        
        Performs cleanup of:
        - Distributed processor resources
        - GPU memory
        - Temporary files
        """
        self.distributed_processor.cleanup()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

def create_web_ui(rag_system: YouTubeRAG, port: int = 8080):
    """Create a web UI for the RAG system.
    
    Provides a user interface for:
    - Adding videos and playlists
    - Querying video content
    - Generating responses
    - Managing system settings
    
    Args:
        rag_system: Initialized RAG system
        port: Port to run the interface on
    """
    # ... rest of the implementation ...

def main():
    """Main function to run the RAG system."""
    # Initialize RAG system
    rag_system = YouTubeRAG()
    
    # Create web UI
    create_web_ui(rag_system, port=8081)

if __name__ == "__main__":
    main() 