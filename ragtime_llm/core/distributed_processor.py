"""
Distributed processing utilities using Ray for parallel computation.

This module provides distributed processing capabilities for:
1. Embedding generation using sentence transformers
2. Video transcription using Whisper
3. Parallel batch processing of documents and media

Key Components:
- DistributedEmbeddingWorker: Handles parallel embedding generation
- DistributedTranscriptionWorker: Manages distributed video transcription
- DistributedProcessor: Coordinates distributed operations

Usage Context:
- Used by YouTubeRAG for processing video content
- Integrates with VectorStore for distributed vector operations
- Supports both CPU and GPU acceleration

Performance Notes:
- Automatically scales based on available CPU cores
- Supports GPU acceleration when available
- Implements efficient batch processing
"""

import ray
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from sentence_transformers import SentenceTransformer
import torch
from concurrent.futures import ThreadPoolExecutor
import whisper
from moviepy.editor import VideoFileClip

logger = logging.getLogger(__name__)

@ray.remote
class DistributedEmbeddingWorker:
    """Worker for distributed embedding generation.
    
    Handles parallel generation of embeddings using sentence transformers.
    Supports both CPU and GPU acceleration.
    
    Performance Context:
    - Optimized for batch processing
    - Automatic device selection (CPU/GPU/MPS)
    - Memory-efficient processing
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the embedding worker.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model = SentenceTransformer(model_name)
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        elif torch.backends.mps.is_available():
            self.model = self.model.to("mps")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            Array of embeddings
        """
        return self.model.encode(texts, convert_to_tensor=True).cpu().numpy()

@ray.remote
class DistributedTranscriptionWorker:
    """Worker for distributed video transcription.
    
    Handles parallel transcription of video segments using Whisper.
    Implements efficient resource management and cleanup.
    
    Performance Context:
    - Processes video segments in parallel
    - Automatic cleanup of temporary files
    - Error handling and recovery
    """
    
    def __init__(self, model_size: str = "base"):
        """Initialize the transcription worker.
        
        Args:
            model_size: Size of the Whisper model to use
        """
        self.model = whisper.load_model(model_size)
    
    def transcribe_segment(self, video_path: str, start_time: float, end_time: float) -> Dict:
        """Transcribe a video segment.
        
        Args:
            video_path: Path to the video file
            start_time: Start time of the segment
            end_time: End time of the segment
            
        Returns:
            Dictionary containing transcription and metadata
        """
        try:
            # Load video segment
            video = VideoFileClip(video_path).subclip(start_time, end_time)
            temp_path = f"/tmp/segment_{start_time}_{end_time}.mp4"
            video.write_videofile(temp_path, codec='libx264')
            
            # Transcribe
            result = self.model.transcribe(temp_path)
            
            # Cleanup
            video.close()
            import os
            os.remove(temp_path)
            
            return {
                "text": result["text"],
                "start_time": start_time,
                "end_time": end_time
            }
        except Exception as e:
            logger.error(f"Error transcribing segment: {e}")
            return {
                "text": "",
                "start_time": start_time,
                "end_time": end_time,
                "error": str(e)
            }

class DistributedProcessor:
    """Main distributed processing coordinator.
    
    Manages distributed workers and coordinates parallel processing tasks.
    Implements efficient resource allocation and task distribution.
    
    System Context:
    - Integrates with Ray for distributed computing
    - Supports dynamic worker scaling
    - Implements fault tolerance and recovery
    
    Performance Context:
    - Automatic worker allocation based on system resources
    - Efficient batch processing
    - Resource cleanup and management
    """
    
    def __init__(self, num_workers: Optional[int] = None):
        """Initialize the distributed processor.
        
        Args:
            num_workers: Number of Ray workers to spawn (defaults to CPU count)
        """
        if not ray.is_initialized():
            ray.init()
        
        self.num_workers = num_workers or ray.available_resources().get("CPU", 1)
        self.embedding_workers = [
            DistributedEmbeddingWorker.remote()
            for _ in range(self.num_workers)
        ]
        self.transcription_workers = [
            DistributedTranscriptionWorker.remote()
            for _ in range(self.num_workers)
        ]
    
    def generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a batch of texts in parallel.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            Array of embeddings
        """
        # Split texts among workers
        chunk_size = len(texts) // self.num_workers + 1
        text_chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
        
        # Process chunks in parallel
        futures = [
            worker.generate_embeddings.remote(chunk)
            for worker, chunk in zip(self.embedding_workers, text_chunks)
        ]
        
        # Collect results
        results = ray.get(futures)
        return np.vstack(results)
    
    def transcribe_video_segments(self, 
                                video_path: str,
                                segments: List[Dict[str, float]]) -> List[Dict]:
        """Transcribe video segments in parallel.
        
        Args:
            video_path: Path to the video file
            segments: List of video segments to transcribe
            
        Returns:
            List of transcription results
        """
        # Distribute segments among workers
        futures = []
        for i, segment in enumerate(segments):
            worker = self.transcription_workers[i % self.num_workers]
            futures.append(
                worker.transcribe_segment.remote(
                    video_path,
                    segment["start_time"],
                    segment["end_time"]
                )
            )
        
        # Collect results
        return ray.get(futures)
    
    def cleanup(self):
        """Clean up Ray resources."""
        ray.shutdown() 