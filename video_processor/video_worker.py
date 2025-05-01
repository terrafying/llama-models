"""
Video processing worker for parallel video generation.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import cv2
import numpy as np
from pathlib import Path
import tempfile
import subprocess
from concurrent.futures import ThreadPoolExecutor
import json
import logging

# Setup basic logging if logger module is not available
logger = logging.getLogger("video_worker")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

@dataclass
class VideoChunk:
    """Represents a chunk of video processing work."""
    chunk_id: str
    input_path: str
    output_path: str
    start_time: float
    end_time: float
    metadata: Dict
    status: str = "pending"
    error: Optional[str] = None

class VideoWorker:
    """Handles video processing in a stateless, parallelizable way."""
    
    def __init__(self, max_workers: int = 4):
        """Initialize the video worker.
        
        Args:
            max_workers: Maximum number of parallel workers
        """
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
    def validate_paths(self, paths: List[str]) -> Tuple[bool, List[str]]:
        """Validate that all video paths exist and are accessible.
        
        Args:
            paths: List of video file paths
            
        Returns:
            Tuple of (success, list of invalid paths)
        """
        invalid_paths = []
        for path in paths:
            if not Path(path).exists():
                invalid_paths.append(path)
        return len(invalid_paths) == 0, invalid_paths
        
    def create_chunks(self, 
                     video_path: str,
                     chunk_duration: float = 30.0,
                     output_dir: str = "output") -> List[VideoChunk]:
        """Split video into processing chunks.
        
        Args:
            video_path: Path to input video
            chunk_duration: Duration of each chunk in seconds
            output_dir: Directory for output files
            
        Returns:
            List of VideoChunk objects
        """
        try:
            cap = cv2.VideoCapture(video_path)
            total_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            chunks = []
            current_time = 0.0
            
            while current_time < total_duration:
                chunk_id = f"chunk_{len(chunks)}"
                output_path = str(Path(output_dir) / f"{chunk_id}.mp4")
                
                chunk = VideoChunk(
                    chunk_id=chunk_id,
                    input_path=video_path,
                    output_path=output_path,
                    start_time=current_time,
                    end_time=min(current_time + chunk_duration, total_duration),
                    metadata={"chunk_index": len(chunks)}
                )
                chunks.append(chunk)
                current_time += chunk_duration
                
            return chunks
            
        except Exception as e:
            logger.error(f"Error creating chunks: {str(e)}", video_path=video_path)
            return []
            
    def process_chunk(self, chunk: VideoChunk) -> bool:
        """Process a single video chunk.
        
        Args:
            chunk: VideoChunk to process
            
        Returns:
            True if processing succeeds
        """
        try:
            # Create output directory
            Path(chunk.output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Extract chunk using ffmpeg
            cmd = [
                "ffmpeg", "-y",
                "-i", chunk.input_path,
                "-ss", str(chunk.start_time),
                "-t", str(chunk.end_time - chunk.start_time),
                "-c:v", "libx264",
                "-c:a", "aac",
                chunk.output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                chunk.status = "failed"
                chunk.error = result.stderr
                logger.error(f"FFmpeg error: {result.stderr}", chunk_id=chunk.chunk_id)
                return False
                
            chunk.status = "completed"
            return True
            
        except Exception as e:
            chunk.status = "failed"
            chunk.error = str(e)
            logger.error(f"Error processing chunk: {str(e)}", chunk_id=chunk.chunk_id)
            return False
            
    def process_chunks_parallel(self, chunks: List[VideoChunk]) -> Dict[str, str]:
        """Process multiple chunks in parallel.
        
        Args:
            chunks: List of VideoChunk objects to process
            
        Returns:
            Dictionary mapping chunk IDs to their status
        """
        futures = []
        for chunk in chunks:
            future = self.executor.submit(self.process_chunk, chunk)
            futures.append((chunk.chunk_id, future))
            
        results = {}
        for chunk_id, future in futures:
            try:
                success = future.result()
                results[chunk_id] = "completed" if success else "failed"
            except Exception as e:
                results[chunk_id] = "failed"
                logger.error(f"Error in chunk {chunk_id}: {str(e)}")
                
        return results
        
    def combine_chunks(self, 
                      chunks: List[VideoChunk],
                      output_path: str) -> bool:
        """Combine processed chunks into final video.
        
        Args:
            chunks: List of processed VideoChunk objects
            output_path: Path for final output video
            
        Returns:
            True if combination succeeds
        """
        try:
            # Create file list for ffmpeg
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt') as f:
                for chunk in chunks:
                    if chunk.status == "completed":
                        f.write(f"file '{chunk.output_path}'\n")
                f.flush()
                
                # Combine chunks
                cmd = [
                    "ffmpeg", "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", f.name,
                    "-c", "copy",
                    output_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.error(f"FFmpeg error combining chunks: {result.stderr}")
                    return False
                    
                return True
                
        except Exception as e:
            logger.error(f"Error combining chunks: {str(e)}")
            return False
            
    def cleanup_chunks(self, chunks: List[VideoChunk]):
        """Clean up temporary chunk files.
        
        Args:
            chunks: List of VideoChunk objects to clean up
        """
        for chunk in chunks:
            try:
                if Path(chunk.output_path).exists():
                    Path(chunk.output_path).unlink()
            except Exception as e:
                logger.error(f"Error cleaning up chunk {chunk.chunk_id}: {str(e)}")
                
    def __del__(self):
        """Cleanup executor on deletion."""
        self.executor.shutdown(wait=True) 