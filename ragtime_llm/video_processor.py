"""
Video processing module for the RAG-LLM system.

This module provides:
1. Video downloading and processing
2. Audio extraction and transcription
3. Efficient storage management
4. Distributed processing capabilities
"""

import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import ray
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
from ragtime_llm.utils.logger import logger
from ragtime_llm.utils.storage_manager import StorageManager

@ray.remote
class VideoProcessor:
    """Processes YouTube videos with distributed capabilities."""
    
    def __init__(self, 
                 output_dir: str = "output",
                 storage_manager: Optional[StorageManager] = None):
        """Initialize the video processor.
        
        Args:
            output_dir: Base directory for output files
            storage_manager: Optional storage manager for efficient file handling
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.storage_manager = storage_manager or StorageManager()
    
    def download_video(self, video_url: str) -> Tuple[Path, Dict]:
        """Download a YouTube video.
        
        Args:
            video_url: URL of the YouTube video
            
        Returns:
            Tuple of (video path, video info)
        """
        try:
            yt = YouTube(video_url)
            video_info = {
                "title": yt.title,
                "author": yt.author,
                "length": yt.length,
                "views": yt.views,
                "description": yt.description
            }
            
            # Download video
            video = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
            if not video:
                raise ValueError("No suitable video stream found")
            
            # Create temporary directory for download
            with tempfile.TemporaryDirectory() as temp_dir:
                video_path = Path(temp_dir) / f"{yt.video_id}.mp4"
                video.download(output_path=temp_dir, filename=f"{yt.video_id}.mp4")
                
                # Store video efficiently
                storage_info = self.storage_manager.store_file(
                    video_path,
                    use_ipfs=True,
                    min_size_ipfs=50 * 1024 * 1024  # 50MB minimum for IPFS
                )
                
                logger.info(f"Video downloaded and stored: {video_info['title']}")
                return Path(storage_info["location"]), video_info
                
        except Exception as e:
            logger.error(f"Error downloading video: {e}")
            raise
    
    def get_transcript(self, video_id: str) -> str:
        """Get video transcript.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Video transcript text
        """
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            return " ".join([entry["text"] for entry in transcript])
        except Exception as e:
            logger.error(f"Error getting transcript: {e}")
            raise
    
    def process_video(self, video_url: str) -> Dict:
        """Process a YouTube video.
        
        Args:
            video_url: URL of the YouTube video
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Download video
            video_path, video_info = self.download_video(video_url)
            
            # Get transcript
            video_id = video_url.split("v=")[-1]
            transcript = self.get_transcript(video_id)
            
            # Store transcript
            transcript_path = self.output_dir / f"{video_id}_transcript.txt"
            with open(transcript_path, "w") as f:
                f.write(transcript)
            
            # Store transcript efficiently
            storage_info = self.storage_manager.store_file(
                transcript_path,
                use_ipfs=False  # Transcripts are small, use local storage
            )
            
            # Clean up original files
            if video_path.exists():
                video_path.unlink()
            if transcript_path.exists():
                transcript_path.unlink()
            
            return {
                "video_info": video_info,
                "transcript": transcript,
                "storage": {
                    "video": storage_info,
                    "transcript": storage_info
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            raise
    
    def cleanup(self):
        """Clean up resources."""
        try:
            self.storage_manager.cleanup()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            raise

# Initialize Ray
ray.init(ignore_reinit_error=True)

# Create video processor
video_processor = VideoProcessor.remote()

def process_video(video_url: str) -> Dict:
    """Process a video using the distributed processor.
    
    Args:
        video_url: URL of the YouTube video
        
    Returns:
        Dictionary with processing results
    """
    return ray.get(video_processor.process_video.remote(video_url))

def cleanup():
    """Clean up resources."""
    ray.get(video_processor.cleanup.remote())
    ray.shutdown() 