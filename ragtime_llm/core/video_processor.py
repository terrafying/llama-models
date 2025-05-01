"""
Video processing component for handling YouTube videos and playlists.
"""

import os
from typing import List, Dict, Any, Optional
from pytube import YouTube, Playlist
from ragtime_llm.utils.logger import logger
from ragtime_llm.core.distributed_processor import DistributedProcessor

class VideoProcessor:
    """Handles video processing and metadata extraction."""
    
    def __init__(self, cache_dir: str = ".cache"):
        """Initialize the video processor.
        
        Args:
            cache_dir: Directory to store cached data
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize distributed processor
        self.distributed_processor = DistributedProcessor(cache_dir=cache_dir)
        
    def process_playlist(self, playlist_url: str) -> List[Dict[str, Any]]:
        """Process all videos in a playlist.
        
        Args:
            playlist_url: URL of the YouTube playlist
            
        Returns:
            List of processing results for each video
        """
        try:
            logger.info(f"Processing playlist: {playlist_url}")
            results = self.distributed_processor.process_playlist(playlist_url)
            
            # Filter successful results
            successful_results = [r for r in results if r["success"]]
            logger.info(f"Successfully processed {len(successful_results)} videos from playlist")
            
            return successful_results
            
        except Exception as e:
            logger.error(f"Error processing playlist {playlist_url}: {e}")
            raise
    
    def process_video(self, video_url: str) -> Dict[str, Any]:
        """Process a single video.
        
        Args:
            video_url: URL of the video to process
            
        Returns:
            Dictionary containing video metadata
        """
        try:
            logger.info(f"Processing video: {video_url}")
            results = self.distributed_processor.process_videos([video_url])
            
            if not results or not results[0]["success"]:
                raise Exception(f"Failed to process video: {results[0]['error'] if results else 'Unknown error'}")
            
            return results[0]["metadata"]
            
        except Exception as e:
            logger.error(f"Error processing video {video_url}: {e}")
            raise
    
    def cleanup(self):
        """Clean up resources."""
        self.distributed_processor.cleanup() 