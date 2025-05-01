"""
Video generation module for creating video responses.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import cv2
import numpy as np
from pathlib import Path
import tempfile
import subprocess
from ragtime_llm.utils.logger import logger
import os

@dataclass
class VideoSegment:
    """Represents a segment of video with metadata."""
    start_time: float
    end_time: float
    transcript: str
    video_path: str
    metadata: Dict

class VideoGenerator:
    """Class for generating videos from segments."""
    
    def __init__(self, fps: int = 30, transition_type: str = "fade", 
                 transition_duration: float = 1.0):
        """Initialize the video generator.
        
        Args:
            fps: Frames per second
            transition_type: Type of transition between segments
            transition_duration: Duration of transitions in seconds
        """
        self.fps = fps
        self.transition_type = transition_type
        self.transition_duration = transition_duration
    
    def extract_frame(self, video_path: str, timestamp: float) -> np.ndarray:
        """Extract a frame from a video at a specific timestamp.
        
        Args:
            video_path: Path to the video file
            timestamp: Timestamp in seconds
            
        Returns:
            Frame as a numpy array
        """
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Failed to extract frame at {timestamp}s from {video_path}")
        
        return frame
    
    def extract_keyframes(self, video_path: str, num_frames: int = 10) -> List[np.ndarray]:
        """Extract keyframes from a video.
        
        Args:
            video_path: Path to the video file
            num_frames: Number of keyframes to extract
            
        Returns:
            List of keyframes as numpy arrays
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        cap.release()
        return frames
    
    def create_transition(self, frame1: np.ndarray, frame2: np.ndarray, 
                         progress: float) -> np.ndarray:
        """Create a transition between two frames.
        
        Args:
            frame1: First frame
            frame2: Second frame
            progress: Transition progress (0 to 1)
            
        Returns:
            Transitioned frame
        """
        if self.transition_type == "fade":
            return cv2.addWeighted(frame1, 1 - progress, frame2, progress, 0)
        elif self.transition_type == "slide":
            h, w = frame1.shape[:2]
            shift = int(w * progress)
            result = np.zeros_like(frame1)
            result[:, :w-shift] = frame1[:, shift:]
            result[:, w-shift:] = frame2[:, :shift]
            return result
        else:
            raise ValueError(f"Unknown transition type: {self.transition_type}")
    
    def generate(self, text: str, video_path: str, output_path: str) -> str:
        """Generate a video response.
        
        Args:
            text: Text to generate video for
            video_path: Path to source video
            output_path: Path to save output video
            
        Returns:
            Path to generated video
        """
        try:
            # Extract keyframes
            keyframes = self.extract_keyframes(video_path)
            if not keyframes:
                raise ValueError("No keyframes extracted")
            
            # Create temporary directory for frames
            with tempfile.TemporaryDirectory() as temp_dir:
                # Generate frames with transitions
                frame_paths = []
                for i in range(len(keyframes) - 1):
                    # Create transition frames
                    num_transition_frames = int(self.transition_duration * self.fps)
                    for j in range(num_transition_frames):
                        progress = j / num_transition_frames
                        frame = self.create_transition(
                            keyframes[i],
                            keyframes[i + 1],
                            progress
                        )
                        frame_path = os.path.join(temp_dir, f"frame_{len(frame_paths):04d}.jpg")
                        cv2.imwrite(frame_path, frame)
                        frame_paths.append(frame_path)
                
                # Combine frames into video
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-framerate", str(self.fps),
                    "-i", os.path.join(temp_dir, "frame_%04d.jpg"),
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    output_path
                ]
                subprocess.run(cmd, check=True)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating video: {e}")
            raise 