"""
Phonetic indexing module for video segments.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import librosa
import cv2
from pathlib import Path
import tempfile
import subprocess
from ragtime_llm.utils.logger import logger

@dataclass
class PhoneticSegment:
    """Represents a segment of phonetic data."""
    start_time: float
    end_time: float
    text: str
    phonemes: List[str]
    video_path: str
    metadata: Dict

@dataclass
class PhoneticIndex:
    """Index for phonetic data in video segments."""
    video_path: str
    segments: List[Dict]
    mfcc_features: np.ndarray
    sample_rate: int
    
    @classmethod
    def from_video(cls, video_path: str, sample_rate: int = 16000) -> 'PhoneticIndex':
        """Create a phonetic index from a video file.
        
        Args:
            video_path: Path to video file
            sample_rate: Audio sample rate
            
        Returns:
            PhoneticIndex instance
        """
        try:
            # Extract audio from video
            with tempfile.NamedTemporaryFile(suffix='.wav') as temp_audio:
                cmd = [
                    "ffmpeg",
                    "-i", video_path,
                    "-vn",
                    "-acodec", "pcm_s16le",
                    "-ar", str(sample_rate),
                    "-ac", "1",
                    temp_audio.name
                ]
                subprocess.run(cmd, check=True)
                
                # Load audio
                audio, sr = librosa.load(temp_audio.name, sr=sample_rate)
                
                # Extract MFCC features
                mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
                
                # Segment audio based on silence
                intervals = librosa.effects.split(audio, top_db=20)
                
                # Create segments
                segments = []
                for start, end in intervals:
                    start_time = librosa.samples_to_time(start, sr=sr)
                    end_time = librosa.samples_to_time(end, sr=sr)
                    
                    # Extract segment MFCC features
                    segment_mfcc = mfcc[:, start:end]
                    
                    segments.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'mfcc_features': segment_mfcc,
                        'duration': end_time - start_time
                    })
                
                return cls(
                    video_path=video_path,
                    segments=segments,
                    mfcc_features=mfcc,
                    sample_rate=sr
                )
                
        except Exception as e:
            logger.error(f"Error creating phonetic index: {e}")
            raise
    
    def find_similar_segments(self, query_mfcc: np.ndarray, 
                            top_k: int = 5) -> List[Dict]:
        """Find segments similar to query MFCC features.
        
        Args:
            query_mfcc: Query MFCC features
            top_k: Number of results to return
            
        Returns:
            List of similar segments
        """
        try:
            similarities = []
            
            for segment in self.segments:
                # Compute cosine similarity
                segment_mfcc = segment['mfcc_features']
                similarity = np.mean([
                    np.dot(q, s) / (np.linalg.norm(q) * np.linalg.norm(s))
                    for q, s in zip(query_mfcc.T, segment_mfcc.T)
                ])
                
                similarities.append((similarity, segment))
            
            # Sort by similarity
            similarities.sort(reverse=True)
            
            return [segment for _, segment in similarities[:top_k]]
            
        except Exception as e:
            logger.error(f"Error finding similar segments: {e}")
            raise
    
    def get_segment_frames(self, segment: Dict) -> List[np.ndarray]:
        """Get frames for a segment.
        
        Args:
            segment: Segment dictionary
            
        Returns:
            List of frames
        """
        try:
            frames = []
            cap = cv2.VideoCapture(self.video_path)
            
            # Set start time
            cap.set(cv2.CAP_PROP_POS_MSEC, segment['start_time'] * 1000)
            
            while cap.get(cv2.CAP_PROP_POS_MSEC) < segment['end_time'] * 1000:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            
            cap.release()
            return frames
            
        except Exception as e:
            logger.error(f"Error getting segment frames: {e}")
            raise 