"""
Video composition module for creating video responses.
"""

import os
import json
from moviepy import ImageClip, TextClip, VideoClip
import numpy as np
import cv2
from moviepy.video.tools.subtitles import SubtitlesClip
from typing import List, Dict, Optional, Union, Tuple
import tempfile
from dataclasses import dataclass
from enum import Enum
import librosa
import soundfile as sf
from pathlib import Path
import subprocess
from ragtime_llm.utils.logger import logger

class VideoMode(Enum):
    BRAINROT = "brainrot"
    PODCAST = "podcast"
    MONOLOGUE = "monologue"
    RAP = "rap"

@dataclass
class SubtitleEntry:
    text: str
    start_time: float
    end_time: float
    srt_file_index: int

@dataclass
class VideoSegment:
    """Represents a segment of video with metadata."""
    start_time: float
    end_time: float
    transcript: str
    video_path: str
    metadata: Dict

class VideoComposer:
    """Class for composing videos from segments."""
    
    def __init__(self, fps: int = 30, transition_type: str = "fade", 
                 transition_duration: float = 1.0):
        """Initialize the video composer.
        
        Args:
            fps: Frames per second
            transition_type: Type of transition between segments
            transition_duration: Duration of transitions in seconds
        """
        self.fps = fps
        self.transition_type = transition_type
        self.transition_duration = transition_duration
    
    def compose(self, segments: List[VideoSegment], output_path: str) -> str:
        """Compose a video from segments.
        
        Args:
            segments: List of video segments
            output_path: Path to save output video
            
        Returns:
            Path to composed video
        """
        try:
            # Create temporary directory for frames
            with tempfile.TemporaryDirectory() as temp_dir:
                frame_paths = []
                
                # Process each segment
                for i, segment in enumerate(segments):
                    # Extract frames from segment
                    cap = cv2.VideoCapture(segment.video_path)
                    cap.set(cv2.CAP_PROP_POS_MSEC, segment.start_time * 1000)
                    
                    while cap.get(cv2.CAP_PROP_POS_MSEC) < segment.end_time * 1000:
                        ret, frame = cap.read()
                        if not ret:
                            break
                            
                        frame_path = os.path.join(temp_dir, f"frame_{len(frame_paths):04d}.jpg")
                        cv2.imwrite(frame_path, frame)
                        frame_paths.append(frame_path)
                    
                    cap.release()
                    
                    # Create transition if not last segment
                    if i < len(segments) - 1:
                        next_segment = segments[i + 1]
                        cap = cv2.VideoCapture(next_segment.video_path)
                        cap.set(cv2.CAP_PROP_POS_MSEC, next_segment.start_time * 1000)
                        ret, next_frame = cap.read()
                        cap.release()
                        
                        if ret:
                            # Create transition frames
                            num_transition_frames = int(self.transition_duration * self.fps)
                            for j in range(num_transition_frames):
                                progress = j / num_transition_frames
                                frame = self.create_transition(
                                    frame,
                                    next_frame,
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
            logger.error(f"Error composing video: {e}")
            raise
    
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
        
    def _load_character_assets(self) -> Dict:
        """Load character assets and metadata."""
        assets_dir = os.path.join("assets", "characters")
        if not os.path.exists(assets_dir):
            return {}
            
        assets = {}
        for char_dir in os.listdir(assets_dir):
            char_path = os.path.join(assets_dir, char_dir)
            if os.path.isdir(char_dir):
                assets[char_dir] = {
                    "images": {
                        "left": os.path.join(char_path, "left.png"),
                        "right": os.path.join(char_path, "right.png")
                    },
                    "color": "#ffffff",  # Default color
                    "voice_id": f"{char_dir}_VOICE_ID"
                }
        return assets
        
    def _get_audio_amplitude(self, audio_path: str, frame: int) -> float:
        """Get audio amplitude at a specific frame for animation."""
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=None)
            
            # Calculate frame time
            frame_time = frame / self.fps
            
            # Get amplitude at frame time
            frame_idx = int(frame_time * sr)
            if frame_idx >= len(y):
                return 0
                
            # Get amplitude in a small window around the frame
            window_size = int(0.1 * sr)  # 100ms window
            start_idx = max(0, frame_idx - window_size // 2)
            end_idx = min(len(y), frame_idx + window_size // 2)
            
            # Calculate RMS amplitude
            amplitude = np.sqrt(np.mean(y[start_idx:end_idx]**2))
            
            # Normalize and scale
            return min(1.0, amplitude * 50)  # Scale factor can be adjusted
            
        except Exception as e:
            print(f"Error calculating audio amplitude: {e}")
            return 0
            
    def _create_subtitle_clip(self, text: str, start_time: float, end_time: float,
                            position: str = "center", fontsize: int = 70) -> TextClip:
        """Create a subtitle text clip."""
        return TextClip(
            text,
            fontsize=fontsize,
            color='white',
            stroke_color='black',
            stroke_width=2,
            font='Arial-Bold',
            size=(self.width * 0.8, None),
            method='caption'
        ).set_start(start_time).set_end(end_time).set_position(position)
        
    def _create_character_clip(self, image_path: str, start_time: float, end_time: float,
                             position: str, audio_path: str) -> ImageClip:
        """Create an animated character clip."""
        # Load character image
        char_img = ImageClip(image_path)
        
        # Create animation function
        def make_frame(t):
            frame = int(t * self.fps)
            amplitude = self._get_audio_amplitude(audio_path, frame)
            
            # Calculate position with bounce effect
            x, y = self.character_positions[position]
            bounce = -amplitude * 25  # Bounce height
            
            # Apply transform
            return char_img.resize(
                width=int(self.width * 0.4)  # Character size as proportion of screen
            ).set_position(
                (int(x * self.width), int(y * self.height + bounce))
            ).get_frame(t)
            
        return ImageClip(make_frame, duration=end_time-start_time).set_start(start_time)
        
    def compose_from_transcript(self, transcript: str, voice_id: str,
                              background: Optional[str] = None,
                              effects: List[str] = None) -> Dict:
        """Create a video composition from a transcript.
        
        Args:
            transcript: The transcript text
            voice_id: Voice ID to use for speech
            background: Optional background video/image
            effects: List of effects to apply
            
        Returns:
            Dict containing composition metadata
        """
        # Parse transcript into segments
        segments = self._parse_transcript(transcript)
        
        # Create video segments
        video_segments = []
        for segment in segments:
            # Generate audio for segment
            audio_path = self._generate_audio(segment.text, voice_id)
            
            # Create video segment
            video_segment = VideoSegment(
                start_time=segment.start_time,
                end_time=segment.end_time,
                text=segment.text,
                agent_name=voice_id,
                audio_path=audio_path,
                background_path=background,
                effects=effects
            )
            video_segments.append(video_segment)
            
        return {
            "segments": video_segments,
            "duration": sum(s.end_time - s.start_time for s in video_segments),
            "mode": VideoMode.BRAINROT.value
        }
        
    def compose_from_chunks(self, chunks: List[Dict], voice_id: str,
                          background: Optional[str] = None,
                          effects: List[str] = None) -> Dict:
        """Create a video composition from RAG chunks.
        
        Args:
            chunks: List of transcript chunks with metadata
            voice_id: Voice ID to use for speech
            background: Optional background video/image
            effects: List of effects to apply
            
        Returns:
            Dict containing composition metadata
        """
        # Convert chunks to video segments
        video_segments = []
        current_time = 0
        
        for chunk in chunks:
            # Generate audio for chunk
            audio_path = self._generate_audio(chunk["text"], voice_id)
            
            # Get duration from audio file
            audio_duration = AudioFileClip(audio_path).duration
            
            # Create video segment
            video_segment = VideoSegment(
                start_time=current_time,
                end_time=current_time + audio_duration,
                text=chunk["text"],
                agent_name=voice_id,
                audio_path=audio_path,
                background_path=background,
                effects=effects
            )
            video_segments.append(video_segment)
            
            current_time += audio_duration
            
        return {
            "segments": video_segments,
            "duration": current_time,
            "mode": VideoMode.BRAINROT.value
        }
        
    def _parse_transcript(self, transcript: str) -> List[SubtitleEntry]:
        """Parse transcript text into subtitle entries."""
        # Simple parsing - split by sentences and assign timing
        sentences = transcript.split(". ")
        entries = []
        
        current_time = 0
        for i, sentence in enumerate(sentences):
            # Estimate duration based on sentence length
            duration = len(sentence.split()) * 0.3  # 0.3 seconds per word
            
            entry = SubtitleEntry(
                text=sentence,
                start_time=current_time,
                end_time=current_time + duration,
                srt_file_index=0
            )
            entries.append(entry)
            current_time += duration
            
        return entries
        
    def _generate_audio(self, text: str, voice_id: str) -> str:
        """Generate audio for text using specified voice."""
        # This would integrate with your TTS system
        # For now, return a placeholder path
        return os.path.join(self.output_dir, f"temp_audio_{hash(text)}.mp3")
        
    def save_composition(self, composition: Dict, filename: str) -> str:
        """Save composition metadata to file.
        
        Args:
            composition: Composition metadata
            filename: Output filename
            
        Returns:
            Path to saved composition file
        """
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w') as f:
            json.dump(composition, f, indent=2)
        return output_path
        
    def _apply_effects(self, clip: VideoClip, effects: List[str], intensity: float) -> VideoClip:
        """Apply visual effects to a clip based on intensity."""
        if not effects:
            return clip
            
        for effect in effects:
            if effect == "pulse":
                clip = self._apply_pulse_effect(clip, intensity)
            elif effect == "glow":
                clip = self._apply_glow_effect(clip, intensity)
            elif effect == "shake":
                clip = self._apply_shake_effect(clip, intensity)
            elif effect == "subtle_pulse":
                clip = self._apply_pulse_effect(clip, intensity * 0.5)
            elif effect == "soft_glow":
                clip = self._apply_glow_effect(clip, intensity * 0.5)
                
        return clip
        
    def _apply_pulse_effect(self, clip: VideoClip, intensity: float) -> VideoClip:
        """Apply a pulsing effect to the clip."""
        def make_frame(t):
            frame = clip.get_frame(t)
            pulse = np.sin(t * 2 * np.pi) * intensity * 0.1
            return frame * (1 + pulse)
            
        return VideoClip(make_frame, duration=clip.duration)
        
    def _apply_glow_effect(self, clip: VideoClip, intensity: float) -> VideoClip:
        """Apply a glowing effect to the clip."""
        def make_frame(t):
            frame = clip.get_frame(t)
            # Apply Gaussian blur for glow
            blurred = cv2.GaussianBlur(frame, (0, 0), intensity * 10)
            # Blend original and blurred
            return cv2.addWeighted(frame, 1, blurred, intensity, 0)
            
        return VideoClip(make_frame, duration=clip.duration)
        
    def _apply_shake_effect(self, clip: VideoClip, intensity: float) -> VideoClip:
        """Apply a shaking effect to the clip."""
        def make_frame(t):
            frame = clip.get_frame(t)
            # Calculate shake offset
            offset_x = np.sin(t * 10) * intensity * 10
            offset_y = np.cos(t * 10) * intensity * 10
            
            # Create transformation matrix
            M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
            
            # Apply transformation
            return cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
            
        return VideoClip(make_frame, duration=clip.duration)
        
    def _adjust_character_animation(self, char_clip: ImageClip, intensity: float) -> ImageClip:
        """Adjust character animation based on intensity."""
        def make_frame(t):
            frame = char_clip.get_frame(t)
            
            # Increase bounce height with intensity
            bounce_height = -intensity * 50
            
            # Add subtle rotation with intensity
            rotation = np.sin(t * 2 * np.pi) * intensity * 5
            
            # Apply transformations
            frame = cv2.warpAffine(
                frame,
                cv2.getRotationMatrix2D(
                    (frame.shape[1]/2, frame.shape[0]/2),
                    rotation,
                    1.0
                ),
                (frame.shape[1], frame.shape[0])
            )
            
            return frame
            
        return VideoClip(make_frame, duration=char_clip.duration)
        
    def render_composition(self, composition: Dict, output_path: str) -> str:
        """Render a video composition to file with enhanced psychodynamic elements.
        
        Args:
            composition: Composition metadata
            output_path: Output video path
            
        Returns:
            Path to rendered video
        """
        # Create video clips for each segment
        video_clips = []
        audio_clips = []
        
        # Get overall intensity
        intensity = composition.get("intensity", 0.5)
        
        for segment in composition["segments"]:
            # Create background clip
            if segment.background_path:
                if segment.background_path.endswith(('.mp4', '.avi', '.mov')):
                    bg_clip = VideoFileClip(segment.background_path)
                else:
                    bg_clip = ImageClip(segment.background_path)
                bg_clip = bg_clip.set_duration(segment.end_time - segment.start_time)
            else:
                # Create solid color background
                bg_clip = ColorClip(size=(self.width, self.height), color=(0, 0, 0))
                bg_clip = bg_clip.set_duration(segment.end_time - segment.start_time)
                
            # Apply effects to background
            if segment.effects:
                bg_clip = self._apply_effects(bg_clip, segment.effects, intensity)
                
            # Create subtitle clip with intensity-based styling
            subtitle_clip = self._create_subtitle_clip(
                segment.text,
                segment.start_time,
                segment.end_time,
                fontsize=int(70 * (1 + intensity * 0.2))  # Increase font size with intensity
            )
            
            # Create character clip with enhanced animation
            char_clip = self._create_character_clip(
                self.character_assets[segment.agent_name]["images"]["left"],
                segment.start_time,
                segment.end_time,
                "left",
                segment.audio_path
            )
            
            # Enhance character animation based on intensity
            char_clip = self._adjust_character_animation(char_clip, intensity)
            
            # Create audio clip
            audio_clip = AudioFileClip(segment.audio_path)
            
            # Combine clips
            segment_clip = CompositeVideoClip([
                bg_clip,
                subtitle_clip,
                char_clip
            ]).set_audio(audio_clip)
            
            video_clips.append(segment_clip)
            
        # Concatenate all clips
        final_clip = concatenate_videoclips(video_clips)
        
        # Write output file
        final_clip.write_videofile(
            output_path,
            fps=self.fps,
            codec='libx264',
            audio_codec='aac'
        )
        
        return output_path 