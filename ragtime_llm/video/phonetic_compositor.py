"""
Phonetic Analysis and Syllable Composition System
Focuses on extracting and compositing specific phonemes/syllables from video content
"""

import numpy as np
import torch
import torchaudio
from moviepy import VideoFileClip, concatenate_videoclips
import librosa
import python_speech_features
from pydub import AudioSegment
import tempfile
import os
from typing import List, Tuple, Optional, Dict
import json
import whisper
from datetime import timedelta
from ragtime_llm.video.phonetic_index import PhoneticIndex, PhoneticSegment
from dataclasses import dataclass
import cv2
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

class PhoneticAnalyzer:
    """Analyzes audio for specific phonemes and syllables."""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.sample_rate = 16000
        self.frame_length = 0.025  # 25ms frames
        self.frame_stride = 0.010  # 10ms stride
        self.min_chunk_duration = 0.5  # Minimum chunk duration in seconds
        self.max_chunk_duration = 10.0  # Maximum chunk duration in seconds
        self.silence_threshold = -20  # dB threshold for silence detection
        
        # Initialize Whisper model
        print("Loading Whisper model...")
        try:
            self.whisper_model = whisper.load_model("base")
            print("Whisper model loaded successfully")
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            self.whisper_model = None
        
        # Initialize phonetic index
        self.phonetic_index = PhoneticIndex(time_resolution=0.01)
        
    def extract_mfcc_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract MFCC features from audio."""
        print(f"Extracting MFCC features from audio of shape {audio.shape}")
        try:
            mfcc = python_speech_features.mfcc(
                audio,
                samplerate=self.sample_rate,
                winlen=self.frame_length,
                winstep=self.frame_stride,
                numcep=13,
                nfilt=26,
                nfft=512
            )
            print(f"Generated MFCC features of shape {mfcc.shape}")
            return mfcc
        except Exception as e:
            print(f"Error extracting MFCC features: {e}")
            return np.array([])
    
    def get_whisper_transcript(self, audio: np.ndarray) -> List[Dict]:
        """Get timestamped transcript using Whisper."""
        print(f"Getting transcript for audio of shape {audio.shape}")
        
        if self.whisper_model is None:
            print("Whisper model not loaded, skipping transcription")
            return []
        
        try:
            # Convert audio to the format Whisper expects
            audio = audio.astype(np.float32)
            
            # Normalize audio
            audio = audio / np.max(np.abs(audio))
            
            # Pad or trim to 30 seconds
            audio = whisper.pad_or_trim(audio)
            
            # Make log-Mel spectrogram
            mel = whisper.log_mel_spectrogram(audio).to(self.device)
            
            # Decode the audio
            result = self.whisper_model.decode(mel, whisper.DecodingOptions())
            
            # Get timestamped segments
            print("Transcribing audio...")
            result = self.whisper_model.transcribe(audio)
            segments = result["segments"]
            
            print(f"Got {len(segments)} transcript segments")
            
            # Debug: Print first few segments
            print("\nFirst few transcript segments:")
            for seg in segments[:3]:
                print(f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['text']}")
            
            return segments
        except Exception as e:
            print(f"Error getting transcript: {e}")
            return []
    
    def extract_meaningful_chunks(self, audio: np.ndarray, transcript_segments: List[Dict]) -> List[Tuple[float, float]]:
        """Extract meaningful chunks from audio based on content and silence.
        
        Args:
            audio: Audio data
            transcript_segments: List of transcript segments
            
        Returns:
            List of (start_time, end_time) tuples for meaningful chunks
        """
        # First, detect silence regions
        intervals = librosa.effects.split(audio, top_db=self.silence_threshold)
        
        # Convert intervals to time
        silence_regions = [(start/self.sample_rate, end/self.sample_rate) 
                          for start, end in intervals]
        
        # Group transcript segments into meaningful chunks
        chunks = []
        current_chunk = None
        
        for segment in transcript_segments:
            start_time = segment["start"]
            end_time = segment["end"]
            text = segment["text"].strip()
            
            # Skip empty segments
            if not text:
                continue
            
            # Check if this segment should start a new chunk
            if current_chunk is None:
                current_chunk = [start_time, end_time, text]
            else:
                # Check if we should merge with current chunk
                if (end_time - current_chunk[0] <= self.max_chunk_duration and
                    not self._contains_significant_silence(silence_regions, 
                                                         current_chunk[1], 
                                                         start_time)):
                    # Merge with current chunk
                    current_chunk[1] = end_time
                    current_chunk[2] += " " + text
                else:
                    # Finalize current chunk if it meets minimum duration
                    if current_chunk[1] - current_chunk[0] >= self.min_chunk_duration:
                        chunks.append((current_chunk[0], current_chunk[1]))
                    
                    # Start new chunk
                    current_chunk = [start_time, end_time, text]
        
        # Add final chunk if it meets minimum duration
        if current_chunk and current_chunk[1] - current_chunk[0] >= self.min_chunk_duration:
            chunks.append((current_chunk[0], current_chunk[1]))
        
        # Merge overlapping chunks
        chunks = self._merge_overlapping_chunks(chunks)
        
        # Ensure chunks don't exceed maximum duration
        final_chunks = []
        for start, end in chunks:
            if end - start > self.max_chunk_duration:
                # Split into smaller chunks at silence points
                sub_chunks = self._split_at_silence(start, end, silence_regions)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append((start, end))
        
        return final_chunks
    
    def _contains_significant_silence(self, silence_regions: List[Tuple[float, float]], 
                                    start_time: float, end_time: float) -> bool:
        """Check if there is significant silence between two time points."""
        for silence_start, silence_end in silence_regions:
            if (silence_start <= start_time <= silence_end or
                silence_start <= end_time <= silence_end or
                (start_time <= silence_start and end_time >= silence_end)):
                silence_duration = min(silence_end, end_time) - max(silence_start, start_time)
                if silence_duration > 0.3:  # 300ms threshold for significant silence
                    return True
        return False
    
    def _merge_overlapping_chunks(self, chunks: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Merge overlapping chunks."""
        if not chunks:
            return []
        
        # Sort chunks by start time
        chunks.sort(key=lambda x: x[0])
        
        merged = []
        current_start, current_end = chunks[0]
        
        for start, end in chunks[1:]:
            if start <= current_end:
                # Overlapping chunks, merge them
                current_end = max(current_end, end)
            else:
                # Non-overlapping chunk, add current and start new
                merged.append((current_start, current_end))
                current_start, current_end = start, end
        
        merged.append((current_start, current_end))
        return merged
    
    def _split_at_silence(self, start_time: float, end_time: float,
                         silence_regions: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Split a chunk at silence points to ensure maximum duration."""
        chunks = []
        current_start = start_time
        
        for silence_start, silence_end in silence_regions:
            if silence_start > current_start and silence_end < end_time:
                # Check if we need to split here
                if silence_end - current_start > self.max_chunk_duration:
                    # Find the best split point within the silence region
                    split_point = silence_start + (silence_end - silence_start) / 2
                    chunks.append((current_start, split_point))
                    current_start = split_point
        
        # Add the final chunk
        if end_time - current_start >= self.min_chunk_duration:
            chunks.append((current_start, end_time))
        
        return chunks

    def analyze_video(self, video_path: str) -> Dict:
        """Analyze video and extract meaningful chunks.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Load video
            video = VideoFileClip(video_path)
            
            # Extract audio
            audio = video.audio.to_soundarray(fps=self.sample_rate)
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)  # Convert to mono
            
            # Get transcript
            transcript_segments = self.get_whisper_transcript(audio)
            
            # Extract meaningful chunks
            chunks = self.extract_meaningful_chunks(audio, transcript_segments)
            
            # Extract MFCC features
            mfcc_features = self.extract_mfcc_features(audio)
            
            # Analyze each chunk
            chunk_analysis = []
            for start, end in chunks:
                # Get transcript for this chunk
                chunk_text = " ".join(seg["text"] for seg in transcript_segments 
                                    if seg["start"] >= start and seg["end"] <= end)
                
                # Get MFCC features for this chunk
                start_frame = int(start / self.frame_stride)
                end_frame = int(end / self.frame_stride)
                chunk_mfcc = mfcc_features[start_frame:end_frame]
                
                # Create phonetic segment
                segment = PhoneticSegment(
                    start_time=start,
                    end_time=end,
                    text=chunk_text,
                    phonemes=[],  # Will be filled by phonetic analysis
                    video_path=video_path,
                    metadata={
                        'mfcc_stats': {
                            'mean': float(np.mean(chunk_mfcc)),
                            'std': float(np.std(chunk_mfcc)),
                            'max': float(np.max(chunk_mfcc)),
                            'min': float(np.min(chunk_mfcc))
                        },
                        'duration': end - start,
                        'frame_count': len(chunk_mfcc)
                    }
                )
                
                chunk_analysis.append(segment)
            
            video.close()
            
            return {
                'chunks': chunk_analysis,
                'total_chunks': len(chunks),
                'total_duration': sum(end - start for start, end in chunks),
                'average_chunk_duration': np.mean([end - start for start, end in chunks]),
                'transcript_segments': transcript_segments
            }
            
        except Exception as e:
            logger.error(f"Error analyzing video: {e}")
            raise
    
    def detect_vowel_e(self, mfcc_features: np.ndarray, transcript_segments: List[Dict]) -> List[Tuple[float, float]]:
        """
        Detect segments containing 'e' vowel sound using both MFCC patterns and transcript.
        Returns list of (start_time, end_time) tuples.
        """
        segments = []
        
        # First pass: Use transcript to find potential segments
        print("\nAnalyzing transcript for 'e' vowels...")
        for segment in transcript_segments:
            text = segment["text"].lower()
            start_time = segment["start"]
            end_time = segment["end"]
            
            # Check if segment contains 'e' vowel
            if any(c in text for c in ['e', 'ee', 'ea', 'ie', 'ei']):
                print(f"Found potential 'e' in segment: {text}")
                # Get corresponding MFCC frames
                start_frame = int(start_time / self.frame_stride)
                end_frame = int(end_time / self.frame_stride)
                
                # Refine segment using MFCC features
                refined_segments = self._refine_segment_with_mfcc(
                    mfcc_features[start_frame:end_frame],
                    start_time,
                    end_time
                )
                if refined_segments:
                    print(f"Refined into {len(refined_segments)} segments")
                segments.extend(refined_segments)
        
        # Second pass: Look for 'e' sounds that might have been missed in transcript
        print("\nPerforming audio-only detection...")
        audio_segments = self._detect_e_from_audio(mfcc_features)
        if audio_segments:
            print(f"Found {len(audio_segments)} additional segments from audio")
        
        # Merge overlapping segments
        all_segments = segments + audio_segments
        merged_segments = self._merge_close_segments(all_segments)
        print(f"\nTotal segments after merging: {len(merged_segments)}")
        
        # Add segments to phonetic index with analysis
        for start, end in merged_segments:
            # Get MFCC features for this segment
            start_frame = int(start / self.frame_stride)
            end_frame = int(end / self.frame_stride)
            segment_mfcc = mfcc_features[start_frame:end_frame]
            
            # Create phonetic segment
            phonetic_segment = PhoneticSegment(
                start_time=start,
                end_time=end,
                text='e',
                phonemes=['e'],
                video_path='',
                metadata={
                    'mfcc_stats': {
                        'mean': float(np.mean(segment_mfcc)),
                        'std': float(np.std(segment_mfcc)),
                        'max': float(np.max(segment_mfcc)),
                        'min': float(np.min(segment_mfcc))
                    },
                    'duration': end - start,
                    'frame_count': len(segment_mfcc)
                }
            )
            
            # Add to index
            segment_idx = self.phonetic_index.add_segment(phonetic_segment)
            
            # Add analysis results
            analysis_results = {
                'mfcc_stats': {
                    'mean': float(np.mean(segment_mfcc)),
                    'std': float(np.std(segment_mfcc)),
                    'max': float(np.max(segment_mfcc)),
                    'min': float(np.min(segment_mfcc))
                },
                'duration': end - start,
                'frame_count': len(segment_mfcc)
            }
            
            self.phonetic_index.add_analysis(segment_idx, 'basic', analysis_results)
        
        return merged_segments
    
    def _refine_segment_with_mfcc(self, mfcc_segment: np.ndarray, 
                                start_time: float, end_time: float) -> List[Tuple[float, float]]:
        """Refine transcript-based segment using MFCC features."""
        segments = []
        frame_duration = self.frame_stride
        
        # Find sub-segments with strong 'e' vowel characteristics
        current_start = None
        for i in range(len(mfcc_segment)):
            if self._is_e_vowel_pattern(mfcc_segment[i]):
                if current_start is None:
                    current_start = start_time + i * frame_duration
            elif current_start is not None:
                end = start_time + i * frame_duration
                segments.append((current_start, end))
                current_start = None
        
        # Handle case where segment ends with 'e' sound
        if current_start is not None:
            segments.append((current_start, end_time))
        
        return segments
    
    def _detect_e_from_audio(self, mfcc_features: np.ndarray) -> List[Tuple[float, float]]:
        """Detect 'e' sounds purely from audio features."""
        segments = []
        frame_duration = self.frame_stride
        
        # Debug: Print MFCC statistics
        print(f"Analyzing {len(mfcc_features)} MFCC frames")
        
        # Lower the threshold for audio-only detection
        threshold = 0.5  # More lenient threshold for audio detection
        
        for i in range(len(mfcc_features)):
            if self._is_e_vowel_pattern(mfcc_features[i], threshold):
                start_time = i * frame_duration
                # Look for end of vowel
                j = i + 1
                while j < len(mfcc_features) and self._is_e_vowel_pattern(mfcc_features[j], threshold):
                    j += 1
                end_time = j * frame_duration
                segments.append((start_time, end_time))
                i = j
        
        return segments
    
    def _is_e_vowel_pattern(self, mfcc_frame: np.ndarray, threshold: float = 0.7) -> bool:
        """
        Check if MFCC frame matches pattern typical of 'e' vowel.
        This is a simplified detection - would need training data for better accuracy.
        """
        # Focus on coefficients that typically represent 'e' vowel formants
        e_pattern = np.array([1.2, -0.8, 0.5, -0.3, 0.2, -0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        correlation = np.corrcoef(mfcc_frame[:len(e_pattern)], e_pattern)[0,1]
        return correlation > threshold
    
    def _merge_close_segments(self, segments: List[Tuple[float, float]], 
                            max_gap: float = 0.05) -> List[Tuple[float, float]]:
        """Merge segments that are very close together."""
        if not segments:
            return []
            
        # Sort segments by start time
        segments.sort(key=lambda x: x[0])
        
        merged = []
        current_start, current_end = segments[0]
        
        for start, end in segments[1:]:
            if start - current_end <= max_gap:
                current_end = max(current_end, end)
            else:
                merged.append((current_start, current_end))
                current_start, current_end = start, end
        
        merged.append((current_start, current_end))
        return merged
    
    def get_phoneme_density(self, phoneme: str = 'e', window_size: float = 1.0) -> List[Tuple[float, float]]:
        """Get density of a phoneme over time."""
        return self.phonetic_index.get_phoneme_density(phoneme, window_size)
    
    def get_phoneme_transitions(self, phoneme: str = 'e', window_size: float = 0.1) -> List[Tuple[str, float]]:
        """Get transitions from a phoneme to other phonemes."""
        return self.phonetic_index.get_phoneme_transitions(phoneme, window_size)
    
    def get_phoneme_sequence(self, start_time: float, end_time: float) -> List[Tuple[str, float, float]]:
        """Get sequence of phonemes within a time range."""
        return self.phonetic_index.get_phoneme_sequence(start_time, end_time)
    
    def get_segments_with_analysis(self, analysis_type: str) -> List[Tuple[PhoneticSegment, Dict]]:
        """Get all segments with a specific type of analysis."""
        return self.phonetic_index.get_segments_with_analysis(analysis_type)
    
    def save_index(self, filepath: str):
        """Save the phonetic index to disk."""
        self.phonetic_index.save(filepath)
    
    def load_index(self, filepath: str):
        """Load the phonetic index from disk."""
        self.phonetic_index.load(filepath)

class SyllableCompositor:
    """Composites video segments containing specific syllables."""
    
    def __init__(self):
        self.analyzer = PhoneticAnalyzer()
        
    def extract_syllable_from_video(self, video_path: str, syllable: str = "e",
                                  output_path: Optional[str] = None,
                                  min_chunk_duration: float = 0.5,
                                  max_chunk_duration: float = 10.0) -> str:
        """
        Extract all instances of a specific syllable from a video and create a composite.
        Returns path to the composite video.
        
        Args:
            video_path: Path to input video
            syllable: Syllable to extract
            output_path: Path to save output video
            min_chunk_duration: Minimum duration for chunks in seconds
            max_chunk_duration: Maximum duration for chunks in seconds
        """
        # Update chunk duration settings
        self.analyzer.min_chunk_duration = min_chunk_duration
        self.analyzer.max_chunk_duration = max_chunk_duration
        
        # Analyze video and get meaningful chunks
        print(f"\nAnalyzing video: {video_path}")
        analysis = self.analyzer.analyze_video(video_path)
        
        if not analysis['chunks']:
            print("No meaningful chunks found in video")
            return ""
        
        print(f"\nFound {analysis['total_chunks']} meaningful chunks")
        print(f"Average chunk duration: {analysis['average_chunk_duration']:.2f} seconds")
        
        # Extract video clips for each chunk
        print("\nExtracting video clips...")
        clips = []
        video = VideoFileClip(video_path)
        
        for chunk in analysis['chunks']:
            try:
                # Check if chunk contains the target syllable
                if syllable.lower() in chunk.text.lower():
                    clip = video.subclip(chunk.start_time, chunk.end_time)
                    clips.append(clip)
                    print(f"Added clip: {chunk.start_time:.2f}s - {chunk.end_time:.2f}s")
            except Exception as e:
                print(f"Error extracting clip from {chunk.start_time} to {chunk.end_time}: {e}")
        
        if not clips:
            print(f"No '{syllable}' syllables found in meaningful chunks")
            video.close()
            return ""
            
        # Composite clips together
        print("\nCompositing clips...")
        try:
            final_video = concatenate_videoclips(clips)
        except Exception as e:
            print(f"Error compositing clips: {e}")
            video.close()
            for clip in clips:
                clip.close()
            return ""
        
        # Save result
        if output_path is None:
            base, ext = os.path.splitext(video_path)
            output_path = f"{base}_{syllable}_syllables{ext}"
            
        print(f"\nSaving composite video to {output_path}...")
        try:
            final_video.write_videofile(output_path)
        except Exception as e:
            print(f"Error saving video: {e}")
            video.close()
            final_video.close()
            for clip in clips:
                clip.close()
            return ""
        
        # Cleanup
        video.close()
        final_video.close()
        for clip in clips:
            clip.close()
            
        return output_path
    
    def analyze_syllable_distribution(self, video_path: str,
                                    min_chunk_duration: float = 0.5,
                                    max_chunk_duration: float = 10.0) -> dict:
        """Analyze the distribution and timing of syllables in the video.
        
        Args:
            video_path: Path to input video
            min_chunk_duration: Minimum duration for chunks in seconds
            max_chunk_duration: Maximum duration for chunks in seconds
        """
        # Update chunk duration settings
        self.analyzer.min_chunk_duration = min_chunk_duration
        self.analyzer.max_chunk_duration = max_chunk_duration
        
        print(f"\nAnalyzing video: {video_path}")
        analysis = self.analyzer.analyze_video(video_path)
        
        if not analysis['chunks']:
            print("No meaningful chunks found in video")
            return {}
        
        # Analyze syllable distribution within chunks
        syllable_analysis = {
            'total_chunks': analysis['total_chunks'],
            'total_duration': analysis['total_duration'],
            'average_chunk_duration': analysis['average_chunk_duration'],
            'chunks_with_syllables': [],
            'syllable_density': [],
            'chunk_statistics': {
                'duration_distribution': [],
                'syllable_frequency': {}
            }
        }
        
        # Analyze each chunk
        for chunk in analysis['chunks']:
            # Count syllables in chunk
            syllable_count = chunk.text.lower().count('e')
            if syllable_count > 0:
                syllable_analysis['chunks_with_syllables'].append({
                    'start_time': chunk.start_time,
                    'end_time': chunk.end_time,
                    'duration': chunk.end_time - chunk.start_time,
                    'syllable_count': syllable_count,
                    'text': chunk.text
                })
                
                # Calculate syllable density
                density = syllable_count / (chunk.end_time - chunk.start_time)
                syllable_analysis['syllable_density'].append({
                    'time': chunk.start_time,
                    'density': density
                })
                
                # Update syllable frequency
                syllable_analysis['chunk_statistics']['syllable_frequency'][chunk.start_time] = syllable_count
            
            # Add to duration distribution
            syllable_analysis['chunk_statistics']['duration_distribution'].append(
                chunk.end_time - chunk.start_time
            )
        
        # Calculate statistics
        if syllable_analysis['chunks_with_syllables']:
            syllable_analysis['average_syllables_per_chunk'] = np.mean(
                [chunk['syllable_count'] for chunk in syllable_analysis['chunks_with_syllables']]
            )
            syllable_analysis['average_syllable_density'] = np.mean(
                [d['density'] for d in syllable_analysis['syllable_density']]
            )
        
        return syllable_analysis

class PhoneticCompositor:
    """Class for composing videos based on phonetic data."""
    
    def __init__(self, fps: int = 30, transition_type: str = "fade", 
                 transition_duration: float = 1.0):
        """Initialize the phonetic compositor.
        
        Args:
            fps: Frames per second
            transition_type: Type of transition between segments
            transition_duration: Duration of transitions in seconds
        """
        self.fps = fps
        self.transition_type = transition_type
        self.transition_duration = transition_duration
    
    def compose(self, segments: List[PhoneticSegment], output_path: str) -> str:
        """Compose a video from phonetic segments.
        
        Args:
            segments: List of phonetic segments
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

def main():
    """Example usage of the syllable compositor."""
    compositor = SyllableCompositor()
    
    # Example video path
    video_path = "downloads/kaku.mp4"
    
    if os.path.exists(video_path):
        # Extract and composite 'e' syllables
        output_path = compositor.extract_syllable_from_video(video_path)
        if output_path:
            print(f"Created composite video at: {output_path}")
            
        # Analyze syllable distribution
        analysis = compositor.analyze_syllable_distribution(video_path)
        print("\nSyllable Analysis:")
        print(f"Total 'e' syllables found: {analysis['total_syllables']}")
        print(f"Total duration of 'e' sounds: {analysis['total_duration']:.2f} seconds")
        print(f"Average syllable duration: {analysis['average_duration']:.3f} seconds")
        
        print("\nTranscript Analysis:")
        print(f"Total words: {analysis['transcript_analysis']['total_words']}")
        print(f"Total segments: {analysis['transcript_analysis']['total_segments']}")
        print("\nFirst few transcript segments:")
        for seg in analysis['transcript_analysis']['transcript'][:3]:
            print(f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['text']}")
        
        print("\nPhonetic Analysis:")
        print("Phoneme density over time:")
        for time, density in analysis['phonetic_analysis']['density'][:5]:
            print(f"Time {time:.2f}s: {density:.2f}")
        
        print("\nPhoneme transitions:")
        for next_phoneme, time in analysis['phonetic_analysis']['transitions'][:5]:
            print(f"Time {time:.2f}s: e -> {next_phoneme}")
        
        print("\nSegment Analysis:")
        for segment, analysis_results in analysis['phonetic_analysis']['segments_with_analysis'][:3]:
            print(f"\nSegment {segment.start_time:.2f}s - {segment.end_time:.2f}s:")
            print(f"MFCC Stats: {analysis_results['mfcc_stats']}")
            print(f"Duration: {analysis_results['duration']:.3f}s")
            print(f"Frame Count: {analysis_results['frame_count']}")
    else:
        print(f"Video file not found: {video_path}")

if __name__ == "__main__":
    main() 