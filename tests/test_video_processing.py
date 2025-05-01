"""
Tests for video processing in the RAG system.
"""

import os
import pytest
import tempfile
from pathlib import Path
import numpy as np
from ragtime_llm.core.unified_rag_system import YouTubeRAG
from ragtime_llm.video.phonetic_compositor import PhoneticCompositor
from ragtime_llm.video.phonetic_index import PhoneticIndex
from ragtime_llm.core.llm_utils import LLMConfig

@pytest.fixture
def test_video_path():
    """Create a test video file."""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        # Create a simple test video using ffmpeg
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "lavfi",
            "-i", "testsrc=duration=5:size=1280x720:rate=30",
            "-f", "lavfi",
            "-i", "sine=frequency=440:duration=5",
            "-c:v", "libx264",
            "-c:a", "aac",
            f.name
        ]
        os.system(" ".join(cmd))
        return f.name

@pytest.fixture
def rag_system():
    """Create a RAG system instance."""
    llm_config = LLMConfig(
        model_id="llama-2-7b-chat.Q4_0.gguf",  # Using Llama 2 quantized model
        provider="local"  # Use local provider instead of OpenAI
    )
    return YouTubeRAG(llm_config=llm_config)

def test_video_loading_and_processing(rag_system, test_video_path):
    """Test that a video can be loaded and processed for both RAG and phonetic composition."""
    output_path = os.path.join(tempfile.gettempdir(), "test_output.mp4")
    try:
        # Add video to RAG system
        rag_system.add_video(test_video_path)
        
        # Verify RAG processing
        assert len(rag_system.vectors) > 0, "No vectors were generated"
        assert len(rag_system.documents) > 0, "No documents were generated"
        assert len(rag_system.metadata) > 0, "No metadata was generated"
        
        # Create phonetic index
        phonetic_index = PhoneticIndex.from_video(test_video_path)
        
        # Verify phonetic processing
        assert len(phonetic_index.segments) > 0, "No phonetic segments were generated"
        assert phonetic_index.mfcc_features.shape[0] > 0, "No MFCC features were generated"
        
        # Test finding similar segments
        query_mfcc = phonetic_index.mfcc_features[:, :10]  # Use first 10 frames as query
        similar_segments = phonetic_index.find_similar_segments(query_mfcc, top_k=3)
        assert len(similar_segments) > 0, "No similar segments were found"
        
        # Test getting segment frames
        segment = similar_segments[0]
        frames = phonetic_index.get_segment_frames(segment)
        assert len(frames) > 0, "No frames were extracted from segment"
        
        # Test phonetic composition
        compositor = PhoneticCompositor()
        
        # Create test segments
        segments = []
        for seg in similar_segments[:3]:
            segments.append({
                'start_time': seg['start_time'],
                'end_time': seg['end_time'],
                'text': 'test',
                'phonemes': ['t', 'e', 's', 't'],
                'video_path': test_video_path,
                'metadata': {}
            })
        
        # Compose video
        composed_path = compositor.compose(segments, output_path)
        assert os.path.exists(composed_path), "Composed video was not created"
        
    finally:
        # Cleanup
        if os.path.exists(test_video_path):
            os.unlink(test_video_path)
        if os.path.exists(output_path):
            os.unlink(output_path)

def test_video_processing_with_invalid_file(rag_system):
    """Test handling of invalid video files."""
    with pytest.raises(Exception):
        rag_system.add_video("nonexistent_video.mp4")
    
    with pytest.raises(Exception):
        PhoneticIndex.from_video("nonexistent_video.mp4")

def test_video_processing_with_empty_file(rag_system):
    """Test handling of empty video files."""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        empty_path = f.name
    
    try:
        with pytest.raises(Exception):
            rag_system.add_video(empty_path)
        
        with pytest.raises(Exception):
            PhoneticIndex.from_video(empty_path)
    finally:
        if os.path.exists(empty_path):
            os.unlink(empty_path) 