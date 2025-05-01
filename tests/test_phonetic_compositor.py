"""
Tests for the phonetic compositor functionality.
"""

import os
import pytest
import numpy as np
from moviepy.editor import VideoFileClip
import tempfile
from ragtime_llm.video.phonetic_compositor import SyllableCompositor, PhoneticAnalyzer

@pytest.fixture
def test_video_path():
    """Create a test video file."""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
        # Create a simple test video with some audio
        video = VideoFileClip("tests/data/test_video.mp4")
        video.write_videofile(temp_file.name)
        video.close()
        return temp_file.name

@pytest.fixture
def compositor():
    """Create a SyllableCompositor instance."""
    return SyllableCompositor()

def test_meaningful_chunk_extraction(compositor, test_video_path):
    """Test that meaningful chunks are correctly extracted."""
    # Analyze video with different chunk duration settings
    analysis = compositor.analyzer.analyze_video(test_video_path)
    
    # Verify basic analysis results
    assert 'chunks' in analysis
    assert 'total_chunks' in analysis
    assert 'total_duration' in analysis
    assert 'average_chunk_duration' in analysis
    
    # Verify chunk properties
    for chunk in analysis['chunks']:
        # Check chunk duration constraints
        duration = chunk.end_time - chunk.start_time
        assert duration >= compositor.analyzer.min_chunk_duration
        assert duration <= compositor.analyzer.max_chunk_duration
        
        # Check chunk content
        assert hasattr(chunk, 'text')
        assert hasattr(chunk, 'start_time')
        assert hasattr(chunk, 'end_time')
        assert hasattr(chunk, 'metadata')

def test_syllable_extraction_with_chunks(compositor, test_video_path):
    """Test syllable extraction using the new chunking system."""
    # Extract syllables with different chunk duration settings
    output_path = compositor.extract_syllable_from_video(
        test_video_path,
        syllable='e',
        min_chunk_duration=0.5,
        max_chunk_duration=5.0
    )
    
    # Verify output
    assert os.path.exists(output_path)
    output_video = VideoFileClip(output_path)
    assert output_video.duration > 0
    output_video.close()

def test_syllable_distribution_analysis(compositor, test_video_path):
    """Test syllable distribution analysis with the new chunking system."""
    # Analyze syllable distribution
    analysis = compositor.analyze_syllable_distribution(
        test_video_path,
        min_chunk_duration=0.5,
        max_chunk_duration=5.0
    )
    
    # Verify analysis results
    assert 'total_chunks' in analysis
    assert 'total_duration' in analysis
    assert 'average_chunk_duration' in analysis
    assert 'chunks_with_syllables' in analysis
    assert 'syllable_density' in analysis
    assert 'chunk_statistics' in analysis
    
    # Verify chunk statistics
    stats = analysis['chunk_statistics']
    assert 'duration_distribution' in stats
    assert 'syllable_frequency' in stats
    
    # Verify duration distribution
    durations = stats['duration_distribution']
    assert len(durations) > 0
    assert all(d >= compositor.analyzer.min_chunk_duration for d in durations)
    assert all(d <= compositor.analyzer.max_chunk_duration for d in durations)

def test_chunk_merging(compositor):
    """Test that overlapping chunks are correctly merged."""
    # Create test chunks
    chunks = [
        (0.0, 1.0),
        (0.8, 2.0),
        (2.5, 3.5),
        (3.4, 4.0)
    ]
    
    # Test merging
    merged = compositor.analyzer._merge_overlapping_chunks(chunks)
    
    # Verify merged chunks
    assert len(merged) == 2  # Should merge into two chunks
    assert merged[0] == (0.0, 2.0)  # First two chunks merged
    assert merged[1] == (2.5, 4.0)  # Last two chunks merged

def test_silence_detection(compositor):
    """Test silence detection in audio chunks."""
    # Create test silence regions
    silence_regions = [
        (1.0, 1.5),
        (2.0, 2.3),
        (3.0, 3.8)
    ]
    
    # Test silence detection
    assert compositor.analyzer._contains_significant_silence(silence_regions, 1.0, 1.5)
    assert not compositor.analyzer._contains_significant_silence(silence_regions, 1.6, 1.9)
    assert compositor.analyzer._contains_significant_silence(silence_regions, 2.1, 2.2)

def test_chunk_splitting(compositor):
    """Test splitting of long chunks at silence points."""
    # Create test chunk and silence regions
    start_time = 0.0
    end_time = 15.0
    silence_regions = [
        (4.0, 4.5),
        (8.0, 8.3),
        (12.0, 12.8)
    ]
    
    # Test splitting
    chunks = compositor.analyzer._split_at_silence(start_time, end_time, silence_regions)
    
    # Verify split chunks
    assert len(chunks) > 1  # Should be split into multiple chunks
    for start, end in chunks:
        assert end - start <= compositor.analyzer.max_chunk_duration
        assert end - start >= compositor.analyzer.min_chunk_duration

def test_cleanup(test_video_path):
    """Clean up test files."""
    if os.path.exists(test_video_path):
        os.unlink(test_video_path) 