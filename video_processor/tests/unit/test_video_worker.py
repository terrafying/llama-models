"""
Unit tests for VideoWorker class.
"""

import pytest
from pathlib import Path
from video_processor.video_worker import VideoWorker, VideoChunk

def test_worker_initialization():
    """Test VideoWorker initialization."""
    worker = VideoWorker(max_workers=4)
    assert worker.max_workers == 4
    assert worker.executor is not None

def test_validate_paths(video_worker, test_video_path, video_dir):
    """Test path validation."""
    # Test valid path
    valid, invalid = video_worker.validate_paths([test_video_path])
    assert valid
    assert len(invalid) == 0
    
    # Test invalid path
    invalid_path = str(video_dir / "nonexistent.mp4")
    valid, invalid = video_worker.validate_paths([invalid_path])
    assert not valid
    assert len(invalid) == 1
    assert invalid[0] == invalid_path

def test_create_chunks(video_worker, test_video_path, output_dir):
    """Test chunk creation."""
    chunks = video_worker.create_chunks(
        test_video_path,
        chunk_duration=3.0,
        output_dir=str(output_dir)
    )
    
    assert len(chunks) > 0
    assert chunks[0].input_path == test_video_path
    assert chunks[0].start_time == 0.0
    assert chunks[0].end_time == 3.0
    assert chunks[0].status == "pending"
    
    # Verify chunk output paths
    for i, chunk in enumerate(chunks):
        expected_path = str(output_dir / f"chunk_{i}.mp4")
        assert chunk.output_path == expected_path

def test_process_chunk(video_worker, sample_chunk):
    """Test chunk processing."""
    success = video_worker.process_chunk(sample_chunk)
    assert success
    assert sample_chunk.status == "completed"
    assert Path(sample_chunk.output_path).exists()

def test_process_chunks_parallel(video_worker, test_video_path, output_dir):
    """Test parallel chunk processing."""
    chunks = [
        VideoChunk(
            chunk_id=f"chunk_{i}",
            input_path=test_video_path,
            output_path=str(output_dir / f"chunk_{i}.mp4"),
            start_time=i * 3.0,
            end_time=(i + 1) * 3.0,
            metadata={}
        )
        for i in range(3)
    ]
    
    results = video_worker.process_chunks_parallel(chunks)
    assert len(results) == 3
    assert all(status == "completed" for status in results.values())
    
    # Verify all chunk files exist
    for chunk in chunks:
        assert Path(chunk.output_path).exists()

def test_combine_chunks(video_worker, test_video_path, output_dir):
    """Test chunk combination."""
    # Create and process chunks
    chunks = video_worker.create_chunks(
        test_video_path,
        chunk_duration=3.0,
        output_dir=str(output_dir)
    )
    
    video_worker.process_chunks_parallel(chunks)
    
    # Combine chunks
    final_output = str(output_dir / "final.mp4")
    success = video_worker.combine_chunks(chunks, final_output)
    
    assert success
    assert Path(final_output).exists()

def test_cleanup_chunks(video_worker, sample_chunk):
    """Test chunk cleanup."""
    # Process the chunk first
    video_worker.process_chunk(sample_chunk)
    assert Path(sample_chunk.output_path).exists()
    
    # Clean up
    video_worker.cleanup_chunks([sample_chunk])
    assert not Path(sample_chunk.output_path).exists()

def test_error_handling(video_worker, video_dir):
    """Test error handling."""
    # Test with non-existent video
    invalid_path = str(video_dir / "nonexistent.mp4")
    chunks = video_worker.create_chunks(invalid_path)
    assert len(chunks) == 0
    
    # Test processing invalid chunk
    invalid_chunk = VideoChunk(
        chunk_id="invalid",
        input_path=invalid_path,
        output_path=str(video_dir / "invalid.mp4"),
        start_time=0.0,
        end_time=3.0,
        metadata={}
    )
    
    success = video_worker.process_chunk(invalid_chunk)
    assert not success
    assert invalid_chunk.status == "failed"
    assert invalid_chunk.error is not None 