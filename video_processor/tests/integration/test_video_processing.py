"""
Integration tests for video processing.
"""

import pytest
from pathlib import Path
from video_processor.video_worker import VideoWorker
import cv2
import numpy as np
import shutil

def test_end_to_end_processing(video_worker, test_video_path, output_dir):
    """Test complete video processing workflow."""
    # Process video
    success = video_worker.process_video(
        test_video_path,
        output_dir=str(output_dir),
        chunk_duration=3.0
    )
    
    assert success
    
    # Verify output files
    output_files = list(output_dir.glob("*.mp4"))
    assert len(output_files) > 0
    
    # Check final output
    final_output = output_dir / "final.mp4"
    assert final_output.exists()
    
    # Verify video properties
    cap = cv2.VideoCapture(str(final_output))
    assert cap.isOpened()
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cap.release()
    
    # Verify properties
    assert width > 0
    assert height > 0
    assert fps > 0
    assert frame_count > 0

def test_parallel_processing(video_worker, test_video_path, output_dir):
    """Test parallel processing of multiple videos."""
    # Create multiple test videos
    test_videos = []
    for i in range(3):
        output_path = output_dir / f"test_video_{i}.mp4"
        # Copy test video
        shutil.copy2(test_video_path, output_path)
        test_videos.append(str(output_path))
    
    # Process videos in parallel
    results = video_worker.process_videos_parallel(
        test_videos,
        output_dir=str(output_dir),
        chunk_duration=3.0
    )
    
    # Verify results
    assert len(results) == 3
    assert all(success for success in results.values())
    
    # Verify output files
    for i in range(3):
        final_output = output_dir / f"test_video_{i}_final.mp4"
        assert final_output.exists()

def test_error_recovery(video_worker, test_video_path, output_dir):
    """Test error recovery during processing."""
    # Create a corrupted video file
    corrupted_path = output_dir / "corrupted.mp4"
    with open(corrupted_path, 'w') as f:
        f.write("This is not a valid video file")
    
    # Process both valid and invalid videos
    videos = [test_video_path, str(corrupted_path)]
    results = video_worker.process_videos_parallel(
        videos,
        output_dir=str(output_dir),
        chunk_duration=3.0
    )
    
    # Verify results
    assert results[test_video_path]  # Valid video should succeed
    assert not results[str(corrupted_path)]  # Corrupted video should fail
    
    # Verify only valid video was processed
    assert (output_dir / "final.mp4").exists()
    assert not (output_dir / "corrupted_final.mp4").exists()

def test_resource_cleanup(video_worker, test_video_path, output_dir):
    """Test proper resource cleanup after processing."""
    # Process video
    video_worker.process_video(
        test_video_path,
        output_dir=str(output_dir),
        chunk_duration=3.0
    )
    
    # Verify chunk files are cleaned up
    chunk_files = list(output_dir.glob("chunk_*.mp4"))
    assert len(chunk_files) == 0
    
    # Verify only final output remains
    output_files = list(output_dir.glob("*.mp4"))
    assert len(output_files) == 1
    assert output_files[0].name == "final.mp4" 