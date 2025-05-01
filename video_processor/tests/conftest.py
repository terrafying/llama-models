"""
Test configuration and fixtures.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import cv2
import numpy as np
from video_processor.video_worker import VideoWorker, VideoChunk

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def video_dir(temp_dir):
    """Create a directory for test videos."""
    video_dir = Path(temp_dir) / "videos"
    video_dir.mkdir()
    return video_dir

@pytest.fixture
def output_dir(temp_dir):
    """Create a directory for test outputs."""
    output_dir = Path(temp_dir) / "output"
    output_dir.mkdir()
    return output_dir

@pytest.fixture
def test_video_path(video_dir):
    """Create a test video file."""
    video_path = video_dir / "test_video.mp4"
    
    # Create a 10-second test video
    fps = 30
    duration = 10
    width, height = 640, 480
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    
    for _ in range(fps * duration):
        # Create a frame with a moving pattern
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.rectangle(frame, (0, 0), (width, height), (0, 255, 0), -1)
        out.write(frame)
        
    out.release()
    return str(video_path)

@pytest.fixture
def video_worker():
    """Create a VideoWorker instance."""
    return VideoWorker(max_workers=2)

@pytest.fixture
def sample_chunk(test_video_path, output_dir):
    """Create a sample VideoChunk."""
    return VideoChunk(
        chunk_id="test_chunk",
        input_path=test_video_path,
        output_path=str(output_dir / "test_chunk.mp4"),
        start_time=0.0,
        end_time=3.0,
        metadata={"test": "data"}
    ) 