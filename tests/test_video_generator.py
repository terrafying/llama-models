import unittest
import os
import tempfile
import pytest
import numpy as np
from pathlib import Path
from ragtime_llm.video.video_generator import (
    VideoGenerator,
    VideoSegment,
    extract_frame,
    extract_keyframes,
    create_transition,
    generate_video
)

class TestVideoGenerator(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.test_video_path = os.path.join(self.test_dir, "test_video.mp4")
        
        # Create a sample video segment
        self.segment = VideoSegment(
            start_time=0.0,
            end_time=5.0,
            transcript="Test transcript",
            video_path=self.test_video_path,
            metadata={"test": "data"}
        )

    def test_video_segment_creation(self):
        """Test VideoSegment creation and attributes"""
        self.assertEqual(self.segment.start_time, 0.0)
        self.assertEqual(self.segment.end_time, 5.0)
        self.assertEqual(self.segment.transcript, "Test transcript")
        self.assertEqual(self.segment.video_path, self.test_video_path)
        self.assertEqual(self.segment.metadata, {"test": "data"})

    def test_extract_frame(self):
        """Test frame extraction"""
        # Note: This test requires a valid video file
        if os.path.exists(self.test_video_path):
            frame = extract_frame(self.test_video_path, 1.0)
            self.assertIsNotNone(frame)
            self.assertEqual(len(frame.shape), 3)  # Should be a 3D array (height, width, channels)

    def test_extract_keyframes(self):
        """Test keyframe extraction"""
        # Note: This test requires a valid video file
        if os.path.exists(self.test_video_path):
            keyframes = extract_keyframes(self.segment)
            self.assertIsInstance(keyframes, list)
            if keyframes:
                self.assertEqual(len(keyframes[0].shape), 3)

    def test_create_transition(self):
        """Test transition creation"""
        # Create two sample frames
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = np.ones((100, 100, 3), dtype=np.uint8) * 255

        # Test fade transition
        fade = create_transition(frame1, frame2, "fade", 0.5)
        self.assertIsNotNone(fade)
        self.assertEqual(fade.shape, (100, 100, 3))

        # Test slide transition
        slide = create_transition(frame1, frame2, "slide", 0.5)
        self.assertIsNotNone(slide)
        self.assertEqual(slide.shape, (100, 100, 3))

        # Test dissolve transition
        dissolve = create_transition(frame1, frame2, "dissolve", 0.5)
        self.assertIsNotNone(dissolve)
        self.assertEqual(dissolve.shape, (100, 100, 3))

    def test_generate_video(self):
        """Test video generation"""
        # Note: This test requires valid video files
        if os.path.exists(self.test_video_path):
            output_path = os.path.join(self.test_dir, "output.mp4")
            segments = [self.segment]
            
            result = generate_video(
                segments=segments,
                output_path=output_path,
                fps=30,
                transition_type="fade",
                transition_duration=0.5
            )
            
            self.assertTrue(result)
            self.assertTrue(os.path.exists(output_path))

    def tearDown(self):
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.test_dir)

if __name__ == '__main__':
    unittest.main() 