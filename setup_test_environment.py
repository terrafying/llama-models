"""
Setup script for test environment with necessary assets.
"""

import os
import shutil
import cv2
import numpy as np
from moviepy import VideoFileClip, ColorClip, CompositeVideoClip

def create_test_environment():
    """Create test environment with necessary assets."""
    # Create directory structure
    directories = [
        "assets/characters/JOE_ROGAN",
        "assets/characters/JORDAN_PETERSON",
        "backgrounds",
        "test_output"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
    # Create test backgrounds
    create_test_backgrounds()
    
    # Create test character assets
    create_test_character_assets()
    
def create_test_backgrounds():
    """Create test background videos."""
    backgrounds = {
        "calm": (0.3, 0.3, 0.3),  # Dark gray
        "neutral": (0.5, 0.5, 0.5),  # Medium gray
        "intense": (0.7, 0.7, 0.7)  # Light gray
    }
    
    for name, color in backgrounds.items():
        # Create a simple gradient background
        width, height = 1080, 1920
        duration = 10  # seconds
        
        # Create gradient
        gradient = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            alpha = y / height
            gradient[y, :] = [int(c * 255) for c in color]
            
        # Add some subtle movement
        def make_frame(t):
            frame = gradient.copy()
            # Add subtle wave effect
            wave = np.sin(t * 2 * np.pi) * 20
            frame = cv2.warpAffine(
                frame,
                cv2.getRotationMatrix2D(
                    (width/2, height/2),
                    wave,
                    1.0
                ),
                (width, height)
            )
            return frame
            
        # Create video clip
        clip = VideoClip(make_frame, duration=duration)
        
        # Write to file
        output_path = os.path.join("backgrounds", f"{name}.mp4")
        clip.write_videofile(
            output_path,
            fps=30,
            codec='libx264',
            audio=False
        )
        
def create_test_character_assets():
    """Create test character images."""
    characters = ["JOE_ROGAN", "JORDAN_PETERSON"]
    positions = ["left", "right"]
    
    for character in characters:
        for position in positions:
            # Create a simple character image
            width, height = 600, 600
            image = np.zeros((height, width, 4), dtype=np.uint8)
            
            # Add some basic shape
            cv2.circle(
                image,
                (width//2, height//2),
                width//3,
                (255, 255, 255, 255),
                -1
            )
            
            # Add character name
            cv2.putText(
                image,
                character,
                (width//4, height//2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0, 255),
                2
            )
            
            # Save image
            output_path = os.path.join(
                "assets",
                "characters",
                character,
                f"{position}.png"
            )
            cv2.imwrite(output_path, image)
            
def cleanup_test_environment():
    """Clean up test environment."""
    directories = [
        "assets",
        "backgrounds",
        "test_output"
    ]
    
    for directory in directories:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            
if __name__ == "__main__":
    create_test_environment() 