"""
YouTube utilities for downloading, parsing, and processing YouTube videos.
"""

import os
import subprocess
from typing import List, Dict, Optional
from pytube import YouTube, Playlist
from youtube_transcript_api import YouTubeTranscriptApi
import whisper
import tempfile
from dataclasses import dataclass
from ragtime_llm.utils.logger import logger

@dataclass
class VideoMetadata:
    """Container for video metadata."""
    id: str
    url: str
    title: str
    description: str
    author: str
    publish_date: str
    views: int
    keywords: List[str]

class YouTubeDownloader:
    """Handles YouTube video downloading and processing."""
    
    def __init__(self):
        """Initialize the YouTube downloader."""
        self.whisper_model = whisper.load_model("base")
        
    def get_video_id(self, url: str) -> str:
        """Extract video ID from YouTube URL."""
        if "youtu.be" in url:
            return url.split("/")[-1]
        elif "youtube.com" in url:
            if "v=" in url:
                return url.split("v=")[1].split("&")[0]
            elif "embed/" in url:
                return url.split("embed/")[1].split("?")[0]
        return url
    
    def is_playlist_url(self, url: str) -> bool:
        """Check if URL is a YouTube playlist."""
        return "playlist" in url or "list=" in url
    
    def get_playlist_videos(self, playlist_url: str) -> List[str]:
        """Get all video URLs from a playlist."""
        playlist = Playlist(playlist_url)
        return list(playlist.video_urls)
    
    def download_audio(self, url: str, output_dir: str = "downloads", video: bool = False) -> Optional[str]:
        """Download audio from YouTube video."""
        try:
            yt = YouTube(url)
            if video:
                stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
            else:
                stream = yt.streams.filter(only_audio=True).first()
            
            if not stream:
                logger.error(f"No suitable stream found for {url}")
                return None
            
            os.makedirs(output_dir, exist_ok=True)
            output_path = stream.download(output_path=output_dir)
            
            if not video:
                # Convert to mp3 if it's audio only
                base, _ = os.path.splitext(output_path)
                new_path = base + '.mp3'
                os.rename(output_path, new_path)
                return new_path
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error downloading {url}: {str(e)}")
            return None
    
    def get_video_metadata(self, url: str) -> Dict:
        """Get video metadata."""
        try:
            yt = YouTube(url)
            return {
                "title": yt.title,
                "author": yt.author,
                "length": yt.length,
                "views": yt.views,
                "publish_date": yt.publish_date.isoformat() if yt.publish_date else None
            }
        except Exception as e:
            logger.error(f"Error getting metadata for {url}: {str(e)}")
            return {}
    
    def get_video_transcript(self, video_id: str) -> List[Dict]:
        """Get video transcript using YouTube API."""
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            return transcript_list
        except Exception as e:
            logger.error(f"Error getting transcript for {video_id}: {str(e)}")
            return []
    
    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio using Whisper."""
        try:
            result = self.whisper_model.transcribe(audio_path)
            return result["text"]
        except Exception as e:
            logger.error(f"Error transcribing {audio_path}: {str(e)}")
            return ""

    def download_with_youtube_dl(self, url: str, video: bool = False) -> Optional[str]:
        """Download video/audio using yt-dlp as a fallback."""
        try:
            video_id = self.get_video_id(url)
            output_path = os.path.join("downloads", f"{video_id}.{'mp4' if video else 'mp3'}")
            
            if os.path.exists(output_path):
                print(f"File already exists: {output_path}")
                return output_path
            
            # Construct yt-dlp command
            format_option = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best" if video else "bestaudio[ext=m4a]/bestaudio/best"
            cmd = [
                "yt-dlp",
                "-f", format_option,
                "-o", output_path,
                url
            ]
            
            # Execute yt-dlp
            subprocess.run(cmd, check=True)
            
            print(f"Successfully downloaded to: {output_path}")
            return output_path
        except Exception as e:
            print(f"Error downloading with yt-dlp: {str(e)}")
            return None
    
    def download_audio(self, url: str, video: bool = False) -> Optional[str]:
        """Download audio/video from YouTube video with fallback to youtube-dl."""
        try:
            video_id = self.get_video_id(url)
            output_path = os.path.join("downloads", f"{video_id}.{'mp4' if video else 'mp3'}")
            
            if os.path.exists(output_path):
                print(f"File already exists: {output_path}")
                return output_path
                
            print(f"Downloading {'video' if video else 'audio'} from: {url}")
            yt = YouTube(url)
            
            # Try different stream options
            if video:
                streams = yt.streams.filter(progressive=True, file_extension='mp4')
                if not streams:
                    streams = yt.streams.filter(file_extension='mp4')
            else:
                streams = yt.streams.filter(only_audio=True)
            
            if not streams:
                print("No suitable streams found with pytube, trying youtube-dl...")
                return self.download_with_youtube_dl(url, video)
                
            # Get the highest quality stream
            stream = streams.order_by('resolution').desc().first()
            if not stream:
                print("Failed to get stream with pytube, trying youtube-dl...")
                return self.download_with_youtube_dl(url, video)
            
            print(f"Downloading stream: {stream.resolution if video else 'audio'}")
            stream.download(output_path=output_path)
            
            print(f"Successfully downloaded to: {output_path}")
            return output_path
        except Exception as e:
            print(f"Error downloading with pytube: {str(e)}")
            print("Trying youtube-dl as fallback...")
            return self.download_with_youtube_dl(url, video)
    
    def get_video_metadata(self, url: str) -> VideoMetadata:
        """Extract metadata from a YouTube video."""
        try:
            print(f"Attempting to get metadata for video: {url}")
            yt = YouTube(url)
            
            # Try to get basic metadata first
            metadata = {
                "id": self.get_video_id(url),
                "url": url
            }
            
            # Add additional metadata if available
            try:
                metadata["title"] = yt.title or "Unknown Title"
            except:
                metadata["title"] = "Unknown Title"
                
            try:
                metadata["description"] = yt.description or "No description available"
            except:
                metadata["description"] = "No description available"
                
            try:
                metadata["author"] = yt.author or "Unknown Author"
            except:
                metadata["author"] = "Unknown Author"
                
            try:
                metadata["publish_date"] = str(yt.publish_date) if yt.publish_date else "Unknown Date"
            except:
                metadata["publish_date"] = "Unknown Date"
                
            try:
                metadata["views"] = yt.views or 0
            except:
                metadata["views"] = 0
                
            try:
                metadata["keywords"] = yt.keywords or []
            except:
                metadata["keywords"] = []
            
            print(f"Successfully extracted metadata for video: {metadata['title']}")
            return VideoMetadata(**metadata)
            
        except Exception as e:
            print(f"Error extracting metadata: {e}")
            # Return basic metadata even if extraction fails
            return VideoMetadata(
                id=self.get_video_id(url),
                url=url,
                title="Unknown Title",
                description="No description available",
                author="Unknown Author",
                publish_date="Unknown Date",
                views=0,
                keywords=[]
            )
    
    def _get_segment_transcript(self, video_path: str, start_time: float, end_time: float) -> str:
        """Extract audio segment and transcribe it."""
        try:
            if not self.whisper_model:
                print("Error: Whisper model not initialized")
                return ""

            print(f"\nTranscribing segment from {start_time:.1f}s to {end_time:.1f}s")
            
            # Create temporary file for audio segment
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_audio:
                temp_audio_path = temp_audio.name

            if not os.path.exists(video_path):
                print(f"Error: Video file not found: {video_path}")
                return ""
            
            # Extract audio segment
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-ss', str(start_time),
                '-t', str(end_time - start_time),
                '-vn',
                '-acodec', 'libmp3lame',
                '-ar', '16000',
                '-ac', '1',
                '-b:a', '128k',
                temp_audio_path
            ]
            
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"FFmpeg error: {result.stderr}")
                return ""
            
            # Transcribe with Whisper
            result = self.whisper_model.transcribe(
                temp_audio_path,
                language="en",
                fp16=False,
                temperature=0.2
            )
            
            transcript = result["text"].strip()
            
            # Clean up
            try:
                os.unlink(temp_audio_path)
            except OSError:
                pass

            return transcript

        except Exception as e:
            print(f"Error in _get_segment_transcript: {str(e)}")
            return "" 