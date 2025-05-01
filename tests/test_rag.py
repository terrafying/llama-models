from ragtime_llm.core.unified_rag_system import YouTubeRAG
import os
import argparse
import requests
import json
import pytest

@pytest.fixture
def endpoint():
    return "http://localhost:52415"

@pytest.fixture
def model():
    return "llama-2-11b"

def test_exo_endpoint(endpoint: str, model: str):
    """Test the Exo endpoint with a simple request."""
    try:
        response = requests.post(
            f"{endpoint}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 0.3,
                "max_tokens": 10
            }
        )
        print(f"Exo endpoint test status code: {response.status_code}")
        if response.status_code == 200:
            print("Exo endpoint test successful!")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        else:
            print(f"Response text: {response.text}")
    except Exception as e:
        print(f"Error testing Exo endpoint: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Test YouTube RAG system with Exo')
    parser.add_argument('--video-url', type=str, default="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                      help='YouTube video URL to process')
    parser.add_argument('--exo-endpoint', type=str, default="http://localhost:52415",
                      help='Exo server endpoint')
    parser.add_argument('--model', type=str, default="llama-2-11b",
                      help='Model to use (e.g., llama-2-11b, llama-2-7b)')
    parser.add_argument('--query', type=str, default="What is the main topic of this video?",
                      help='Query to ask about the video')
    args = parser.parse_args()

    # Initialize RAG system
    rag_system = YouTubeRAG()
    
    print(f"\nProcessing video: {args.video_url}")
    try:
        # Download video
        print("\nDownloading video...")
        video_path = rag_system.download_audio(args.video_url, output_dir="downloads", video=True)
        if not video_path:
            raise ValueError(f"Failed to download video: {args.video_url}")
        print(f"Video downloaded to: {video_path}")
        
        # Extract video clips and get transcripts
        print("\nExtracting video clips and transcripts...")
        video_chunks = rag_system.extract_video_clips(video_path)
        if not video_chunks:
            raise ValueError("No video chunks were extracted")
        print(f"Extracted {len(video_chunks)} chunks")
        
        # Print transcripts
        print("\nTranscripts:")
        for i, chunk in enumerate(video_chunks, 1):
            print(f"\nChunk {i} ({chunk.start_time:.1f}s - {chunk.end_time:.1f}s):")
            print(f"Transcript: {chunk.transcript}")
            print(f"Length: {len(chunk.transcript)} characters")
            print(f"Tokens: {len(chunk.tokens)}")
        
    except Exception as e:
        print(f"Error processing video: {e}")
        return

if __name__ == "__main__":
    main() 