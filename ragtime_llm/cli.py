"""
Command-line interface for the RAG-LLM system.

This module provides a command-line interface for:
1. Processing YouTube videos and playlists
2. Querying video content
3. Generating video responses
4. Managing the system
5. Configuring storage options

Usage:
    ragtime process-video <video_url> [--output-dir OUTPUT_DIR] [--use-ipfs] [--cache-dir CACHE_DIR]
    ragtime process-playlist <playlist_url> [--output-dir OUTPUT_DIR] [--use-ipfs] [--cache-dir CACHE_DIR]
    ragtime query <query> [--video-id VIDEO_ID] [--max-tokens MAX_TOKENS] [--temperature TEMP]
    ragtime generate-video <query> [--output-path OUTPUT_PATH] [--max-tokens MAX_TOKENS] [--temperature TEMP]
    ragtime serve [--port PORT] [--cache-dir CACHE_DIR]
    ragtime cleanup [--cache-dir CACHE_DIR] [--max-cache-size MAX_CACHE_SIZE]
"""

import argparse
import os
from pathlib import Path
from typing import Optional
from ragtime_llm.core.unified_rag_system import YouTubeRAG, create_web_ui
from ragtime_llm.utils.logger import logger
from ragtime_llm.utils.storage_manager import StorageManager

def get_storage_manager(args) -> StorageManager:
    """Create storage manager from CLI arguments."""
    return StorageManager(
        cache_dir=args.cache_dir,
        ipfs_api=args.ipfs_api,
        local_volumes=args.local_volumes
    )

def process_video(args):
    """Process a single YouTube video."""
    try:
        storage_manager = get_storage_manager(args)
        rag_system = YouTubeRAG(storage_manager=storage_manager)
        result = rag_system.add_video(args.video_url)
        logger.info(f"Successfully processed video: {args.video_url}")
        logger.info(f"Video stored at: {result['storage']['video']['location']}")
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        raise

def process_playlist(args):
    """Process a YouTube playlist."""
    try:
        storage_manager = get_storage_manager(args)
        rag_system = YouTubeRAG(storage_manager=storage_manager)
        results = rag_system.add_playlist(args.playlist_url)
        logger.info(f"Successfully processed playlist: {args.playlist_url}")
        for video_result in results:
            logger.info(f"Video {video_result['video_info']['title']} stored at: {video_result['storage']['video']['location']}")
    except Exception as e:
        logger.error(f"Error processing playlist: {e}")
        raise

def query(args):
    """Query the RAG system."""
    try:
        storage_manager = get_storage_manager(args)
        rag_system = YouTubeRAG(storage_manager=storage_manager)
        response = rag_system.generate_response(
            query=args.query,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )
        print("\nResponse:")
        print(response)
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise

def generate_video(args):
    """Generate a video response."""
    try:
        storage_manager = get_storage_manager(args)
        rag_system = YouTubeRAG(storage_manager=storage_manager)
        output_path = rag_system.generate_video_response(
            query=args.query,
            output_path=args.output_path,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )
        logger.info(f"Video response generated: {output_path}")
    except Exception as e:
        logger.error(f"Error generating video response: {e}")
        raise

def serve(args):
    """Start the web interface."""
    try:
        storage_manager = get_storage_manager(args)
        rag_system = YouTubeRAG(storage_manager=storage_manager)
        create_web_ui(rag_system, port=args.port)
    except Exception as e:
        logger.error(f"Error starting web interface: {e}")
        raise

def cleanup(args):
    """Clean up storage."""
    try:
        storage_manager = get_storage_manager(args)
        storage_manager.cleanup(max_cache_size=args.max_cache_size)
        logger.info("Storage cleanup completed")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="RAG-LLM System CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--cache-dir", default=".cache", help="Cache directory")
    common_parser.add_argument("--ipfs-api", default="/ip4/127.0.0.1/tcp/5001", help="IPFS API endpoint")
    common_parser.add_argument("--local-volumes", nargs="+", help="Local volume paths")

    # Process video command
    process_video_parser = subparsers.add_parser("process-video", parents=[common_parser], help="Process a YouTube video")
    process_video_parser.add_argument("video_url", help="URL of the YouTube video")
    process_video_parser.add_argument("--output-dir", default="output", help="Output directory")
    process_video_parser.set_defaults(func=process_video)

    # Process playlist command
    process_playlist_parser = subparsers.add_parser("process-playlist", parents=[common_parser], help="Process a YouTube playlist")
    process_playlist_parser.add_argument("playlist_url", help="URL of the YouTube playlist")
    process_playlist_parser.add_argument("--output-dir", default="output", help="Output directory")
    process_playlist_parser.set_defaults(func=process_playlist)

    # Query command
    query_parser = subparsers.add_parser("query", parents=[common_parser], help="Query the RAG system")
    query_parser.add_argument("query", help="Query text")
    query_parser.add_argument("--max-tokens", type=int, default=500, help="Maximum tokens in response")
    query_parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    query_parser.set_defaults(func=query)

    # Generate video command
    generate_video_parser = subparsers.add_parser("generate-video", parents=[common_parser], help="Generate a video response")
    generate_video_parser.add_argument("query", help="Query text")
    generate_video_parser.add_argument("--output-path", default="response.mp4", help="Output video path")
    generate_video_parser.add_argument("--max-tokens", type=int, default=500, help="Maximum tokens in response")
    generate_video_parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    generate_video_parser.set_defaults(func=generate_video)

    # Serve command
    serve_parser = subparsers.add_parser("serve", parents=[common_parser], help="Start the web interface")
    serve_parser.add_argument("--port", type=int, default=8080, help="Port to run the interface on")
    serve_parser.set_defaults(func=serve)

    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", parents=[common_parser], help="Clean up storage")
    cleanup_parser.add_argument("--max-cache-size", type=int, default=10 * 1024 * 1024 * 1024, help="Maximum cache size in bytes")
    cleanup_parser.set_defaults(func=cleanup)

    args = parser.parse_args()
    if args.command:
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 