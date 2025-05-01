from video_processor import VideoWorker
from pathlib import Path
import os
import argparse

def resolve_paths(base_dir: str, video_paths: list) -> list:
    """Resolve video paths relative to base directory."""
    return [str(Path(base_dir) / path) for path in video_paths]

def main():
    parser = argparse.ArgumentParser(description="Process video segments in parallel")
    parser.add_argument("--base-dir", default="videos", help="Base directory for video files")
    parser.add_argument("--output-dir", default="output", help="Output directory for processed videos")
    parser.add_argument("--chunk-duration", type=float, default=30.0, help="Duration of each chunk in seconds")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum number of parallel workers")
    args = parser.parse_args()

    # Create worker
    worker = VideoWorker(max_workers=args.max_workers)

    # Example video paths (relative to base directory)
    video_paths = [
        "segment1.mp4",
        "segment2.mp4",
        "segment3.mp4"
    ]

    # Resolve paths
    resolved_paths = resolve_paths(args.base_dir, video_paths)
    
    # Validate paths
    valid, invalid_paths = worker.validate_paths(resolved_paths)
    if not valid:
        print(f"Error: The following video files were not found:")
        for path in invalid_paths:
            print(f"  - {path}")
        return

    # Process each video
    all_chunks = []
    for video_path in resolved_paths:
        # Create chunks for this video
        chunks = worker.create_chunks(
            video_path=video_path,
            chunk_duration=args.chunk_duration,
            output_dir=args.output_dir
        )
        all_chunks.extend(chunks)

    if not all_chunks:
        print("No chunks created. Check video files and paths.")
        return

    # Process chunks in parallel
    print(f"Processing {len(all_chunks)} chunks with {args.max_workers} workers...")
    results = worker.process_chunks_parallel(all_chunks)

    # Count successes and failures
    success_count = sum(1 for status in results.values() if status == "completed")
    print(f"Completed {success_count}/{len(all_chunks)} chunks successfully")

    # Combine successful chunks
    if success_count > 0:
        final_output = str(Path(args.output_dir) / "final_video.mp4")
        if worker.combine_chunks(all_chunks, final_output):
            print(f"Final video generated at: {final_output}")
        else:
            print("Failed to combine chunks into final video")

    # Cleanup temporary files
    worker.cleanup_chunks(all_chunks)

if __name__ == "__main__":
    main() 