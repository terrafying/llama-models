# RAG-LLM System with Distributed Processing

A powerful RAG (Retrieval-Augmented Generation) system for processing and querying YouTube videos using distributed computing capabilities.

## Features

- Distributed video processing using Ray
- Parallel embedding generation
- Efficient vector storage with FAISS
- Multiple LLM provider support
- GPU acceleration when available
- Web interface for easy interaction
- Command-line interface for automation
- Space-efficient storage with IPFS and local volumes
- Automatic storage management and cleanup
- Topic and creator deep-dive analysis
- Content synthesis and reporting

## Installation

1. Clone the repository:
```bash
git clone git@github.com:terrafying/llama-models.git
cd llama-models
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install IPFS for distributed storage:
```bash
# macOS
brew install ipfs

# Linux
sudo apt-get install ipfs

# Start IPFS daemon
ipfs daemon
```

## Storage Configuration

The system supports multiple storage backends:

1. **IPFS Storage** (for large files >50MB):
   - Automatically used for video files
   - Requires IPFS daemon running
   - Default API endpoint: `/ip4/127.0.0.1/tcp/5001`

2. **Local Volumes** (for medium-sized files):
   - Automatically detects mounted volumes in `/Volumes` and `/mnt`
   - Can be configured manually
   - Example configuration:
   ```python
   from ragtime_llm.utils.storage_manager import StorageManager
   
   storage_manager = StorageManager(
       cache_dir=".cache",
       ipfs_api="/ip4/127.0.0.1/tcp/5001",
       local_volumes=["/Volumes/my_drive"]
   )
   ```

3. **Cache Storage** (for small files):
   - Default location: `.cache`
   - Configurable size limit (default: 10GB)
   - Automatic cleanup of oldest files

## Usage

### Command Line Interface

The system provides a command-line interface for common operations:

1. Process a YouTube video:
```bash
# Basic usage
ragtime process-video "https://www.youtube.com/watch?v=..."

# With custom output directory
ragtime process-video "https://www.youtube.com/watch?v=..." --output-dir /Volumes/my_drive/output
```

2. Process a YouTube playlist:
```bash
# Basic usage
ragtime process-playlist "https://www.youtube.com/playlist?list=..."

# With custom output directory
ragtime process-playlist "https://www.youtube.com/playlist?list=..." --output-dir /Volumes/my_drive/output
```

3. Query the system:
```bash
# Basic query
ragtime query "What is the main topic of the video?"

# With custom parameters
ragtime query "What is the main topic of the video?" --max-tokens 500 --temperature 0.7
```

4. Generate a video response:
```bash
# Basic usage
ragtime generate-video "Summarize the key points"

# With custom output
ragtime generate-video "Summarize the key points" --output-path /Volumes/my_drive/responses/summary.mp4
```

5. Start the web interface:
```bash
# Default port (8080)
ragtime serve

# Custom port
ragtime serve --port 9000
```

6. Topic Deep-Dive Analysis:
```bash
# Basic topic analysis
ragtime topic-deep-dive "artificial intelligence"

# With custom parameters
ragtime topic-deep-dive "quantum computing" \
  --max-videos 15 \
  --max-tokens 3000 \
  --temperature 0.8 \
  --output-format html \
  --output-file analysis.html
```

7. Creator Deep-Dive Analysis:
```bash
# Basic creator analysis
ragtime creator-deep-dive "https://www.youtube.com/c/3blue1brown"

# With custom parameters
ragtime creator-deep-dive "https://www.youtube.com/c/veritasium" \
  --max-videos 30 \
  --max-tokens 3000 \
  --temperature 0.8 \
  --output-format markdown \
  --output-file creator_analysis.md
```

### Python API

1. Initialize the RAG system with storage:
```python
from ragtime_llm.core.unified_rag_system import YouTubeRAG
from ragtime_llm.utils.storage_manager import StorageManager

# Initialize storage manager
storage_manager = StorageManager(
    cache_dir=".cache",
    ipfs_api="/ip4/127.0.0.1/tcp/5001",
    local_volumes=["/Volumes/my_drive"]
)

# Initialize RAG system with storage
rag_system = YouTubeRAG(
    embedding_model="all-MiniLM-L6-v2",
    num_workers=4,
    storage_manager=storage_manager
)
```

2. Add videos with storage management:
```python
# Add a single video
result = rag_system.add_video("https://www.youtube.com/watch?v=...")
print(f"Video stored at: {result['storage']['video']['location']}")

# Add a playlist
results = rag_system.add_playlist("https://www.youtube.com/playlist?list=...")
for video_result in results:
    print(f"Video {video_result['video_info']['title']} stored at: {video_result['storage']['video']['location']}")
```

3. Query videos:
```python
response = rag_system.generate_response(
    query="What is the main topic of the video?",
    max_tokens=500,
    temperature=0.7
)
print(response)
```

4. Generate video responses:
```python
video_path = rag_system.generate_video_response(
    query="Summarize the key points",
    output_path="/Volumes/my_drive/responses/summary.mp4"
)
print(f"Response video generated at: {video_path}")
```

5. Clean up resources:
```python
# Clean up storage
rag_system.cleanup()

# Or with custom cache size
rag_system.storage_manager.cleanup(max_cache_size=5 * 1024 * 1024 * 1024)  # 5GB
```

6. Topic and Creator Analysis:
```python
from ragtime_llm.core.unified_rag_system import YouTubeRAG
from ragtime_llm.content_synthesis import ContentSynthesizer

# Initialize RAG system
rag_system = YouTubeRAG()

# Create content synthesizer
synthesizer = ContentSynthesizer(rag_system)

# Topic deep-dive
topic_analysis = synthesizer.topic_deep_dive(
    topic="machine learning",
    max_videos=15,
    max_tokens=3000,
    temperature=0.8
)

# Generate report
report = synthesizer.generate_report(
    analysis=topic_analysis,
    output_format="markdown"
)

# Save report
with open("topic_analysis.md", "w") as f:
    f.write(report)

# Creator deep-dive
creator_analysis = synthesizer.creator_deep_dive(
    creator_url="https://www.youtube.com/c/3blue1brown",
    max_videos=30,
    max_tokens=3000,
    temperature=0.8
)

# Generate HTML report
html_report = synthesizer.generate_report(
    analysis=creator_analysis,
    output_format="html"
)

# Save report
with open("creator_analysis.html", "w") as f:
    f.write(html_report)
```

### Analysis Features

1. Topic Deep-Dive:
   - Comprehensive analysis of a topic across multiple videos
   - Key points extraction and synthesis
   - Chronological timeline of topic development
   - Source tracking and citation
   - Customizable analysis depth and scope

2. Creator Deep-Dive:
   - Analysis of a creator's body of work
   - Content theme identification
   - Presentation style analysis
   - Content evolution tracking
   - Source management and citation

3. Report Generation:
   - Multiple output formats (Markdown, HTML)
   - Structured and formatted reports
   - Source citations and references
   - Customizable report sections
   - Timestamp and metadata inclusion

## Web Interface

Start the web interface:
```python
from ragtime_llm.core.unified_rag_system import create_web_ui

create_web_ui(rag_system, port=8080)
```

## Distributed Processing

The system uses Ray for distributed processing:
- Video transcription is parallelized across workers
- Embedding generation is distributed
- Vector store operations are optimized for parallel processing
- Storage operations are distributed across available volumes

## Hardware Requirements

- CPU: Multi-core processor recommended
- RAM: 8GB minimum, 16GB+ recommended
- GPU: Optional but recommended for faster processing
- Storage: 
  - SSD recommended for better performance
  - Multiple volumes for distributed storage
  - IPFS node for distributed storage (optional)

## Contributing

We welcome contributions to improve the RAG-LLM system! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest tests/`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Setup

1. Install development dependencies:
```bash
pip install -r requirements.txt
pip install -e ".[dev]"
```

2. Set up pre-commit hooks:
```bash
pre-commit install
```

3. Run tests:
```bash
pytest tests/
```

### Code Style

We use:
- Black for code formatting
- isort for import sorting
- flake8 for linting
- mypy for type checking

Run the formatters:
```bash
black .
isort .
```

### Documentation

- Keep docstrings up to date
- Update README.md for new features
- Add type hints to all functions
- Include examples in docstrings

## License

MIT License
