# RAG-LLM System with Distributed Processing

A powerful RAG (Retrieval-Augmented Generation) system for processing and querying YouTube videos using distributed computing capabilities.

## Features

- Distributed video processing using Ray
- Parallel embedding generation
- Efficient vector storage with FAISS
- Multiple LLM provider support
- GPU acceleration when available
- Web interface for easy interaction

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

## Usage

1. Initialize the RAG system:
```python
from ragtime_llm.core.unified_rag_system import YouTubeRAG

# Initialize with default settings
rag_system = YouTubeRAG()

# Or customize settings
rag_system = YouTubeRAG(
    embedding_model="all-MiniLM-L6-v2",
    num_workers=4  # Number of Ray workers
)
```

2. Add videos:
```python
# Add a single video
rag_system.add_video("https://www.youtube.com/watch?v=...")

# Add a playlist
rag_system.add_playlist("https://www.youtube.com/playlist?list=...")
```

3. Query videos:
```python
response = rag_system.generate_response(
    query="What is the main topic of the video?",
    max_tokens=500,
    temperature=0.7
)
```

4. Generate video responses:
```python
video_path = rag_system.generate_video_response(
    query="Summarize the key points",
    output_path="response.mp4"
)
```

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

## Hardware Requirements

- CPU: Multi-core processor recommended
- RAM: 8GB minimum, 16GB+ recommended
- GPU: Optional but recommended for faster processing
- Storage: SSD recommended for better performance

## License

MIT License
