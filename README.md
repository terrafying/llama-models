# RAGTime LLM

A unified Retrieval-Augmented Generation (RAG) system for YouTube videos that combines GPU acceleration, multiple LLM options, and a user-friendly web interface.

## Features

- GPU-accelerated vector store using PyTorch
- Support for multiple LLM providers:
  - OpenAI API
  - Local LLMs (llama-cpp-python)
  - Exo Labs/Exo instance
- YouTube video content extraction and processing
- Automatic transcript generation using Whisper
- Efficient text chunking and embedding
- Web UI built with Gradio
- Save/load functionality for the RAG system

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ragtime-llm.git
cd ragtime-llm
```

2. Install the package:
```bash
pip install -e .
```

## Usage

1. Start the web interface:
```bash
python -m ragtime_llm.core.unified_rag_system
```

2. Open your browser and navigate to `http://localhost:8081`

3. Use the interface to:
   - Add YouTube videos to the RAG system
   - Ask questions about the video content
   - Save and load the RAG system state

## Configuration

The system can be configured through the following parameters:

- `model_name`: The sentence transformer model to use (default: "all-MiniLM-L6-v2")
- `dimension`: The dimension of the embeddings (default: 384)
- `llm_model_path`: Path to the local LLM model file (default: "models/llama-2-7b-chat.gguf")

## LLM Options

The system supports three LLM providers:

1. OpenAI API:
   - Requires an API key
   - Supports various models (e.g., gpt-3.5-turbo, gpt-4)
   - Configure through the web interface

2. Local LLM:
   - Uses llama-cpp-python
   - Supports various GGUF models
   - Specify the model path in the web interface

3. Exo Labs/Exo instance:
   - Requires an API key
   - Supports custom endpoint configuration
   - Configure through the web interface
   - Ideal for private deployments

## GPU Acceleration

The system automatically detects and uses GPU acceleration when available:
- Vector operations are accelerated using PyTorch
- LLM inference can use GPU layers
- Sentence transformer models can use GPU

## Package Structure

```
ragtime_llm/
├── core/
│   ├── __init__.py
│   └── unified_rag_system.py
├── video/
│   ├── __init__.py
│   ├── video_generator.py
│   └── video_composer.py
├── utils/
│   ├── __init__.py
│   └── logger.py
└── __init__.py
```

## API Reference

### YouTubeRAG Class

Core RAG system for processing YouTube videos and generating responses.

```python
def __init__(self, model_name: str = "all-MiniLM-L6-v2", dimension: int = 384, 
             llm_model_path: str = "models/llama-2-7b-chat.gguf")

def add_video(self, url: str) -> int

def add_playlist(self, playlist_url: str) -> int

def extract_video_clips(self, video_path: str, chunk_duration: float = 15.0) -> List[VideoChunk]

def generate_response(self, query: str, llm_provider: str = "local", 
                     model: str = None, video_url: str = None, k: int = 3) -> Dict

def save(self, path: str = "youtube_rag")

def load(self, path: str = "youtube_rag")
```

### VectorStore Class

GPU-accelerated vector store for managing embeddings and documents.

```python
def __init__(self, dimension: int = 384, initial_capacity: int = 1000)

def add_vectors(self, vectors: np.ndarray, documents: List[str], metadata: List[Dict], 
                video_chunks: Optional[List[VideoChunk]] = None)

def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict]

def save(self, path: str)

def load(self, path: str)
```

### VideoChunk Class

Container for video segment data.

```python
def __init__(self, video_path: str, start_time: float, end_time: float, 
             transcript: str, tokens: List[int], metadata: Dict)
```

### Web UI Functions

```python
def create_web_ui(rag_system: YouTubeRAG, port: int = 8081)

def add_video(youtube_url: str)

def query_videos(question: str, llm_provider: str, model: str, 
                api_key: str = "", exo_endpoint: str = DEFAULT_EXO_ENDPOINT)

def save_rag(path)

def load_rag(path)
```

## Testing

Run the test suite using pytest:

```bash
pytest -v
```

The test suite includes:
- Video generation tests
- RAG system tests
- Local video processing tests
- Video generator unit tests

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 