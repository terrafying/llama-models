"""
Vector store implementation using FAISS for efficient similarity search.

This module provides a distributed vector store implementation with:
1. Efficient similarity search using FAISS
2. Distributed vector operations using Ray
3. Metadata management and filtering
4. Persistent storage capabilities

Key Components:
- VectorStoreWorker: Handles distributed vector operations
- VectorStore: Main vector store implementation with distributed support

System Context:
- Integrates with FAISS for efficient similarity search
- Uses Ray for distributed operations
- Supports both local and distributed processing

Performance Context:
- Optimized for high-dimensional vectors
- Efficient batch processing
- Automatic scaling based on available resources
- Memory-efficient storage and retrieval
"""

import numpy as np
import torch
import faiss
from typing import List, Dict, Optional, Tuple
import logging
import ray
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

@ray.remote
class VectorStoreWorker:
    """Worker for distributed vector store operations.
    
    Handles parallel vector operations including:
    - Vector addition and indexing
    - Similarity search
    - Metadata management
    
    Performance Context:
    - Optimized for batch operations
    - Efficient memory management
    - Automatic resource cleanup
    """
    
    def __init__(self, dimension: int = 384):
        """Initialize the vector store worker.
        
        Args:
            dimension: Dimension of the vectors to store
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []
        self.metadata = []
        self.video_chunks = []
    
    def add_vectors(self, 
                   vectors: np.ndarray,
                   documents: List[str],
                   metadata: List[Dict],
                   video_chunks: List[Dict]) -> int:
        """Add vectors to the worker's index.
        
        Args:
            vectors: Array of vectors to add
            documents: List of document texts
            metadata: List of metadata dictionaries
            video_chunks: List of video chunk information
            
        Returns:
            Number of vectors added
        """
        if len(vectors) != len(documents) or len(vectors) != len(metadata):
            raise ValueError("Input lists must have the same length")
        
        # Add to FAISS index
        self.index.add(vectors.astype('float32'))
        
        # Store metadata
        self.documents.extend(documents)
        self.metadata.extend(metadata)
        self.video_chunks.extend(video_chunks)
        
        return len(documents)
    
    def search(self, 
              query_embedding: np.ndarray,
              k: int = 5,
              filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """Search for similar vectors in the worker's index.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of similar documents with scores
        """
        # Search in FAISS index
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'),
            k
        )
        
        # Format results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
                
            # Apply metadata filter if specified
            if filter_metadata:
                doc_metadata = self.metadata[idx]
                if not all(doc_metadata.get(k) == v for k, v in filter_metadata.items()):
                    continue
            
            # Convert distance to similarity score (FAISS uses L2 distance)
            similarity = 1.0 / (1.0 + distance)
            
            results.append({
                "text": self.documents[idx],
                "score": float(similarity),
                "metadata": self.metadata[idx],
                "video_chunk": self.video_chunks[idx]
            })
        
        return results

class VectorStore:
    """Efficient vector store implementation using FAISS with distributed support.
    
    Provides a unified interface for vector storage and retrieval with:
    - Distributed vector operations
    - Efficient similarity search
    - Metadata management
    - Persistent storage
    
    System Context:
    - Integrates with FAISS for efficient similarity search
    - Uses Ray for distributed operations
    - Supports both local and distributed processing
    
    Performance Context:
    - Optimized for high-dimensional vectors
    - Efficient batch processing
    - Automatic scaling based on available resources
    - Memory-efficient storage and retrieval
    """
    
    def __init__(self, 
                 dimension: int = 384, 
                 initial_capacity: int = 1000,
                 num_workers: Optional[int] = None):
        """Initialize the vector store.
        
        Args:
            dimension: Dimension of the vectors
            initial_capacity: Initial capacity of the store
            num_workers: Number of Ray workers for distributed operations
        """
        self.dimension = dimension
        self.capacity = initial_capacity
        self.count = 0
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(dimension)
        
        # Store metadata
        self.documents = []
        self.metadata = []
        self.video_chunks = []
        
        # Initialize distributed workers if Ray is available
        if ray.is_initialized():
            self.num_workers = num_workers or ray.available_resources().get("CPU", 1)
            self.workers = [
                VectorStoreWorker.remote(dimension)
                for _ in range(self.num_workers)
            ]
        else:
            self.num_workers = 1
            self.workers = None
    
    def add_document(self, 
                    document: str,
                    embedding: np.ndarray,
                    metadata: Optional[Dict] = None,
                    video_chunk: Optional[Dict] = None) -> int:
        """Add a document to the store.
        
        Args:
            document: Document text
            embedding: Document embedding
            metadata: Optional metadata dictionary
            video_chunk: Optional video chunk information
            
        Returns:
            Index of the added document
        """
        # Ensure embedding is the right shape
        if embedding.shape != (self.dimension,):
            raise ValueError(f"Expected embedding shape ({self.dimension},), got {embedding.shape}")
        
        # Add to FAISS index
        self.index.add(embedding.reshape(1, -1).astype('float32'))
        
        # Store metadata
        self.documents.append(document)
        self.metadata.append(metadata or {})
        self.video_chunks.append(video_chunk or {})
        
        # Update count
        self.count += 1
        
        return self.count - 1
    
    def add_documents_batch(self,
                          documents: List[str],
                          embeddings: np.ndarray,
                          metadata: Optional[List[Dict]] = None,
                          video_chunks: Optional[List[Dict]] = None) -> List[int]:
        """Add multiple documents to the store in parallel.
        
        Args:
            documents: List of document texts
            embeddings: Array of document embeddings
            metadata: Optional list of metadata dictionaries
            video_chunks: Optional list of video chunk information
            
        Returns:
            List of indices of added documents
        """
        if self.workers is None:
            # Fall back to sequential processing
            indices = []
            for doc, emb, meta, chunk in zip(
                documents,
                embeddings,
                metadata or [{}] * len(documents),
                video_chunks or [{}] * len(documents)
            ):
                idx = self.add_document(doc, emb, meta, chunk)
                indices.append(idx)
            return indices
        
        # Prepare data for distributed processing
        if metadata is None:
            metadata = [{}] * len(documents)
        if video_chunks is None:
            video_chunks = [{}] * len(documents)
        
        # Split data among workers
        chunk_size = len(documents) // self.num_workers + 1
        futures = []
        
        for i in range(0, len(documents), chunk_size):
            end = min(i + chunk_size, len(documents))
            worker = self.workers[i // chunk_size % self.num_workers]
            
            futures.append(
                worker.add_vectors.remote(
                    embeddings[i:end],
                    documents[i:end],
                    metadata[i:end],
                    video_chunks[i:end]
                )
            )
        
        # Collect results
        results = ray.get(futures)
        
        # Update local index
        self.index.add(embeddings.astype('float32'))
        self.documents.extend(documents)
        self.metadata.extend(metadata)
        self.video_chunks.extend(video_chunks)
        self.count += len(documents)
        
        return list(range(self.count - len(documents), self.count))
    
    def search(self, 
              query_embedding: np.ndarray,
              k: int = 5,
              filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """Search for similar documents.
        
        Args:
            query_embedding: Query embedding
            k: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of results with scores and metadata
        """
        if self.workers is None:
            # Fall back to sequential search
            return self._search_local(query_embedding, k, filter_metadata)
        
        # Search in parallel across workers
        futures = [
            worker.search.remote(query_embedding, k, filter_metadata)
            for worker in self.workers
        ]
        
        # Collect and merge results
        all_results = ray.get(futures)
        merged_results = []
        for results in all_results:
            merged_results.extend(results)
        
        # Sort by score and take top k
        merged_results.sort(key=lambda x: x["score"], reverse=True)
        return merged_results[:k]
    
    def _search_local(self,
                     query_embedding: np.ndarray,
                     k: int = 5,
                     filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """Local search implementation.
        
        Args:
            query_embedding: Query embedding
            k: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of results with scores and metadata
        """
        # Ensure query embedding is the right shape
        if query_embedding.shape != (self.dimension,):
            raise ValueError(f"Expected query embedding shape ({self.dimension},), got {query_embedding.shape}")
        
        # Search in FAISS index
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'),
            k
        )
        
        # Format results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
                
            # Apply metadata filter if specified
            if filter_metadata:
                doc_metadata = self.metadata[idx]
                if not all(doc_metadata.get(k) == v for k, v in filter_metadata.items()):
                    continue
            
            # Convert distance to similarity score (FAISS uses L2 distance)
            similarity = 1.0 / (1.0 + distance)
            
            results.append({
                "text": self.documents[idx],
                "score": float(similarity),
                "metadata": self.metadata[idx],
                "video_chunk": self.video_chunks[idx]
            })
        
        return results
    
    def save(self, path: str) -> bool:
        """Save the vector store to disk.
        
        Args:
            path: Path to save the store
            
        Returns:
            True if successful
        """
        try:
            # Save FAISS index
            faiss.write_index(self.index, f"{path}.index")
            
            # Save metadata
            import pickle
            with open(f"{path}.meta", "wb") as f:
                pickle.dump({
                    "documents": self.documents,
                    "metadata": self.metadata,
                    "video_chunks": self.video_chunks,
                    "count": self.count
                }, f)
            
            return True
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
            return False
    
    def load(self, path: str) -> bool:
        """Load the vector store from disk.
        
        Args:
            path: Path to load the store from
            
        Returns:
            True if successful
        """
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{path}.index")
            
            # Load metadata
            import pickle
            with open(f"{path}.meta", "rb") as f:
                data = pickle.load(f)
                self.documents = data["documents"]
                self.metadata = data["metadata"]
                self.video_chunks = data["video_chunks"]
                self.count = data["count"]
            
            return True
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return False 