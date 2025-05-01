"""
Storage manager module for efficient data storage and retrieval.

This module provides:
1. Integration with IPFS for distributed storage
2. Local volume management
3. Cache management
4. Content deduplication
5. Global content addressing and indexing
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import ipfshttpclient

from ragtime_llm.utils.logger import logger
from ragtime_llm.utils.storage_partitioning import StoragePartitioner, StorageMetadata
from ragtime_llm.utils.content_indexer import ContentIndexer, ContentIndex

class StorageManager:
    """Manages storage across IPFS and local volumes."""

    def __init__(
        self,
        cache_dir: str = ".cache",
        ipfs_api: str = "/ip4/127.0.0.1/tcp/5001",
        local_volumes: Optional[List[str]] = None,
        ipfs_pin_threshold: int = 50 * 1024 * 1024,  # 50MB
        local_volume_threshold: int = 10 * 1024 * 1024,  # 10MB
        replication_factor: int = 2,
        index_dir: str = ".index"
    ):
        """Initialize the storage manager.
        
        Args:
            cache_dir: Cache directory path
            ipfs_api: IPFS API endpoint
            local_volumes: List of local volume paths
            ipfs_pin_threshold: Size threshold for IPFS pinning
            local_volume_threshold: Size threshold for local volume storage
            replication_factor: Number of copies to maintain
            index_dir: Directory for content indices
        """
        self.partitioner = StoragePartitioner(
            cache_dir=cache_dir,
            ipfs_api=ipfs_api,
            local_volumes=local_volumes,
            ipfs_pin_threshold=ipfs_pin_threshold,
            local_volume_threshold=local_volume_threshold,
            replication_factor=replication_factor
        )
        
        self.indexer = ContentIndexer(
            index_dir=index_dir,
            storage_manager=self
        )
        
        self.logger = logging.getLogger(__name__)

    def store_file(
        self,
        file_path: Union[str, Path],
        content_type: str,
        metadata: Optional[Dict] = None
    ) -> Tuple[StorageMetadata, ContentIndex]:
        """Store a file with intelligent partitioning and indexing.
        
        Args:
            file_path: Path to the file to store
            content_type: Type of content (e.g., 'video', 'transcript')
            metadata: Additional metadata
            
        Returns:
            Tuple of (StorageMetadata, ContentIndex)
        """
        try:
            file_path = Path(file_path)
            content = file_path.read_bytes()
            
            # Store content
            storage_metadata = self.partitioner.store_content(content, content_type, metadata)
            
            # Index content
            content_index = self.indexer.index_content(
                content_hash=storage_metadata.content_hash,
                content_type=content_type,
                storage_metadata=storage_metadata,
                metadata=metadata
            )
            
            return storage_metadata, content_index
        except Exception as e:
            self.logger.error(f"Error storing file {file_path}: {e}")
            raise

    def store_bytes(
        self,
        content: bytes,
        content_type: str,
        metadata: Optional[Dict] = None
    ) -> Tuple[StorageMetadata, ContentIndex]:
        """Store bytes with intelligent partitioning and indexing.
        
        Args:
            content: Content to store
            content_type: Type of content
            metadata: Additional metadata
            
        Returns:
            Tuple of (StorageMetadata, ContentIndex)
        """
        try:
            # Store content
            storage_metadata = self.partitioner.store_content(content, content_type, metadata)
            
            # Index content
            content_index = self.indexer.index_content(
                content_hash=storage_metadata.content_hash,
                content_type=content_type,
                storage_metadata=storage_metadata,
                metadata=metadata
            )
            
            return storage_metadata, content_index
        except Exception as e:
            self.logger.error(f"Error storing content: {e}")
            raise

    def retrieve_file(
        self,
        content_hash: str,
        output_path: Optional[Union[str, Path]] = None
    ) -> Tuple[bytes, StorageMetadata, ContentIndex]:
        """Retrieve content by hash.
        
        Args:
            content_hash: Hash of the content to retrieve
            output_path: Optional path to save the content
            
        Returns:
            Tuple of (content_bytes, StorageMetadata, ContentIndex)
        """
        try:
            # Get content index
            content_index = self.indexer.get_content(content_hash)
            if not content_index:
                raise KeyError(f"Content not found: {content_hash}")
            
            # Retrieve content
            content, storage_metadata = self.partitioner.retrieve_content(content_hash)
            
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(content)
            
            return content, storage_metadata, content_index
        except Exception as e:
            self.logger.error(f"Error retrieving content {content_hash}: {e}")
            raise

    def get_metadata(self, content_hash: str) -> Optional[Tuple[StorageMetadata, ContentIndex]]:
        """Get metadata for content hash."""
        try:
            storage_metadata = self.partitioner._get_metadata(content_hash)
            content_index = self.indexer.get_content(content_hash)
            
            if storage_metadata and content_index:
                return storage_metadata, content_index
            return None
        except Exception as e:
            self.logger.error(f"Error getting metadata for {content_hash}: {e}")
            return None

    def get_by_type(self, content_type: str) -> List[ContentIndex]:
        """Get all content of a specific type."""
        return self.indexer.get_by_type(content_type)

    def get_by_tag(self, tag: str) -> List[ContentIndex]:
        """Get all content with a specific tag."""
        return self.indexer.get_by_tag(tag)

    def get_references(self, content_hash: str) -> List[ContentIndex]:
        """Get all content referenced by the given content."""
        return self.indexer.get_references(content_hash)

    def get_referenced_by(self, content_hash: str) -> List[ContentIndex]:
        """Get all content that references the given content."""
        return self.indexer.get_referenced_by(content_hash)

    def update_metadata(
        self,
        content_hash: str,
        metadata: Dict
    ) -> Optional[ContentIndex]:
        """Update content metadata."""
        return self.indexer.update_metadata(content_hash, metadata)

    def delete_content(self, content_hash: str) -> bool:
        """Delete content from storage and index."""
        try:
            # Delete from storage
            # Note: Storage deletion is handled by the partitioner's cleanup
            
            # Delete from index
            return self.indexer.delete_content(content_hash)
        except Exception as e:
            self.logger.error(f"Error deleting content {content_hash}: {e}")
            return False

    def cleanup(self, max_cache_size: Optional[int] = None):
        """Clean up storage.
        
        Args:
            max_cache_size: Maximum cache size in bytes
        """
        try:
            self.partitioner.cleanup(max_cache_size)
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            raise

    def get_storage_stats(self) -> Dict:
        """Get storage and index statistics."""
        try:
            storage_stats = self.partitioner.get_storage_stats()
            index_stats = self.indexer.get_stats()
            
            return {
                "storage": storage_stats,
                "index": index_stats
            }
        except Exception as e:
            self.logger.error(f"Error getting storage stats: {e}")
            return {
                "storage": {
                    "ipfs": {"total_size": 0, "file_count": 0},
                    "local": {"total_size": 0, "file_count": 0},
                    "cache": {"total_size": 0, "file_count": 0}
                },
                "index": {
                    "total_content": 0,
                    "content_types": {},
                    "tags": {},
                    "total_references": 0
                }
            } 