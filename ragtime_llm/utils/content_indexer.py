"""
Content indexing and addressing module for global content management.

This module provides:
1. Global addressing scheme for all content
2. Content indexing and metadata management
3. Cross-referencing between different content types
4. Efficient content lookup and retrieval
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Union
from datetime import datetime
from dataclasses import dataclass, asdict
import hashlib

from ragtime_llm.utils.logger import logger
from ragtime_llm.utils.storage_partitioning import StorageMetadata

@dataclass
class ContentAddress:
    """Global address for content."""
    content_type: str  # 'video', 'transcript', 'embedding', etc.
    content_hash: str  # Hash of the content
    parent_hash: Optional[str] = None  # Hash of parent content (e.g., video for transcript)
    metadata_hash: Optional[str] = None  # Hash of metadata
    created_at: str = ""  # ISO format timestamp
    updated_at: str = ""  # ISO format timestamp
    tags: List[str] = None  # Content tags
    references: List[str] = None  # References to other content

@dataclass
class ContentIndex:
    """Index entry for content."""
    address: ContentAddress
    storage_metadata: StorageMetadata
    title: Optional[str] = None
    description: Optional[str] = None
    source_url: Optional[str] = None
    duration: Optional[float] = None
    size_bytes: int = 0
    access_count: int = 0
    last_accessed: Optional[str] = None
    is_public: bool = True
    custom_metadata: Dict = None

class ContentIndexer:
    """Manages global content indexing and addressing."""

    def __init__(
        self,
        index_dir: str = ".index",
        storage_manager = None
    ):
        """Initialize the content indexer.
        
        Args:
            index_dir: Directory for index files
            storage_manager: Storage manager instance
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self.storage_manager = storage_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize indices
        self._init_indices()
        
        # Load existing indices
        self._load_indices()

    def _init_indices(self):
        """Initialize index structures."""
        self.content_index: Dict[str, ContentIndex] = {}  # content_hash -> ContentIndex
        self.type_index: Dict[str, Set[str]] = {}  # content_type -> set of content_hashes
        self.tag_index: Dict[str, Set[str]] = {}  # tag -> set of content_hashes
        self.reference_index: Dict[str, Set[str]] = {}  # content_hash -> set of referenced hashes

    def _load_indices(self):
        """Load existing indices from disk."""
        try:
            # Load main index
            index_file = self.index_dir / "content_index.json"
            if index_file.exists():
                with open(index_file) as f:
                    data = json.load(f)
                    for hash_val, index_data in data.items():
                        self.content_index[hash_val] = ContentIndex(
                            address=ContentAddress(**index_data["address"]),
                            storage_metadata=StorageMetadata(**index_data["storage_metadata"]),
                            **{k: v for k, v in index_data.items() 
                               if k not in ["address", "storage_metadata"]}
                        )
            
            # Load type index
            type_file = self.index_dir / "type_index.json"
            if type_file.exists():
                with open(type_file) as f:
                    self.type_index = {k: set(v) for k, v in json.load(f).items()}
            
            # Load tag index
            tag_file = self.index_dir / "tag_index.json"
            if tag_file.exists():
                with open(tag_file) as f:
                    self.tag_index = {k: set(v) for k, v in json.load(f).items()}
            
            # Load reference index
            ref_file = self.index_dir / "reference_index.json"
            if ref_file.exists():
                with open(ref_file) as f:
                    self.reference_index = {k: set(v) for k, v in json.load(f).items()}
        
        except Exception as e:
            self.logger.error(f"Error loading indices: {e}")
            self._init_indices()  # Reset on error

    def _save_indices(self):
        """Save indices to disk."""
        try:
            # Save main index
            index_file = self.index_dir / "content_index.json"
            with open(index_file, "w") as f:
                json.dump({
                    hash_val: {
                        "address": asdict(index.address),
                        "storage_metadata": asdict(index.storage_metadata),
                        **{k: v for k, v in asdict(index).items() 
                           if k not in ["address", "storage_metadata"]}
                    }
                    for hash_val, index in self.content_index.items()
                }, f, indent=2)
            
            # Save type index
            type_file = self.index_dir / "type_index.json"
            with open(type_file, "w") as f:
                json.dump({k: list(v) for k, v in self.type_index.items()}, f, indent=2)
            
            # Save tag index
            tag_file = self.index_dir / "tag_index.json"
            with open(tag_file, "w") as f:
                json.dump({k: list(v) for k, v in self.tag_index.items()}, f, indent=2)
            
            # Save reference index
            ref_file = self.index_dir / "reference_index.json"
            with open(ref_file, "w") as f:
                json.dump({k: list(v) for k, v in self.reference_index.items()}, f, indent=2)
        
        except Exception as e:
            self.logger.error(f"Error saving indices: {e}")

    def index_content(
        self,
        content_hash: str,
        content_type: str,
        storage_metadata: StorageMetadata,
        parent_hash: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> ContentIndex:
        """Index new content.
        
        Args:
            content_hash: Hash of the content
            content_type: Type of content
            storage_metadata: Storage metadata
            parent_hash: Hash of parent content
            metadata: Additional metadata
            
        Returns:
            ContentIndex object
        """
        try:
            # Create content address
            address = ContentAddress(
                content_type=content_type,
                content_hash=content_hash,
                parent_hash=parent_hash,
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                tags=metadata.get("tags", []) if metadata else [],
                references=metadata.get("references", []) if metadata else []
            )
            
            # Create index entry
            index = ContentIndex(
                address=address,
                storage_metadata=storage_metadata,
                title=metadata.get("title") if metadata else None,
                description=metadata.get("description") if metadata else None,
                source_url=metadata.get("source_url") if metadata else None,
                duration=metadata.get("duration") if metadata else None,
                size_bytes=storage_metadata.size_bytes,
                custom_metadata=metadata
            )
            
            # Update indices
            self.content_index[content_hash] = index
            
            # Update type index
            if content_type not in self.type_index:
                self.type_index[content_type] = set()
            self.type_index[content_type].add(content_hash)
            
            # Update tag index
            for tag in address.tags:
                if tag not in self.tag_index:
                    self.tag_index[tag] = set()
                self.tag_index[tag].add(content_hash)
            
            # Update reference index
            if address.references:
                self.reference_index[content_hash] = set(address.references)
            
            # Save indices
            self._save_indices()
            
            return index
        
        except Exception as e:
            self.logger.error(f"Error indexing content {content_hash}: {e}")
            raise

    def get_content(self, content_hash: str) -> Optional[ContentIndex]:
        """Get content by hash."""
        return self.content_index.get(content_hash)

    def get_by_type(self, content_type: str) -> List[ContentIndex]:
        """Get all content of a specific type."""
        hashes = self.type_index.get(content_type, set())
        return [self.content_index[h] for h in hashes if h in self.content_index]

    def get_by_tag(self, tag: str) -> List[ContentIndex]:
        """Get all content with a specific tag."""
        hashes = self.tag_index.get(tag, set())
        return [self.content_index[h] for h in hashes if h in self.content_index]

    def get_references(self, content_hash: str) -> List[ContentIndex]:
        """Get all content referenced by the given content."""
        hashes = self.reference_index.get(content_hash, set())
        return [self.content_index[h] for h in hashes if h in self.content_index]

    def get_referenced_by(self, content_hash: str) -> List[ContentIndex]:
        """Get all content that references the given content."""
        return [
            index for index in self.content_index.values()
            if content_hash in (index.address.references or [])
        ]

    def update_metadata(
        self,
        content_hash: str,
        metadata: Dict
    ) -> Optional[ContentIndex]:
        """Update content metadata."""
        if content_hash not in self.content_index:
            return None
        
        index = self.content_index[content_hash]
        
        # Update address
        if "tags" in metadata:
            # Remove old tags
            for tag in index.address.tags:
                if tag in self.tag_index:
                    self.tag_index[tag].discard(content_hash)
            
            # Add new tags
            index.address.tags = metadata["tags"]
            for tag in index.address.tags:
                if tag not in self.tag_index:
                    self.tag_index[tag] = set()
                self.tag_index[tag].add(content_hash)
        
        if "references" in metadata:
            index.address.references = metadata["references"]
            self.reference_index[content_hash] = set(metadata["references"])
        
        # Update index
        for key, value in metadata.items():
            if hasattr(index, key):
                setattr(index, key, value)
        
        index.address.updated_at = datetime.now().isoformat()
        
        # Save indices
        self._save_indices()
        
        return index

    def delete_content(self, content_hash: str) -> bool:
        """Delete content from index."""
        if content_hash not in self.content_index:
            return False
        
        index = self.content_index[content_hash]
        
        # Remove from type index
        if index.address.content_type in self.type_index:
            self.type_index[index.address.content_type].discard(content_hash)
        
        # Remove from tag index
        for tag in index.address.tags:
            if tag in self.tag_index:
                self.tag_index[tag].discard(content_hash)
        
        # Remove from reference index
        if content_hash in self.reference_index:
            del self.reference_index[content_hash]
        
        # Remove from content index
        del self.content_index[content_hash]
        
        # Save indices
        self._save_indices()
        
        return True

    def get_stats(self) -> Dict:
        """Get index statistics."""
        return {
            "total_content": len(self.content_index),
            "content_types": {
                type_: len(hashes)
                for type_, hashes in self.type_index.items()
            },
            "tags": {
                tag: len(hashes)
                for tag, hashes in self.tag_index.items()
            },
            "total_references": sum(len(refs) for refs in self.reference_index.values())
        } 