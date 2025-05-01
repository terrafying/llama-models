"""
Storage partitioning module for efficient data distribution across IPFS and local storage.

This module provides:
1. Smart partitioning of data based on size, type, and access patterns
2. IPFS integration with pinning and garbage collection
3. Local storage management with volume balancing
4. Metadata tracking and deduplication
"""

import os
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import ipfshttpclient
from dataclasses import dataclass, asdict
import shutil

from ragtime_llm.utils.logger import logger

@dataclass
class StorageMetadata:
    """Metadata for stored content."""
    content_hash: str
    size_bytes: int
    content_type: str
    storage_type: str  # 'ipfs', 'local', or 'cache'
    location: str
    created_at: str
    last_accessed: str
    access_count: int
    pin_status: Optional[bool] = None  # For IPFS content
    replication_factor: int = 1  # Number of copies

class StoragePartitioner:
    """Handles intelligent partitioning of data across storage backends."""

    def __init__(
        self,
        ipfs_api: str = "/ip4/127.0.0.1/tcp/5001",
        local_volumes: Optional[List[str]] = None,
        cache_dir: str = ".cache",
        ipfs_pin_threshold: int = 50 * 1024 * 1024,  # 50MB
        local_volume_threshold: int = 10 * 1024 * 1024,  # 10MB
        replication_factor: int = 2
    ):
        """Initialize the storage partitioner.
        
        Args:
            ipfs_api: IPFS API endpoint
            local_volumes: List of local volume paths
            cache_dir: Cache directory path
            ipfs_pin_threshold: Size threshold for IPFS pinning
            local_volume_threshold: Size threshold for local volume storage
            replication_factor: Number of copies to maintain
        """
        self.ipfs_api = ipfs_api
        self.local_volumes = local_volumes or []
        self.cache_dir = Path(cache_dir)
        self.ipfs_pin_threshold = ipfs_pin_threshold
        self.local_volume_threshold = local_volume_threshold
        self.replication_factor = replication_factor
        
        # Initialize storage
        self._init_storage()
        
        # Initialize IPFS client
        try:
            self.ipfs_client = ipfshttpclient.connect(ipfs_api)
            logger.info(f"Connected to IPFS API at {ipfs_api}")
        except Exception as e:
            logger.warning(f"Failed to connect to IPFS: {e}")
            self.ipfs_client = None

    def _init_storage(self):
        """Initialize storage directories and metadata."""
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create metadata directory
        self.metadata_dir = self.cache_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)
        
        # Initialize local volumes
        for volume in self.local_volumes:
            volume_path = Path(volume)
            if not volume_path.exists():
                logger.warning(f"Local volume not found: {volume}")
                continue
            (volume_path / "ragtime").mkdir(exist_ok=True)

    def _calculate_hash(self, content: bytes) -> str:
        """Calculate content hash."""
        return hashlib.sha256(content).hexdigest()

    def _get_storage_decision(
        self,
        content: bytes,
        content_type: str
    ) -> Tuple[str, List[str]]:
        """Determine optimal storage location(s) for content.
        
        Returns:
            Tuple of (primary_storage_type, [storage_locations])
        """
        size = len(content)
        
        # Determine storage type based on size and content type
        if size >= self.ipfs_pin_threshold:
            storage_type = "ipfs"
        elif size >= self.local_volume_threshold:
            storage_type = "local"
        else:
            storage_type = "cache"
        
        # Select storage locations
        locations = []
        if storage_type == "ipfs":
            locations = ["ipfs"]  # IPFS is distributed by nature
        elif storage_type == "local":
            # Select volumes based on available space
            volumes = self._get_available_volumes()
            locations = volumes[:self.replication_factor]
        else:  # cache
            locations = [str(self.cache_dir)]
        
        return storage_type, locations

    def _get_available_volumes(self) -> List[str]:
        """Get list of available volumes sorted by free space."""
        volumes = []
        for volume in self.local_volumes:
            try:
                volume_path = Path(volume)
                if not volume_path.exists():
                    continue
                free_space = shutil.disk_usage(volume_path).free
                volumes.append((volume, free_space))
            except Exception as e:
                logger.warning(f"Error checking volume {volume}: {e}")
        
        # Sort by free space (descending)
        volumes.sort(key=lambda x: x[1], reverse=True)
        return [v[0] for v in volumes]

    def store_content(
        self,
        content: bytes,
        content_type: str,
        metadata: Optional[Dict] = None
    ) -> StorageMetadata:
        """Store content with intelligent partitioning.
        
        Args:
            content: Content to store
            content_type: Type of content (e.g., 'video', 'transcript')
            metadata: Additional metadata
            
        Returns:
            StorageMetadata object
        """
        content_hash = self._calculate_hash(content)
        
        # Check if content already exists
        existing_metadata = self._get_metadata(content_hash)
        if existing_metadata:
            return existing_metadata
        
        # Determine storage location
        storage_type, locations = self._get_storage_decision(content, content_type)
        
        # Store content
        stored_locations = []
        for location in locations:
            try:
                if location == "ipfs":
                    if self.ipfs_client:
                        result = self.ipfs_client.add_bytes(content)
                        if len(content) >= self.ipfs_pin_threshold:
                            self.ipfs_client.pin.add(result)
                        stored_locations.append(f"ipfs://{result}")
                else:
                    # Store in local volume or cache
                    store_path = Path(location) / "ragtime" / content_hash[:2] / content_hash
                    store_path.parent.mkdir(parents=True, exist_ok=True)
                    store_path.write_bytes(content)
                    stored_locations.append(str(store_path))
            except Exception as e:
                logger.error(f"Error storing content in {location}: {e}")
        
        if not stored_locations:
            raise RuntimeError("Failed to store content in any location")
        
        # Create metadata
        metadata = StorageMetadata(
            content_hash=content_hash,
            size_bytes=len(content),
            content_type=content_type,
            storage_type=storage_type,
            location=stored_locations[0],  # Primary location
            created_at=datetime.now().isoformat(),
            last_accessed=datetime.now().isoformat(),
            access_count=0,
            pin_status=storage_type == "ipfs",
            replication_factor=len(stored_locations)
        )
        
        # Save metadata
        self._save_metadata(metadata)
        
        return metadata

    def retrieve_content(self, content_hash: str) -> Tuple[bytes, StorageMetadata]:
        """Retrieve content by hash.
        
        Args:
            content_hash: Hash of the content to retrieve
            
        Returns:
            Tuple of (content_bytes, metadata)
        """
        metadata = self._get_metadata(content_hash)
        if not metadata:
            raise KeyError(f"Content not found: {content_hash}")
        
        # Try locations in order
        for location in self._get_all_locations(metadata):
            try:
                if location.startswith("ipfs://"):
                    if self.ipfs_client:
                        ipfs_hash = location[7:]  # Remove "ipfs://" prefix
                        content = self.ipfs_client.cat(ipfs_hash)
                        break
                else:
                    content = Path(location).read_bytes()
                    break
            except Exception as e:
                logger.warning(f"Error retrieving from {location}: {e}")
        else:
            raise RuntimeError(f"Failed to retrieve content from any location: {content_hash}")
        
        # Update metadata
        metadata.last_accessed = datetime.now().isoformat()
        metadata.access_count += 1
        self._save_metadata(metadata)
        
        return content, metadata

    def _get_all_locations(self, metadata: StorageMetadata) -> List[str]:
        """Get all locations where content is stored."""
        locations = [metadata.location]
        
        # Check metadata directory for additional locations
        metadata_path = self.metadata_dir / f"{metadata.content_hash}.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                data = json.load(f)
                if "additional_locations" in data:
                    locations.extend(data["additional_locations"])
        
        return locations

    def _get_metadata(self, content_hash: str) -> Optional[StorageMetadata]:
        """Get metadata for content hash."""
        metadata_path = self.metadata_dir / f"{content_hash}.json"
        if not metadata_path.exists():
            return None
        
        with open(metadata_path) as f:
            data = json.load(f)
            return StorageMetadata(**data)

    def _save_metadata(self, metadata: StorageMetadata):
        """Save metadata to disk."""
        metadata_path = self.metadata_dir / f"{metadata.content_hash}.json"
        with open(metadata_path, "w") as f:
            json.dump(asdict(metadata), f, indent=2)

    def cleanup(self, max_cache_size: Optional[int] = None):
        """Clean up storage.
        
        Args:
            max_cache_size: Maximum cache size in bytes
        """
        if max_cache_size:
            self._cleanup_cache(max_cache_size)
        
        if self.ipfs_client:
            self._cleanup_ipfs()

    def _cleanup_cache(self, max_size: int):
        """Clean up cache directory."""
        try:
            total_size = 0
            files = []
            
            # Get all files in cache
            for path in self.cache_dir.rglob("*"):
                if path.is_file():
                    size = path.stat().st_size
                    total_size += size
                    files.append((path, size, path.stat().st_mtime))
            
            # Sort by last access time
            files.sort(key=lambda x: x[2])
            
            # Remove oldest files until under max size
            while total_size > max_size and files:
                path, size, _ = files.pop(0)
                try:
                    path.unlink()
                    total_size -= size
                except Exception as e:
                    logger.warning(f"Error removing file {path}: {e}")
        
        except Exception as e:
            logger.error(f"Error cleaning up cache: {e}")

    def _cleanup_ipfs(self):
        """Clean up IPFS storage."""
        try:
            # Get list of pinned items
            pinned = self.ipfs_client.pin.ls()
            
            # Check metadata for each pinned item
            for item in pinned:
                try:
                    metadata = self._get_metadata(item)
                    if not metadata or metadata.access_count == 0:
                        self.ipfs_client.pin.rm(item)
                except Exception as e:
                    logger.warning(f"Error checking pinned item {item}: {e}")
        
        except Exception as e:
            logger.error(f"Error cleaning up IPFS: {e}")

    def get_storage_stats(self) -> Dict:
        """Get storage statistics."""
        stats = {
            "ipfs": {"total_size": 0, "file_count": 0},
            "local": {"total_size": 0, "file_count": 0},
            "cache": {"total_size": 0, "file_count": 0}
        }
        
        # Count files and sizes
        for metadata_path in self.metadata_dir.glob("*.json"):
            try:
                with open(metadata_path) as f:
                    metadata = StorageMetadata(**json.load(f))
                    stats[metadata.storage_type]["total_size"] += metadata.size_bytes
                    stats[metadata.storage_type]["file_count"] += 1
            except Exception as e:
                logger.warning(f"Error reading metadata {metadata_path}: {e}")
        
        return stats 