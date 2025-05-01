"""
Storage manager for efficient data handling using IPFS and local volumes.

This module provides:
1. Automatic storage selection based on size and availability
2. IPFS integration for distributed storage
3. Local volume management for large files
4. Cache management and cleanup
"""

import os
import shutil
import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, List, Union
import ipfshttpclient
from ragtime_llm.utils.logger import logger

class StorageManager:
    """Manages storage across IPFS and local volumes."""
    
    def __init__(self, 
                 cache_dir: str = ".cache",
                 ipfs_api: str = "/ip4/127.0.0.1/tcp/5001",
                 local_volumes: Optional[List[str]] = None):
        """Initialize the storage manager.
        
        Args:
            cache_dir: Base directory for cache
            ipfs_api: IPFS API endpoint
            local_volumes: List of local volume paths to use
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize IPFS client
        try:
            self.ipfs_client = ipfshttpclient.connect(ipfs_api)
            self.ipfs_available = True
            logger.info("IPFS client initialized successfully")
        except Exception as e:
            self.ipfs_available = False
            logger.warning(f"IPFS client initialization failed: {e}")
        
        # Set up local volumes
        self.local_volumes = local_volumes or []
        if not self.local_volumes:
            # Auto-detect volumes
            for path in ["/Volumes", "/mnt"]:
                if os.path.exists(path):
                    for vol in os.listdir(path):
                        vol_path = os.path.join(path, vol)
                        if os.path.ismount(vol_path):
                            self.local_volumes.append(vol_path)
        
        logger.info(f"Using local volumes: {self.local_volumes}")
        
        # Initialize storage metadata
        self.metadata_file = self.cache_dir / "storage_metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load storage metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading metadata: {e}")
        return {"files": {}, "volumes": {}}
    
    def _save_metadata(self):
        """Save storage metadata."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def _get_file_hash(self, file_path: Union[str, Path]) -> str:
        """Calculate file hash."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _get_available_volume(self, size_bytes: int) -> Optional[str]:
        """Get available volume with sufficient space."""
        for volume in self.local_volumes:
            try:
                stat = os.statvfs(volume)
                free_space = stat.f_frsize * stat.f_bavail
                if free_space > size_bytes * 1.1:  # 10% buffer
                    return volume
            except Exception as e:
                logger.warning(f"Error checking volume {volume}: {e}")
        return None
    
    def store_file(self, 
                  file_path: Union[str, Path],
                  use_ipfs: bool = True,
                  min_size_ipfs: int = 10 * 1024 * 1024) -> Dict:
        """Store a file efficiently.
        
        Args:
            file_path: Path to the file
            use_ipfs: Whether to use IPFS for storage
            min_size_ipfs: Minimum file size for IPFS storage (10MB default)
            
        Returns:
            Dictionary with storage information
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_size = file_path.stat().st_size
        file_hash = self._get_file_hash(file_path)
        
        # Check if already stored
        if file_hash in self.metadata["files"]:
            return self.metadata["files"][file_hash]
        
        storage_info = {
            "hash": file_hash,
            "size": file_size,
            "original_path": str(file_path),
            "storage_type": None,
            "location": None
        }
        
        # Determine storage method
        if use_ipfs and self.ipfs_available and file_size >= min_size_ipfs:
            try:
                # Add to IPFS
                result = self.ipfs_client.add(str(file_path))
                storage_info.update({
                    "storage_type": "ipfs",
                    "location": result["Hash"]
                })
                logger.info(f"File stored in IPFS: {result['Hash']}")
            except Exception as e:
                logger.warning(f"IPFS storage failed: {e}")
                use_ipfs = False
        
        if not use_ipfs or not self.ipfs_available:
            # Use local volume
            volume = self._get_available_volume(file_size)
            if volume:
                target_dir = Path(volume) / "ragtime_storage" / file_hash[:2]
                target_dir.mkdir(parents=True, exist_ok=True)
                target_path = target_dir / file_path.name
                
                # Copy file
                shutil.copy2(file_path, target_path)
                storage_info.update({
                    "storage_type": "local",
                    "location": str(target_path)
                })
                logger.info(f"File stored locally: {target_path}")
            else:
                # Fallback to cache
                target_dir = self.cache_dir / "files" / file_hash[:2]
                target_dir.mkdir(parents=True, exist_ok=True)
                target_path = target_dir / file_path.name
                
                # Copy file
                shutil.copy2(file_path, target_path)
                storage_info.update({
                    "storage_type": "cache",
                    "location": str(target_path)
                })
                logger.info(f"File stored in cache: {target_path}")
        
        # Update metadata
        self.metadata["files"][file_hash] = storage_info
        self._save_metadata()
        
        return storage_info
    
    def retrieve_file(self, 
                     file_hash: str,
                     target_path: Optional[Union[str, Path]] = None) -> Path:
        """Retrieve a stored file.
        
        Args:
            file_hash: Hash of the file to retrieve
            target_path: Optional target path for the file
            
        Returns:
            Path to the retrieved file
        """
        if file_hash not in self.metadata["files"]:
            raise KeyError(f"File not found: {file_hash}")
        
        storage_info = self.metadata["files"][file_hash]
        
        if target_path is None:
            target_path = self.cache_dir / "retrieved" / file_hash[:2]
            target_path.mkdir(parents=True, exist_ok=True)
            target_path = target_path / Path(storage_info["original_path"]).name
        
        target_path = Path(target_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        if storage_info["storage_type"] == "ipfs":
            try:
                self.ipfs_client.get(storage_info["location"], str(target_path))
            except Exception as e:
                logger.error(f"Error retrieving from IPFS: {e}")
                raise
        else:
            # Local or cache storage
            shutil.copy2(storage_info["location"], target_path)
        
        return target_path
    
    def cleanup(self, max_cache_size: int = 10 * 1024 * 1024 * 1024):  # 10GB default
        """Clean up cache directory.
        
        Args:
            max_cache_size: Maximum cache size in bytes
        """
        cache_dir = self.cache_dir / "files"
        if not cache_dir.exists():
            return
        
        # Calculate current size
        total_size = 0
        files = []
        for path in cache_dir.rglob("*"):
            if path.is_file():
                size = path.stat().st_size
                total_size += size
                files.append((path, size))
        
        # Sort by modification time
        files.sort(key=lambda x: x[0].stat().st_mtime)
        
        # Remove oldest files until under limit
        while total_size > max_cache_size and files:
            path, size = files.pop(0)
            try:
                path.unlink()
                total_size -= size
                logger.info(f"Removed cached file: {path}")
            except Exception as e:
                logger.error(f"Error removing file {path}: {e}")
        
        self._save_metadata() 