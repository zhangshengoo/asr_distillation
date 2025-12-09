"""Batch media processor for efficient media handling"""

import asyncio
import hashlib
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import pickle
import os

from loguru import logger

from .media_detector import MediaDetector, MediaInfo, MediaType
from .media_extractor import MediaExtractor, MediaConfig


@dataclass
class MediaItem:
    """Media item container"""
    item_id: str
    file_bytes: bytes
    filename: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def cache_key(self) -> str:
        """Generate cache key for this media item"""
        # Create hash based on file content
        content_hash = hashlib.md5(self.file_bytes).hexdigest()
        return f"{self.item_id}_{content_hash}"


@dataclass
class AudioData:
    """Processed audio data container"""
    item_id: str
    audio_bytes: bytes
    sample_rate: int
    channels: int
    duration: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class CacheConfig:
    """Cache configuration"""
    enabled: bool = True
    cache_dir: str = "./cache/media"
    max_size_gb: float = 50.0
    ttl_hours: int = 24
    cleanup_interval_hours: int = 6


class MediaCache:
    """LRU cache for processed media"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache_dir = config.cache_dir
        self.max_size_bytes = int(config.max_size_gb * 1024 * 1024 * 1024)
        self.ttl_seconds = config.ttl_hours * 3600
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Cache index
        self._index = self._load_index()
        self._cleanup_expired()
    
    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load cache index from disk"""
        index_file = os.path.join(self.cache_dir, "index.pkl")
        try:
            if os.path.exists(index_file):
                with open(index_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache index: {e}")
        
        return {}
    
    def _save_index(self):
        """Save cache index to disk"""
        index_file = os.path.join(self.cache_dir, "index.pkl")
        try:
            with open(index_file, 'wb') as f:
                pickle.dump(self._index, f)
        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")
    
    def _get_cache_path(self, cache_key: str) -> str:
        """Get cache file path for key"""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def _is_expired(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if cache entry is expired"""
        return time.time() - cache_entry['timestamp'] > self.ttl_seconds
    
    def _cleanup_expired(self):
        """Remove expired cache entries"""
        expired_keys = []
        for key, entry in self._index.items():
            if self._is_expired(entry):
                expired_keys.append(key)
        
        for key in expired_keys:
            self._remove_entry(key)
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _remove_entry(self, cache_key: str):
        """Remove cache entry"""
        cache_path = self._get_cache_path(cache_key)
        try:
            if os.path.exists(cache_path):
                os.unlink(cache_path)
            if cache_key in self._index:
                del self._index[cache_key]
        except Exception as e:
            logger.error(f"Failed to remove cache entry {cache_key}: {e}")
    
    def _enforce_size_limit(self):
        """Enforce cache size limit using LRU"""
        total_size = sum(entry['size'] for entry in self._index.values())
        
        if total_size > self.max_size_bytes:
            # Sort by last access time (LRU)
            sorted_entries = sorted(
                self._index.items(),
                key=lambda x: x[1]['last_access']
            )
            
            # Remove oldest entries until under limit
            for cache_key, entry in sorted_entries:
                self._remove_entry(cache_key)
                total_size -= entry['size']
                
                if total_size <= self.max_size_bytes * 0.8:  # Leave 20% headroom
                    break
            
            logger.info(f"Cache cleanup completed. Current size: {total_size / (1024**2):.1f} MB")
    
    def get(self, cache_key: str) -> Optional[AudioData]:
        """Get cached audio data"""
        if not self.config.enabled:
            return None
        
        if cache_key not in self._index:
            return None
        
        entry = self._index[cache_key]
        
        # Check expiration
        if self._is_expired(entry):
            self._remove_entry(cache_key)
            return None
        
        # Load cached data
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, 'rb') as f:
                audio_data = pickle.load(f)
            
            # Update access time
            entry['last_access'] = time.time()
            self._save_index()
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Failed to load cached data {cache_key}: {e}")
            self._remove_entry(cache_key)
            return None
    
    def put(self, cache_key: str, audio_data: AudioData):
        """Cache audio data"""
        if not self.config.enabled:
            return
        
        cache_path = self._get_cache_path(cache_key)
        
        try:
            # Save audio data
            with open(cache_path, 'wb') as f:
                pickle.dump(audio_data, f)
            
            # Update index
            file_size = os.path.getsize(cache_path)
            self._index[cache_key] = {
                'timestamp': time.time(),
                'last_access': time.time(),
                'size': file_size
            }
            
            # Enforce size limit
            self._enforce_size_limit()
            self._save_index()
            
        except Exception as e:
            logger.error(f"Failed to cache data {cache_key}: {e}")
            try:
                if os.path.exists(cache_path):
                    os.unlink(cache_path)
            except:
                pass


class BatchMediaProcessor:
    """Efficient batch media processor with caching and parallel processing"""
    
    def __init__(self, config: MediaConfig, cache_config: Optional[CacheConfig] = None):
        self.config = config
        self.detector = MediaDetector()
        self.extractor = MediaExtractor(config)
        
        # Setup cache
        if cache_config is None:
            cache_config = CacheConfig()
        self.cache = MediaCache(cache_config)
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=config.ffmpeg_num_workers)
        
        # Statistics
        self.stats = {
            'processed': 0,
            'cache_hits': 0,
            'errors': 0,
            'processing_time': 0.0
        }
    
    def process_batch(self, media_items: List[MediaItem]) -> List[AudioData]:
        """
        Process a batch of media items
        
        Args:
            media_items: List of media items to process
            
        Returns:
            List of processed audio data
        """
        start_time = time.time()
        
        # Preprocess items
        filtered_items = self._preprocess_items(media_items)
        
        # Check cache
        cached_results, pending_items = self._check_cache(filtered_items)
        
        # Process pending items
        processed_results = self._process_pending_items(pending_items)
        
        # Combine results
        all_results = cached_results + processed_results
        
        # Update statistics
        self.stats['processed'] += len(media_items)
        self.stats['cache_hits'] += len(cached_results)
        self.stats['processing_time'] += time.time() - start_time
        
        logger.info(
            f"Batch processed: {len(media_items)} items, "
            f"cache hits: {len(cached_results)}, "
            f"errors: {len(media_items) - len(all_results)}"
        )
        
        return all_results
    
    def _preprocess_items(self, media_items: List[MediaItem]) -> List[MediaItem]:
        """Preprocess media items"""
        filtered_items = []
        
        for item in media_items:
            # Validate file size
            if not self.extractor.validate_file_size(item.file_bytes):
                logger.warning(f"Skipping oversized file: {item.item_id}")
                continue
            
            # Detect format
            media_info = self.detector.detect_format(item.file_bytes, item.filename)
            
            # Check if format is supported
            if not self.detector.is_supported(media_info):
                logger.warning(f"Unsupported format: {media_info.format_name} for {item.item_id}")
                continue
            
            # Store media info in metadata
            item.metadata['media_info'] = media_info
            filtered_items.append(item)
        
        return filtered_items
    
    def _check_cache(self, media_items: List[MediaItem]) -> Tuple[List[AudioData], List[MediaItem]]:
        """Check cache for existing processed data"""
        cached_results = []
        pending_items = []
        
        for item in media_items:
            cache_key = item.cache_key
            cached_data = self.cache.get(cache_key)
            
            if cached_data:
                cached_results.append(cached_data)
            else:
                pending_items.append(item)
        
        return cached_results, pending_items
    
    def _process_pending_items(self, media_items: List[MediaItem]) -> List[AudioData]:
        """Process items that are not in cache"""
        if not media_items:
            return []
        
        # Submit all tasks to thread pool
        future_to_item = {}
        for item in media_items:
            future = self.executor.submit(self._process_single_item, item)
            future_to_item[future] = item
        
        # Collect results
        results = []
        for future in as_completed(future_to_item):
            item = future_to_item[future]
            
            try:
                audio_data = future.result()
                results.append(audio_data)
                
                # Cache the result
                self.cache.put(item.cache_key, audio_data)
                
            except Exception as e:
                logger.error(f"Failed to process item {item.item_id}: {e}")
                self.stats['errors'] += 1
        
        return results
    
    def _process_single_item(self, media_item: MediaItem) -> AudioData:
        """Process a single media item"""
        try:
            # Get media info from metadata
            media_info = media_item.metadata['media_info']
            
            # Extract audio
            audio_bytes = self.extractor.extract_audio(media_item.file_bytes, media_info)
            
            # Extract metadata
            audio_metadata = self.extractor.extract_audio_metadata(
                media_item.file_bytes, media_info
            )
            
            # Create audio data object
            audio_data = AudioData(
                item_id=media_item.item_id,
                audio_bytes=audio_bytes,
                sample_rate=self.config.target_sample_rate,
                channels=self.config.target_channels,
                duration=audio_metadata.get('duration'),
                metadata=audio_metadata
            )
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Error processing media item {media_item.item_id}: {e}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        total_processed = self.stats['processed']
        if total_processed > 0:
            cache_hit_rate = self.stats['cache_hits'] / total_processed
            error_rate = self.stats['errors'] / total_processed
            avg_processing_time = self.stats['processing_time'] / total_processed
        else:
            cache_hit_rate = 0.0
            error_rate = 0.0
            avg_processing_time = 0.0
        
        return {
            'total_processed': total_processed,
            'cache_hits': self.stats['cache_hits'],
            'cache_hit_rate': cache_hit_rate,
            'errors': self.stats['errors'],
            'error_rate': error_rate,
            'avg_processing_time': avg_processing_time,
            'total_processing_time': self.stats['processing_time']
        }
    
    def reset_statistics(self):
        """Reset processing statistics"""
        self.stats = {
            'processed': 0,
            'cache_hits': 0,
            'errors': 0,
            'processing_time': 0.0
        }
    
    def cleanup(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)
        self.cache._save_index()


# Utility functions for integration with existing pipeline
def create_media_items_from_batch(batch: Dict[str, Any]) -> List[MediaItem]:
    """Create MediaItem list from pipeline batch"""
    media_items = []
    
    for item in batch.get('items', []):
        media_item = MediaItem(
            item_id=item.get('id', str(len(media_items))),
            file_bytes=item.get('file_bytes', b''),
            filename=item.get('filename'),
            metadata=item.get('metadata', {})
        )
        media_items.append(media_item)
    
    return media_items


def update_batch_with_audio_data(batch: Dict[str, Any], audio_data_list: List[AudioData]):
    """Update pipeline batch with processed audio data"""
    # Create mapping from item_id to audio data
    audio_map = {data.item_id: data for data in audio_data_list}
    
    # Update batch items
    for item in batch.get('items', []):
        item_id = item.get('id')
        if item_id in audio_map:
            audio_data = audio_map[item_id]
            item['audio_data'] = audio_data.audio_bytes
            item['audio_metadata'] = audio_data.metadata
            item['sample_rate'] = audio_data.sample_rate
            item['channels'] = audio_data.channels