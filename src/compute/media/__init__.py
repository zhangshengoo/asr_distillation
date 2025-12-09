"""Media processing module for handling various audio and video formats"""

from .media_detector import MediaDetector, MediaType, MediaInfo
from .media_extractor import MediaExtractor, MediaConfig
from .batch_media_processor import (
    BatchMediaProcessor, 
    MediaItem, 
    AudioData, 
    CacheConfig,
    create_media_items_from_batch,
    update_batch_with_audio_data
)

__all__ = [
    # Core classes
    'MediaDetector',
    'MediaExtractor', 
    'BatchMediaProcessor',
    
    # Data structures
    'MediaType',
    'MediaInfo',
    'MediaConfig',
    'MediaItem',
    'AudioData',
    'CacheConfig',
    
    # Utility functions
    'create_media_items_from_batch',
    'update_batch_with_audio_data'
]