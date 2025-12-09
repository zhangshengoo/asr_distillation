"""Test media indexing functionality"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch

from src.data.media_indexer import (
    ParquetIndexer, 
    MediaMetadata,
    MediaCache,
    MediaDataLoader
)


class TestParquetIndexer:
    """Test ParquetIndexer class"""
    
    def setup_method(self):
        """Setup test environment"""
        self.index_path = "./test_index"
        self.indexer = ParquetIndexer(self.index_path)
        
    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        if os.path.exists(self.index_path):
            shutil.rmtree(self.index_path)
    
    def test_create_index(self):
        """Test index creation"""
        media_files = [
            {
                'file_id': 'test_001',
                'oss_path': 'oss://bucket/test_001.wav',
                'media_type': 'audio',
                'duration': 10.5,
                'size_bytes': 1024000,
                'format': 'wav',
                'sample_rate': 16000
            },
            {
                'file_id': 'test_002',
                'oss_path': 'oss://bucket/test_002.mp4',
                'media_type': 'video',
                'duration': 15.2,
                'size_bytes': 1536000,
                'format': 'mp4',
                'video_codec': 'h264',
                'resolution': (1920, 1080)
            }
        ]
        
        df = self.indexer.create_index(media_files)
        
        assert len(df) == 2
        assert df.iloc[0]['file_id'] == 'test_001'
        assert df.iloc[1]['file_id'] == 'test_002'
        assert df.iloc[0]['media_type'] == 'audio'
        assert df.iloc[1]['media_type'] == 'video'
        assert df.iloc[0]['duration'] == 10.5
        assert df.iloc[1]['duration'] == 15.2
    
    def test_load_empty_index(self):
        """Test loading empty index"""
        df = self.indexer.load_index()
        assert df.empty
    
    def test_save_and_load_index(self):
        """Test saving and loading index"""
        media_files = [
            {
                'file_id': 'test_001',
                'oss_path': 'oss://bucket/test_001.wav',
                'media_type': 'audio',
                'duration': 10.5,
                'size_bytes': 1024000,
                'format': 'wav',
                'sample_rate': 16000
            }
        ]
        
        # Create and save index
        original_df = self.indexer.create_index(media_files)
        
        # Create new indexer instance and load
        new_indexer = ParquetIndexer(self.index_path)
        loaded_df = new_indexer.load_index()
        
        assert len(loaded_df) == len(original_df)
        assert loaded_df.iloc[0]['file_id'] == original_df.iloc[0]['file_id']


class TestMediaMetadata:
    """Test MediaMetadata class"""
    
    def test_audio_metadata_creation(self):
        """Test audio metadata creation"""
        metadata = MediaMetadata(
            file_id='test_001',
            oss_path='oss://bucket/test_001.wav',
            media_type='audio',
            duration=10.5,
            size_bytes=1024000,
            format='wav',
            sample_rate=16000,
            channels=2,
            bitrate=128000,
            audio_codec='pcm'
        )
        
        assert metadata.file_id == 'test_001'
        assert metadata.oss_path == 'oss://bucket/test_001.wav'
        assert metadata.media_type == 'audio'
        assert metadata.duration == 10.5
        assert metadata.size_bytes == 1024000
        assert metadata.format == 'wav'
        assert metadata.sample_rate == 16000
        assert metadata.channels == 2
        assert metadata.bitrate == 128000
        assert metadata.audio_codec == 'pcm'
        assert metadata.checksum is None
    
    def test_video_metadata_creation(self):
        """Test video metadata creation"""
        metadata = MediaMetadata(
            file_id='test_002',
            oss_path='oss://bucket/test_002.mp4',
            media_type='video',
            duration=30.0,
            size_bytes=5120000,
            format='mp4',
            video_codec='h264',
            resolution=(1920, 1080),
            frame_rate=30.0,
            audio_codec='aac'
        )
        
        assert metadata.file_id == 'test_002'
        assert metadata.media_type == 'video'
        assert metadata.video_codec == 'h264'
        assert metadata.resolution == (1920, 1080)
        assert metadata.frame_rate == 30.0
        assert metadata.audio_codec == 'aac'
    
    def test_metadata_with_checksum(self):
        """Test metadata with checksum"""
        metadata = MediaMetadata(
            file_id='test_001',
            oss_path='oss://bucket/test_001.wav',
            media_type='audio',
            duration=10.5,
            size_bytes=1024000,
            format='wav',
            checksum='abc123'
        )
        
        assert metadata.checksum == 'abc123'


class TestMediaCache:
    """Test MediaCache class"""
    
    def setup_method(self):
        """Setup test environment"""
        self.cache_dir = "./test_cache"
        self.cache = MediaCache(self.cache_dir, max_size_gb=0.001)  # 1MB
        
    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
    
    def test_cache_put_and_get_audio(self):
        """Test cache put and get for audio"""
        file_id = 'test_001'
        data = b'test audio data'
        
        # Put data in cache
        cached_path = self.cache.put(file_id, data, media_type='audio')
        assert cached_path.exists()
        assert cached_path.suffix == '.wav'
        
        # Get data from cache
        retrieved_path = self.cache.get(file_id, media_type='audio')
        assert retrieved_path == cached_path
        
        # Verify data
        with open(retrieved_path, 'rb') as f:
            cached_data = f.read()
        assert cached_data == data
    
    def test_cache_put_and_get_video(self):
        """Test cache put and get for video"""
        file_id = 'test_002'
        data = b'test video data'
        
        # Put data in cache
        cached_path = self.cache.put(file_id, data, media_type='video')
        assert cached_path.exists()
        assert cached_path.suffix == '.mp4'
        
        # Get data from cache
        retrieved_path = self.cache.get(file_id, media_type='video')
        assert retrieved_path == cached_path
        
        # Verify data
        with open(retrieved_path, 'rb') as f:
            cached_data = f.read()
        assert cached_data == data
    
    def test_cache_get_nonexistent(self):
        """Test getting non-existent file from cache"""
        result = self.cache.get('nonexistent_file')
        assert result is None
    
    def test_cache_clear(self):
        """Test cache clearing"""
        file_id = 'test_001'
        data = b'test audio data'
        
        # Put data in cache
        self.cache.put(file_id, data)
        
        # Clear cache
        self.cache.clear()
        
        # Verify cache is empty
        result = self.cache.get(file_id)
        assert result is None
    
    def test_cache_eviction(self):
        """Test cache eviction when full"""
        # Fill cache beyond limit
        large_data = b'x' * (2 * 1024 * 1024)  # 2MB > 1MB limit
        
        # Put large data
        self.cache.put('large_file', large_data)
        
        # Put another file to trigger eviction
        self.cache.put('another_file', b'small data')
        
        # Check that cache size is within limits
        stats = self.cache._get_cache_size()
        assert stats <= self.cache.max_size_bytes


class TestMediaDataLoader:
    """Test MediaDataLoader class"""
    
    def setup_method(self):
        """Setup test environment"""
        self.config = {
            'index_path': './test_index',
            'cache_dir': './test_cache',
            'cache_size_gb': 0.001
        }
        self.loader = MediaDataLoader(self.config)
        
    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        for path in ['./test_index', './test_cache']:
            if os.path.exists(path):
                shutil.rmtree(path)
    
    def test_initialization(self):
        """Test loader initialization"""
        assert self.loader.index_path == './test_index'
        assert self.loader.cache_dir == './test_cache'
        assert self.loader.cache_size_gb == 0.001
        assert self.loader.indexer is not None
        assert self.loader.cache is not None
    
    def test_load_empty_index(self):
        """Test loading empty index"""
        df = self.loader.load_index()
        assert df.empty
    
    def test_cache_methods(self):
        """Test cache methods"""
        file_id = 'test_001'
        data = b'test audio data'
        
        # Cache data
        cached_path = self.loader.cache_media(file_id, data, media_type='audio')
        assert cached_path.exists()
        
        # Get cached data
        retrieved_path = self.loader.get_cached_media(file_id, media_type='audio')
        assert retrieved_path == cached_path
        
        # Verify data
        with open(retrieved_path, 'rb') as f:
            cached_data = f.read()
        assert cached_data == data
    
    def test_clear_cache(self):
        """Test cache clearing"""
        file_id = 'test_001'
        data = b'test audio data'
        
        # Cache data
        self.loader.cache_media(file_id, data)
        
        # Clear cache
        self.loader.clear_cache()
        
        # Verify cache is empty
        result = self.loader.get_cached_media(file_id)
        assert result is None
    
    def test_get_cache_stats(self):
        """Test getting cache statistics"""
        file_id = 'test_001'
        data = b'test audio data'
        
        # Cache data
        self.loader.cache_media(file_id, data)
        
        # Get stats
        stats = self.loader.get_cache_stats()
        
        assert 'num_files' in stats
        assert 'total_size_mb' in stats
        assert 'total_size_gb' in stats
        assert 'max_size_gb' in stats
        assert 'utilization' in stats
        assert stats['num_files'] == 1
    
    def test_filter_by_media_type(self):
        """Test filtering by media type"""
        # Create test data
        media_files = [
            {
                'file_id': 'audio_001',
                'oss_path': 'oss://bucket/audio_001.wav',
                'media_type': 'audio',
                'duration': 10.5,
                'size_bytes': 1024000,
                'format': 'wav',
                'sample_rate': 16000
            },
            {
                'file_id': 'video_001',
                'oss_path': 'oss://bucket/video_001.mp4',
                'media_type': 'video',
                'duration': 30.0,
                'size_bytes': 5120000,
                'format': 'mp4',
                'video_codec': 'h264'
            }
        ]
        
        # Create index
        self.loader.create_index(media_files)
        
        # Test filtering
        audio_df = self.loader.get_audio_files()
        video_df = self.loader.get_video_files()
        
        assert len(audio_df) == 1
        assert len(video_df) == 1
        assert audio_df.iloc[0]['media_type'] == 'audio'
        assert video_df.iloc[0]['media_type'] == 'video'