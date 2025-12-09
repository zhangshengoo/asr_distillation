"""Tests for batch media processor"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from src.compute.media import (
    BatchMediaProcessor, 
    MediaConfig, 
    CacheConfig, 
    MediaItem, 
    AudioData,
    MediaInfo,
    MediaType
)


class TestBatchMediaProcessor:
    """Test cases for BatchMediaProcessor"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.media_config = MediaConfig(
            ffmpeg_num_workers=2,
            ffmpeg_timeout=60,
            target_sample_rate=16000,
            target_channels=1,
            max_file_size_mb=500
        )
        self.cache_config = CacheConfig(
            enabled=True,
            cache_dir=tempfile.mkdtemp(),
            max_size_gb=1.0,
            ttl_hours=24
        )
        self.processor = BatchMediaProcessor(self.media_config, self.cache_config)
    
    def teardown_method(self):
        """Cleanup after tests"""
        self.processor.cleanup()
        # Clean up cache directory
        import shutil
        if os.path.exists(self.cache_config.cache_dir):
            shutil.rmtree(self.cache_config.cache_dir)
    
    @patch('src.compute.media.MediaExtractor')
    @patch('src.compute.media.MediaDetector')
    def test_process_batch_success(self, mock_detector_class, mock_extractor_class):
        """Test successful batch processing"""
        # Mock detector
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector
        mock_detector.detect_format.return_value = MediaInfo(
            media_type=MediaType.AUDIO,
            format_name="mp3",
            extension="mp3",
            mime_type="audio/mpeg"
        )
        mock_detector.is_supported.return_value = True
        
        # Mock extractor
        mock_extractor = Mock()
        mock_extractor_class.return_value = mock_extractor
        mock_extractor.validate_file_size.return_value = True
        mock_extractor.extract_audio.return_value = b'processed_audio_data'
        mock_extractor.extract_audio_metadata.return_value = {
            'duration': 10.5,
            'sample_rate': 16000,
            'channels': 1
        }
        
        # Create test media items
        media_items = [
            MediaItem(
                item_id="test1",
                file_bytes=b'fake_mp3_data',
                filename="test1.mp3"
            ),
            MediaItem(
                item_id="test2",
                file_bytes=b'fake_wav_data',
                filename="test2.wav"
            )
        ]
        
        # Process batch
        results = self.processor.process_batch(media_items)
        
        # Verify results
        assert len(results) == 2
        assert all(isinstance(r, AudioData) for r in results)
        assert results[0].item_id == "test1"
        assert results[1].item_id == "test2"
        assert results[0].audio_bytes == b'processed_audio_data'
        assert results[1].audio_bytes == b'processed_audio_data'
    
    @patch('src.compute.media.MediaExtractor')
    @patch('src.compute.media.MediaDetector')
    def test_process_batch_with_cache(self, mock_detector_class, mock_extractor_class):
        """Test batch processing with cache hits"""
        # Mock detector
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector
        mock_detector.detect_format.return_value = MediaInfo(
            media_type=MediaType.AUDIO,
            format_name="mp3",
            extension="mp3",
            mime_type="audio/mpeg"
        )
        mock_detector.is_supported.return_value = True
        
        # Mock extractor
        mock_extractor = Mock()
        mock_extractor_class.return_value = mock_extractor
        mock_extractor.validate_file_size.return_value = True
        mock_extractor.extract_audio.return_value = b'processed_audio_data'
        mock_extractor.extract_audio_metadata.return_value = {}
        
        # Create test media item
        media_item = MediaItem(
            item_id="test_cached",
            file_bytes=b'fake_mp3_data',
            filename="test.mp3"
        )
        
        # First processing - should cache result
        results1 = self.processor.process_batch([media_item])
        assert len(results1) == 1
        
        # Second processing - should use cache
        results2 = self.processor.process_batch([media_item])
        assert len(results2) == 1
        
        # Extractor should only be called once due to cache
        assert mock_extractor.extract_audio.call_count == 1
    
    @patch('src.compute.media.MediaExtractor')
    @patch('src.compute.media.MediaDetector')
    def test_process_batch_unsupported_format(self, mock_detector_class, mock_extractor_class):
        """Test processing with unsupported format"""
        # Mock detector
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector
        mock_detector.detect_format.return_value = MediaInfo(
            media_type=MediaType.UNKNOWN,
            format_name="unknown",
            extension="xyz",
            mime_type="application/octet-stream"
        )
        mock_detector.is_supported.return_value = False
        
        # Create test media item with unsupported format
        media_item = MediaItem(
            item_id="test_unsupported",
            file_bytes=b'fake_data',
            filename="test.xyz"
        )
        
        # Process batch
        results = self.processor.process_batch([media_item])
        
        # Should return empty results for unsupported format
        assert len(results) == 0
    
    @patch('src.compute.media.MediaExtractor')
    @patch('src.compute.media.MediaDetector')
    def test_process_batch_oversized_file(self, mock_detector_class, mock_extractor_class):
        """Test processing with oversized file"""
        # Mock detector
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector
        mock_detector.detect_format.return_value = MediaInfo(
            media_type=MediaType.AUDIO,
            format_name="mp3",
            extension="mp3",
            mime_type="audio/mpeg"
        )
        mock_detector.is_supported.return_value = True
        
        # Mock extractor - file too large
        mock_extractor = Mock()
        mock_extractor_class.return_value = mock_extractor
        mock_extractor.validate_file_size.return_value = False
        
        # Create test media item with oversized file
        media_item = MediaItem(
            item_id="test_oversized",
            file_bytes=b'x' * (600 * 1024 * 1024),  # 600MB
            filename="test.mp3"
        )
        
        # Process batch
        results = self.processor.process_batch([media_item])
        
        # Should return empty results for oversized file
        assert len(results) == 0
    
    def test_get_statistics(self):
        """Test statistics collection"""
        # Initial stats should be zero
        stats = self.processor.get_statistics()
        assert stats['total_processed'] == 0
        assert stats['cache_hits'] == 0
        assert stats['errors'] == 0
        assert stats['cache_hit_rate'] == 0.0
        assert stats['error_rate'] == 0.0
    
    def test_reset_statistics(self):
        """Test statistics reset"""
        # Process some items to generate stats
        with patch('src.compute.media.MediaExtractor') as mock_extractor_class, \
             patch('src.compute.media.MediaDetector') as mock_detector_class:
            
            # Mock detector and extractor
            mock_detector = Mock()
            mock_detector_class.return_value = mock_detector
            mock_detector.detect_format.return_value = MediaInfo(
                media_type=MediaType.AUDIO,
                format_name="mp3",
                extension="mp3",
                mime_type="audio/mpeg"
            )
            mock_detector.is_supported.return_value = True
            
            mock_extractor = Mock()
            mock_extractor_class.return_value = mock_extractor
            mock_extractor.validate_file_size.return_value = True
            mock_extractor.extract_audio.return_value = b'processed_audio_data'
            mock_extractor.extract_audio_metadata.return_value = {}
            
            # Process batch
            media_item = MediaItem(
                item_id="test",
                file_bytes=b'fake_data',
                filename="test.mp3"
            )
            self.processor.process_batch([media_item])
            
            # Verify stats are non-zero
            stats = self.processor.get_statistics()
            assert stats['total_processed'] == 1
            
            # Reset stats
            self.processor.reset_statistics()
            
            # Verify stats are reset
            stats = self.processor.get_statistics()
            assert stats['total_processed'] == 0
    
    def test_media_item_cache_key(self):
        """Test MediaItem cache key generation"""
        # Same content should generate same cache key
        item1 = MediaItem(
            item_id="test1",
            file_bytes=b'same_content',
            filename="file1.mp3"
        )
        
        item2 = MediaItem(
            item_id="test2",
            file_bytes=b'same_content',
            filename="file2.mp3"
        )
        
        # Cache keys should be different due to different item_id
        assert item1.cache_key != item2.cache_key
        
        # Same item_id and content should generate same key
        item3 = MediaItem(
            item_id="test1",
            file_bytes=b'same_content',
            filename="file3.mp3"
        )
        
        assert item1.cache_key == item3.cache_key
    
    def test_audio_data_creation(self):
        """Test AudioData object creation"""
        audio_data = AudioData(
            item_id="test",
            audio_bytes=b'audio_data',
            sample_rate=16000,
            channels=1,
            duration=10.5,
            metadata={'format': 'wav'}
        )
        
        assert audio_data.item_id == "test"
        assert audio_data.audio_bytes == b'audio_data'
        assert audio_data.sample_rate == 16000
        assert audio_data.channels == 1
        assert audio_data.duration == 10.5
        assert audio_data.metadata['format'] == 'wav'