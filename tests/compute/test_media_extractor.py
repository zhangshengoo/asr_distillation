"""Tests for media extractor"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.compute.media import MediaExtractor, MediaConfig, MediaInfo, MediaType


class TestMediaExtractor:
    """Test cases for MediaExtractor"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.config = MediaConfig(
            ffmpeg_num_workers=2,
            ffmpeg_timeout=60,
            target_sample_rate=16000,
            target_channels=1
        )
        self.extractor = MediaExtractor(self.config)
    
    @patch('subprocess.run')
    def test_extract_audio_from_mp3(self, mock_run):
        """Test extracting audio from MP3 file"""
        # Mock successful FFmpeg execution
        mock_run.return_value = Mock(
            returncode=0,
            stderr='',
            stdout=''
        )
        
        # Mock file operations
        with patch('tempfile.NamedTemporaryFile') as mock_temp:
            mock_temp.return_value.__enter__.return_value.name = '/tmp/test.mp3'
            mock_temp.return_value.__enter__.return_value.write = Mock()
            mock_temp.return_value.__enter__.return_value.flush = Mock()
            
            with patch('builtins.open', create=True) as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = b'fake_audio_data'
                
                # Test extraction
                media_info = MediaInfo(
                    media_type=MediaType.AUDIO,
                    format_name="mp3",
                    extension="mp3",
                    mime_type="audio/mpeg"
                )
                
                result = self.extractor.extract_audio(b'fake_mp3_data', media_info)
                
                # Verify FFmpeg was called with correct parameters
                mock_run.assert_called_once()
                args = mock_run.call_args[0][0]
                assert 'ffmpeg' in args
                assert '-ar' in args
                assert str(self.config.target_sample_rate) in args
                assert '-ac' in args
                assert str(self.config.target_channels) in args
    
    @patch('subprocess.run')
    def test_extract_audio_from_video(self, mock_run):
        """Test extracting audio from video file"""
        # Mock successful FFmpeg execution
        mock_run.return_value = Mock(
            returncode=0,
            stderr='',
            stdout=''
        )
        
        # Mock file operations
        with patch('tempfile.NamedTemporaryFile') as mock_temp:
            mock_temp.return_value.__enter__.return_value.name = '/tmp/test.mp4'
            mock_temp.return_value.__enter__.return_value.write = Mock()
            mock_temp.return_value.__enter__.return_value.flush = Mock()
            
            with patch('builtins.open', create=True) as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = b'fake_audio_data'
                
                # Test extraction
                media_info = MediaInfo(
                    media_type=MediaType.VIDEO,
                    format_name="mp4",
                    extension="mp4",
                    mime_type="video/mp4"
                )
                
                result = self.extractor.extract_audio(b'fake_mp4_data', media_info)
                
                # Verify FFmpeg was called with -vn flag for video
                mock_run.assert_called_once()
                args = mock_run.call_args[0][0]
                assert 'ffmpeg' in args
                assert '-vn' in args  # No video flag
    
    @patch('subprocess.run')
    def test_extract_audio_ffmpeg_error(self, mock_run):
        """Test handling FFmpeg errors"""
        # Mock FFmpeg error
        mock_run.return_value = Mock(
            returncode=1,
            stderr='FFmpeg error: Invalid data found',
            stdout=''
        )
        
        media_info = MediaInfo(
            media_type=MediaType.AUDIO,
            format_name="mp3",
            extension="mp3",
            mime_type="audio/mpeg"
        )
        
        # Should raise exception on FFmpeg error
        with pytest.raises(Exception):
            self.extractor.extract_audio(b'invalid_data', media_info)
    
    def test_validate_file_size(self):
        """Test file size validation"""
        # Valid size
        valid_data = b'x' * (100 * 1024 * 1024)  # 100MB
        assert self.extractor.validate_file_size(valid_data) is True
        
        # Invalid size (too large)
        invalid_data = b'x' * (600 * 1024 * 1024)  # 600MB
        assert self.extractor.validate_file_size(invalid_data) is False
    
    @patch('subprocess.run')
    def test_extract_audio_metadata(self, mock_run):
        """Test extracting metadata from media file"""
        # Mock successful ffprobe execution
        mock_run.return_value = Mock(
            returncode=0,
            stderr='',
            stdout='''{
                "format": {
                    "duration": "120.5",
                    "bit_rate": "128000"
                },
                "streams": [
                    {
                        "codec_type": "audio",
                        "sample_rate": "44100",
                        "channels": 2,
                        "codec_name": "mp3"
                    }
                ]
            }'''
        )
        
        media_info = MediaInfo(
            media_type=MediaType.AUDIO,
            format_name="mp3",
            extension="mp3",
            mime_type="audio/mpeg"
        )
        
        metadata = self.extractor.extract_audio_metadata(b'fake_data', media_info)
        
        assert metadata['duration'] == 120.5
        assert metadata['bitrate'] == 128000
        assert metadata['sample_rate'] == 44100
        assert metadata['channels'] == 2
        assert metadata['audio_codec'] == "mp3"
    
    def test_media_config_defaults(self):
        """Test MediaConfig default values"""
        config = MediaConfig()
        
        assert "mp3" in config.audio_formats
        assert "wav" in config.audio_formats
        assert "mp4" in config.video_formats
        assert "avi" in config.video_formats
        assert config.ffmpeg_num_workers == 4
        assert config.ffmpeg_timeout == 300
        assert config.target_sample_rate == 16000
        assert config.target_channels == 1