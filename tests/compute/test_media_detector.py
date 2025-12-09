"""Tests for media detector"""

import pytest
from src.compute.media import MediaDetector, MediaType, MediaInfo


class TestMediaDetector:
    """Test cases for MediaDetector"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.detector = MediaDetector()
    
    def test_detect_mp3_by_signature(self):
        """Test MP3 detection by file signature"""
        # MP3 with ID3 tag
        mp3_bytes = b'ID3\x04\x00\x00\x00\x00\x00\x00' + b'\x00' * 100
        media_info = self.detector.detect_format(mp3_bytes, "test.mp3")
        
        assert media_info.media_type == MediaType.AUDIO
        assert media_info.format_name == "mp3"
        assert media_info.extension == "mp3"
        assert self.detector.is_audio(media_info)
        assert not self.detector.is_video(media_info)
    
    def test_detect_wav_by_signature(self):
        """Test WAV detection by file signature"""
        wav_bytes = b'RIFF\x24\x08\x00\x00WAVE' + b'\x00' * 100
        media_info = self.detector.detect_format(wav_bytes, "test.wav")
        
        assert media_info.media_type == MediaType.AUDIO
        assert media_info.format_name == "wav"
        assert media_info.extension == "wav"
    
    def test_detect_mp4_by_signature(self):
        """Test MP4 detection by file signature"""
        mp4_bytes = b'\x00\x00\x00\x18ftypmp42' + b'\x00' * 100
        media_info = self.detector.detect_format(mp4_bytes, "test.mp4")
        
        assert media_info.media_type == MediaType.VIDEO
        assert media_info.format_name == "mp4"
        assert media_info.extension == "mp4"
        assert not self.detector.is_audio(media_info)
        assert self.detector.is_video(media_info)
    
    def test_detect_by_extension_fallback(self):
        """Test format detection by extension when signature fails"""
        # Invalid signature but valid extension
        invalid_bytes = b'INVALID_SIGNATURE' + b'\x00' * 100
        media_info = self.detector.detect_format(invalid_bytes, "test.flac")
        
        assert media_info.media_type == MediaType.AUDIO
        assert media_info.format_name == "flac"
        assert media_info.extension == "flac"
    
    def test_detect_unknown_format(self):
        """Test detection of unknown format"""
        unknown_bytes = b'UNKNOWN_FORMAT_BYTES' + b'\x00' * 100
        media_info = self.detector.detect_format(unknown_bytes)
        
        assert media_info.media_type == MediaType.UNKNOWN
        assert media_info.format_name == "unknown"
        assert media_info.extension == ""
    
    def test_supported_formats(self):
        """Test supported format lists"""
        audio_formats = self.detector.get_supported_audio_formats()
        video_formats = self.detector.get_supported_video_formats()
        
        assert "mp3" in audio_formats
        assert "wav" in audio_formats
        assert "flac" in audio_formats
        
        assert "mp4" in video_formats
        assert "avi" in video_formats
        assert "mov" in video_formats
    
    def test_is_supported(self):
        """Test format support checking"""
        # Supported audio
        audio_info = MediaInfo(
            media_type=MediaType.AUDIO,
            format_name="mp3",
            extension="mp3",
            mime_type="audio/mpeg"
        )
        assert self.detector.is_supported(audio_info)
        
        # Supported video
        video_info = MediaInfo(
            media_type=MediaType.VIDEO,
            format_name="mp4",
            extension="mp4",
            mime_type="video/mp4"
        )
        assert self.detector.is_supported(video_info)
        
        # Unknown format
        unknown_info = MediaInfo(
            media_type=MediaType.UNKNOWN,
            format_name="unknown",
            extension="xyz",
            mime_type="application/octet-stream"
        )
        assert not self.detector.is_supported(unknown_info)