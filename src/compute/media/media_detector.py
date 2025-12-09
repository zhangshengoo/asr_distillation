"""Media format detector for audio and video files"""

import os
import struct
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from loguru import logger


class MediaType(Enum):
    """Media type enumeration"""
    AUDIO = "audio"
    VIDEO = "video"
    UNKNOWN = "unknown"


@dataclass
class MediaInfo:
    """Media information container"""
    media_type: MediaType
    format_name: str
    extension: str
    mime_type: str
    duration: Optional[float] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    bitrate: Optional[int] = None
    video_codec: Optional[str] = None
    audio_codec: Optional[str] = None
    resolution: Optional[Tuple[int, int]] = None


class MediaDetector:
    """Media format detector using file signatures and metadata"""
    
    # File signatures for common formats
    FILE_SIGNATURES = {
        # Audio formats
        b'ID3': ('mp3', 'audio/mpeg'),
        b'\xff\xfb': ('mp3', 'audio/mpeg'),
        b'\xff\xf3': ('mp3', 'audio/mpeg'),
        b'\xff\xf2': ('mp3', 'audio/mpeg'),
        b'RIFF': ('wav', 'audio/wav'),
        b'fLaC': ('flac', 'audio/flac'),
        b'OggS': ('ogg', 'audio/ogg'),
        b'\x00\x00\x00\x18ftypmp42': ('mp4', 'video/mp4'),
        b'\x00\x00\x00\x1cftypisom': ('mp4', 'video/mp4'),
        b'ftyp': ('mp4', 'video/mp4'),
        
        # Video formats
        b'\x1a\x45\xdf\xa3': ('webm', 'video/webm'),
        b'RIFF....AVI ': ('avi', 'video/avi'),
        b'\x00\x00\x00\x14ftypqt  ': ('mov', 'video/quicktime'),
    }
    
    # Extension mappings
    EXTENSION_MAP = {
        # Audio
        'mp3': ('audio', 'audio/mpeg'),
        'wav': ('audio', 'audio/wav'),
        'flac': ('audio', 'audio/flac'),
        'aac': ('audio', 'audio/aac'),
        'm4a': ('audio', 'audio/mp4'),
        'ogg': ('audio', 'audio/ogg'),
        'wma': ('audio', 'audio/x-ms-wma'),
        
        # Video
        'mp4': ('video', 'video/mp4'),
        'avi': ('video', 'video/avi'),
        'mov': ('video', 'video/quicktime'),
        'mkv': ('video', 'video/x-matroska'),
        'webm': ('video', 'video/webm'),
        'flv': ('video', 'video/x-flv'),
        '3gp': ('video', 'video/3gpp'),
    }
    
    def __init__(self):
        self._supported_audio_formats = set()
        self._supported_video_formats = set()
        self._initialize_supported_formats()
    
    def _initialize_supported_formats(self):
        """Initialize supported format lists"""
        for ext, (media_type, _) in self.EXTENSION_MAP.items():
            if media_type == 'audio':
                self._supported_audio_formats.add(ext)
            elif media_type == 'video':
                self._supported_video_formats.add(ext)
    
    def detect_format(self, file_bytes: bytes, filename: Optional[str] = None) -> MediaInfo:
        """
        Detect media format from file bytes and optional filename
        
        Args:
            file_bytes: Raw file bytes
            filename: Optional filename for extension-based detection
            
        Returns:
            MediaInfo object with detected format information
        """
        # Try detection by file signature first
        format_info = self._detect_by_signature(file_bytes)
        
        # Fallback to extension detection
        if format_info is None and filename:
            format_info = self._detect_by_extension(filename)
        
        # Default to unknown if detection fails
        if format_info is None:
            logger.warning(f"Could not detect media format for file: {filename}")
            return MediaInfo(
                media_type=MediaType.UNKNOWN,
                format_name="unknown",
                extension="",
                mime_type="application/octet-stream"
            )
        
        media_type, format_name, mime_type = format_info
        
        # Determine media type
        if media_type == 'audio':
            media_type_enum = MediaType.AUDIO
        elif media_type == 'video':
            media_type_enum = MediaType.VIDEO
        else:
            media_type_enum = MediaType.UNKNOWN
        
        # Extract extension from filename if available
        extension = ""
        if filename:
            extension = os.path.splitext(filename)[1].lower().lstrip('.')
        
        return MediaInfo(
            media_type=media_type_enum,
            format_name=format_name,
            extension=extension,
            mime_type=mime_type
        )
    
    def _detect_by_signature(self, file_bytes: bytes) -> Optional[Tuple[str, str, str]]:
        """Detect format by file signature"""
        # Check for common signatures
        for signature, (format_name, mime_type) in self.FILE_SIGNATURES.items():
            if file_bytes.startswith(signature):
                # Determine media type from mime type
                if mime_type.startswith('audio/'):
                    media_type = 'audio'
                elif mime_type.startswith('video/'):
                    media_type = 'video'
                else:
                    media_type = 'unknown'
                
                return (media_type, format_name, mime_type)
        
        # Special case for WAV files (RIFF format)
        if file_bytes.startswith(b'RIFF') and len(file_bytes) > 12:
            # Check for WAVE format identifier
            if file_bytes[8:12] == b'WAVE':
                return ('audio', 'wav', 'audio/wav')
        
        return None
    
    def _detect_by_extension(self, filename: str) -> Optional[Tuple[str, str, str]]:
        """Detect format by file extension"""
        extension = os.path.splitext(filename)[1].lower().lstrip('.')
        
        if extension in self.EXTENSION_MAP:
            media_type, mime_type = self.EXTENSION_MAP[extension]
            return (media_type, extension, mime_type)
        
        return None
    
    def is_audio(self, media_info: MediaInfo) -> bool:
        """Check if media is audio"""
        return media_info.media_type == MediaType.AUDIO
    
    def is_video(self, media_info: MediaInfo) -> bool:
        """Check if media is video"""
        return media_info.media_type == MediaType.VIDEO
    
    def is_supported(self, media_info: MediaInfo) -> bool:
        """Check if media format is supported"""
        if media_info.media_type == MediaType.AUDIO:
            return media_info.extension in self._supported_audio_formats
        elif media_info.media_type == MediaType.VIDEO:
            return media_info.extension in self._supported_video_formats
        return False
    
    def get_supported_audio_formats(self) -> List[str]:
        """Get list of supported audio formats"""
        return sorted(list(self._supported_audio_formats))
    
    def get_supported_video_formats(self) -> List[str]:
        """Get list of supported video formats"""
        return sorted(list(self._supported_video_formats))
    
    def extract_metadata(self, file_bytes: bytes, media_info: MediaInfo) -> MediaInfo:
        """
        Extract additional metadata from media file
        
        This is a placeholder for more sophisticated metadata extraction.
        In production, you might use ffmpeg-python or similar libraries.
        """
        # For now, return the original media_info
        # This can be extended to extract duration, bitrate, etc.
        return media_info