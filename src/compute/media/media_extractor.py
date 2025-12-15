"""Media extractor for audio and video files - 同步版本"""

import os
import subprocess
import tempfile
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import time

import ffmpeg

from .media_detector import MediaDetector, MediaInfo, MediaType
from src.config.manager import MediaConfig


class MediaExtractor:
    """Unified media extraction interface using FFmpeg - 同步版本"""
    
    def __init__(self, config: MediaConfig):
        self.config = config
        self.detector = MediaDetector()
        # 移除线程池，使用同步处理
        
        # FFmpeg quality presets
        self.quality_settings = {
            "low": {"audio_bitrate": "64k", "compression": 6},
            "medium": {"audio_bitrate": "128k", "compression": 3},
            "high": {"audio_bitrate": "192k", "compression": 0}
        }
    
    def extract_audio(self, media_bytes: bytes, media_info: MediaInfo) -> bytes:
        """
        Extract audio from media file
        
        Args:
            media_bytes: Raw media file bytes
            media_info: Media information from detector
            
        Returns:
            Extracted audio bytes in target format
        """
        if media_info.media_type == MediaType.AUDIO:
            return self._convert_audio_format(media_bytes, media_info)
        elif media_info.media_type == MediaType.VIDEO:
            return self._extract_from_video(media_bytes, media_info)
        else:
            raise ValueError(f"Unsupported media type: {media_info.media_type}")
    
    def _convert_audio_format(self, audio_bytes: bytes, media_info: MediaInfo) -> bytes:
        """Convert audio to target format"""
        # Create temporary files
        with tempfile.NamedTemporaryFile(
            suffix=f".{media_info.extension}", 
            delete=False
        ) as input_file, tempfile.NamedTemporaryFile(
            suffix=f".{self.config.target_format}", 
            delete=False
        ) as output_file:
            
            input_path = input_file.name
            output_path = output_file.name
            
            # Write input data
            input_file.write(audio_bytes)
            input_file.flush()
            
            # Build FFmpeg command
            quality = self.quality_settings[self.config.ffmpeg_quality]
            
            cmd = [
                'ffmpeg', '-y',  # Overwrite output file
                '-i', input_path,
                '-ar', str(self.config.target_sample_rate),
                '-ac', str(self.config.target_channels),
                '-f', self.config.target_format,
                '-ab', quality["audio_bitrate"],
                '-loglevel', 'error',
                output_path
            ]
            
            # Run FFmpeg
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.ffmpeg_timeout
            )
            
            if process.returncode != 0:
                error_msg = process.stderr.strip() or process.stdout.strip()
                raise subprocess.CalledProcessError(
                    process.returncode, cmd, error_msg
                )
            
            # Read output
            with open(output_path, 'rb') as f:
                output_bytes = f.read()
            
            return output_bytes
    
    def _extract_from_video(self, video_bytes: bytes, media_info: MediaInfo) -> bytes:
        """Extract audio from video file"""
        # Create temporary files
        with tempfile.NamedTemporaryFile(
            suffix=f".{media_info.extension}", 
            delete=False
        ) as input_file, tempfile.NamedTemporaryFile(
            suffix=f".{self.config.target_format}", 
            delete=False
        ) as output_file:
            
            input_path = input_file.name
            output_path = output_file.name
            
            # Write input data
            input_file.write(video_bytes)
            input_file.flush()
            
            # Build FFmpeg command for audio extraction
            quality = self.quality_settings[self.config.ffmpeg_quality]
            
            cmd = [
                'ffmpeg', '-y',
                '-i', input_path,
                '-vn',  # No video
                '-ar', str(self.config.target_sample_rate),
                '-ac', str(self.config.target_channels),
                '-f', self.config.target_format,
                '-ab', quality["audio_bitrate"],
                '-loglevel', 'error',
                output_path
            ]
            
            # Run FFmpeg
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.ffmpeg_timeout
            )
            
            if process.returncode != 0:
                error_msg = process.stderr.strip() or process.stdout.strip()
                raise subprocess.CalledProcessError(
                    process.returncode, cmd, error_msg
                )
            
            # Read output
            with open(output_path, 'rb') as f:
                output_bytes = f.read()
            
            return output_bytes
    
    def extract_audio_metadata(self, media_bytes: bytes, media_info: MediaInfo) -> Dict[str, Any]:
        """Extract metadata from media file"""
        with tempfile.NamedTemporaryFile(
            suffix=f".{media_info.extension}", 
            delete=False
        ) as input_file:
            
            input_path = input_file.name
            input_file.write(media_bytes)
            input_file.flush()
            
            # Use ffprobe to get metadata
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', input_path
            ]
            
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if process.returncode == 0:
                import json
                metadata = json.loads(process.stdout)
                result = self._parse_metadata(metadata, media_info)
            else:
                result = {}
        
        # Cleanup temporary file
        try:
            os.unlink(input_path)
        except:
            pass
            
        return result
    
    def _parse_metadata(self, ffprobe_output: Dict[str, Any], media_info: MediaInfo) -> Dict[str, Any]:
        """Parse ffprobe output into structured metadata"""
        metadata = {
            'format_name': media_info.format_name,
            'media_type': media_info.media_type.value,
            'duration': None,
            'bitrate': None,
            'sample_rate': None,
            'channels': None,
            'audio_codec': None,
            'video_codec': None,
            'resolution': None
        }
        
        # Extract format information
        if 'format' in ffprobe_output:
            format_info = ffprobe_output['format']
            metadata['duration'] = float(format_info.get('duration', 0)) or None
            metadata['bitrate'] = int(format_info.get('bit_rate', 0)) or None
        
        # Extract stream information
        if 'streams' in ffprobe_output:
            for stream in ffprobe_output['streams']:
                codec_type = stream.get('codec_type')
                
                if codec_type == 'audio':
                    metadata['sample_rate'] = int(stream.get('sample_rate', 0)) or None
                    metadata['channels'] = int(stream.get('channels', 0)) or None
                    metadata['audio_codec'] = stream.get('codec_name')
                    
                elif codec_type == 'video':
                    metadata['video_codec'] = stream.get('codec_name')
                    width = stream.get('width')
                    height = stream.get('height')
                    if width and height:
                        metadata['resolution'] = (width, height)
        
        return metadata
    
    def validate_file_size(self, file_bytes: bytes) -> bool:
        """Validate file size against configured limits"""
        size_mb = len(file_bytes) / (1024 * 1024)
        return size_mb <= self.config.max_file_size_mb
    
    def extract_audio_sync(self, media_bytes: bytes, media_info: MediaInfo) -> bytes:
        """Synchronous version of extract_audio - 直接调用extract_audio方法"""
        return self.extract_audio(media_bytes, media_info)
    
    def cleanup(self):
        """Cleanup resources - 同步版本无需特殊清理"""
        pass