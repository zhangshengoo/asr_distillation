"""CPU audio preprocessing workers with multimedia support"""

import os
import io
import tempfile
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torchaudio
import numpy as np
import logging

from ..scheduling.pipeline import PipelineStage
from ..common import BatchData, SourceItem, RawAudioItem, TensorItem, SegmentItem
from .media import (
    MediaDetector, 
    MediaType, 
    MediaConfig, 
    MediaExtractor, 
    BatchMediaProcessor, 
    MediaItem, 
    AudioData, 
    CacheConfig, 
    create_media_items_from_batch, 
    update_batch_with_audio_data
)


from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class AudioConfig:
    """Audio processing configuration"""
    target_sample_rate: int = 16000
    max_duration: float = 30.0  # seconds
    normalize: bool = True
    remove_silence: bool = False
    audio_format: str = 'wav'
    features: Dict[str, Any] = field(default_factory=lambda: {
        'feature_type': 'mel_spectrogram',
        'sample_rate': 16000,
        'n_fft': 400,
        'hop_length': 160,
        'n_mels': 80
    })


class AudioPreprocessor:
    """Audio preprocessing utilities"""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        
    def load_audio_from_bytes(self, audio_bytes: bytes) -> Tuple[torch.Tensor, int]:
        """Load audio tensor from bytes"""
        # Create temporary file to load with torchaudio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file.flush()
            
            # Load audio
            waveform, sample_rate = torchaudio.load(tmp_file.name)
            
            # Cleanup temp file
            os.unlink(tmp_file.name)
            
            return waveform, sample_rate
    
    def resample(self, waveform: torch.Tensor, original_sr: int) -> torch.Tensor:
        """Resample audio to target sample rate"""
        if original_sr == self.config.target_sample_rate:
            return waveform
            
        resampler = torchaudio.transforms.Resample(
            orig_freq=original_sr, 
            new_freq=self.config.target_sample_rate
        )
        return resampler(waveform)
    
    def normalize_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """Normalize audio waveform"""
        if self.config.normalize:
            # RMS normalization
            rms = torch.sqrt(torch.mean(waveform ** 2))
            if rms > 0:
                waveform = waveform / rms * 0.1  # Target RMS
        return waveform
    
    def trim_silence(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Trim silence from audio"""
        if not self.config.remove_silence:
            return waveform
            
        # Simple energy-based silence removal
        frame_length = int(0.025 * sample_rate)  # 25ms frames
        hop_length = int(0.01 * sample_rate)     # 10ms hop
        
        # Compute energy
        energy = torch.tensor([
            torch.sum(waveform[i:i+frame_length] ** 2) 
            for i in range(0, waveform.shape[-1] - frame_length, hop_length)
        ])
        
        # Find non-silent regions
        threshold = torch.max(energy) * 0.01  # 1% of max energy
        non_silent = energy > threshold
        
        if torch.any(non_silent):
            # Find start and end of non-silent region
            start_idx = torch.where(non_silent)[0][0] * hop_length
            end_idx = torch.where(non_silent)[0][-1] * hop_length + frame_length
            waveform = waveform[..., start_idx:end_idx]
            
        return waveform
    
    def truncate_or_pad(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Truncate or pad audio to max duration"""
        max_samples = int(self.config.max_duration * sample_rate)
        current_samples = waveform.shape[-1]
        
        if current_samples > max_samples:
            # Truncate
            waveform = waveform[..., :max_samples]
        elif current_samples < max_samples:
            # Pad with zeros
            pad_amount = max_samples - current_samples
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
            
        return waveform
    
    def process_audio(self, audio_bytes: bytes) -> Tuple[torch.Tensor, int]:
        """Process audio bytes and return tensor"""
        # Load audio
        waveform, sample_rate = self.load_audio_from_bytes(audio_bytes)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample
        waveform = self.resample(waveform, sample_rate)
        sample_rate = self.config.target_sample_rate
        
        # Trim silence (optional)
        waveform = self.trim_silence(waveform, sample_rate)
        
        # Normalize
        waveform = self.normalize_audio(waveform)
        
        # Return original length audio - no truncation or padding
        # vLLM handles variable sequence lengths efficiently
        return waveform.squeeze(0), sample_rate  # Remove channel dimension

class AudioDownloadStage(PipelineStage):
    """Stage for downloading audio and multimedia files from storage"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        from ..data.storage import MediaStorageManager
        from ..data.media_indexer import MediaDataLoader
        
        self.logger = logging.getLogger("AudioDownloadStage")
        self.storage_manager = MediaStorageManager(config['storage'])
        self.data_loader = MediaDataLoader(config['data'])
        
        # Initialize multimedia processing if enabled
        self.enable_multimedia = config.get('enable_multimedia', True)
        if self.enable_multimedia:
            media_config = MediaConfig(**config.get('media', {}))
            cache_config = CacheConfig(
                enabled=media_config.cache_enable,
                cache_dir=config.get('media', {}).get('cache', {}).get('cache_dir', './cache/media'),
                max_size_gb=media_config.cache_max_size_gb,
                ttl_hours=media_config.cache_ttl_hours
            )
            
            self.detector = MediaDetector()
            self.extractor = MediaExtractor(media_config)
            self.batch_processor = BatchMediaProcessor(media_config, cache_config)
        
    def process(self, batch: BatchData[SourceItem]) -> BatchData[RawAudioItem]:
        """Download and process audio/multimedia files for batch"""
        if self.enable_multimedia:
            # Use multimedia processing pipeline
            return self._process_multimedia_batch(batch)
        else:
            # Use original audio-only pipeline
            return self._process_audio_batch(batch)
    
    def _process_audio_batch(self, batch: BatchData) -> BatchData:
        """Process audio-only batch"""
        processed_items = []
        
        for item in batch.items:
            file_id = item.file_id
            oss_path = item.oss_path
            audio_bytes = None  # 初始化为 None
            tmp_file_path = None  # 追踪临时文件路径
            
            # Check cache first
            # item is SourceItem, dot notation required
            cached_audio = self.data_loader.get_cached_media(file_id, 'audio')
            if cached_audio and cached_audio.exists():
                try:
                    with open(cached_audio, 'rb') as f:
                        audio_bytes = f.read()
                except Exception as e:
                    self.logger.error(f"Failed to read cached audio for {file_id}: {e}")
                    raise
            else:
                # Download from storage
                try:
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                        tmp_file_path = tmp_file.name
                        success = self.storage_manager.download_audio(oss_path, tmp_file.name)
                        if not success:
                            error_msg = f"Failed to download {oss_path}"
                            self.logger.error(error_msg)
                            raise ValueError(error_msg)
                    
                    # 在 with 块外读取文件，确保文件已关闭
                    with open(tmp_file_path, 'rb') as f:
                        audio_bytes = f.read()
                    
                    # Cache the downloaded audio
                    self.data_loader.cache_media(file_id, audio_bytes, 'audio')
                    
                except Exception as e:
                    self.logger.error(f"Error processing audio download for {file_id} ({oss_path}): {e}")
                    raise
                finally:
                    # 清理临时文件
                    if tmp_file_path and os.path.exists(tmp_file_path):
                        try:
                            os.unlink(tmp_file_path)
                        except Exception as e:
                            self.logger.warning(f"Failed to cleanup temp file {tmp_file_path}: {e}")

            # 确保 audio_bytes 是 bytes 类型
            if isinstance(audio_bytes, bytes):
                # Create proper RawAudioItem instead of modifying dict
                raw_item = RawAudioItem(
                    file_id=item.file_id,
                    oss_path=item.oss_path,
                    format=item.format,
                    duration=item.duration,
                    metadata=item.metadata,
                    audio_bytes=audio_bytes
                )
                processed_items.append(raw_item)
            else:
                error_msg = f"Invalid audio data type for {file_id}: {type(audio_bytes)}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
        
        # Create new batch with downloaded audio
        new_batch = BatchData(
            batch_id=batch.batch_id,
            items=processed_items,
            metadata={**batch.metadata, 'stage': 'audio_download'}
        )
        
        return new_batch
    
    def _process_multimedia_batch(self, batch: BatchData[SourceItem]) -> BatchData[RawAudioItem]:
        """Process multimedia batch with format conversion"""
        processed_items = []
        
        # Create media items from batch
        media_items = []
        item_mapping = {}  # Map item_id to original SourceItem
        
        for item in batch.items:
            file_id = item.file_id
            oss_path = item.oss_path
            filename = item.metadata.get('filename', os.path.basename(oss_path))
            file_bytes = None
            tmp_file_path = None
            
            # Check cache first
            cached_media = self.data_loader.get_cached_media(file_id, 'audio')
            if cached_media and cached_media.exists():
                try:
                    with open(cached_media, 'rb') as f:
                        file_bytes = f.read()
                except Exception as e:
                    self.logger.error(f"Failed to read cached media for {file_id}: {e}")
                    raise
            else:
                # Download from storage
                try:
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                        tmp_file_path = tmp_file.name
                        success = self.storage_manager.download_audio(oss_path, tmp_file.name)
                        if not success:
                            error_msg = f"Failed to download {oss_path}"
                            self.logger.error(error_msg)
                            raise ValueError(error_msg)
                    
                    # 在 with 块外读取文件
                    with open(tmp_file_path, 'rb') as f:
                        file_bytes = f.read()
                    
                    # Cache the downloaded media
                    self.data_loader.cache_media(file_id, file_bytes, 'audio')
                    
                except Exception as e:
                    self.logger.error(f"Error processing multimedia download for {file_id} ({oss_path}): {e}")
                    raise
                finally:
                    # 清理临时文件
                    if tmp_file_path and os.path.exists(tmp_file_path):
                        try:
                            os.unlink(tmp_file_path)
                        except Exception as e:
                            self.logger.warning(f"Failed to cleanup temp file {tmp_file_path}: {e}")
            
            # 确保 file_bytes 是 bytes 类型
            if isinstance(file_bytes, bytes):
                # Create media item
                media_item = MediaItem(
                    item_id=file_id,
                    file_bytes=file_bytes,
                    filename=filename,
                    metadata=item.metadata
                )
                media_items.append(media_item)
                item_mapping[file_id] = item
            else:
                error_msg = f"Invalid media data type for {file_id}: {type(file_bytes)}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
    
        # Process media items in batch
        audio_data_list = self.batch_processor.process_batch(media_items)
        
        # Create RawAudioItem objects with processed audio data
        for audio_data in audio_data_list:
            item_id = audio_data.item_id
            if item_id in item_mapping:
                original_item = item_mapping[item_id]
                # Create RawAudioItem with audio_bytes
                raw_audio_item = RawAudioItem(
                    file_id=original_item.file_id,
                    oss_path=original_item.oss_path,
                    format=original_item.format,
                    duration=original_item.duration,
                    metadata={
                        **original_item.metadata,
                        'audio_metadata': audio_data.metadata,
                        'sample_rate': audio_data.sample_rate,
                        'channels': audio_data.channels
                    },
                    audio_bytes=audio_data.audio_bytes
                )
                processed_items.append(raw_audio_item)
    
        # Create new batch with processed multimedia
        new_batch = BatchData(
            batch_id=batch.batch_id,
            items=processed_items,
            metadata={**batch.metadata, 'stage': 'multimedia_download'}
        )
        
        return new_batch


class AudioPreprocessingStage(PipelineStage):
    """Stage for audio preprocessing and tensor conversion"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger = logging.getLogger("AudioPreprocessingStage")
        audio_config = AudioConfig(**config.get('audio', {}))
        self.preprocessor = AudioPreprocessor(audio_config)
        
    def _preprocess_item(self, item: RawAudioItem) -> TensorItem:
        """Preprocess single item"""
        try:
            waveform, sample_rate = self.preprocessor.process_audio(item.audio_bytes)
            
            # Create TensorItem
            # Optimization: Convert to numpy for shared memory efficiency if needed, 
            # but TensorItem expects numpy as per common.py definition.
            # self.preprocessor returns Tensor, need conversion
            waveform_np = waveform.numpy() 
            
            return TensorItem(
                file_id=item.file_id,
                oss_path=item.oss_path,
                format=item.format,
                duration=item.duration,
                metadata=item.metadata,
                waveform=waveform_np,
                sample_rate=sample_rate
            )
        except Exception as e:
            self.logger.error(f"Preprocessing failed for item {item.file_id}: {e}")
            return None

    def process(self, batch: BatchData[RawAudioItem]) -> BatchData[TensorItem]:
        """Preprocess audio and convert to tensors"""
        return batch.map(self._preprocess_item, new_batch_id=f"{batch.batch_id}_preprocessed")


class AudioFeatureExtractor:
    """Extract audio features for model input"""
    
    def __init__(self, config: Dict[str, Any]):
        # 从配置中获取features部分
        features_config = config.get('features', config)  # 如果没有单独的features配置，则使用整个config
        self.config = features_config
        self.feature_type = features_config.get('feature_type', 'mel_spectrogram')
        self.sample_rate = features_config.get('sample_rate', 16000)
        self.n_fft = features_config.get('n_fft', 400)
        self.hop_length = features_config.get('hop_length', 160)
        self.n_mels = features_config.get('n_mels', 80)
        
        # Initialize feature extractors
        if self.feature_type == 'mel_spectrogram':
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels
            )
        elif self.feature_type == 'mfcc':
            self.mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=self.sample_rate,
                n_mfcc=self.n_mels,
                melkwargs={
                    'n_fft': self.n_fft,
                    'hop_length': self.hop_length,
                    'n_mels': self.n_mels
                }
            )
    
    def extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract audio features"""
        if self.feature_type == 'mel_spectrogram':
            features = self.mel_transform(waveform)
            # Log mel spectrogram
            features = torch.log(features + 1e-8)
        elif self.feature_type == 'mfcc':
            features = self.mfcc_transform(waveform)
        else:
            # Raw waveform
            features = waveform.unsqueeze(0)  # Add channel dimension
            
        return features


class AudioFeatureStage(PipelineStage):
    """Stage for preparing audio data for Qwen3-Omni model"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger = logging.getLogger("AudioFeatureStage")
        # Qwen3-Omni expects raw audio waveform, no feature extraction needed
        
    def _prepare_item(self, item: SegmentItem) -> SegmentItem:
        """Prepare item (add features if needed, or pass through for Qwen3-Omni)"""
        try:
            # For Qwen3-Omni, we might just pass through or ensure format
            # If we need to add 'feature' field, we would need a new Item type or modify SegmentItem
            # But per design doc, AudioFeatureStage output is SegmentItem-like but with features.
            # Ideally we might have a FeatureItem, but for now let's reuse/enrich.
            
            # Actually Note.md says AudioFeatureStage output has 'audio_features' etc.
            # But common.py SegmentItem definition doesn't have 'audio_features'.
            # For strict typing, we should either update SegmentItem or use dynamic getattr
            # or stick to what is in common.py.
            # Assuming we can dynamically add or we update common.py later.
            # For now, let's assume valid SegmentItem in -> SegmentItem out (maybe with updated metadata?)
            # Or returns InferenceReadyItem?
            
            # NOTE: Refactoring to use functional map implies returning a new Typed Item.
            # InferenceItem requires 'transcription', so we can't use that yet.
            # We will return SegmentItem, but perhaps use metadata to store features if needed, 
            # OR just update common.py to include audio_features in SegmentItem (optional).
            
            # For now, let's keep it simple: Just pass through or do minimal updates.
            # Qwen3-Omni takes raw waveform which SegmentItem already has in 'waveform' field.
            # So this stage might be a confusing "pass-through" or "verification" stage.
            return item
        except Exception as e:
            self.logger.error(f"Feature preparation failed for item {item.file_id}: {e}")
            return None

    def process(self, batch: BatchData[SegmentItem]) -> BatchData[SegmentItem]:
        """Prepare audio data for Qwen3-Omni model"""
        # This stage effectively might just be a NOP or validation for Qwen3-Omni
        # if the model takes raw waveforms which are already in SegmentItems.
        return batch.map(self._prepare_item, new_batch_id=f"{batch.batch_id}_features")