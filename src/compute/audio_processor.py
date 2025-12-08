"""CPU audio preprocessing workers"""

import os
import io
import tempfile
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

import torch
import torchaudio
import numpy as np
from loguru import logger

from ..scheduling.pipeline import PipelineStage, DataBatch


@dataclass
class AudioConfig:
    """Audio processing configuration"""
    target_sample_rate: int = 16000
    max_duration: float = 30.0  # seconds
    normalize: bool = True
    remove_silence: bool = False
    audio_format: str = 'wav'


class AudioPreprocessor:
    """Audio preprocessing utilities"""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        
    def load_audio_from_bytes(self, audio_bytes: bytes) -> torch.Tensor:
        """Load audio tensor from bytes"""
        try:
            # Create temporary file to load with torchaudio
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_file.flush()
                
                # Load audio
                waveform, sample_rate = torchaudio.load(tmp_file.name)
                
                # Cleanup temp file
                os.unlink(tmp_file.name)
                
                return waveform, sample_rate
                
        except Exception as e:
            logger.error(f"Error loading audio from bytes: {e}")
            raise
    
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
        try:
            # Load audio
            waveform, sample_rate = self.load_audio_from_bytes(audio_bytes)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample
            waveform = self.resample(waveform, sample_rate)
            sample_rate = self.config.target_sample_rate
            
            # Trim silence
            waveform = self.trim_silence(waveform, sample_rate)
            
            # Normalize
            waveform = self.normalize_audio(waveform)
            
            # Truncate or pad
            waveform = self.truncate_or_pad(waveform, sample_rate)
            
            return waveform.squeeze(0), sample_rate  # Remove channel dimension
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            raise


class AudioDownloadStage(PipelineStage):
    """Stage for downloading audio from storage"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        from ..data.storage import AudioStorageManager
        from ..data.audio_indexer import DataLoader
        
        self.storage_manager = AudioStorageManager(config['storage'])
        self.data_loader = DataLoader(config)
        
    def process(self, batch: DataBatch) -> DataBatch:
        """Download audio files for batch"""
        processed_items = []
        
        for item in batch.items:
            try:
                file_id = item['file_id']
                oss_path = item['oss_path']
                
                # Check cache first
                cached_audio = self.data_loader.get_cached_audio(file_id)
                if cached_audio and cached_audio.exists():
                    with open(cached_audio, 'rb') as f:
                        audio_bytes = f.read()
                else:
                    # Download from storage
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                        success = self.storage_manager.download_audio(oss_path, tmp_file.name)
                        if not success:
                            raise ValueError(f"Failed to download {oss_path}")
                        
                        with open(tmp_file.name, 'rb') as f:
                            audio_bytes = f.read()
                        
                        # Cache the downloaded audio
                        self.data_loader.cache_audio(file_id, audio_bytes)
                        
                        os.unlink(tmp_file.name)
                
                # Add audio bytes to item
                item['audio_bytes'] = audio_bytes
                processed_items.append(item)
                
            except Exception as e:
                logger.error(f"Error downloading audio for {item['file_id']}: {e}")
                item['error'] = str(e)
                processed_items.append(item)
        
        # Create new batch with downloaded audio
        new_batch = DataBatch(
            batch_id=batch.batch_id,
            items=processed_items,
            metadata={**batch.metadata, 'stage': 'audio_download'}
        )
        
        return new_batch


class AudioPreprocessingStage(PipelineStage):
    """Stage for audio preprocessing and tensor conversion"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        audio_config = AudioConfig(**config.get('audio', {}))
        self.preprocessor = AudioPreprocessor(audio_config)
        
    def process(self, batch: DataBatch) -> DataBatch:
        """Preprocess audio and convert to tensors"""
        processed_items = []
        
        for item in batch.items:
            try:
                if 'error' in item:
                    processed_items.append(item)
                    continue
                    
                audio_bytes = item['audio_bytes']
                
                # Process audio
                waveform, sample_rate = self.preprocessor.process_audio(audio_bytes)
                
                # Convert to tensor for GPU processing
                audio_tensor = {
                    'waveform': waveform,
                    'sample_rate': sample_rate,
                    'duration': waveform.shape[-1] / sample_rate,
                    'format': 'tensor'
                }
                
                # Update item with processed audio
                item['audio_tensor'] = audio_tensor
                item.pop('audio_bytes', None)  # Remove raw bytes to save memory
                
                processed_items.append(item)
                
            except Exception as e:
                logger.error(f"Error preprocessing audio for {item['file_id']}: {e}")
                item['error'] = str(e)
                processed_items.append(item)
        
        # Create new batch with preprocessed audio
        new_batch = DataBatch(
            batch_id=batch.batch_id,
            items=processed_items,
            metadata={**batch.metadata, 'stage': 'audio_preprocessing'}
        )
        
        return new_batch


class AudioFeatureExtractor:
    """Extract audio features for model input"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.feature_type = config.get('feature_type', 'mel_spectrogram')
        
        # Initialize feature extractors
        if self.feature_type == 'mel_spectrogram':
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=config.get('sample_rate', 16000),
                n_fft=config.get('n_fft', 400),
                hop_length=config.get('hop_length', 160),
                n_mels=config.get('n_mels', 80)
            )
        elif self.feature_type == 'mfcc':
            self.mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=config.get('sample_rate', 16000),
                n_mfcc=config.get('n_mfcc', 40),
                melkwargs={
                    'n_fft': config.get('n_fft', 400),
                    'hop_length': config.get('hop_length', 160),
                    'n_mels': config.get('n_mels', 80)
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
    """Stage for extracting audio features"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.feature_extractor = AudioFeatureExtractor(config.get('features', {}))
        
    def process(self, batch: DataBatch) -> DataBatch:
        """Extract features from preprocessed audio"""
        processed_items = []
        
        for item in batch.items:
            try:
                if 'error' in item:
                    processed_items.append(item)
                    continue
                    
                waveform = item['audio_tensor']['waveform']
                
                # Extract features
                features = self.feature_extractor.extract_features(waveform)
                
                # Update item with features
                item['audio_features'] = features
                item['feature_type'] = self.feature_extractor.feature_type
                
                processed_items.append(item)
                
            except Exception as e:
                logger.error(f"Error extracting features for {item['file_id']}: {e}")
                item['error'] = str(e)
                processed_items.append(item)
        
        # Create new batch with features
        new_batch = DataBatch(
            batch_id=batch.batch_id,
            items=processed_items,
            metadata={**batch.metadata, 'stage': 'feature_extraction'}
        )
        
        return new_batch