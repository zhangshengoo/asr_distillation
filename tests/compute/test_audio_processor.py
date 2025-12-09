"""
AudioProcessor 简单单元测试

测试音频处理功能
"""

import pytest
import torch
from unittest.mock import Mock, patch

from src.compute.audio_processor import AudioPreprocessor, AudioConfig


class TestAudioPreprocessor:
    """AudioPreprocessor简单测试"""

    def test_init(self):
        """测试初始化"""
        config = AudioConfig(
            target_sample_rate=16000,
            max_duration=30.0,
            normalize=True,
            remove_silence=False,
            audio_format='wav'
        )
        
        preprocessor = AudioPreprocessor(config)
        assert preprocessor.config.target_sample_rate == 16000
        assert preprocessor.config.max_duration == 30.0
        assert preprocessor.config.normalize == True

    def test_normalize_audio(self):
        """测试音频归一化"""
        config = AudioConfig(normalize=True)
        preprocessor = AudioPreprocessor(config)
        
        # 创建测试音频
        waveform = torch.randn(1, 16000) * 0.1
        
        # 归一化处理
        normalized = preprocessor.normalize_audio(waveform)
        
        # 验证结果
        assert normalized.shape == waveform.shape
        assert torch.allclose(torch.sqrt(torch.mean(normalized ** 2)), 0.1, atol=0.01)

    def test_truncate_or_pad(self):
        """测试音频截断和填充"""
        config = AudioConfig(max_duration=2.0)
        preprocessor = AudioPreprocessor(config)
        
        sample_rate = 16000
        max_samples = int(2.0 * sample_rate)  # 2秒
        
        # 测试截断
        long_waveform = torch.randn(1, max_samples * 2)
        truncated = preprocessor.truncate_or_pad(long_waveform, sample_rate)
        assert truncated.shape[-1] == max_samples
        
        # 测试填充
        short_waveform = torch.randn(1, max_samples // 2)
        padded = preprocessor.truncate_or_pad(short_waveform, sample_rate)
        assert padded.shape[-1] == max_samples

    def test_resample(self):
        """测试重采样"""
        config = AudioConfig(target_sample_rate=16000)
        preprocessor = AudioPreprocessor(config)
        
        # 创建8kHz音频
        original_sr = 8000
        waveform = torch.randn(1, 8000)
        
        # 重采样到16kHz
        resampled = preprocessor.resample(waveform, original_sr)
        
        # 验证结果
        assert resampled.shape[-1] == 16000  # 长度加倍