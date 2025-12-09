"""测试VAD处理功能"""

import pytest
import numpy as np
from pathlib import Path

from src.compute.vad import (
    VADProcessor, 
    VADResult, 
    AudioSegment,
    VADModelManager,
    VADCache,
    TimestampManager
)


class TestVADProcessor:
    """VAD处理器测试"""
    
    @pytest.fixture
    def vad_config(self):
        """VAD配置"""
        return {
            'sampling_rate': 16000,
            'threshold': 0.4,
            'min_speech_duration_ms': 1500,
            'min_silence_duration_ms': 1000,
            'speech_pad_ms': 100,
            'cache_enabled': False,  # 测试时禁用缓存
            'cache_dir': './test_cache/vad',
            'cache_max_size_gb': 1.0
        }
    
    @pytest.fixture
    def sample_audio(self):
        """生成测试音频数据"""
        # 生成10秒的16kHz音频
        duration = 10.0
        sample_rate = 16000
        t = np.linspace(0, duration, int(duration * sample_rate))
        
        # 添加一些语音段（正弦波模拟）
        audio = np.zeros_like(t)
        
        # 添加语音段：1-3秒，5-7秒
        speech_mask = ((t >= 1.0) & (t <= 3.0)) | ((t >= 5.0) & (t <= 7.0))
        audio[speech_mask] = 0.5 * np.sin(2 * np.pi * 440 * t[speech_mask])  # 440Hz正弦波
        
        return audio, sample_rate
    
    def test_vad_processor_init(self, vad_config):
        """测试VAD处理器初始化"""
        processor = VADProcessor(vad_config)
        assert processor is not None
        assert processor.vad_params['sampling_rate'] == 16000
        assert processor.vad_params['threshold'] == 0.4
    
    def test_process_audio(self, vad_config, sample_audio):
        """测试音频处理"""
        audio_data, sample_rate = sample_audio
        processor = VADProcessor(vad_config)
        
        result = processor.process_audio("test_001", audio_data, sample_rate)
        
        assert isinstance(result, VADResult)
        assert result.file_id == "test_001"
        assert result.original_duration == 10.0
        assert len(result.speech_segments) > 0
        assert result.total_speech_duration > 0
        assert 0 <= result.speech_ratio <= 1
    
    def test_segment_audio(self, vad_config, sample_audio):
        """测试音频切分"""
        audio_data, sample_rate = sample_audio
        processor = VADProcessor(vad_config)
        
        # 首先进行VAD检测
        vad_result = processor.process_audio("test_001", audio_data, sample_rate)
        
        # 然后切分音频
        segments = processor.segment_audio("test_001", audio_data, sample_rate, vad_result)
        
        assert len(segments) > 0
        
        for segment in segments:
            assert isinstance(segment, AudioSegment)
            assert segment.file_id == "test_001"
            assert segment.start_time < segment.end_time
            assert segment.duration > 0
            assert segment.audio_data is not None
            assert len(segment.audio_data) > 0
    
    def test_batch_process(self, vad_config, sample_audio):
        """测试批量处理"""
        audio_data, sample_rate = sample_audio
        processor = VADProcessor(vad_config)
        
        # 准备批次数据
        batch = [
            ("test_001", audio_data.copy(), sample_rate),
            ("test_002", audio_data.copy(), sample_rate),
            ("test_003", audio_data.copy(), sample_rate)
        ]
        
        results = processor.batch_process(batch)
        
        assert len(results) == 3
        
        for result in results:
            assert isinstance(result, VADResult)
            assert result.file_id.startswith("test_")
            assert len(result.speech_segments) > 0


class TestVADCache:
    """VAD缓存测试"""
    
    @pytest.fixture
    def cache_dir(self, tmp_path):
        """临时缓存目录"""
        cache_path = tmp_path / "vad_cache"
        cache_path.mkdir(exist_ok=True)
        return str(cache_path)
    
    def test_cache_put_and_get(self, cache_dir):
        """测试缓存存取"""
        cache = VADCache(cache_dir, max_size_gb=0.1)
        
        # 准备测试数据
        audio_data = np.random.randn(16000).astype(np.float32)  # 1秒音频
        config = {'threshold': 0.4}
        
        result = VADResult(
            file_id="test_001",
            speech_segments=[{'start': 0.0, 'end': 1.0}],
            total_speech_duration=1.0,
            speech_ratio=1.0,
            original_duration=1.0,
            sample_rate=16000
        )
        
        # 存入缓存
        cache.put(audio_data, config, result)
        
        # 从缓存获取
        cached_result = cache.get(audio_data, config)
        
        assert cached_result is not None
        assert cached_result.file_id == result.file_id
        assert len(cached_result.speech_segments) == len(result.speech_segments)
    
    def test_cache_eviction(self, cache_dir):
        """测试缓存淘汰"""
        cache = VADCache(cache_dir, max_size_gb=0.001)  # 很小的缓存
        
        # 添加多个缓存项，触发淘汰
        for i in range(10):
            audio_data = np.random.randn(16000).astype(np.float32)
            config = {'threshold': 0.4}
            
            result = VADResult(
                file_id=f"test_{i:03d}",
                speech_segments=[{'start': 0.0, 'end': 1.0}],
                total_speech_duration=1.0,
                speech_ratio=1.0,
                original_duration=1.0,
                sample_rate=16000
            )
            
            cache.put(audio_data, config, result)
        
        # 验证缓存大小被限制
        assert cache._get_cache_size() <= cache.max_size_bytes


class TestTimestampManager:
    """时间戳管理器测试"""
    
    def test_map_asr_to_timestamps(self):
        """测试ASR结果映射到时间戳"""
        # 准备测试片段
        segments = [
            AudioSegment(
                file_id="test_001",
                segment_id="test_001_seg_000",
                audio_data=np.random.randn(16000),
                start_time=0.0,
                end_time=1.0,
                duration=1.0,
                sample_rate=16000,
                original_duration=3.0
            ),
            AudioSegment(
                file_id="test_001",
                segment_id="test_001_seg_001",
                audio_data=np.random.randn(16000),
                start_time=2.0,
                end_time=3.0,
                duration=1.0,
                sample_rate=16000,
                original_duration=3.0
            )
        ]
        
        # ASR识别结果
        asr_results = ["第一段语音", "第二段语音"]
        
        # 映射结果
        mapped_results = TimestampManager.map_asr_to_timestamps(segments, asr_results)
        
        assert len(mapped_results) == 2
        
        for i, result in enumerate(mapped_results):
            assert result.segment_id == segments[i].segment_id
            assert result.transcription == asr_results[i]
            assert result.start_time == segments[i].start_time
            assert result.end_time == segments[i].end_time
    
    def test_merge_results_by_file(self):
        """测试按文件合并结果"""
        from src.compute.vad import ASRResult
        
        # 准备测试结果
        results = [
            ASRResult("file_001", "seg_000", "语音1", 0.9, 0.0, 1.0, 1.0),
            ASRResult("file_002", "seg_000", "语音2", 0.8, 0.5, 1.5, 1.0),
            ASRResult("file_001", "seg_001", "语音3", 0.95, 2.0, 3.0, 1.0)
        ]
        
        # 合并结果
        merged = TimestampManager.merge_results_by_file(results)
        
        assert len(merged) == 2
        assert "file_001" in merged
        assert "file_002" in merged
        assert len(merged["file_001"]) == 2
        assert len(merged["file_002"]) == 1
        
        # 验证时间戳排序
        file_001_results = merged["file_001"]
        assert file_001_results[0].start_time < file_001_results[1].start_time
    
    def test_format_output(self):
        """测试输出格式化"""
        from src.compute.vad import ASRResult
        
        # 准备测试数据
        results = [
            ASRResult("file_001", "seg_000", "语音1", 0.9, 0.0, 1.0, 1.0),
            ASRResult("file_001", "seg_001", "语音2", 0.95, 2.0, 3.0, 1.0)
        ]
        
        file_results = {"file_001": results}
        formatted = TimestampManager.format_output(file_results)
        
        assert len(formatted) == 1
        
        output = formatted[0]
        assert output["file_id"] == "file_001"
        assert output["original_duration"] == 3.0
        assert output["total_speech_duration"] == 2.0
        assert output["speech_ratio"] == 2.0 / 3.0
        assert len(output["speech_segments"]) == 2
        
        # 验证片段格式
        segment = output["speech_segments"][0]
        assert "segment_id" in segment
        assert "start_time" in segment
        assert "end_time" in segment
        assert "duration" in segment
        assert "transcription" in segment
        assert "confidence" in segment


if __name__ == "__main__":
    pytest.main([__file__])