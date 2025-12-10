"""VAD (Voice Activity Detection) processing components"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

import numpy as np
import torch
import torchaudio
from loguru import logger
from silero_vad import load_silero_vad, get_speech_timestamps

from ray import remote
from ray.util.queue import Queue


@dataclass
class AudioSegment:
    """音频片段数据结构"""
    file_id: str
    segment_id: str
    audio_data: np.ndarray
    start_time: float  # 片段开始时间（秒）
    end_time: float    # 片段结束时间（秒）
    duration: float    # 片段时长（秒）
    sample_rate: int   # 采样率
    original_duration: float  # 原始音频总时长


@dataclass
class VADResult:
    """VAD检测结果"""
    file_id: str
    speech_segments: List[Dict[str, Any]]  # 语音片段时间戳列表
    total_speech_duration: float  # 总语音时长
    speech_ratio: float  # 语音占比
    original_duration: float  # 原始音频总时长
    sample_rate: int  # 采样率


@dataclass
class ASRResult:
    """ASR识别结果"""
    file_id: str
    segment_id: str
    transcription: str
    confidence: float
    start_time: float
    end_time: float
    duration: float


class VADModelManager:
    """VAD模型管理器 - 每个Worker独立加载模型"""
    
    def __init__(self):
        self._model = None
        self._config = None
    
    def load_model(self, config: Dict[str, Any]) -> None:
        """加载VAD模型"""
        if self._model is None or self._config != config:
            try:
                # 不使用ONNX，使用PyTorch模型
                self._model = load_silero_vad(onnx=False)
                self._config = config
                logger.info("VAD模型加载成功（PyTorch版本）")
            except Exception as e:
                logger.error(f"VAD模型加载失败: {e}")
                raise
    
    def get_model(self):
        """获取VAD模型"""
        if self._model is None:
            raise RuntimeError("VAD模型未加载，请先调用load_model")
        return self._model


class VADCache:
    """VAD结果缓存"""
    
    def __init__(self, cache_dir: str, max_size_gb: float = 10.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self.cache_index_file = self.cache_dir / "vad_cache_index.json"
        
    def _get_cache_key(self, audio_data: np.ndarray, config: Dict[str, Any]) -> str:
        """生成缓存键"""
        # 使用音频数据和配置的哈希值作为缓存键
        audio_hash = hashlib.md5(audio_data.tobytes()).hexdigest()
        config_str = json.dumps(config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()
        return f"{audio_hash}_{config_hash}"
    
    def get(self, audio_data: np.ndarray, config: Dict[str, Any]) -> Optional[VADResult]:
        """获取缓存的VAD结果"""
        cache_key = self._get_cache_key(audio_data, config)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return VADResult(**data)
            except Exception as e:
                logger.warning(f"读取VAD缓存失败: {e}")
        return None
    
    def put(self, audio_data: np.ndarray, config: Dict[str, Any], result: VADResult) -> None:
        """缓存VAD结果"""
        # 检查缓存大小
        self._evict_if_needed()
        
        cache_key = self._get_cache_key(audio_data, config)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(result), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存VAD缓存失败: {e}")
    
    def _evict_if_needed(self) -> None:
        """如果需要，清理旧缓存"""
        total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.json"))
        
        if total_size > self.max_size_bytes:
            # 按修改时间排序，删除最旧的文件
            cache_files = list(self.cache_dir.glob("*.json"))
            cache_files.sort(key=lambda f: f.stat().st_mtime)
            
            for cache_file in cache_files:
                cache_file.unlink()
                total_size -= cache_file.stat().st_size
                if total_size <= self.max_size_bytes * 0.8:  # 保留20%空间
                    break
            
            logger.info(f"VAD缓存清理完成，当前大小: {total_size / 1024 / 1024:.1f} MB")
    
    def clear(self) -> None:
        """清空所有缓存"""
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
        logger.info("VAD缓存已清空")


class VADProcessor:
    """VAD处理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_manager = VADModelManager()
        self.cache = VADCache(
            cache_dir=config.get('cache_dir', './cache/vad'),
            max_size_gb=config.get('cache_max_size_gb', 10.0)
        )
        
        # 加载模型
        self.model_manager.load_model(config)
        
        # VAD参数
        self.vad_params = {
            'sampling_rate': config.get('sampling_rate', 16000),
            'return_seconds': True,
            'min_speech_duration_ms': config.get('min_speech_duration_ms', 1500),
            'min_silence_duration_ms': config.get('min_silence_duration_ms', 1000),
            'threshold': config.get('threshold', 0.4),
            'neg_threshold': config.get('neg_threshold', 0.15),
            'speech_pad_ms': config.get('speech_pad_ms', 100)
        }
    
    def process_audio(self, file_id: str, audio_data: np.ndarray, 
                     sample_rate: int) -> VADResult:
        """处理单个音频文件"""
        try:
            # 重采样到目标采样率
            if sample_rate != self.vad_params['sampling_rate']:
                audio_tensor = torch.from_numpy(audio_data).float()
                if len(audio_tensor.shape) == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)
                
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=self.vad_params['sampling_rate']
                )
                audio_tensor = resampler(audio_tensor)
                audio_data = audio_tensor.squeeze().numpy()
                sample_rate = self.vad_params['sampling_rate']
            
            # 检查缓存
            if self.config.get('cache_enabled', True):
                cached_result = self.cache.get(audio_data, self.vad_params)
                if cached_result:
                    logger.debug(f"使用VAD缓存结果: {file_id}")
                    return cached_result
            
            # 执行VAD检测
            model = self.model_manager.get_model()
            speech_timestamps = get_speech_timestamps(
                audio_data,
                model,
                **self.vad_params
            )
            
            # 计算统计信息
            original_duration = len(audio_data) / sample_rate
            total_speech_duration = sum(
                seg['end'] - seg['start'] for seg in speech_timestamps
            )
            speech_ratio = total_speech_duration / original_duration if original_duration > 0 else 0
            
            # 构建结果
            result = VADResult(
                file_id=file_id,
                speech_segments=speech_timestamps,
                total_speech_duration=total_speech_duration,
                speech_ratio=speech_ratio,
                original_duration=original_duration,
                sample_rate=sample_rate
            )
            
            # 缓存结果
            if self.config.get('cache_enabled', True):
                self.cache.put(audio_data, self.vad_params, result)
            
            logger.info(f"VAD处理完成: {file_id}, 语音占比: {speech_ratio:.2%}")
            return result
            
        except Exception as e:
            logger.error(f"VAD处理失败 {file_id}: {e}")
            raise
    
    def segment_audio(self, file_id: str, audio_data: np.ndarray, 
                     sample_rate: int, vad_result: VADResult) -> List[AudioSegment]:
        """根据VAD结果切分音频"""
        segments = []
        
        for i, segment_info in enumerate(vad_result.speech_segments):
            start_sample = int(segment_info['start'] * sample_rate)
            end_sample = int(segment_info['end'] * sample_rate)
            
            # 确保索引不越界
            start_sample = max(0, start_sample)
            end_sample = min(len(audio_data), end_sample)
            
            if end_sample > start_sample:
                segment_audio = audio_data[start_sample:end_sample]
                
                audio_segment = AudioSegment(
                    file_id=file_id,
                    segment_id=f"{file_id}_seg_{i:03d}",
                    audio_data=segment_audio,
                    start_time=segment_info['start'],
                    end_time=segment_info['end'],
                    duration=segment_info['end'] - segment_info['start'],
                    sample_rate=sample_rate,
                    original_duration=vad_result.original_duration
                )
                segments.append(audio_segment)
        
        logger.info(f"音频切分完成: {file_id}, 生成 {len(segments)} 个语音片段")
        return segments
    
    def batch_process(self, audio_batch: List[Tuple[str, np.ndarray, int]]) -> List[VADResult]:
        """批量处理音频"""
        results = []
        
        for file_id, audio_data, sample_rate in audio_batch:
            try:
                result = self.process_audio(file_id, audio_data, sample_rate)
                results.append(result)
            except Exception as e:
                logger.error(f"批量VAD处理失败 {file_id}: {e}")
                # 创建空结果，保持批次完整性
                results.append(VADResult(
                    file_id=file_id,
                    speech_segments=[],
                    total_speech_duration=0.0,
                    speech_ratio=0.0,
                    original_duration=len(audio_data) / sample_rate,
                    sample_rate=sample_rate
                ))
        
        return results


class TimestampManager:
    """时间戳管理器"""
    
    @staticmethod
    def map_asr_to_timestamps(segments: List[AudioSegment], 
                             asr_results: List[str]) -> List[ASRResult]:
        """将ASR结果映射到时间戳"""
        if len(segments) != len(asr_results):
            logger.warning(f"片段数量({len(segments)})与ASR结果数量({len(asr_results)})不匹配")
        
        results = []
        for i, segment in enumerate(segments):
            transcription = asr_results[i] if i < len(asr_results) else ""
            
            asr_result = ASRResult(
                file_id=segment.file_id,
                segment_id=segment.segment_id,
                transcription=transcription,
                confidence=0.0,  # 需要从ASR模型获取
                start_time=segment.start_time,
                end_time=segment.end_time,
                duration=segment.duration
            )
            results.append(asr_result)
        
        return results
    
    @staticmethod
    def merge_results_by_file(asr_results: List[ASRResult]) -> Dict[str, List[ASRResult]]:
        """按文件ID合并结果"""
        merged_results = {}
        
        for result in asr_results:
            if result.file_id not in merged_results:
                merged_results[result.file_id] = []
            merged_results[result.file_id].append(result)
        
        # 按时间戳排序
        for file_id in merged_results:
            merged_results[file_id].sort(key=lambda x: x.start_time)
        
        return merged_results
    
    @staticmethod
    def format_output(file_results: Dict[str, List[ASRResult]]) -> List[Dict[str, Any]]:
        """格式化输出结果"""
        formatted_results = []
        
        for file_id, results in file_results.items():
            if not results:
                continue
            
            # 计算总时长和语音时长
            original_duration = max(result.end_time for result in results)
            total_speech_duration = sum(result.duration for result in results)
            speech_ratio = total_speech_duration / original_duration if original_duration > 0 else 0
            
            # 构建输出格式
            output = {
                "file_id": file_id,
                "original_duration": original_duration,
                "speech_segments": [
                    {
                        "segment_id": result.segment_id,
                        "start_time": result.start_time,
                        "end_time": result.end_time,
                        "duration": result.duration,
                        "transcription": result.transcription,
                        "confidence": result.confidence
                    }
                    for result in results
                ],
                "total_speech_duration": total_speech_duration,
                "speech_ratio": speech_ratio
            }
            
            formatted_results.append(output)
        
        return formatted_results


# 向后兼容的别名
VAD = VADProcessor