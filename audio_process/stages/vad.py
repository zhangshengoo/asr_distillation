"""VAD处理Stage"""
import multiprocessing
from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import torch
import torchaudio

from base import Stage
from tools.data_structures import ProcessingItem

# 进程级VAD模型缓存
_process_vad_models = {}


def _init_vad_worker(vad_config: dict):
    """初始化工作进程的VAD模型"""
    pid = multiprocessing.current_process().pid
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=False
    )
    get_speech_timestamps = utils[0]
    _process_vad_models[pid] = (model, get_speech_timestamps, vad_config)


def _process_vad_worker(item_data: tuple) -> tuple:
    """在工作进程中处理VAD"""
    file_id, audio_data, sample_rate = item_data
    pid = multiprocessing.current_process().pid
    model, get_speech_timestamps, vad_config = _process_vad_models[pid]
    
    with torch.no_grad():
        # VAD检测
        speech_timestamps = get_speech_timestamps(
            audio_data,
            model,
            sampling_rate=vad_config.get('sampling_rate', 16000),
            return_seconds=True,
            min_speech_duration_ms=vad_config.get('min_speech_duration_ms', 500),
            min_silence_duration_ms=vad_config.get('min_silence_duration_ms', 8000),
            threshold=vad_config.get('threshold', 0.25),
            neg_threshold=vad_config.get('neg_threshold', 0.1),
            speech_pad_ms=vad_config.get('speech_pad_ms', 1000)
        )
        
        # 转换为segment格式
        segments = []
        for seg_info in speech_timestamps:
            start_idx = int(seg_info['start'] * sample_rate)
            end_idx = int(seg_info['end'] * sample_rate)
            
            segments.append({
                'start_time': seg_info['start'],
                'end_time': seg_info['end'],
                'duration': seg_info['end'] - seg_info['start'],
                'start_idx': start_idx,
                'end_idx': end_idx
            })
        
        # 计算统计信息
        original_duration = len(audio_data) / sample_rate
        total_speech_duration = sum(seg['duration'] for seg in segments)
        speech_ratio = total_speech_duration / original_duration if original_duration > 0 else 0
        
        vad_result = {
            'total_speech_duration': total_speech_duration,
            'speech_ratio': speech_ratio,
            'num_segments': len(segments)
        }
        
        return (file_id, segments, vad_result)


class CoarseVADStage(Stage):
    """粗VAD处理：产生较长的语音片段"""
    
    def __init__(self, vad_config: dict, num_workers: int = 4):
        self.vad_config = vad_config
        self.num_workers = num_workers
        self._local_model = None
        self._local_utils = None
    
    def name(self) -> str:
        return "coarse_vad"
    
    def _ensure_local_model(self):
        """懒加载本地VAD模型（用于单个处理）"""
        if self._local_model is None:
            self._local_model, self._local_utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
    
    def process(self, item: ProcessingItem) -> ProcessingItem:
        """执行VAD检测（单个处理）"""
        self._ensure_local_model()
        get_speech_timestamps = self._local_utils[0]
        
        with torch.no_grad():
            audio_data = item.audio_data
            
            # VAD检测
            speech_timestamps = get_speech_timestamps(
                audio_data,
                self._local_model,
                sampling_rate=self.vad_config.get('sampling_rate', 16000),
                return_seconds=True,
                min_speech_duration_ms=self.vad_config.get('min_speech_duration_ms', 500),
                min_silence_duration_ms=self.vad_config.get('min_silence_duration_ms', 8000),
                threshold=self.vad_config.get('threshold', 0.25),
                neg_threshold=self.vad_config.get('neg_threshold', 0.1),
                speech_pad_ms=self.vad_config.get('speech_pad_ms', 1000)
            )
            
            # 转换为segment格式
            segments = []
            for seg_info in speech_timestamps:
                start_idx = int(seg_info['start'] * item.sample_rate)
                end_idx = int(seg_info['end'] * item.sample_rate)
                
                segments.append({
                    'start_time': seg_info['start'],
                    'end_time': seg_info['end'],
                    'duration': seg_info['end'] - seg_info['start'],
                    'start_idx': start_idx,
                    'end_idx': end_idx
                })
            
            # 统计信息
            original_duration = len(audio_data) / item.sample_rate
            total_speech_duration = sum(seg['duration'] for seg in segments)
            speech_ratio = total_speech_duration / original_duration if original_duration > 0 else 0
            
            item.metadata['vad_result'] = {
                'total_speech_duration': total_speech_duration,
                'speech_ratio': speech_ratio,
                'num_segments': len(segments)
            }
            item.segments = segments
            
            return item
    
    def process_batch(self, items: List[ProcessingItem]) -> List[ProcessingItem]:
        """并行处理批次样本"""
        if not items:
            return []
        
        # 准备输入数据
        item_data = [
            (item.file_id, item.audio_data, item.sample_rate)
            for item in items
        ]
        
        # 并行处理
        with ProcessPoolExecutor(
            max_workers=self.num_workers,
            initializer=_init_vad_worker,
            initargs=(self.vad_config,)
        ) as executor:
            futures = {
                executor.submit(_process_vad_worker, data): i
                for i, data in enumerate(item_data)
            }
            
            results = [None] * len(items)
            for future in as_completed(futures):
                idx = futures[future]
                file_id, segments, vad_result = future.result()
                results[idx] = (segments, vad_result)
        
        # 更新items
        for i, item in enumerate(items):
            segments, vad_result = results[i]
            item.segments = segments
            item.metadata['vad_result'] = vad_result
        
        return items