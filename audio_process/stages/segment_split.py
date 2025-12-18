"""Segment切分Stage"""
import numpy as np
from typing import List, Dict
from ..stages.base import Stage
from ..tools.data_structures import ProcessingItem
from src.compute.vad import VADProcessor


class SegmentSplitStage(Stage):
    """
    检测超长segment并用细VAD切分
    目标：尽量产生接近target_duration的segments，但不超过max_duration
    """
    
    def __init__(self, max_duration: float = 178, target_duration: float = 120):
        self.max_duration = max_duration
        self.target_duration = target_duration
        
        # 细VAD配置（更严格）
        self.fine_vad_config = {
            'sampling_rate': 16000,
            'min_speech_duration_ms': 1500,
            'min_silence_duration_ms': 300,  # 300ms静音即可切分
            'threshold': 0.4,
            'neg_threshold': 0.15,
            'speech_pad_ms': 100,
            'cache_enabled': False  # 不缓存细VAD结果
        }
        self.fine_vad = None  # 懒加载
    
    def name(self) -> str:
        return "segment_split"
    
    def _ensure_fine_vad(self):
        """懒加载细VAD模型"""
        if self.fine_vad is None:
            self.fine_vad = VADProcessor(self.fine_vad_config)
    
    def process(self, item: ProcessingItem) -> ProcessingItem:
        """检测并切分超长segments"""
        new_segments = []
        
        for seg in item.segments:
            if seg['duration'] <= self.max_duration:
                # 不需要切分
                new_segments.append(seg)
            else:
                # 需要切分
                split_segs = self._split_long_segment(seg, item)
                new_segments.extend(split_segs)
        
        item.segments = new_segments
        return item
    
    def _split_long_segment(self, segment: Dict, item: ProcessingItem) -> List[Dict]:
        """切分超长segment"""
        self._ensure_fine_vad()
        
        # 提取该segment的音频数据
        seg_audio = item.audio_data[segment['start_idx']:segment['end_idx']]
        
        try:
            # 用细VAD重新检测
            from silero_vad import get_speech_timestamps
            
            speech_timestamps = get_speech_timestamps(
                seg_audio,
                self.fine_vad.model_manager.get_model(),
                sampling_rate=self.fine_vad_config['sampling_rate'],
                min_speech_duration_ms=self.fine_vad_config['min_speech_duration_ms'],
                min_silence_duration_ms=self.fine_vad_config['min_silence_duration_ms'],
                threshold=self.fine_vad_config['threshold'],
                neg_threshold=self.fine_vad_config['neg_threshold'],
                speech_pad_ms=self.fine_vad_config['speech_pad_ms'],
                return_seconds=False  # 返回采样点索引
            )
            
            if not speech_timestamps:
                # 没有检测到语音，返回原segment
                return [segment]
            
            # 收集所有潜在切分点（语音段起始点）
            split_points = {0, len(seg_audio)}
            for ts in speech_timestamps:
                split_points.add(ts['start'])
            
            sorted_points = sorted(list(split_points))
            
            # 按target_duration寻找切分点
            target_samples = int(self.target_duration * item.sample_rate)
            max_samples = int(self.max_duration * item.sample_rate)
            
            final_splits = [0]
            target_pos = target_samples
            
            while target_pos < len(seg_audio):
                # 找最接近target的切分点
                closest = min(sorted_points, key=lambda p: abs(p - target_pos))
                final_splits.append(closest)
                target_pos = closest + target_samples
            
            final_splits.append(len(seg_audio))
            final_splits = sorted(list(set(final_splits)))
            
            # 硬性保护：确保不超过max_duration
            protected_splits = [0]
            for i in range(1, len(final_splits)):
                segment_len = final_splits[i] - final_splits[i-1]
                if segment_len <= max_samples:
                    protected_splits.append(final_splits[i])
                else:
                    # 强制等分
                    num_parts = int(np.ceil(segment_len / max_samples))
                    part_len = segment_len / num_parts
                    for j in range(1, num_parts):
                        protected_splits.append(int(final_splits[i-1] + j * part_len))
                    protected_splits.append(final_splits[i])
            
            # 构建新的segments
            result = []
            base_time = segment['start_time']
            base_idx = segment['start_idx']
            
            for i in range(len(protected_splits) - 1):
                start_offset = protected_splits[i]
                end_offset = protected_splits[i + 1]
                
                result.append({
                    'start_time': base_time + start_offset / item.sample_rate,
                    'end_time': base_time + end_offset / item.sample_rate,
                    'duration': (end_offset - start_offset) / item.sample_rate,
                    'start_idx': base_idx + start_offset,
                    'end_idx': base_idx + end_offset
                })
            
            return result
            
        except Exception as e:
            # 细VAD失败，回退到等分
            print("Vad fail")
            duration = segment['duration']
            num_parts = int(np.ceil(duration / self.max_duration))
            part_duration = duration / num_parts
            part_samples = int(part_duration * item.sample_rate)
            
            result = []
            for i in range(num_parts):
                start_idx = segment['start_idx'] + i * part_samples
                end_idx = min(segment['start_idx'] + (i + 1) * part_samples, segment['end_idx'])
                
                result.append({
                    'start_time': segment['start_time'] + i * part_duration,
                    'end_time': segment['start_time'] + (i + 1) * part_duration,
                    'duration': (end_idx - start_idx) / item.sample_rate,
                    'start_idx': start_idx,
                    'end_idx': end_idx
                })
            
            return result