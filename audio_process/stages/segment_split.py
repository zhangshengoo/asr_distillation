"""Segment切分Stage"""
import numpy as np
import multiprocessing
from typing import List, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from base import Stage
from tools.data_structures import ProcessingItem
from stages.vad import VADProcessor

# 进程级VAD模型缓存
_process_vad_models = {}


def _init_vad_worker(vad_config: dict):
    """初始化工作进程的VAD模型"""
    pid = multiprocessing.current_process().pid
    _process_vad_models[pid] = VADProcessor(vad_config)


def _split_segment_worker(seg_data: Tuple[Dict, np.ndarray, int, float, float, dict]) -> List[Dict]:
    """在工作进程中切分单个segment"""
    segment, audio_data, sample_rate, max_duration, target_duration, vad_config = seg_data
    
    pid = multiprocessing.current_process().pid
    vad_processor = _process_vad_models[pid]
    
    seg_audio = audio_data[segment['start_idx']:segment['end_idx']]
    
    try:
        from silero_vad import get_speech_timestamps
        
        speech_timestamps = get_speech_timestamps(
            seg_audio,
            vad_processor.model_manager.get_model(),
            sampling_rate=vad_config['sampling_rate'],
            min_speech_duration_ms=vad_config['min_speech_duration_ms'],
            min_silence_duration_ms=vad_config['min_silence_duration_ms'],
            threshold=vad_config['threshold'],
            neg_threshold=vad_config['neg_threshold'],
            speech_pad_ms=vad_config['speech_pad_ms'],
            return_seconds=False
        )
        
        if not speech_timestamps:
            return [segment]
        
        split_points = {0, len(seg_audio)}
        for ts in speech_timestamps:
            split_points.add(ts['start'])
        sorted_points = sorted(list(split_points))
        
        target_samples = int(target_duration * sample_rate)
        max_samples = int(max_duration * sample_rate)
        
        final_splits = [0]
        target_pos = target_samples
        while target_pos < len(seg_audio):
            closest = min(sorted_points, key=lambda p: abs(p - target_pos))
            final_splits.append(closest)
            target_pos = closest + target_samples
        final_splits.append(len(seg_audio))
        final_splits = sorted(list(set(final_splits)))
        
        protected_splits = [0]
        for i in range(1, len(final_splits)):
            segment_len = final_splits[i] - final_splits[i-1]
            if segment_len <= max_samples:
                protected_splits.append(final_splits[i])
            else:
                num_parts = int(np.ceil(segment_len / max_samples))
                part_len = segment_len / num_parts
                for j in range(1, num_parts):
                    protected_splits.append(int(final_splits[i-1] + j * part_len))
                protected_splits.append(final_splits[i])
        
        result = []
        base_time = segment['start_time']
        base_idx = segment['start_idx']
        for i in range(len(protected_splits) - 1):
            start_offset = protected_splits[i]
            end_offset = protected_splits[i + 1]
            result.append({
                'start_time': base_time + start_offset / sample_rate,
                'end_time': base_time + end_offset / sample_rate,
                'duration': (end_offset - start_offset) / sample_rate,
                'start_idx': base_idx + start_offset,
                'end_idx': base_idx + end_offset
            })
        return result
        
    except Exception:
        duration = segment['duration']
        num_parts = int(np.ceil(duration / max_duration))
        part_duration = duration / num_parts
        part_samples = int(part_duration * sample_rate)
        
        result = []
        for i in range(num_parts):
            start_idx = segment['start_idx'] + i * part_samples
            end_idx = min(segment['start_idx'] + (i + 1) * part_samples, segment['end_idx'])
            result.append({
                'start_time': segment['start_time'] + i * part_duration,
                'end_time': segment['start_time'] + (i + 1) * part_duration,
                'duration': (end_idx - start_idx) / sample_rate,
                'start_idx': start_idx,
                'end_idx': end_idx
            })
        return result


class SegmentSplitStage(Stage):
    """
    检测超长segment并用细VAD切分
    目标：尽量产生接近target_duration的segments，但不超过max_duration
    """
    
    def __init__(self, max_duration: float = 178, target_duration: float = 120, 
                 num_workers: int = 4):
        self.max_duration = max_duration
        self.target_duration = target_duration
        self.num_workers = num_workers
        
        self.fine_vad_config = {
            'sampling_rate': 16000,
            'min_speech_duration_ms': 1500,
            'min_silence_duration_ms': 300,
            'threshold': 0.4,
            'neg_threshold': 0.15,
            'speech_pad_ms': 100,
            'cache_enabled': False
        }
        self.fine_vad = None
    
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
                new_segments.append(seg)
            else:
                split_segs = self._split_long_segment(seg, item)
                new_segments.extend(split_segs)
        item.segments = new_segments
        return item
    
    def process_batch(self, items: List[ProcessingItem]) -> List[ProcessingItem]:
        """并行处理批次样本中的所有segments"""
        if not items:
            return []
        
        # 收集所有需要切分的segments
        split_tasks = []  # [(item_idx, seg_idx, segment, audio_data, sample_rate)]
        for item_idx, item in enumerate(items):
            for seg_idx, seg in enumerate(item.segments):
                if seg['duration'] > self.max_duration:
                    split_tasks.append((
                        item_idx, seg_idx, seg,
                        item.audio_data, item.sample_rate
                    ))
        
        if not split_tasks:
            return items
        
        # 并行切分所有超长segments
        with ProcessPoolExecutor(
            max_workers=self.num_workers,
            initializer=_init_vad_worker,
            initargs=(self.fine_vad_config,)
        ) as executor:
            futures = {
                executor.submit(
                    _split_segment_worker,
                    (seg, audio_data, sample_rate, self.max_duration, 
                     self.target_duration, self.fine_vad_config)
                ): (item_idx, seg_idx)
                for item_idx, seg_idx, seg, audio_data, sample_rate in split_tasks
            }
            
            # 收集切分结果
            split_results = {}  # {(item_idx, seg_idx): [new_segments]}
            for future in as_completed(futures):
                item_idx, seg_idx = futures[future]
                split_results[(item_idx, seg_idx)] = future.result()
        
        # 重组每个item的segments
        for item_idx, item in enumerate(items):
            new_segments = []
            for seg_idx, seg in enumerate(item.segments):
                if (item_idx, seg_idx) in split_results:
                    new_segments.extend(split_results[(item_idx, seg_idx)])
                else:
                    new_segments.append(seg)
            item.segments = new_segments
        
        return items
    
    def _split_long_segment(self, segment: Dict, item: ProcessingItem) -> List[Dict]:
        """切分超长segment（单个处理时使用）"""
        self._ensure_fine_vad()
        seg_audio = item.audio_data[segment['start_idx']:segment['end_idx']]
        
        try:
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
                return_seconds=False
            )
            
            if not speech_timestamps:
                return [segment]
            
            split_points = {0, len(seg_audio)}
            for ts in speech_timestamps:
                split_points.add(ts['start'])
            sorted_points = sorted(list(split_points))
            
            target_samples = int(self.target_duration * item.sample_rate)
            max_samples = int(self.max_duration * item.sample_rate)
            
            final_splits = [0]
            target_pos = target_samples
            while target_pos < len(seg_audio):
                closest = min(sorted_points, key=lambda p: abs(p - target_pos))
                final_splits.append(closest)
                target_pos = closest + target_samples
            final_splits.append(len(seg_audio))
            final_splits = sorted(list(set(final_splits)))
            
            protected_splits = [0]
            for i in range(1, len(final_splits)):
                segment_len = final_splits[i] - final_splits[i-1]
                if segment_len <= max_samples:
                    protected_splits.append(final_splits[i])
                else:
                    num_parts = int(np.ceil(segment_len / max_samples))
                    part_len = segment_len / num_parts
                    for j in range(1, num_parts):
                        protected_splits.append(int(final_splits[i-1] + j * part_len))
                    protected_splits.append(final_splits[i])
            
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
            
        except Exception:
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