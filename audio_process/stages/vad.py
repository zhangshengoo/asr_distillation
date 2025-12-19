"""VAD处理Stage"""
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List
from base import Stage
from tools.data_structures import ProcessingItem
from stages.vad import VADProcessor


# 全局VAD处理器字典（用于多进程）
_vad_processors = {}


class CoarseVADStage(Stage):
    """粗VAD处理：产生较长的语音片段"""
    
    def __init__(self, vad_config: dict, num_workers: int = 4):
        self.vad_config = vad_config
        self.num_workers = num_workers
        self.vad_processor = VADProcessor(vad_config)  # 单进程使用
    
    def name(self) -> str:
        return "coarse_vad"
    
    def process(self, item: ProcessingItem) -> ProcessingItem:
        """执行VAD检测（单个样本）"""
        vad_result = self.vad_processor.process_audio(
            item.file_id,
            item.audio_data,
            item.sample_rate
        )
        
        item.metadata['vad_result'] = {
            'total_speech_duration': vad_result.total_speech_duration,
            'speech_ratio': vad_result.speech_ratio,
            'num_segments': len(vad_result.speech_segments)
        }
        
        segments = []
        for seg_info in vad_result.speech_segments:
            start_idx = int(seg_info['start'] * item.sample_rate)
            end_idx = int(seg_info['end'] * item.sample_rate)
            
            segments.append({
                'start_time': seg_info['start'],
                'end_time': seg_info['end'],
                'duration': seg_info['end'] - seg_info['start'],
                'start_idx': start_idx,
                'end_idx': end_idx
            })
        
        item.segments = segments
        return item
    
    def process_batch(self, items: List[ProcessingItem]) -> List[ProcessingItem]:
        """并行处理批次样本"""
        if not items:
            return []
        
        with ProcessPoolExecutor(
            max_workers=self.num_workers,
            initializer=_init_worker,
            initargs=(self.vad_config,)
        ) as executor:
            future_to_item = {
                executor.submit(_process_single_item, item): item
                for item in items
            }
            
            results = []
            for future in as_completed(future_to_item):
                try:
                    results.append(future.result())
                except Exception as e:
                    item = future_to_item[future]
                    item.mark_failed("vad", e)
                    results.append(item)
        
        return results


def _init_worker(vad_config: dict):
    """初始化工作进程的VAD处理器"""
    pid = multiprocessing.current_process().pid
    _vad_processors[pid] = VADProcessor(vad_config)


def _process_single_item(item: ProcessingItem) -> ProcessingItem:
    """单个样本的VAD处理（在工作进程中执行）"""
    pid = multiprocessing.current_process().pid
    vad_processor = _vad_processors[pid]
    
    vad_result = vad_processor.process_audio(
        item.file_id,
        item.audio_data,
        item.sample_rate
    )
    
    item.metadata['vad_result'] = {
        'total_speech_duration': vad_result.total_speech_duration,
        'speech_ratio': vad_result.speech_ratio,
        'num_segments': len(vad_result.speech_segments)
    }
    
    segments = []
    for seg_info in vad_result.speech_segments:
        start_idx = int(seg_info['start'] * item.sample_rate)
        end_idx = int(seg_info['end'] * item.sample_rate)
        
        segments.append({
            'start_time': seg_info['start'],
            'end_time': seg_info['end'],
            'duration': seg_info['end'] - seg_info['start'],
            'start_idx': start_idx,
            'end_idx': end_idx
        })
    
    item.segments = segments
    return item