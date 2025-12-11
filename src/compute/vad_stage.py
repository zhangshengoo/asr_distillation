"""VAD处理阶段 - Ray流水线集成"""

import time
from typing import Dict, List, Any, Tuple, Iterator
from pathlib import Path
import logging

import ray
import numpy as np

from src.compute.vad import VADProcessor, AudioSegment, VADResult
from src.scheduling.pipeline import PipelineStage
from src.common import BatchData, TensorItem, SegmentItem


class VADProcessingStage(PipelineStage):
    """VAD处理阶段 - Ray Actor"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger = logging.getLogger("VADProcessingStage")
        self.vad_config = config.get('vad', {})
        self.batch_size = config.get('batch_size', 32)
        self.cache_enabled = self.vad_config.get('cache_enabled', True)
        
        # 初始化VAD处理器
        self.vad_processor = None
        self._init_vad_processor()
        
        # 统计信息
        self.stats = {
            'processed_files': 0,
            'total_segments': 0,
            'total_speech_duration': 0.0,
            'processing_time': 0.0
        }
    
    def _init_vad_processor(self) -> None:
        """初始化VAD处理器"""
        # 确保缓存目录存在
        cache_dir = self.vad_config.get('cache_dir', './cache/vad')
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        
        self.vad_processor = VADProcessor(self.vad_config)
    
    def _vad_process(self, item: TensorItem) -> TensorItem:
        """Process VAD for single item"""
        try:
             # TensorItem has waveform as numpy array as per definition
             audio_data = item.waveform
             sample_rate = item.sample_rate
             
             vad_result = self.vad_processor.process_audio(item.file_id, audio_data, sample_rate)
             segments = self.vad_processor.segment_audio(
                item.file_id, audio_data, sample_rate, vad_result
             )
             
             # Enrich item metadata with VAD results
             # But TensorItem is typed. We can't just add fields dynamically if we want strict typing.
             # However, common implementation allows metadata dict.
             # Or we assume downstream expansion stage handles this.
             # But existing logic puts 'segments' directly on item dict.
             # Let's verify common.py again. 
             # TensorItem inherits SourceItem (metadata dict).
             # We can put vad results in metadata.
             
             new_metadata = item.metadata.copy()
             new_metadata.update({
                 'vad_result': vad_result,
                 'segments': segments,
                 'original_duration': vad_result.original_duration,
                 'speech_ratio': vad_result.speech_ratio,
                 'num_segments': len(segments)
             })

             return TensorItem(
                 file_id=item.file_id,
                 oss_path=item.oss_path,
                 format=item.format,
                 duration=item.duration,
                 metadata=new_metadata,
                 waveform=item.waveform,
                 sample_rate=item.sample_rate
             )
        except Exception as e:
            self.logger.error(f"VAD processing failed for item {item.file_id}: {e}")
            return None

    def process(self, batch: BatchData[TensorItem]) -> BatchData[TensorItem]:
        """处理音频批次"""
        start_time = time.time()
        
        # 使用 map 处理
        new_batch = batch.map(self._vad_process, new_batch_id=f"{batch.batch_id}_vad_processed")

        # 更新处理时间
        self.stats['processing_time'] += time.time() - start_time
        # Use existing metadata update logic if needed or just return new_batch
        new_batch.metadata.update({
             'stage': 'vad_processing',
             'processing_time': self.stats['processing_time']
        })
        
        return new_batch
    
    def get_stats(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        stats = self.stats.copy()
        
        # 计算平均处理时间
        if stats['processed_files'] > 0:
            stats['avg_processing_time'] = stats['processing_time'] / stats['processed_files']
            stats['avg_segments_per_file'] = stats['total_segments'] / stats['processed_files']
            stats['avg_speech_duration'] = stats['total_speech_duration'] / stats['processed_files']
        
        # 获取VAD缓存统计
        if self.vad_processor and hasattr(self.vad_processor, 'cache'):
            # 检查cache对象是否有get_cache_stats方法
            if hasattr(self.vad_processor.cache, 'get_cache_stats'):
                cache_stats = self.vad_processor.cache.get_cache_stats()
                stats['cache_stats'] = cache_stats
        
        return stats
    
    def cleanup(self) -> None:
        """清理资源"""
        try:
            if self.vad_processor and hasattr(self.vad_processor, 'cache'):
                self.vad_processor.cache.clear()
        except Exception as e:
            self.logger.error(f"Failed to cleanup VAD resources: {e}")


class VADStageAdapter:
    """VAD阶段适配器 - 用于与非Ray系统集成"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vad_stage = None
    
    def initialize(self) -> None:
        """初始化VAD阶段"""
        self.vad_stage = VADProcessingStage.remote(self.config)
    
    def process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """处理批次"""
        if self.vad_stage is None:
            self.initialize()
        
        try:
            # 使用Ray远程调用
            results_ref = self.vad_stage.process_batch.remote(batch)
            results = ray.get(results_ref)
            return results
        except Exception as e:
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        if self.vad_stage is None:
            return {}
        
        try:
            stats_ref = self.vad_stage.get_stats.remote()
            return ray.get(stats_ref)
        except Exception as e:
            return {}
    
    def cleanup(self) -> None:
        """清理资源"""
        if self.vad_stage is not None:
            try:
                self.vad_stage.cleanup.remote()
                ray.kill(self.vad_stage)
                self.vad_stage = None
            except Exception as e:
                pass


# 工厂函数
def create_vad_stage(config: Dict[str, Any]) -> VADProcessingStage:
    """创建VAD处理阶段"""
    return VADProcessingStage(config)


def create_vad_adapter(config: Dict[str, Any]) -> VADStageAdapter:
    """创建VAD阶段适配器"""
    return VADStageAdapter(config)


# 向后兼容的别名
VADStage = VADProcessingStage