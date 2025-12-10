"""VAD处理阶段 - Ray流水线集成"""

import time
from typing import Dict, List, Any, Tuple, Iterator
from pathlib import Path

import ray
import numpy as np

from src.compute.vad import VADProcessor, AudioSegment, VADResult
from src.compute.audio_processor import PipelineStage
from src.scheduling.pipeline import DataBatch


class VADProcessingStage(PipelineStage):
    """VAD处理阶段 - Ray Actor"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
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
    
    def process(self, batch: DataBatch) -> DataBatch:
        """处理音频批次"""
        start_time = time.time()
        results = []
        
        # 从batch.items中获取数据
        for item in batch.items:
            file_id = item.get('file_id')
            
            # 检查是否有错误 - 不处理有错误的项目，直接跳过
            if 'error' in item:
                continue  # 不将错误项目加入结果
                
            # 处理音频数据
            audio_data = item.get('audio_data')
            sample_rate = item.get('sample_rate')
            
            # 验证音频数据有效性
            assert audio_data is not None, f"Audio data is None for file {file_id}"
            assert sample_rate is not None, f"Sample rate is None for file {file_id}"
            
            # 执行VAD处理
            vad_result = self.vad_processor.process_audio(file_id, audio_data, sample_rate)
            
            # 根据VAD结果切分音频
            segments = self.vad_processor.segment_audio(
                file_id, audio_data, sample_rate, vad_result
            )
            
            # 构建输出结果
            result = {
                'file_id': file_id,
                'vad_result': vad_result,
                'segments': segments,
                'original_duration': vad_result.original_duration,
                'speech_ratio': vad_result.speech_ratio,
                'num_segments': len(segments),
                # 继承原始项目中的其他字段
                **{k: v for k, v in item.items() if k not in ['audio_data', 'sample_rate']}
            }
            
            results.append(result)
            
            # 更新统计信息
            self.stats['processed_files'] += 1
            self.stats['total_segments'] += len(segments)
            self.stats['total_speech_duration'] += vad_result.total_speech_duration
    
        # 更新处理时间
        self.stats['processing_time'] += time.time() - start_time
        
        # 创建新的DataBatch并返回
        new_batch = DataBatch(
            batch_id=f"{batch.batch_id}_vad_processed",
            items=results,
            metadata={
                **batch.metadata,
                'stage': 'vad_processing',
                'processed_items': len(results),
                'processing_time': self.stats['processing_time']
            }
        )
        
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
            pass


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