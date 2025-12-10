"""VAD处理阶段 - Ray流水线集成"""

import time
from typing import Dict, List, Any, Tuple, Iterator
from pathlib import Path

import ray
import numpy as np
from loguru import logger

from src.compute.vad import VADProcessor, AudioSegment, VADResult
from src.compute.audio_processor import PipelineStage


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
        try:
            # 确保缓存目录存在
            cache_dir = self.vad_config.get('cache_dir', './cache/vad')
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            
            self.vad_processor = VADProcessor(self.vad_config)
            logger.info(f"VAD处理器初始化成功，缓存目录: {cache_dir}")
        except Exception as e:
            logger.error(f"VAD处理器初始化失败: {e}")
            raise
    
    def process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """处理音频批次"""
        start_time = time.time()
        results = []
        
        try:
            # 准备音频数据
            audio_batch = []
            for item in batch:
                file_id = item.get('file_id')
                audio_data = item.get('audio_data')
                sample_rate = item.get('sample_rate')
                
                if audio_data is not None and sample_rate is not None:
                    audio_batch.append((file_id, audio_data, sample_rate))
                else:
                    logger.warning(f"跳过无效音频数据: {file_id}")
                    results.append({
                        'file_id': file_id,
                        'error': 'Invalid audio data',
                        'segments': []
                    })
            
            # 批量VAD处理
            if audio_batch:
                vad_results = self.vad_processor.batch_process(audio_batch)
                
                # 处理每个文件的VAD结果
                for (file_id, audio_data, sample_rate), vad_result in zip(audio_batch, vad_results):
                    try:
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
                            'num_segments': len(segments)
                        }
                        
                        results.append(result)
                        
                        # 更新统计信息
                        self.stats['processed_files'] += 1
                        self.stats['total_segments'] += len(segments)
                        self.stats['total_speech_duration'] += vad_result.total_speech_duration
                        
                    except Exception as e:
                        logger.error(f"音频切分失败 {file_id}: {e}")
                        results.append({
                            'file_id': file_id,
                            'error': str(e),
                            'segments': []
                        })
            
            # 更新处理时间
            self.stats['processing_time'] += time.time() - start_time
            
            # 记录批次统计
            batch_size = len(batch)
            total_segments = sum(r.get('num_segments', 0) for r in results)
            logger.info(f"VAD批次处理完成: {batch_size}个文件, {total_segments}个语音片段")
            
            return results
            
        except Exception as e:
            logger.error(f"VAD批次处理失败: {e}")
            # 返回错误结果，保持批次完整性
            error_results = []
            for item in batch:
                error_results.append({
                    'file_id': item.get('file_id'),
                    'error': str(e),
                    'segments': []
                })
            return error_results
    
    def process(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """处理单个音频文件（用于单文件处理）"""
        return self.process_batch([item])[0]
    
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
            cache_stats = self.vad_processor.cache.get_cache_stats()
            stats['cache_stats'] = cache_stats
        
        return stats
    
    def cleanup(self) -> None:
        """清理资源"""
        try:
            if self.vad_processor and hasattr(self.vad_processor, 'cache'):
                self.vad_processor.cache.clear()
            logger.info("VAD处理阶段资源清理完成")
        except Exception as e:
            logger.error(f"VAD处理阶段资源清理失败: {e}")


class VADStageAdapter:
    """VAD阶段适配器 - 用于与非Ray系统集成"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vad_stage = None
    
    def initialize(self) -> None:
        """初始化VAD阶段"""
        self.vad_stage = VADProcessingStage.remote(self.config)
        logger.info("VAD阶段适配器初始化完成")
    
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
            logger.error(f"VAD批次处理失败: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        if self.vad_stage is None:
            return {}
        
        try:
            stats_ref = self.vad_stage.get_stats.remote()
            return ray.get(stats_ref)
        except Exception as e:
            logger.error(f"获取VAD统计信息失败: {e}")
            return {}
    
    def cleanup(self) -> None:
        """清理资源"""
        if self.vad_stage is not None:
            try:
                self.vad_stage.cleanup.remote()
                ray.kill(self.vad_stage)
                self.vad_stage = None
                logger.info("VAD阶段适配器清理完成")
            except Exception as e:
                logger.error(f"VAD阶段适配器清理失败: {e}")


# 工厂函数
def create_vad_stage(config: Dict[str, Any]) -> VADProcessingStage:
    """创建VAD处理阶段"""
    return VADProcessingStage(config)


def create_vad_adapter(config: Dict[str, Any]) -> VADStageAdapter:
    """创建VAD阶段适配器"""
    return VADStageAdapter(config)


# 向后兼容的别名
VADStage = VADProcessingStage