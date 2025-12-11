"""Segment处理阶段 - 音频片段展开与聚合"""

import time
from typing import Dict, List, Any
from dataclasses import dataclass

import ray

from src.compute.audio_processor import PipelineStage
from src.scheduling.pipeline import DataBatch


class SegmentExpansionStage(PipelineStage):
    """音频片段展开阶段 - 将VAD结果展开为segment级别的items
    
    功能：
    1. 将文件级别的VAD结果展开为segment级别的items
    2. 为每个segment添加完整的元数据
    3. 过滤无效的segments
    4. 保持与后续阶段的接口兼容性
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.min_segment_duration = config.get('min_segment_duration', 0.1)  # 最小片段时长(秒)
        self.preserve_order = config.get('preserve_order', True)  # 保持片段顺序
        
        # 统计信息
        self.stats = {
            'processed_files': 0,
            'total_segments': 0,
            'filtered_segments': 0,
            'processing_time': 0.0
        }
    
    def process(self, batch: DataBatch) -> DataBatch:
        """处理批次数据，展开为segment级别"""
        start_time = time.time()
        segment_items = []
        
        for item in batch.items:
            file_id = item.get('file_id')
            
            # 处理错误情况 - 不处理有错误的项目，直接跳过
            if 'error' in item:
                continue  # 不将错误项目加入结果
                
            # 检查是否有segments
            segments = item.get('segments', [])
            if not segments:
                continue  # 没有有效segments的项目也不加入结果
                
            # 展开segments
            valid_segments = self._filter_segments(segments)
            for i, segment in enumerate(valid_segments):
                segment_item = self._create_segment_item(item, segment, i)
                segment_items.append(segment_item)
            
            # 更新统计信息
            self.stats['processed_files'] += 1
            self.stats['total_segments'] += len(segments)
            self.stats['filtered_segments'] += len(segments) - len(valid_segments)
        
        # 更新处理时间
        self.stats['processing_time'] += time.time() - start_time
        
        # 创建新的批次
        new_batch = DataBatch(
            batch_id=f"{batch.batch_id}_expanded",
            items=segment_items,
            metadata={
                **batch.metadata,
                'stage': 'segment_expansion',
                'original_batch_size': len(batch.items),
                'expanded_batch_size': len(segment_items),
                'expansion_ratio': len(segment_items) / len(batch.items) if batch.items else 0
            }
        )
        
        return new_batch
    
    def _filter_segments(self, segments: List[Any]) -> List[Any]:
        """过滤无效的segments"""
        valid_segments = []
        
        for segment in segments:
            duration = getattr(segment, 'duration', 0)
            
            # 检查时长范围 - 只过滤过短的segment
            if duration < self.min_segment_duration:
                continue
            
            # 检查音频数据
            if not hasattr(segment, 'audio_data') or segment.audio_data is None:
                continue
            
            valid_segments.append(segment)
        
        # 保持顺序（如果需要）
        if self.preserve_order:
            valid_segments.sort(key=lambda s: getattr(s, 'start_time', 0))
        
        return valid_segments
    
    def _create_segment_item(self, original_item: Dict[str, Any], segment: Any, index: int) -> Dict[str, Any]:
        """创建segment级别的item"""
        segment_item = {
            # 基本信息继承
            'file_id': original_item['file_id'],
            'segment_id': segment.segment_id,
            
            # Segment特定信息
            'start_time': segment.start_time,
            'end_time': segment.end_time,
            
            # 为AudioFeatureStage创建audio_tensor格式
            'audio_tensor': {
                'waveform': segment.audio_data,
                'sample_rate': segment.sample_rate,
                'duration': segment.duration,
                'format': 'tensor'
            },
            
            # 原始文件信息
            'original_duration': segment.original_duration,
            'segment_index': index,
            
            # 保留原始item中的所有重要字段，确保后续阶段能访问
            'oss_path': original_item.get('oss_path', ''),
            'format': original_item.get('format', 'wav'),
            'audio_metadata': original_item.get('audio_metadata', {}),
            
            # VAD相关信息
            'speech_ratio': original_item.get('speech_ratio', 0.0),
            'num_segments': original_item.get('num_segments', 0),
            
            # 其他元数据
            'metadata': original_item.get('metadata', {}),
            'processing_timestamp': time.time()
        }
        
        return segment_item
    
    def get_stats(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        stats = self.stats.copy()
        
        if stats['processed_files'] > 0:
            stats['avg_segments_per_file'] = stats['total_segments'] / stats['processed_files']
            stats['avg_processing_time'] = stats['processing_time'] / stats['processed_files']
            stats['filter_rate'] = stats['filtered_segments'] / stats['total_segments'] if stats['total_segments'] > 0 else 0
        
        return stats


class SegmentAggregationStage(PipelineStage):
    """音频片段聚合阶段 - 将segment级别的结果聚合到文件级别
    
    功能：
    1. 将segment级别的处理结果重新聚合到文件级别
    2. 按时间戳排序segments
    3. 计算文件级别的统计信息
    4. 生成最终的输出格式
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.sort_by_timestamp = config.get('sort_by_timestamp', True)
        self.include_segment_details = config.get('include_segment_details', True)
        self.calculate_file_stats = config.get('calculate_file_stats', True)
        
        # 统计信息
        self.stats = {
            'processed_batches': 0,
            'aggregated_files': 0,
            'total_segments': 0,
            'processing_time': 0.0
        }
    
    def process(self, batch: DataBatch) -> DataBatch:
        """处理批次数据，聚合到文件级别"""
        start_time = time.time()
        
        # 按文件ID分组
        file_groups = self._group_by_file(batch.items)
        
        # 聚合每个文件的结果
        aggregated_items = []
        for file_id, segments in file_groups.items():
            aggregated_item = self._aggregate_file_results(file_id, segments)
            # 只有在没有错误的情况下才添加到结果中
            if 'error' not in aggregated_item:
                aggregated_items.append(aggregated_item)
        
        # 更新统计信息
        self.stats['processed_batches'] += 1
        self.stats['aggregated_files'] += len(aggregated_items)  # 只统计成功聚合的文件
        self.stats['total_segments'] += sum(len(segments) for segments in file_groups.values())
        self.stats['processing_time'] += time.time() - start_time
        
        # 创建新的批次
        new_batch = DataBatch(
            batch_id=f"{batch.batch_id}_aggregated",
            items=aggregated_items,
            metadata={
                **batch.metadata,
                'stage': 'segment_aggregation',
                'aggregated_files': len(aggregated_items),
                'total_segments': self.stats['total_segments']
            }
        )
        
        return new_batch
    
    def _group_by_file(self, items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """按文件ID分组items"""
        file_groups = {}
        
        for item in items:
            file_id = item.get('file_id')
            if file_id not in file_groups:
                file_groups[file_id] = []
            file_groups[file_id].append(item)
        
        return file_groups
    
    def _aggregate_file_results(self, file_id: str, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """聚合单个文件的结果"""
        # 检查是否有错误
        error_segments = [s for s in segments if 'error' in s]
        if error_segments:
            return {
                'file_id': file_id,
                'error': f"Segments contain errors: {len(error_segments)} failed segments",
                'segments': [],
                'transcription': '',
                'confidence': 0.0
            }
        
        # 排序segments（如果需要）
        if self.sort_by_timestamp:
            segments.sort(key=lambda s: s.get('start_time', 0))
        
        # 提取转录结果
        transcriptions = []
        total_confidence = 0.0
        total_speech_duration = 0.0
        
        segment_details = []
        for segment in segments:
            transcription = segment.get('transcription', '')
            confidence = segment.get('confidence', 0.0)
            duration = segment.get('duration', 0.0)
            
            transcriptions.append(transcription)
            total_confidence += confidence
            total_speech_duration += duration
            
            if self.include_segment_details:
                segment_details.append({
                    'segment_id': segment.get('segment_id'),
                    'start_time': segment.get('start_time'),
                    'end_time': segment.get('end_time'),
                    'duration': duration,
                    'transcription': transcription,
                    'confidence': confidence
                })
        
        # 计算聚合统计
        avg_confidence = total_confidence / len(segments) if segments else 0.0
        full_transcription = ' '.join(transcriptions)
        
        # 从第一个segment获取原始元数据（所有segment共享相同的原始元数据）
        first_segment = segments[0] if segments else {}
        
        # 构建聚合结果
        aggregated_result = {
            'file_id': file_id,
            'transcription': full_transcription,
            'confidence': avg_confidence,
            'num_segments': len(segments),
            'total_speech_duration': total_speech_duration,
            'processing_timestamp': time.time(),
            # 保留原始item中的所有重要字段，确保PostProcessingStage能访问
            'duration': first_segment.get('original_duration', total_speech_duration),
            'sample_rate': first_segment.get('sample_rate', 16000),
            'oss_path': first_segment.get('oss_path', ''),
            'format': first_segment.get('format', 'wav'),
            'audio_metadata': first_segment.get('audio_metadata', {}),
            'audio_tensor': first_segment.get('audio_tensor', {}),
            'metadata': first_segment.get('metadata', {})
        }
        
        # 添加详细信息（如果需要）
        if self.include_segment_details:
            aggregated_result['segments'] = segment_details
        
        # 添加文件级别统计（如果需要）
        if self.calculate_file_stats:
            original_duration = segments[0].get('original_duration', total_speech_duration) if segments else 0.0
            speech_ratio = total_speech_duration / original_duration if original_duration > 0 else 0.0
            
            aggregated_result.update({
                'original_duration': original_duration,
                'speech_ratio': speech_ratio,
                'avg_segment_duration': total_speech_duration / len(segments) if segments else 0.0
            })
        
        return aggregated_result
    
    def get_stats(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        stats = self.stats.copy()
        
        if stats['aggregated_files'] > 0:
            stats['avg_segments_per_file'] = stats['total_segments'] / stats['aggregated_files']
            stats['avg_processing_time'] = stats['processing_time'] / stats['processed_batches'] if stats['processed_batches'] > 0 else 0
        
        return stats


# 工厂函数
def create_segment_expansion_stage(config: Dict[str, Any]) -> SegmentExpansionStage:
    """创建SegmentExpansionStage"""
    return SegmentExpansionStage(config)


def create_segment_aggregation_stage(config: Dict[str, Any]) -> SegmentAggregationStage:
    """创建SegmentAggregationStage"""
    return SegmentAggregationStage(config)


# 配置模板
SEGMENT_EXPANSION_CONFIG = {
    'min_segment_duration': 0.1,
    'preserve_order': True
}

SEGMENT_AGGREGATION_CONFIG = {
    'sort_by_timestamp': True,
    'include_segment_details': True,
    'calculate_file_stats': True
}