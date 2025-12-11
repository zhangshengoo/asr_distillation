"""Segment处理阶段 - 音频片段展开与聚合"""

import time
from typing import Dict, List, Any
from dataclasses import dataclass
import logging

import ray

from src.scheduling.pipeline import PipelineStage
from src.common import BatchData, TensorItem, SegmentItem, InferenceItem, FileResultItem


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
        self.logger = logging.getLogger("SegmentExpansionStage")
        self.min_segment_duration = config.get('min_segment_duration', 0.1)  # 最小片段时长(秒)
        self.preserve_order = config.get('preserve_order', True)  # 保持片段顺序
        
        # 统计信息
        self.stats = {
            'processed_files': 0,
            'total_segments': 0,
            'filtered_segments': 0,
            'processing_time': 0.0
        }
    
    def process(self, batch: BatchData[TensorItem]) -> BatchData[SegmentItem]:
        """处理批次数据，展开为segment级别"""
        start_time = time.time()
        
        # 使用flat_map展开segments
        new_batch = batch.flat_map(self._expand_item, new_batch_id=f"{batch.batch_id}_expanded")
        
        # 更新处理时间
        self.stats['processing_time'] += time.time() - start_time
        self.stats['processed_files'] += len(batch.items)
        self.stats['total_segments'] += len(new_batch.items)
        
        # 更新metadata
        new_batch.metadata.update({
            'stage': 'segment_expansion',
            'original_batch_size': len(batch.items),
            'expanded_batch_size': len(new_batch.items),
            'expansion_ratio': len(new_batch.items) / len(batch.items) if batch.items else 0
        })
        
        return new_batch
    
    def _expand_item(self, item: TensorItem) -> List[SegmentItem]:
        """展开单个TensorItem为多个SegmentItem"""
        try:
            # 从metadata获取VAD segment结果
            segments_data = item.metadata.get('segments', [])
            if not segments_data:
                return []
            
            # 过滤并创建SegmentItem列表
            valid_segments = self._filter_segments(segments_data)
            segment_items = []
            
            for idx, segment_data in enumerate(valid_segments):
                segment_item = SegmentItem(
                    file_id=item.file_id,
                    parent_file_id=item.file_id,
                    segment_id=segment_data.segment_id,
                    segment_index=idx,
                    start_time=segment_data.start_time,
                    end_time=segment_data.end_time,
                    waveform=segment_data.audio_data,
                    original_duration=segment_data.original_duration,
                    oss_path=item.oss_path,
                    metadata={
                        **item.metadata,
                        'sample_rate': segment_data.sample_rate,
                        'duration': segment_data.duration,
                        'processing_timestamp': time.time()
                    }
                )
                segment_items.append(segment_item)
                
            self.stats['filtered_segments'] += len(segments_data) - len(valid_segments)
            return segment_items
        except Exception as e:
            self.logger.error(f"Failed to expand item {item.file_id}: {e}")
            return []
    
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
            
            # 检查音频数据是否为空数组
            if hasattr(segment.audio_data, '__len__') and len(segment.audio_data) == 0:
                continue  # 空音频数据，跳过该segment
            
            valid_segments.append(segment)
        
        # 保持顺序（如果需要）
        if self.preserve_order:
            valid_segments.sort(key=lambda s: getattr(s, 'start_time', 0))
        
        return valid_segments
    
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
        self.logger = logging.getLogger("SegmentAggregationStage")
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
    
    def process(self, batch: BatchData[InferenceItem]) -> BatchData[FileResultItem]:
        """处理批次数据，聚合到文件级别"""
        start_time = time.time()
        
        # 使用group_by按文件ID分组
        file_groups = batch.group_by(lambda item: item.file_id)
        
        # 聚合每个文件的结果
        aggregated_items = []
        for file_id, segments in file_groups.items():
            try:
                aggregated_item = self._aggregate_segments(file_id, segments)
                aggregated_items.append(aggregated_item)
            except Exception as e:
                self.logger.error(f"Failed to aggregate segments for file {file_id}: {e}")
        
        # 更新统计信息
        self.stats['processed_batches'] += 1
        self.stats['aggregated_files'] += len(aggregated_items)
        self.stats['total_segments'] += len(batch.items)
        self.stats['processing_time'] += time.time() - start_time
        
        # 创建新的批次
        new_batch = BatchData(
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
    
    def _aggregate_segments(self, file_id: str, segments: List[InferenceItem]) -> FileResultItem:
        """聚合单个文件的segment结果"""
        # 排序 segments
        if self.sort_by_timestamp:
            sorted_segments = sorted(segments, key=lambda s: s.start_time)
        else:
            sorted_segments = segments
        
        # 聚合转录
        transcriptions = [s.transcription for s in sorted_segments]
        full_transcription = ' '.join(transcriptions)
        
        # 构建segment详情
        segment_details = []
        if self.include_segment_details:
            for s in sorted_segments:
                segment_details.append({
                    'segment_id': s.segment_id,
                    'start_time': s.start_time,
                    'end_time': s.end_time,
                    'duration': s.metadata.get('duration', 0.0),
                    'transcription': s.transcription,
                    'confidence': s.confidence
                })
        
        # 统计信息
        stats = {
            'num_segments': len(sorted_segments),
            'avg_confidence': sum(s.confidence for s in sorted_segments) / len(sorted_segments) if sorted_segments else 0.0,
            'total_duration': sorted_segments[0].original_duration if sorted_segments else 0.0
        }
        
        # 添加文件级别统计
        if self.calculate_file_stats and sorted_segments:
            total_speech_duration = sum(s.metadata.get('duration', 0.0) for s in sorted_segments)
            original_duration = sorted_segments[0].original_duration
            stats.update({
                'original_duration': original_duration,
                'speech_duration': total_speech_duration,
                'speech_ratio': total_speech_duration / original_duration if original_duration > 0 else 0.0,
                'avg_segment_duration': total_speech_duration / len(sorted_segments)
            })
        
        return FileResultItem(
            file_id=file_id,
            transcription=full_transcription,
            segments=segment_details,
            stats=stats,
            metadata=sorted_segments[0].metadata if sorted_segments else {}
        )
    
    
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