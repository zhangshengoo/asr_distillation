"""Segment处理阶段 - 音频片段展开与聚合"""

import time
import uuid
from typing import Dict, List, Any
from dataclasses import dataclass
import logging

import ray

from src.scheduling.pipeline import PipelineStage
from src.common import BatchData, TensorItem, SegmentItem, InferenceItem, FileResultItem

import numpy as np


class SegmentExpansionStage(PipelineStage):
    """音频片段展开阶段 - 将VAD结果展开为segment级别的items
    
    功能：
    1. 将文件级别的VAD结果展开为segment级别的items
    2. 智能切分超长片段（>178秒）：优先在静音处切分，否则等分
    3. 过滤无效的segments
    4. 保持与后续阶段的接口兼容性
    5. 支持音频片段上传到OSS（可选）
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger = logging.getLogger("SegmentExpansionStage")
        self.min_segment_duration = config.get('min_segment_duration', 0.5)
        self.max_segment_duration = config.get('max_segment_duration', 178)      # 硬性上限
        self.segment_threshold = config.get('segment_threshold', 120)            # 目标切分间隔
        self.preserve_order = config.get('preserve_order', True)
        
        # ✅ 超长片段重新VAD的参数（更严格）
        self.resplit_vad_params = {
            'sampling_rate': config.get('sampling_rate', 16000),
            'min_speech_duration_ms': config.get('resplit_min_speech_ms', 1500),
            'min_silence_duration_ms': config.get('resplit_min_silence_ms', 300),   # 300ms静音即可切分
            'threshold': config.get('resplit_threshold', 0.4),
            'neg_threshold': config.get('resplit_neg_threshold', 0.15),
            'speech_pad_ms': config.get('resplit_speech_pad_ms', 100),
            'return_seconds': False  # 返回采样点索引
        }
        
        # VAD模型（懒加载，仅在需要切分超长片段时使用）
        self.vad_model = None
        
        # 初始化音频片段上传功能（如果启用）
        self.segment_uploader = None
        self.enable_segment_upload = config.get('segment_upload', {}).get('enable_segment_upload', False)
        
        if self.enable_segment_upload and ('input_storage' in config and 'output_storage' in config):
            from src.data.storage import MediaStorageManager
            from src.storage.result_writer import AudioSegmentUploader, SegmentUploadConfig
            
            # 使用分离的输入和输出存储配置 - segment上传使用输出存储
            storage_manager = MediaStorageManager(
                input_config=config['input_storage'],
                output_config=config['output_storage']
            )
            segment_config = SegmentUploadConfig(**config.get('segment_upload', {}))
            self.segment_uploader = AudioSegmentUploader(segment_config, storage_manager)
        
        # 统计信息
        self.stats = {
            'processed_files': 0,
            'total_segments': 0,
            'filtered_segments': 0,
            'split_segments': 0,
            'smart_split_count': 0,      # 智能切分（在静音处）
            'forced_split_count': 0,      # 强制等分
            'processing_time': 0.0
        }
    
    def _ensure_vad_model(self):
        """懒加载VAD模型（仅在需要重新切分时加载）"""
        if self.vad_model is None:
            from silero_vad import load_silero_vad
            self.vad_model = load_silero_vad(onnx=False)
            self.logger.info("Loaded VAD model for resplitting long segments")
    
    def _split_long_segment(self, segment_data, file_id: str) -> List:
        """切分超长片段 - 参考Qwen3-ASR-Toolkit智能切分逻辑
        
        策略：
        1. 用更严格的VAD重新检测，找到所有语音段起始点（静音位置）
        2. 每隔segment_threshold秒，找最近的静音点切分
        3. 如果仍超过max_segment_duration，强制等分
        
        Args:
            segment_data: AudioSegment对象
            file_id: 文件ID
            
        Returns:
            切分后的AudioSegment列表
        """
        from src.compute.vad import AudioSegment
        from silero_vad import get_speech_timestamps
        
        duration = segment_data.duration
        sample_rate = segment_data.sample_rate
        audio_data = segment_data.audio_data
        total_samples = len(audio_data)
        
        # 不需要切分
        if duration <= self.max_segment_duration:
            return [segment_data]
        
        # ✅ Step 1: 用更严格的VAD重新检测，找到所有潜在切分点
        self._ensure_vad_model()
        
        try:
            speech_timestamps = get_speech_timestamps(
                audio_data,
                self.vad_model,
                **self.resplit_vad_params
            )
            
            if not speech_timestamps:
                raise ValueError("No speech detected in resplit VAD")
            
            # 收集所有语音段的起始点（静音后的位置，适合切分）
            potential_split_points = {0, total_samples}
            for ts in speech_timestamps:
                potential_split_points.add(ts['start'])
            
            sorted_splits = sorted(list(potential_split_points))
            
            # ✅ Step 2: 按segment_threshold间隔寻找最近的切分点
            segment_threshold_samples = int(self.segment_threshold * sample_rate)
            final_split_points = {0, total_samples}
            
            target_time = segment_threshold_samples
            while target_time < total_samples:
                # 找到最接近目标时间的VAD切分点
                closest_point = min(sorted_splits, key=lambda p: abs(p - target_time))
                final_split_points.add(closest_point)
                target_time += segment_threshold_samples
            
            final_ordered_splits = sorted(list(final_split_points))
            
            # ✅ Step 3: 硬性保护 - 确保每段不超过max_segment_duration
            max_segment_samples = int(self.max_segment_duration * sample_rate)
            new_split_points = [0]
            
            for i in range(1, len(final_ordered_splits)):
                start = final_ordered_splits[i - 1]
                end = final_ordered_splits[i]
                segment_length = end - start
                
                if segment_length <= max_segment_samples:
                    new_split_points.append(end)
                else:
                    # 仍然超过最大限制，强制等分
                    num_subsegments = int(np.ceil(segment_length / max_segment_samples))
                    subsegment_length = segment_length / num_subsegments
                    
                    for j in range(1, num_subsegments):
                        split_point = int(start + j * subsegment_length)
                        new_split_points.append(split_point)
                    
                    new_split_points.append(end)
                    self.stats['forced_split_count'] += 1
            
            self.stats['smart_split_count'] += 1
            self.logger.debug(
                f"Smart split: segment {segment_data.segment_id} ({duration:.1f}s) "
                f"-> {len(new_split_points)-1} parts using VAD-detected split points"
            )
            
        except Exception as e:
            # VAD检测失败，回退到等分策略
            self.logger.warning(
                f"VAD resplit failed for segment {segment_data.segment_id}: {e}, "
                f"using equal split fallback"
            )
            max_segment_samples = int(self.max_segment_duration * sample_rate)
            num_subsegments = int(np.ceil(total_samples / max_segment_samples))
            subsegment_length = total_samples / num_subsegments
            
            new_split_points = [int(i * subsegment_length) for i in range(num_subsegments + 1)]
            new_split_points[-1] = total_samples
            self.stats['forced_split_count'] += 1
        
        # ✅ Step 4: 根据切分点创建子片段
        sub_segments = []
        for i in range(len(new_split_points) - 1):
            start_sample = new_split_points[i]
            end_sample = new_split_points[i + 1]
            
            # 计算时间范围
            chunk_start_time = segment_data.start_time + (start_sample / sample_rate)
            chunk_end_time = segment_data.start_time + (end_sample / sample_rate)
            chunk_duration = chunk_end_time - chunk_start_time
            
            # 提取音频数据
            chunk_audio = audio_data[start_sample:end_sample]
            
            # 创建子片段
            sub_segment = AudioSegment(
                file_id=file_id,
                segment_id=f"{segment_data.segment_id}_part{i}",
                audio_data=chunk_audio,
                start_time=chunk_start_time,
                end_time=chunk_end_time,
                duration=chunk_duration,
                sample_rate=sample_rate,
                original_duration=segment_data.original_duration
            )
            sub_segments.append(sub_segment)
        
        self.logger.info(
            f"Split long segment {segment_data.segment_id} ({duration:.1f}s) "
            f"into {len(sub_segments)} parts (avg: {duration/len(sub_segments):.1f}s each)"
        )
        
        return sub_segments
    
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
            
            # 处理每个segment（过滤 + 切分超长）
            processed_segments = []
            for segment_data in segments_data:
                # 过滤过短片段
                if segment_data.duration < self.min_segment_duration:
                    self.stats['filtered_segments'] += 1
                    continue
                
                # ✅ 处理超长片段（智能切分）
                if segment_data.duration > self.max_segment_duration:
                    sub_segments = self._split_long_segment(segment_data, item.file_id)
                    processed_segments.extend(sub_segments)
                    self.stats['split_segments'] += 1
                else:
                    processed_segments.append(segment_data)
            
            # 过滤并创建SegmentItem列表
            valid_segments = self._filter_segments(processed_segments)
            segment_items = []
            
            for idx, segment_data in enumerate(valid_segments):
                # 如果启用了segment上传，上传音频片段
                segment_oss_path = None
                if self.segment_uploader and hasattr(segment_data, 'audio_data') and segment_data.audio_data is not None:
                    original_oss_path = getattr(item, 'oss_path', getattr(item, 'file_id', 'unknown'))
                    segment_id = f"seg_{getattr(item, 'file_id', uuid.uuid4().hex[:8])}_{idx}_{int(segment_data.start_time*1000)}_{int(segment_data.end_time*1000)}"
                    
                    segment_oss_path = self.segment_uploader.upload_segment(
                        segment_data.audio_data,
                        original_oss_path,
                        segment_data.start_time,
                        segment_data.end_time,
                        segment_data.sample_rate,
                        segment_id
                    )
                    
                    # 记录元数据
                    if segment_oss_path:
                        segment_metadata = {
                            'segment_id': segment_id,
                            'original_oss_path': original_oss_path,
                            'segment_oss_path': segment_oss_path,
                            'start_time': segment_data.start_time,
                            'end_time': segment_data.end_time,
                            'duration': segment_data.duration,
                            'original_file_id': getattr(item, 'file_id', ''),
                            'timestamp': time.time(),
                            'processing_metadata': getattr(item, 'metadata', {})
                        }
                        
                        self.segment_uploader.write_segment_metadata(segment_metadata)
                
                segment_item = SegmentItem(
                    file_id=f"{item.file_id}_seg_{idx}",
                    parent_file_id=item.file_id,
                    segment_id=segment_data.segment_id,
                    segment_index=idx,
                    start_time=segment_data.start_time,
                    end_time=segment_data.end_time,
                    waveform=segment_data.audio_data,
                    original_duration=segment_data.original_duration,
                    oss_path=segment_oss_path if segment_oss_path else item.oss_path,
                    metadata={
                        **item.metadata,
                        'sample_rate': segment_data.sample_rate,
                        'duration': segment_data.duration,
                        'processing_timestamp': time.time(),
                        'segment_oss_path': segment_oss_path  # 添加segment OSS路径到metadata
                    }
                )
                segment_items.append(segment_item)
            
            return segment_items
            
        except Exception as e:
            self.logger.error(f"Failed to expand item {item.file_id}: {e}")
            return []
    
    def _filter_segments(self, segments: List[Any]) -> List[Any]:
        """过滤无效的segments"""
        valid_segments = []
        
        for segment in segments:
            duration = getattr(segment, 'duration', 0)
            
            # 检查时长范围
            if duration < self.min_segment_duration:
                continue
            
            # 检查音频数据
            if not hasattr(segment, 'audio_data') or segment.audio_data is None:
                continue
            
            # 检查音频数据是否为空数组
            if hasattr(segment.audio_data, '__len__') and len(segment.audio_data) == 0:
                continue
            
            valid_segments.append(segment)
        
        # 保持顺序
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
        
        if stats['split_segments'] > 0:
            stats['smart_split_ratio'] = stats['smart_split_count'] / stats['split_segments']
            stats['forced_split_ratio'] = stats['forced_split_count'] / stats['split_segments']
        
        # 添加segment上传统计
        if self.segment_uploader:
            stats.update(self.segment_uploader.get_stats())
        
        return stats
    
    def cleanup(self):
        """清理资源"""
        if self.segment_uploader:
            self.segment_uploader.close()
            self.logger.info("Segment uploader closed")
        super().cleanup()

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