"""4个音频处理Stage的Processor实现

将现有的Stage类适配到新的StageProcessor接口
"""

import logging
from typing import Dict, List, Any
import numpy as np

from data_process.simple_ray_pipeline import StageProcessor, ProcessBatch
from src.common import BatchData, SourceItem, RawAudioItem, TensorItem, SegmentItem

logger = logging.getLogger(__name__)


# ==================== Stage 1: 音频下载 ====================

class AudioDownloadProcessor(StageProcessor):
    """音频下载处理器"""
    
    def __init__(self, config: Dict[str, Any], stage_name: str):
        super().__init__(config, stage_name)
        
        # 初始化下载相关组件
        from src.data.storage import MediaStorageManager
        from src.data.media_indexer import MediaDataLoader
        
        self.storage_manager = MediaStorageManager(
            input_config=config['input_storage'],
            output_config=config['output_storage']
        )
        
        self.data_loader = MediaDataLoader(config.get('data', {}))
        
        # 多媒体支持
        self.enable_multimedia = config.get('enable_multimedia', True)
        if self.enable_multimedia:
            from src.compute.media import (
                MediaDetector, MediaExtractor, BatchMediaProcessor,
                MediaConfig, CacheConfig
            )
            
            media_config = MediaConfig(**config.get('media', {}))
            cache_config = CacheConfig(
                enabled=media_config.cache_enable,
                cache_dir=config.get('media', {}).get('cache_dir', './cache/media'),
                max_size_gb=media_config.cache_max_size_gb,
                ttl_hours=media_config.cache_ttl_hours
            )
            
            self.detector = MediaDetector()
            self.extractor = MediaExtractor(media_config)
            self.batch_processor = BatchMediaProcessor(media_config, cache_config)
    
    def process(self, batch: ProcessBatch) -> ProcessBatch:
        """下载音频数据
        
        输入: batch.data = List[Dict] (包含file_id, oss_path等)
        输出: batch.data = List[RawAudioItem]
        """
        records = batch.data
        raw_items = []
        
        for record in records:
            file_id = record['file_id']
            oss_path = record['oss_path']
            
            # 检查缓存
            cached = self.data_loader.get_cached_media(file_id, 'audio')
            if cached and cached.exists():
                with open(cached, 'rb') as f:
                    audio_bytes = f.read()
            else:
                # 下载
                import tempfile
                import os
                
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    success = self.storage_manager.download_audio(oss_path, tmp.name)
                    if not success:
                        logger.error(f"下载失败: {oss_path}")
                        continue
                    
                    with open(tmp.name, 'rb') as f:
                        audio_bytes = f.read()
                    
                    os.unlink(tmp.name)
                
                # 缓存
                self.data_loader.cache_media(file_id, audio_bytes, 'audio')
            
            # 创建RawAudioItem
            raw_item = {
                'file_id': file_id,
                'oss_path': oss_path,
                'format': record.get('format', 'wav'),
                'audio_bytes': audio_bytes,
                'metadata': record.get('metadata', {})
            }
            raw_items.append(raw_item)
        
        batch.data = raw_items
        logger.debug(f"下载完成: {len(raw_items)} 个文件")
        return batch


# ==================== Stage 2: 音频预处理 ====================

class AudioPreprocessingProcessor(StageProcessor):
    """音频预处理处理器"""
    
    def __init__(self, config: Dict[str, Any], stage_name: str):
        super().__init__(config, stage_name)
        
        # 初始化预处理器
        from src.compute.audio_processor import AudioPreprocessor, AudioConfig
        
        audio_config = AudioConfig(**config.get('audio', {}))
        self.preprocessor = AudioPreprocessor(audio_config)
    
    def process(self, batch: ProcessBatch) -> ProcessBatch:
        """预处理音频
        
        输入: batch.data = List[Dict] (包含audio_bytes)
        输出: batch.data = List[Dict] (包含waveform, sample_rate)
        """
        raw_items = batch.data
        tensor_items = []
        
        for item in raw_items:
            try:
                audio_bytes = item['audio_bytes']
                
                # 处理音频
                waveform, sample_rate = self.preprocessor.process_audio(audio_bytes)
                
                # 转换为numpy
                waveform_np = waveform.numpy()
                
                # 创建TensorItem
                tensor_item = {
                    'file_id': item['file_id'],
                    'oss_path': item['oss_path'],
                    'format': item['format'],
                    'waveform': waveform_np,
                    'sample_rate': sample_rate,
                    'metadata': item.get('metadata', {})
                }
                tensor_items.append(tensor_item)
                
            except Exception as e:
                logger.error(f"预处理失败 {item.get('file_id')}: {e}")
                continue
        
        batch.data = tensor_items
        logger.debug(f"预处理完成: {len(tensor_items)} 个文件")
        return batch


# ==================== Stage 3: VAD处理 ====================

class VADProcessor(StageProcessor):
    """VAD处理器"""
    
    def __init__(self, config: Dict[str, Any], stage_name: str):
        super().__init__(config, stage_name)
        
        # 初始化VAD
        from src.compute.vad import VADProcessor as VADProc
        
        self.vad = VADProc(config.get('vad', {}))
    
    def process(self, batch: ProcessBatch) -> ProcessBatch:
        """VAD检测
        
        输入: batch.data = List[Dict] (包含waveform, sample_rate)
        输出: batch.data = List[Dict] (添加vad_result和segments)
        """
        tensor_items = batch.data
        processed_items = []
        
        for item in tensor_items:
            try:
                file_id = item['file_id']
                waveform = item['waveform']
                sample_rate = item['sample_rate']
                
                # VAD检测
                vad_result = self.vad.process_audio(file_id, waveform, sample_rate)
                
                # 切分音频
                segments = self.vad.segment_audio(file_id, waveform, sample_rate, vad_result)
                
                # 添加VAD结果
                item['vad_result'] = vad_result
                item['segments'] = segments
                item['metadata']['num_segments'] = len(segments)
                item['metadata']['speech_ratio'] = vad_result.speech_ratio
                
                processed_items.append(item)
                
            except Exception as e:
                logger.error(f"VAD失败 {item.get('file_id')}: {e}")
                continue
        
        batch.data = processed_items
        logger.debug(f"VAD完成: {len(processed_items)} 个文件")
        return batch


# ==================== Stage 4: 片段展开 ====================

class SegmentExpansionProcessor(StageProcessor):
    """片段展开处理器"""
    
    def __init__(self, config: Dict[str, Any], stage_name: str):
        super().__init__(config, stage_name)
        
        self.min_segment_duration = config.get('min_segment_duration', 0.5)
        self.max_segment_duration = config.get('max_segment_duration', 178)
        self.segment_threshold = config.get('segment_threshold', 120)
        self.sampling_rate = config.get('sampling_rate', 16000)
        
        # VAD重切分参数
        self.resplit_params = {
            'sampling_rate': config.get('sampling_rate', 16000),
            'min_speech_duration_ms': config.get('resplit_min_speech_ms', 1500),
            'min_silence_duration_ms': config.get('resplit_min_silence_ms', 300),
            'threshold': config.get('resplit_threshold', 0.4),
            'neg_threshold': config.get('resplit_neg_threshold', 0.15),
            'speech_pad_ms': config.get('resplit_speech_pad_ms', 100),
            'return_seconds': False
        }
        
        # 懒加载VAD模型
        self.vad_model = None
        
        # Segment上传（可选）
        self.enable_upload = config.get('segment_upload', {}).get('enable_segment_upload', False)
        if self.enable_upload:
            from src.data.storage import MediaStorageManager
            from src.storage.result_writer import AudioSegmentUploader, SegmentUploadConfig
            
            output_config = config['output_storage'].copy()
            segment_config = SegmentUploadConfig(**config.get('segment_upload', {}))
            output_config['result_prefix'] = segment_config.metadata_prefix
            
            storage_manager = MediaStorageManager(
                input_config=config['input_storage'],
                output_config=output_config
            )
            self.segment_uploader = AudioSegmentUploader(segment_config, storage_manager)
        else:
            self.segment_uploader = None
    
    def _ensure_vad_model(self):
        """懒加载VAD模型"""
        if self.vad_model is None:
            from silero_vad import load_silero_vad
            self.vad_model = load_silero_vad(onnx=False)
    
    def _split_long_segment(self, segment_data, file_id: str) -> List:
        """切分超长片段"""
        duration = segment_data.duration
        sample_rate = segment_data.sample_rate
        audio_data = segment_data.audio_data
        
        if duration <= self.max_segment_duration:
            return [segment_data]
        
        # 使用VAD重新检测
        self._ensure_vad_model()
        
        from silero_vad import get_speech_timestamps
        
        try:
            speech_timestamps = get_speech_timestamps(
                audio_data, self.vad_model, **self.resplit_params
            )
            
            # 找到切分点
            split_points = [0, len(audio_data)]
            for ts in speech_timestamps:
                split_points.append(ts['start'])
            
            split_points = sorted(set(split_points))
            
            # 生成子片段
            sub_segments = []
            for i in range(len(split_points) - 1):
                start_sample = split_points[i]
                end_sample = split_points[i + 1]
                
                if end_sample - start_sample < self.sampling_rate * 0.5:
                    continue
                
                chunk_audio = audio_data[start_sample:end_sample]
                chunk_start_time = segment_data.start_time + (start_sample / sample_rate)
                chunk_end_time = segment_data.start_time + (end_sample / sample_rate)
                
                from src.compute.vad import AudioSegment
                sub_seg = AudioSegment(
                    file_id=file_id,
                    segment_id=f"{segment_data.segment_id}_part{i}",
                    audio_data=chunk_audio,
                    start_time=chunk_start_time,
                    end_time=chunk_end_time,
                    duration=chunk_end_time - chunk_start_time,
                    sample_rate=sample_rate,
                    original_duration=segment_data.original_duration
                )
                sub_segments.append(sub_seg)
            
            return sub_segments if sub_segments else [segment_data]
            
        except Exception as e:
            logger.warning(f"VAD重切分失败: {e}, 使用原始片段")
            return [segment_data]
    
    def process(self, batch: ProcessBatch) -> ProcessBatch:
        """展开片段
        
        输入: batch.data = List[Dict] (包含segments列表)
        输出: batch.data = List[Dict] (展开后的segment级别数据)
        """
        vad_items = batch.data
        all_segments = []
        
        for item in vad_items:
            file_id = item['file_id']
            oss_path = item['oss_path']
            segments = item.get('segments', [])
            
            # 处理每个segment
            for idx, segment_data in enumerate(segments):
                # 过滤太短的片段
                if segment_data.duration < self.min_segment_duration:
                    continue
                
                # 切分超长片段
                if segment_data.duration > self.max_segment_duration:
                    sub_segments = self._split_long_segment(segment_data, file_id)
                else:
                    sub_segments = [segment_data]
                
                # 处理每个子片段
                for seg_idx, seg in enumerate(sub_segments):
                    segment_dict = {
                        'file_id': f"{file_id}_seg_{idx}_{seg_idx}",
                        'parent_file_id': file_id,
                        'segment_id': seg.segment_id,
                        'segment_index': idx,
                        'start_time': seg.start_time,
                        'end_time': seg.end_time,
                        'duration': seg.duration,
                        'waveform': seg.audio_data,
                        'sample_rate': seg.sample_rate,
                        'oss_path': oss_path,
                        'metadata': {
                            'original_duration': seg.original_duration
                        }
                    }
                    
                    # 可选：上传segment
                    if self.segment_uploader:
                        segment_oss_path = self.segment_uploader.upload_segment(
                            seg.audio_data, oss_path, seg.start_time, seg.end_time,
                            seg.sample_rate, seg.segment_id
                        )
                        segment_dict['segment_oss_path'] = segment_oss_path
                        
                        # 写入元数据
                        self.segment_uploader.write_segment_metadata({
                            'segment_id': seg.segment_id,
                            'original_oss_path': oss_path,
                            'segment_oss_path': segment_oss_path,
                            'start_time': seg.start_time,
                            'end_time': seg.end_time,
                            'duration': seg.duration,
                            'original_file_id': file_id
                        })
                    
                    all_segments.append(segment_dict)
        
        batch.data = all_segments
        logger.debug(f"展开完成: {len(all_segments)} 个片段")
        return batch