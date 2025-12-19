"""Segment展开和上传Stage"""
import io
import wave
import numpy as np
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from .base import ExpandStage
from ..data_structures import ProcessingItem, SegmentItem


class SegmentExpandAndUploadStage(ExpandStage):
    """
    展开segments并并发上传
    关键：使用ThreadPoolExecutor实现文件内并发
    """
    
    def __init__(self, storage_manager, config: dict, max_upload_workers: int = 4):
        self.storage = storage_manager
        self.config = config
        self.max_workers = max_upload_workers
        self.audio_segment_prefix = config.get('audio_segment_prefix', 'audio_segments/')
    
    def name(self) -> str:
        return "segment_expand_upload"
    
    def expand(self, item: ProcessingItem) -> List[SegmentItem]:
        """展开并并发上传"""
        if not item.segments:
            return []
        
        # 1. 创建所有segment items（audio_data是view，不额外占内存）
        seg_items = self._create_segment_items(item)
        
        # 2. 并发上传
        self._concurrent_upload(seg_items)
        
        # 3. ✅ 清理所有segment的audio_data（节省内存）
        for seg_item in seg_items:
            seg_item.clear_audio_data()
        
        return seg_items
    
    def _create_segment_items(self, item: ProcessingItem) -> List[SegmentItem]:
        """创建segment items"""
        seg_items = []
        
        for i, seg_info in enumerate(item.segments):
            seg_id = (f"seg_{item.file_id}_{i}_"
                     f"{int(seg_info['start_time']*1000)}_"
                     f"{int(seg_info['end_time']*1000)}")
            
            seg_item = SegmentItem(
                file_id=seg_id,
                oss_path=item.oss_path,
                parent_file_id=item.file_id,
                segment_index=i,
                audio_data=item.audio_data[seg_info['start_idx']:seg_info['end_idx']],  # numpy view
                sample_rate=item.sample_rate,
                start_time=seg_info['start_time'],
                end_time=seg_info['end_time'],
                segment_duration=seg_info['duration']
            )
            seg_items.append(seg_item)
        
        return seg_items
    
    def _concurrent_upload(self, seg_items: List[SegmentItem]):
        """并发上传所有segments"""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有上传任务
            future_to_seg = {
                executor.submit(self._upload_single_segment, seg): seg
                for seg in seg_items
            }
            
            # 收集结果
            for future in as_completed(future_to_seg):
                seg_item = future_to_seg[future]
                try:
                    seg_item.segment_oss_path = future.result()
                    seg_item.mark_success()
                except Exception as e:
                    seg_item.mark_failed("upload", e)
    
    def _upload_single_segment(self, seg_item: SegmentItem) -> str:
        """
        上传单个segment（线程安全）
        
        Returns:
            上传后的OSS路径
        """
        # 转为WAV bytes
        wav_bytes = self._to_wav_bytes(seg_item.audio_data, seg_item.sample_rate)
        
        # 生成OSS路径
        oss_path = f"{self.audio_segment_prefix}{seg_item.file_id}.wav"
        
        # 上传
        self.storage.upload_bytes(wav_bytes, oss_path)
        
        return oss_path
    
    def _to_wav_bytes(self, audio_data: np.ndarray, sample_rate: int) -> bytes:
        """将numpy数组转为WAV格式bytes"""
        buffer = io.BytesIO()
        
        with wave.open(buffer, 'wb') as wav_file:
            # 设置WAV参数
            wav_file.setnchannels(1)  # 单声道
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            
            # 转为int16
            audio_int16 = (audio_data * 32767).astype(np.int16)
            wav_file.writeframes(audio_int16.tobytes())
        
        buffer.seek(0)
        return buffer.read()