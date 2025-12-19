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
        
        seg_items = self._create_segment_items(item)
        self._concurrent_upload(seg_items)
        
        for seg_item in seg_items:
            seg_item.clear_audio_data()
        
        return seg_items
    
    def process_batch(self, items: List[ProcessingItem]) -> List[List[SegmentItem]]:
        """并行处理批次样本 - 所有segments统一并发上传"""
        if not items:
            return []
        
        # 1. 创建所有segment items
        all_seg_items = []
        item_seg_counts = []
        
        for item in items:
            if not item.segments:
                item_seg_counts.append(0)
                continue
            
            seg_items = self._create_segment_items(item)
            all_seg_items.extend(seg_items)
            item_seg_counts.append(len(seg_items))
        
        # 2. 统一并发上传所有segments
        if all_seg_items:
            self._concurrent_upload_all(all_seg_items)
        
        # 3. 清理audio_data并按item重新组织结果
        results = []
        start_idx = 0
        for count in item_seg_counts:
            if count == 0:
                results.append([])
            else:
                end_idx = start_idx + count
                item_segs = all_seg_items[start_idx:end_idx]
                for seg in item_segs:
                    seg.clear_audio_data()
                results.append(item_segs)
                start_idx = end_idx
        
        return results
    
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
                audio_data=item.audio_data[seg_info['start_idx']:seg_info['end_idx']],
                sample_rate=item.sample_rate,
                start_time=seg_info['start_time'],
                end_time=seg_info['end_time'],
                segment_duration=seg_info['duration']
            )
            seg_items.append(seg_item)
        
        return seg_items
    
    def _concurrent_upload(self, seg_items: List[SegmentItem]):
        """并发上传单个item的所有segments"""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_seg = {
                executor.submit(self._upload_single_segment, seg): seg
                for seg in seg_items
            }
            
            for future in as_completed(future_to_seg):
                seg_item = future_to_seg[future]
                try:
                    seg_item.segment_oss_path = future.result()
                    seg_item.mark_success()
                except Exception as e:
                    seg_item.mark_failed("upload", e)
    
    def _concurrent_upload_all(self, all_seg_items: List[SegmentItem]):
        """并发上传所有segments（跨多个items）"""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_seg = {
                executor.submit(self._upload_single_segment, seg): seg
                for seg in all_seg_items
            }
            
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
        wav_bytes = self._to_wav_bytes(seg_item.audio_data, seg_item.sample_rate)
        oss_path = f"{self.audio_segment_prefix}{seg_item.file_id}.wav"
        self.storage.upload_bytes(wav_bytes, oss_path)
        return oss_path
    
    def _to_wav_bytes(self, audio_data: np.ndarray, sample_rate: int) -> bytes:
        """将numpy数组转为WAV格式bytes"""
        buffer = io.BytesIO()
        
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            audio_int16 = (audio_data * 32767).astype(np.int16)
            wav_file.writeframes(audio_int16.tobytes())
        
        buffer.seek(0)
        return buffer.read()