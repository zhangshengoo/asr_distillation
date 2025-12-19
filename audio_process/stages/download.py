"""下载Stage"""
import tempfile
import os
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from base import Stage
from tools.data_structures import ProcessingItem


class DownloadStage(Stage):
    """从OSS下载音频文件"""
    
    def __init__(self, storage_manager, max_workers: int = 8):
        self.storage = storage_manager
        self.max_workers = max_workers
    
    def name(self) -> str:
        return "download"
    
    def process(self, item: ProcessingItem) -> ProcessingItem:
        """下载音频文件"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            success = self.storage.download_audio(item.oss_path, tmp_path)
            if not success:
                raise ValueError(f"Failed to download {item.oss_path}")
            
            with open(tmp_path, 'rb') as f:
                item.audio_bytes = f.read()
            
            return item
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def process_batch(self, items: List[ProcessingItem]) -> List[ProcessingItem]:
        """并行下载批次样本"""
        if not items:
            return []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._download_single, item): i 
                for i, item in enumerate(items)
            }
            
            results = [None] * len(items)
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
            
            return results
    
    def _download_single(self, item: ProcessingItem) -> ProcessingItem:
        """下载单个文件（线程安全）"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            success = self.storage.download_audio(item.oss_path, tmp_path)
            if not success:
                raise ValueError(f"Failed to download {item.oss_path}")
            
            with open(tmp_path, 'rb') as f:
                item.audio_bytes = f.read()
            
            return item
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)