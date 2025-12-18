"""下载Stage"""
import tempfile
import os
from ..stages.base import Stage
from ..tools.data_structures import ProcessingItem


class DownloadStage(Stage):
    """从OSS下载音频文件"""
    
    def __init__(self, storage_manager):
        self.storage = storage_manager
    
    def name(self) -> str:
        return "download"
    
    def process(self, item: ProcessingItem) -> ProcessingItem:
        """下载音频文件"""
        # 下载到临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            success = self.storage.download_audio(item.oss_path, tmp_path)
            if not success:
                raise ValueError(f"Failed to download {item.oss_path}")
            
            # 读取文件内容
            with open(tmp_path, 'rb') as f:
                item.audio_bytes = f.read()
            
            return item
            
        finally:
            # 清理临时文件
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)