"""Metadata写入器"""
import json
import time
from pathlib import Path
from typing import Dict
import threading
import tempfile
import os


class MetadataWriter:
    """线程安全的metadata写入器，支持本地缓冲+OSS上传"""
    
    def __init__(self, storage_manager, metadata_prefix: str, 
                 local_buffer_size: int = 1000,
                 output_filename: str = None):
        """
        Args:
            storage_manager: OSS存储管理器
            metadata_prefix: OSS上的metadata路径前缀
            local_buffer_size: 本地缓冲多少条后上传一次
            output_filename: 输出文件名，默认使用时间戳
        """
        self.storage = storage_manager
        self.metadata_prefix = metadata_prefix.rstrip('/')
        self.local_buffer_size = local_buffer_size
        
        # 生成输出文件名
        if output_filename is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            output_filename = f'segment_metadata_{timestamp}.jsonl'
        
        self.oss_path = f"{self.metadata_prefix}/{output_filename}"
        
        # 本地临时文件
        self.temp_file = tempfile.NamedTemporaryFile(
            mode='w', 
            encoding='utf-8',
            delete=False,
            suffix='.jsonl'
        )
        self.temp_path = self.temp_file.name
        
        self.lock = threading.Lock()
        self.count = 0
        self.uploaded_count = 0
    
    def write(self, meta_dict: Dict):
        """写入一条metadata"""
        with self.lock:
            line = json.dumps(meta_dict, ensure_ascii=False)
            self.temp_file.write(line + '\n')
            self.count += 1
            
            # 定期flush
            if self.count % 100 == 0:
                self.temp_file.flush()
            
            # 达到缓冲阈值，上传一次
            if self.count - self.uploaded_count >= self.local_buffer_size:
                self._upload_chunk()
    
    def _upload_chunk(self):
        """上传当前缓冲的数据"""
        self.temp_file.flush()
        
        try:
            # 读取临时文件并上传（追加模式）
            with open(self.temp_path, 'rb') as f:
                # 跳过已上传的部分
                if self.uploaded_count > 0:
                    for _ in range(self.uploaded_count):
                        f.readline()
                
                # 读取新增数据
                new_data = f.read()
            
            if new_data:
                # 上传到OSS（追加模式）
                self.storage.append_bytes(new_data, self.oss_path)
                self.uploaded_count = self.count
        
        except Exception as e:
            print(f"Failed to upload metadata chunk: {e}")
    
    def close(self):
        """关闭并上传剩余数据"""
        with self.lock:
            if self.temp_file:
                # 上传最后剩余的数据
                if self.count > self.uploaded_count:
                    self._upload_chunk()
                
                self.temp_file.close()
                
                # 清理临时文件
                try:
                    os.unlink(self.temp_path)
                except:
                    pass
                
                self.temp_file = None
                
                print(f"Metadata uploaded to OSS: {self.oss_path}")
                print(f"Total records: {self.count}")