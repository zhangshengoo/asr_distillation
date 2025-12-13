"""OSS storage interface for audio files"""

import os
import oss2
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse


class OSSClient:
    """阿里云OSS客户端封装"""
    
    def __init__(self, 
                 endpoint: str,
                 access_key_id: str,
                 access_key_secret: str,
                 bucket_name: str):
        
        auth = oss2.Auth(access_key_id, access_key_secret)
        self.bucket = oss2.Bucket(auth, endpoint, bucket_name)
        
    def list_objects(self, prefix: str = "", max_keys: int = 1000) -> List[oss2.models.SimplifiedObjectInfo]:
        """列出指定前缀的对象"""
        objects = []
        
        for obj in oss2.ObjectIterator(self.bucket, prefix=prefix, max_keys=max_keys):
            objects.append(obj)
            
        return objects
    
    def download_object(self, key: str, local_path: str) -> bool:
        """下载对象到本地"""
        try:
            self.bucket.get_object_to_file(key, local_path)
            return True
        except oss2.exceptions.OssError as e:
            return False
    
    def upload_object(self, key: str, local_path: str) -> bool:
        """上传本地文件到OSS"""
        try:
            self.bucket.put_object_from_file(key, local_path)
            return True
        except oss2.exceptions.OssError as e:
            return False
    
    def get_object_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """获取对象元数据"""
        try:
            meta = self.bucket.get_object_meta(key)
            return {
                'size': int(meta.headers.get('Content-Length', 0)),
                'last_modified': meta.headers.get('Last-Modified'),
                'etag': meta.headers.get('Etag', '').strip('"'),
                'content_type': meta.headers.get('Content-Type')
            }
        except oss2.exceptions.OssError as e:
            return None
    
    def object_exists(self, key: str) -> bool:
        """检查对象是否存在"""
        try:
            self.bucket.head_object(key)
            return True
        except oss2.exceptions.NoSuchKey:
            return False
        except oss2.exceptions.OssError as e:
            return False


class MediaStorageManager:
    """多媒体存储管理器 - 支持独立的输入和输出存储配置"""
    
    def __init__(self, 
                 input_config: Dict[str, Any], 
                 output_config: Dict[str, Any]):
        """
        初始化存储管理器
        Args:
            input_config: 输入存储配置（用于数据下载）
            output_config: 输出存储配置（用于结果上传）
        """
        # 输入存储配置
        self.input_oss_client = OSSClient(
            endpoint=input_config['endpoint'],
            access_key_id=input_config['access_key_id'],
            access_key_secret=input_config['access_key_secret'],
            bucket_name=input_config['bucket']
        )
        self.input_audio_prefix = input_config.get('audio_prefix', 'audio/')
        self.input_video_prefix = input_config.get('video_prefix', 'video/')
        
        # 输出存储配置
        self.output_oss_client = OSSClient(
            endpoint=output_config['endpoint'],
            access_key_id=output_config['access_key_id'],
            access_key_secret=output_config['access_key_secret'],
            bucket_name=output_config['bucket']
        )
        self.output_result_prefix = output_config.get('result_prefix', 'results/')
        
        # 保留配置引用
        self.config = input_config
    
    def parse_oss_path(self, oss_path: str) -> str:
        """解析oss://bucket/key路径，返回key"""
        parsed = urlparse(oss_path)
        if parsed.scheme != 'oss':
            raise ValueError(f"无效的OSS路径: {oss_path}")
        return parsed.path.lstrip('/')

    def list_media_files(self, 
                         prefix: Optional[str] = None,
                         file_extensions: List[str] = ['.wav', '.mp3', '.flac', '.mp4', '.avi', '.mov']) -> List[Dict[str, Any]]:
        """列出存储中的多媒体文件（使用输入存储）"""
        if not self.input_oss_client:
            raise ValueError("输入存储客户端未初始化")
        
        prefix = prefix or self.input_audio_prefix
        objects = self.input_oss_client.list_objects(prefix)
        
        media_files = []
        for obj in objects:
            key = obj.key
            if any(key.lower().endswith(ext) for ext in file_extensions):
                oss_path = f"oss://{self.input_oss_client.bucket.bucket_name}/{key}"
                file_id = Path(key).stem
                
                # 确定媒体类型
                media_type = 'audio' if any(key.lower().endswith(ext) for ext in ['.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a', '.wma']) else 'video'
                
                media_files.append({
                    'file_id': file_id,
                    'oss_path': oss_path,
                    'size_bytes': obj.size,
                    'last_modified': obj.last_modified,
                    'key': key,
                    'media_type': media_type
                })
                
        return media_files
    
    def list_audio_files(self, 
                        prefix: Optional[str] = None,
                        file_extensions: List[str] = ['.wav', '.mp3', '.flac']) -> List[Dict[str, Any]]:
        """列出存储中的音频文件（向后兼容）"""
        return self.list_media_files(prefix, file_extensions)
    
    def download_media(self, oss_path: str, local_path: str) -> bool:
        """从OSS下载多媒体文件（使用输入存储）"""
        if not self.input_oss_client:
            raise ValueError("输入存储客户端未初始化")
        
        key = self.parse_oss_path(oss_path)
        return self.input_oss_client.download_object(key, local_path)
    
    def download_audio(self, oss_path: str, local_path: str) -> bool:
        """从OSS下载音频文件（向后兼容）"""
        return self.download_media(oss_path, local_path)
    
    def upload_result(self, 
                     file_id: str, 
                     result_data: str, 
                     suffix: str = '.json') -> bool:
        """上传结果到OSS（使用输出存储）"""
        if not self.output_oss_client:
            raise ValueError("输出存储客户端未初始化")
        
        key = f"{self.output_result_prefix}{file_id}{suffix}"
        
        # 先写入临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
            f.write(result_data)
            temp_path = f.name
        
        try:
            return self.output_oss_client.upload_object(key, temp_path)
        finally:
            os.unlink(temp_path)
    
    def upload_batch_results(self, 
                           batch_file: str, 
                           batch_id: str) -> bool:
        """上传批量结果文件（使用输出存储）"""
        if not self.output_oss_client:
            raise ValueError("输出存储客户端未初始化")
        
        key = f"{self.output_result_prefix}batch_{batch_id}.jsonl"
        return self.output_oss_client.upload_object(key, batch_file)
    
    def get_audio_metadata(self, oss_path: str) -> Optional[Dict[str, Any]]:
        """获取音频文件元数据（使用输入存储）"""
        if not self.input_oss_client:
            raise ValueError("输入存储客户端未初始化")
        
        key = self.parse_oss_path(oss_path)
        return self.input_oss_client.get_object_metadata(key)
    
    def audio_exists(self, oss_path: str) -> bool:
        """检查音频文件是否存在（使用输入存储）"""
        if not self.input_oss_client:
            raise ValueError("输入存储客户端未初始化")
        
        key = self.parse_oss_path(oss_path)
        return self.input_oss_client.object_exists(key)


# 向后兼容的别名 - 但需要使用新构造函数
AudioStorageManager = MediaStorageManager  # 保持向后兼容性