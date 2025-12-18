"""OSS存储管理器"""
import logging
import oss2
from typing import Dict, Any, Optional


class MediaStorageManager:
    """媒体存储管理器，支持输入和输出分离"""
    
    def __init__(self, input_config: Dict[str, Any], output_config: Dict[str, Any]):
        """
        Args:
            input_config: 输入存储配置
            output_config: 输出存储配置
        """
        self.logger = logging.getLogger(__name__)
        
        # 输入存储配置
        self.input_config = input_config
        self.input_auth = oss2.Auth(
            input_config['access_key_id'],
            input_config['access_key_secret']
        )
        self.input_bucket = oss2.Bucket(
            self.input_auth,
            input_config['endpoint'],
            input_config['bucket']
        )
        
        # 输出存储配置
        self.output_config = output_config
        self.output_auth = oss2.Auth(
            output_config['access_key_id'],
            output_config['access_key_secret']
        )
        self.output_bucket = oss2.Bucket(
            self.output_auth,
            output_config['endpoint'],
            output_config['bucket']
        )
    
    def download_audio(self, oss_path: str, local_path: str) -> bool:
        """从输入存储下载音频"""
        try:
            self.input_bucket.get_object_to_file(oss_path, local_path)
            return True
        except Exception as e:
            self.logger.error(f"Failed to download {oss_path}: {e}")
            return False
    
    def upload_bytes(self, data: bytes, oss_path: str) -> bool:
        """上传bytes到输出存储"""
        try:
            self.output_bucket.put_object(oss_path, data)
            return True
        except Exception as e:
            self.logger.error(f"Failed to upload to {oss_path}: {e}")
            return False
    
    def append_bytes(self, data: bytes, oss_path: str) -> bool:
        """追加数据到OSS文件（如果文件存在则追加，否则创建）"""
        try:
            # 检查文件是否存在
            if self.output_bucket.object_exists(oss_path):
                # 下载现有文件
                existing_data = self.output_bucket.get_object(oss_path).read()
                # 合并数据
                merged_data = existing_data + data
                # 上传合并后的数据
                self.output_bucket.put_object(oss_path, merged_data)
            else:
                # 文件不存在，直接上传
                self.output_bucket.put_object(oss_path, data)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to append to OSS {oss_path}: {e}")
            return False
    
    def download_from_input(self, oss_path: str, local_path: str) -> bool:
        """从输入bucket下载"""
        return self.download_audio(oss_path, local_path)
    
    def upload_to_output(self, local_path: str, oss_path: str) -> bool:
        """上传文件到输出bucket"""
        try:
            self.output_bucket.put_object_from_file(oss_path, local_path)
            return True
        except Exception as e:
            self.logger.error(f"Failed to upload {local_path} to {oss_path}: {e}")
            return False