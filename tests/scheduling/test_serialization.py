"""
序列化问题测试

测试Ray actor序列化问题，特别是文件对象序列化问题
"""

import pytest
import ray
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.scheduling.pipeline import DataProducer, DistributedPipeline, PipelineConfig


# 定义全局类，避免本地类序列化问题
class BadObject:
    """包含文件对象的对象，无法序列化"""
    def __init__(self):
        self.file_handle = open('/dev/null', 'r')


class GoodObject:
    """使用延迟初始化的对象，可以序列化"""
    def __init__(self):
        self.config = {'test': 'value'}
        self._file_handle = None
    
    @property
    def file_handle(self):
        if self._file_handle is None:
            self._file_handle = open('/dev/null', 'r')
        return self._file_handle


@ray.remote
class TestActor:
    """测试Ray actor的序列化"""
    def __init__(self, config):
        self.config = config
        self._resource = None
    
    @property
    def resource(self):
        if self._resource is None:
            # 模拟创建包含文件对象的对象
            self._resource = Mock()
            self._resource.file_handle = open('/dev/null', 'r')
        return self._resource
    
    def test_method(self):
        return "success"


class TestSerializationIssues:
    """测试序列化相关问题"""
    
    def test_data_producer_lazy_initialization(self, ray_initialized):
        """测试DataProducer延迟初始化修复序列化问题"""
        
        # 创建测试配置
        data_config = {
            'storage': {
                'bucket': 'test-bucket',
                'endpoint': 'https://oss-cn-beijing.aliyuncs.com',
                'access_key_id': 'test-key',
                'access_key_secret': 'test-secret'
            },
            'index_path': '/tmp/test_index'
        }
        
        # 使用正确的导入路径进行patch
        with patch('src.data.media_indexer.DataLoader') as mock_data_loader, \
             patch('src.data.storage.MediaStorageManager') as mock_storage_manager:
            
            # 设置模拟对象
            mock_loader_instance = Mock()
            mock_storage_instance = Mock()
            mock_data_loader.return_value = mock_loader_instance
            mock_storage_manager.return_value = mock_storage_instance
            
            # 模拟load_index方法
            mock_df = Mock()
            mock_df.empty = True
            mock_df.to_dict.return_value = []
            mock_loader_instance.load_index.return_value = mock_df
            mock_storage_instance.list_audio_files.return_value = []
            
            # 创建DataProducer应该成功（不会立即初始化DataLoader和AudioStorageManager）
            producer = DataProducer.remote(data_config, 32)
            
            # 调用方法应该成功（此时才会初始化DataLoader和AudioStorageManager）
            result = ray.get(producer.load_index.remote())
            assert result is not None
            
            # 验证延迟初始化被调用
            mock_data_loader.assert_called_once()
            mock_storage_manager.assert_called_once()
    
    def test_data_producer_properties(self, ray_initialized):
        """测试DataProducer属性访问"""
        
        data_config = {
            'storage': {'bucket': 'test'},
            'index_path': '/tmp/test_index'
        }
        
        with patch('src.data.media_indexer.DataLoader') as mock_data_loader, \
             patch('src.data.storage.MediaStorageManager') as mock_storage_manager:
            
            mock_loader_instance = Mock()
            mock_storage_instance = Mock()
            mock_data_loader.return_value = mock_loader_instance
            mock_storage_manager.return_value = mock_storage_instance
            
            # 创建DataProducer
            producer = DataProducer.remote(data_config, 32)
            
            # 测试属性访问 - 注意：这些不是remote方法，不能直接调用
            # 我们通过调用其他方法来间接测试属性访问
            result = ray.get(producer.load_index.remote())
            
            # 验证只初始化一次
            mock_data_loader.assert_called_once()
            mock_storage_manager.assert_called_once()
    
    def test_distributed_pipeline_setup_producer_fixed(self, ray_initialized):
        """测试DistributedPipeline.setup_producer修复后的序列化"""
        
        config = PipelineConfig(num_cpu_workers=1, num_gpu_workers=0)
        pipeline = DistributedPipeline(config)
        
        data_config = {
            'storage': {
                'bucket': 'test-bucket',
                'endpoint': 'https://oss-cn-beijing.aliyuncs.com',
                'access_key_id': 'test-key',
                'access_key_secret': 'test-secret'
            },
            'index_path': '/tmp/test_index'
        }
        
        # 模拟DataLoader和AudioStorageManager
        with patch('src.data.media_indexer.DataLoader') as mock_data_loader, \
             patch('src.data.storage.MediaStorageManager') as mock_storage_manager:
            
            mock_loader_instance = Mock()
            mock_storage_instance = Mock()
            mock_data_loader.return_value = mock_loader_instance
            mock_storage_manager.return_value = mock_storage_instance
            
            # 模拟load_index方法
            mock_df = Mock()
            mock_df.empty = True
            mock_df.to_dict.return_value = []
            mock_loader_instance.load_index.return_value = mock_df
            mock_storage_instance.list_audio_files.return_value = []
            
            # 设置producer应该成功
            pipeline.setup_producer(data_config)
            assert pipeline.producer is not None
    
    def test_file_object_serialization_problem(self):
        """测试文件对象序列化问题的基本原理"""
        
        # 测试坏对象
        bad_obj = BadObject()
        try:
            import pickle
            pickle.dumps(bad_obj)
            pytest.fail("坏对象不应该能够序列化")
        except Exception:
            pass  # 预期会失败
        finally:
            bad_obj.file_handle.close()
        
        # 测试好对象
        good_obj = GoodObject()
        try:
            import pickle
            serialized = pickle.dumps(good_obj)
            # 序列化成功
            assert serialized is not None
        except Exception as e:
            pytest.fail(f"好对象应该能够序列化: {e}")
        finally:
            if good_obj._file_handle:
                good_obj._file_handle.close()
    
    def test_ray_actor_serialization(self, ray_initialized):
        """测试Ray actor的序列化问题修复"""
        
        # 创建actor应该成功
        config = {'test': 'value'}
        actor = TestActor.remote(config)
        
        # 调用方法应该成功
        result = ray.get(actor.test_method.remote())
        assert result == "success"


if __name__ == "__main__":
    pytest.main([__file__])
