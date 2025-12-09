"""
调度层单元测试

测试调度层的核心功能：
- PipelineConfig配置验证
- DataBatch数据结构
- PipelineStage抽象基类
- DistributedPipeline分布式流水线
- PipelineOrchestrator高级调度器
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from src.scheduling.pipeline import (
    PipelineConfig,
    DataBatch,
    PipelineStage,
    DistributedPipeline,
    PipelineOrchestrator,
    DataProducer,
    PipelineWorker
)


class TestPipelineConfig:
    """测试PipelineConfig配置类"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = PipelineConfig()
        
        assert config.num_cpu_workers == 10
        assert config.num_gpu_workers == 1
        assert config.batch_size == 32
        assert config.max_concurrent_batches == 4
        assert config.object_store_memory == 1024 * 1024 * 1024
        assert config.checkpoint_interval == 1000
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = PipelineConfig(
            num_cpu_workers=20,
            num_gpu_workers=2,
            batch_size=64,
            object_store_memory=2 * 1024 * 1024 * 1024
        )
        
        assert config.num_cpu_workers == 20
        assert config.num_gpu_workers == 2
        assert config.batch_size == 64
        assert config.object_store_memory == 2 * 1024 * 1024 * 1024


class TestDataBatch:
    """测试DataBatch数据结构"""
    
    def test_data_batch_creation(self):
        """测试DataBatch创建"""
        items = [
            {'file_id': 'test1', 'data': 'audio1'},
            {'file_id': 'test2', 'data': 'audio2'}
        ]
        
        batch = DataBatch(
            batch_id="test_batch",
            items=items,
            metadata={'stage': 'test'}
        )
        
        assert batch.batch_id == "test_batch"
        assert len(batch.items) == 2
        assert batch.metadata['stage'] == 'test'
        assert batch.items[0]['file_id'] == 'test1'
    
    def test_data_batch_empty(self):
        """测试空DataBatch"""
        batch = DataBatch(
            batch_id="empty_batch",
            items=[],
            metadata={}
        )
        
        assert batch.batch_id == "empty_batch"
        assert len(batch.items) == 0
        assert batch.metadata == {}


class MockPipelineStage(PipelineStage):
    """模拟PipelineStage用于测试"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.processed_count = 0
    
    def process(self, batch: DataBatch) -> DataBatch:
        """模拟处理批次"""
        self.processed_count += 1
        
        # 在每个item中添加处理标记
        for item in batch.items:
            item['processed_by'] = self.__class__.__name__
        
        batch.metadata['processed_count'] = self.processed_count
        return batch


class TestPipelineStage:
    """测试PipelineStage抽象基类"""
    
    def test_concrete_stage_implementation(self):
        """测试具体阶段实现"""
        config = {'param1': 'value1'}
        stage = MockPipelineStage(config)
        
        assert stage.config == config
        assert stage.processed_count == 0
        
        # 测试处理
        batch = DataBatch(
            batch_id="test",
            items=[{'file_id': 'test1'}],
            metadata={}
        )
        
        result = stage.process(batch)
        
        assert result.batch_id == "test"
        assert result.items[0]['processed_by'] == 'MockPipelineStage'
        assert result.metadata['processed_count'] == 1
        assert stage.processed_count == 1


class TestDistributedPipeline:
    """测试DistributedPipeline分布式流水线"""
    
    @patch('ray.is_initialized')
    @patch('ray.init')
    def test_pipeline_initialization(self, mock_ray_init, mock_ray_is_initialized):
        """测试流水线初始化"""
        mock_ray_is_initialized.return_value = False
        
        config = PipelineConfig(num_cpu_workers=2, num_gpu_workers=1)
        pipeline = DistributedPipeline(config)
        
        assert pipeline.config == config
        assert pipeline.producer is None
        assert pipeline.stage_workers == {}
        assert pipeline.stage_queues == {}
        
        mock_ray_init.assert_called_once_with(
            object_store_memory=config.object_store_memory,
            ignore_reinit_error=True
        )
    
    @patch('ray.is_initialized')
    def test_pipeline_with_ray_already_initialized(self, mock_ray_is_initialized):
        """测试Ray已初始化时的流水线创建"""
        mock_ray_is_initialized.return_value = True
        
        config = PipelineConfig()
        pipeline = DistributedPipeline(config)
        
        assert pipeline.config == config
    
    @patch('ray.is_initialized')
    @patch('ray.init')
    @patch('src.scheduling.pipeline.DataProducer.remote')
    def test_setup_producer(self, mock_producer_remote, mock_ray_init, mock_ray_is_initialized):
        """测试设置数据生产者"""
        mock_ray_is_initialized.return_value = False
        mock_producer = Mock()
        mock_producer_remote.return_value = mock_producer
        
        config = PipelineConfig()
        pipeline = DistributedPipeline(config)
        
        data_config = {'storage': {'bucket': 'test'}}
        pipeline.setup_producer(data_config)
        
        assert pipeline.producer == mock_producer
        mock_producer_remote.assert_called_once_with(data_config, config.batch_size)
    
    @patch('ray.is_initialized')
    @patch('ray.init')
    @patch('src.scheduling.pipeline.PipelineWorker.options')
    def test_setup_cpu_workers(self, mock_worker_options, mock_ray_init, mock_ray_is_initialized):
        """测试设置CPU工作节点"""
        mock_ray_is_initialized.return_value = False
        mock_worker = Mock()
        mock_worker_options.return_value.remote.return_value = mock_worker
        
        config = PipelineConfig(num_cpu_workers=3)
        pipeline = DistributedPipeline(config)
        
        stage_config = {'param1': 'value1'}
        pipeline.setup_cpu_workers(MockPipelineStage, stage_config, num_workers=3, stage_name='test_cpu')
        
        assert 'test_cpu' in pipeline.stage_workers
        assert len(pipeline.stage_workers['test_cpu']) == 3
        assert 'test_cpu' in pipeline.stage_queues
        
        # 验证资源配置
        mock_worker_options.assert_called_with(CPU=1)
    
    @patch('ray.is_initialized')
    @patch('ray.init')
    @patch('src.scheduling.pipeline.PipelineWorker.options')
    def test_setup_gpu_workers(self, mock_worker_options, mock_ray_init, mock_ray_is_initialized):
        """测试设置GPU工作节点"""
        mock_ray_is_initialized.return_value = False
        mock_worker = Mock()
        mock_worker_options.return_value.remote.return_value = mock_worker
        
        config = PipelineConfig(num_gpu_workers=2)
        pipeline = DistributedPipeline(config)
        
        stage_config = {'param1': 'value1'}
        pipeline.setup_gpu_workers(MockPipelineStage, stage_config, num_workers=2, stage_name='test_gpu')
        
        assert 'test_gpu' in pipeline.stage_workers
        assert len(pipeline.stage_workers['test_gpu']) == 2
        assert 'test_gpu' in pipeline.stage_queues
        
        # 验证资源配置
        mock_worker_options.assert_called_with(CPU=1, GPU=1)
    
    @patch('ray.is_initialized')
    @patch('ray.init')
    def test_get_pipeline_stats(self, mock_ray_init, mock_ray_is_initialized):
        """测试获取流水线统计信息"""
        mock_ray_is_initialized.return_value = False
        
        config = PipelineConfig()
        pipeline = DistributedPipeline(config)
        
        # 模拟添加一些工作节点
        pipeline.stage_workers = {
            'cpu_stage_1': [Mock(), Mock()],
            'cpu_stage_2': [Mock()],
            'gpu_stage_1': [Mock()]
        }
        
        with patch('ray.cluster_resources') as mock_cluster_resources, \
             patch('ray.available_resources') as mock_available_resources:
            
            mock_cluster_resources.return_value = {'CPU': 8, 'GPU': 1}
            mock_available_resources.return_value = {'CPU': 4, 'GPU': 0.5}
            
            stats = pipeline.get_pipeline_stats()
            
            assert stats['stages'] == ['cpu_stage_1', 'cpu_stage_2', 'gpu_stage_1']
            assert stats['num_cpu_workers'] == 3
            assert stats['num_gpu_workers'] == 1
            assert stats['total_workers'] == 4
            assert stats['ray_cluster_resources'] == {'CPU': 8, 'GPU': 1}
            assert stats['available_resources'] == {'CPU': 4, 'GPU': 0.5}


class TestPipelineOrchestrator:
    """测试PipelineOrchestrator高级调度器"""
    
    def test_orchestrator_initialization(self):
        """测试调度器初始化"""
        config = {
            'pipeline': {
                'num_cpu_workers': 5,
                'num_gpu_workers': 2
            },
            'data': {
                'storage': {'bucket': 'test'}
            }
        }
        
        orchestrator = PipelineOrchestrator(config)
        
        assert orchestrator.pipeline_config.num_cpu_workers == 5
        assert orchestrator.pipeline_config.num_gpu_workers == 2
        assert orchestrator.data_config == config['data']
        assert orchestrator.stages_config == []
    
    @patch('src.scheduling.pipeline.DistributedPipeline')
    def test_setup_multi_stage_pipeline(self, mock_distributed_pipeline):
        """测试设置多阶段流水线"""
        mock_pipeline = Mock()
        mock_distributed_pipeline.return_value = mock_pipeline
        
        config = {
            'pipeline': {'num_cpu_workers': 2, 'num_gpu_workers': 1},
            'data': {'storage': {'bucket': 'test'}}
        }
        
        orchestrator = PipelineOrchestrator(config)
        
        stages_config = [
            {
                'type': 'cpu',
                'class': MockPipelineStage,
                'config': {'param1': 'value1'},
                'name': 'stage1',
                'num_workers': 3
            },
            {
                'type': 'gpu',
                'class': MockPipelineStage,
                'config': {'param2': 'value2'},
                'name': 'stage2',
                'num_workers': 2
            }
        ]
        
        orchestrator.setup_multi_stage_pipeline(stages_config)
        
        assert len(orchestrator.stages_config) == 2
        assert orchestrator.stages_config[0]['name'] == 'stage1'
        assert orchestrator.stages_config[1]['name'] == 'stage2'
        
        # 验证流水线设置调用
        mock_pipeline.setup_producer.assert_called_once()
        assert mock_pipeline.setup_cpu_workers.call_count == 1
        assert mock_pipeline.setup_gpu_workers.call_count == 1
    
    @patch('src.scheduling.pipeline.DistributedPipeline')
    def test_get_stats(self, mock_distributed_pipeline):
        """测试获取调度器统计信息"""
        mock_pipeline = Mock()
        mock_pipeline.get_pipeline_stats.return_value = {
            'stages': ['stage1', 'stage2'],
            'num_cpu_workers': 3,
            'num_gpu_workers': 1
        }
        mock_distributed_pipeline.return_value = mock_pipeline
        
        config = {
            'pipeline': {},
            'data': {'storage': {'bucket': 'test'}}
        }
        
        orchestrator = PipelineOrchestrator(config)
        orchestrator.stages_config = [
            {'type': 'cpu', 'class': MockPipelineStage, 'name': 'stage1'},
            {'type': 'gpu', 'class': MockPipelineStage, 'name': 'stage2'}
        ]
        
        stats = orchestrator.get_stats()
        
        assert 'stages' in stats
        assert len(stats['stages']) == 2
        assert stats['stages'][0]['name'] == 'stage1'
        assert stats['stages'][0]['type'] == 'cpu'
        assert stats['stages'][0]['class'] == 'MockPipelineStage'
        assert stats['num_cpu_workers'] == 3
        assert stats['num_gpu_workers'] == 1
    
    @patch('src.scheduling.pipeline.DistributedPipeline')
    def test_cleanup(self, mock_distributed_pipeline):
        """测试清理资源"""
        mock_pipeline = Mock()
        mock_distributed_pipeline.return_value = mock_pipeline
        
        config = {
            'pipeline': {},
            'data': {'storage': {'bucket': 'test'}}
        }
        
        orchestrator = PipelineOrchestrator(config)
        orchestrator.cleanup()
        
        mock_pipeline.shutdown.assert_called_once()


@pytest.mark.unit
class TestPipelineIntegration:
    """调度层集成测试"""
    
    def test_multi_stage_configuration(self):
        """测试多阶段配置"""
        config = {
            'pipeline': {
                'num_cpu_workers': 10,
                'num_gpu_workers': 2,
                'batch_size': 32
            },
            'data': {
                'storage': {'bucket': 'test-bucket'}
            }
        }
        
        stages_config = [
            {
                'type': 'cpu',
                'class': MockPipelineStage,
                'config': {'stage': 'download'},
                'name': 'audio_download',
                'num_workers': 5
            },
            {
                'type': 'cpu',
                'class': MockPipelineStage,
                'config': {'stage': 'preprocess'},
                'name': 'audio_preprocess',
                'num_workers': 10
            },
            {
                'type': 'gpu',
                'class': MockPipelineStage,
                'config': {'stage': 'inference'},
                'name': 'batch_inference',
                'num_workers': 2
            }
        ]
        
        # 验证配置结构
        assert len(stages_config) == 3
        assert stages_config[0]['type'] == 'cpu'
        assert stages_config[1]['type'] == 'cpu'
        assert stages_config[2]['type'] == 'gpu'
        assert stages_config[0]['num_workers'] == 5
        assert stages_config[1]['num_workers'] == 10
        assert stages_config[2]['num_workers'] == 2
    
    def test_data_flow_validation(self):
        """测试数据流验证"""
        # 创建测试数据
        items = [
            {'file_id': 'audio1', 'audio_data': 'raw_data1'},
            {'file_id': 'audio2', 'audio_data': 'raw_data2'}
        ]
        
        batch = DataBatch(
            batch_id="test_batch",
            items=items,
            metadata={'source': 'test'}
        )
        
        # 模拟多阶段处理
        stage1 = MockPipelineStage({'stage': 'download'})
        stage2 = MockPipelineStage({'stage': 'preprocess'})
        
        # 第一阶段处理
        result1 = stage1.process(batch)
        assert result1.items[0]['processed_by'] == 'MockPipelineStage'
        
        # 第二阶段处理
        result2 = stage2.process(result1)
        assert result2.metadata['processed_count'] == 2
        
        # 验证数据流
        assert result2.batch_id == batch.batch_id
        assert len(result2.items) == len(batch.items)


if __name__ == "__main__":
    pytest.main([__file__])