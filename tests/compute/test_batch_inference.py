
"""
BatchInferenceStage 单元测试

测试BatchInferenceStage的各种功能，包括边界条件和错误处理
"""

import pytest
import torch
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from src.compute.inference import BatchInferenceStage, InferenceConfig
from src.scheduling.pipeline import DataBatch


class TestBatchInferenceStage:
    """BatchInferenceStage测试"""

    @pytest.fixture
    def basic_config(self):
        """基本配置"""
        return {
            'inference': {
                'model_name': 'test-model',
                'tensor_parallel_size': 1,
                'max_num_batched_tokens': 1024,
                'max_model_len': 2048,
                'gpu_memory_utilization': 0.8,
                'trust_remote_code': True,
                'dtype': 'float32',
                'temperature': 0.1,
                'max_tokens': 512,
                'top_p': 0.9,
                'repetition_penalty': 1.1,
                'max_num_seqs': 1,
                'seed': 42
            },
            'prompt_template': '请将这段语音转换为纯文本。'
        }

    def test_init(self, basic_config):
        """测试初始化"""
        # 模拟VLLMInferenceEngine和AudioModelProcessor
        with patch('src.compute.inference.VLLMInferenceEngine'), \
             patch('src.compute.inference.AudioModelProcessor'):
            stage = BatchInferenceStage(basic_config)
            assert stage.prompt_template == '请将这段语音转换为纯文本。'

    @pytest.mark.asyncio
    async def test_process_async_normal_batch(self, basic_config):
        """测试正常批次异步处理"""
        items = [
            {
                'file_id': f'test_audio_{i}',
                'audio_features': torch.randn(1, 16000),  # 单通道音频波形
                'sample_rate': 16000,
                'duration': 1.0 + i
            }
            for i in range(3)
        ]
        
        batch = DataBatch(
            batch_id="test_batch_001",
            items=items,
            metadata={'stage': 'test'}
        )
        
        # 模拟推理结果
        mock_results = [
            f"Transcription result {i}" for i in range(len(items))
        ]
        
        with patch('src.compute.inference.VLLMInferenceEngine') as MockEngine, \
             patch('src.compute.inference.AudioModelProcessor') as MockProcessor:
            mock_engine_instance = Mock()
            mock_engine_instance.generate_batch_async = AsyncMock(return_value=mock_results)
            MockEngine.return_value = mock_engine_instance
            
            mock_processor_instance = Mock()
            mock_processor_instance.prepare_inputs = Mock(return_value={'prompt': 'test', 'multi_modal_data': {}})
            MockProcessor.return_value = mock_processor_instance
            
            stage = BatchInferenceStage(basic_config)
            result_batch = await stage.process_async(batch)
            
            # 验证结果
            assert len(result_batch.items) == 3
            for i, item in enumerate(result_batch.items):
                assert 'transcription' in item
                assert item['transcription'] == f"Transcription result {i}"
                assert 'confidence' in item
                assert item['confidence'] == 0.0

    @pytest.mark.asyncio
    async def test_process_async_batch_with_errors(self, basic_config):
        """测试包含错误项目的批次处理"""
        items = [
            {
                'file_id': 'error_audio',
                'audio_features': torch.randn(1, 16000),
                'sample_rate': 16000,
                'error': 'Processing error'
            },
            {
                'file_id': 'normal_audio',
                'audio_features': torch.randn(1, 16000),
                'sample_rate': 16000
            }
        ]
        
        batch = DataBatch(
            batch_id="error_batch",
            items=items,
            metadata={'stage': 'test'}
        )
        
        with patch('src.compute.inference.VLLMInferenceEngine') as MockEngine, \
             patch('src.compute.inference.AudioModelProcessor') as MockProcessor:
            mock_engine_instance = Mock()
            mock_engine_instance.generate_batch_async = AsyncMock(return_value=["Normal result"])
            MockEngine.return_value = mock_engine_instance
            
            mock_processor_instance = Mock()
            mock_processor_instance.prepare_inputs = Mock(return_value={'prompt': 'test', 'multi_modal_data': {}})
            MockProcessor.return_value = mock_processor_instance
            
            stage = BatchInferenceStage(basic_config)
            result_batch = await stage.process_async(batch)
            
            # 验证错误项目被过滤，只有正常项目被处理
            assert len(result_batch.items) == 1  # 只有正常项目被处理
            assert result_batch.items[0]['file_id'] == 'normal_audio'
            assert 'transcription' in result_batch.items[0]

    @pytest.mark.asyncio
    async def test_process_async_empty_batch(self, basic_config):
        """测试空批次"""
        empty_batch = DataBatch(
            batch_id="empty_batch",
            items=[],
            metadata={'stage': 'test'}
        )
        
        with patch('src.compute.inference.VLLMInferenceEngine'), \
             patch('src.compute.inference.AudioModelProcessor'):
            stage = BatchInferenceStage(basic_config)
            result_batch = await stage.process_async(empty_batch)
            
            assert len(result_batch.items) == 0
            assert result_batch.batch_id == "empty_batch"
            assert result_batch.metadata['stage'] == 'batch_inference'

    @pytest.mark.asyncio
    async def test_process_async_batch_with_empty_audio_features(self, basic_config):
        """测试音频特征为空的批次"""
        items = [
            {
                'file_id': 'empty_audio',
                'audio_features': torch.empty(0),  # 空张量
                'sample_rate': 16000
            },
            {
                'file_id': 'normal_audio',
                'audio_features': torch.randn(1, 8000),
                'sample_rate': 16000
            }
        ]
        
        batch = DataBatch(
            batch_id="empty_features_batch",
            items=items,
            metadata={'stage': 'test'}
        )
        
        with patch('src.compute.inference.VLLMInferenceEngine') as MockEngine, \
             patch('src.compute.inference.AudioModelProcessor') as MockProcessor:
            mock_engine_instance = Mock()
            mock_engine_instance.generate_batch_async = AsyncMock(return_value=["Normal result"])
            MockEngine.return_value = mock_engine_instance
            
            mock_processor_instance = Mock()
            mock_processor_instance.prepare_inputs = Mock(return_value={'prompt': 'test', 'multi_modal_data': {}})
            MockProcessor.return_value = mock_processor_instance
            
            stage = BatchInferenceStage(basic_config)
            result_batch = await stage.process_async(batch)
            
            # 验证只有正常项目被处理，空音频项目可能在prepare_inputs中引发异常或被处理
            assert len(result_batch.items) == 1
            assert result_batch.items[0]['file_id'] == 'normal_audio'
            assert 'transcription' in result_batch.items[0]

    @pytest.mark.asyncio
    async def test_process_async_batch_missing_audio_features(self, basic_config):
        """测试缺少audio_features的项目"""
        items = [
            {
                'file_id': 'missing_features_audio',
                # 缺少 'audio_features' 字段
                'sample_rate': 16000
            },
            {
                'file_id': 'normal_audio',
                'audio_features': torch.randn(1, 8000),
                'sample_rate': 16000
            }
        ]
        
        batch = DataBatch(
            batch_id="missing_features_batch",
            items=items,
            metadata={'stage': 'test'}
        )
        
        with patch('src.compute.inference.VLLMInferenceEngine') as MockEngine, \
             patch('src.compute.inference.AudioModelProcessor') as MockProcessor:
            mock_engine_instance = Mock()
            mock_engine_instance.generate_batch_async = AsyncMock(return_value=["Normal result"])
            MockEngine.return_value = mock_engine_instance
            
            mock_processor_instance = Mock()
            mock_processor_instance.prepare_inputs = Mock(return_value={'prompt': 'test', 'multi_modal_data': {}})
            MockProcessor.return_value = mock_processor_instance
            
            stage = BatchInferenceStage(basic_config)
            # 应该引发异常，因为缺少必要的audio_features字段
            with pytest.raises(KeyError):
                await stage.process_async(batch)

    @pytest.mark.asyncio
    async def test_process_async_batch_with_zero_length_audio(self, basic_config):
        """测试零长度音频数据"""
        items = [
            {
                'file_id': 'zero_length_audio',
                'audio_features': torch.zeros(1, 0),  # 零长度音频
                'sample_rate': 16000
            },
            {
                'file_id': 'normal_audio',
                'audio_features': torch.randn(1, 8000),
                'sample_rate': 16000
            }
        ]
        
        batch = DataBatch(
            batch_id="zero_length_batch",
            items=items,
            metadata={'stage': 'test'}
        )
        
        with patch('src.compute.inference.VLLMInferenceEngine') as MockEngine, \
             patch('src.compute.inference.AudioModelProcessor') as MockProcessor:
            mock_engine_instance = Mock()
            mock_engine_instance.generate_batch_async = AsyncMock(return_value=["Normal result"])
            MockEngine.return_value = mock_engine_instance
            
            mock_processor_instance = Mock()
            mock_processor_instance.prepare_inputs = Mock(return_value={'prompt': 'test', 'multi_modal_data': {}})
            MockProcessor.return_value = mock_processor_instance
            
            stage = BatchInferenceStage(basic_config)
            result_batch = await stage.process_async(batch)
            
            # 验证只有正常项目被处理
            assert len(result_batch.items) == 1
            assert result_batch.items[0]['file_id'] == 'normal_audio'

    def test_sync_process_wrapper(self, basic_config):
        """测试同步process包装器"""
        items = [
            {
                'file_id': 'sync_test_audio',
                'audio_features': torch.randn(1, 8000),
                'sample_rate': 16000
            }
        ]
        
        batch = DataBatch(
            batch_id="sync_test_batch",
            items=items,
            metadata={'stage': 'test'}
        )
        
        with patch('src.compute.inference.VLLMInferenceEngine') as MockEngine, \
             patch('src.compute.inference.AudioModelProcessor') as MockProcessor:
            mock_engine_instance = Mock()
            mock_engine_instance.generate_batch_async = AsyncMock(return_value=["Sync result"])
            MockEngine.return_value = mock_engine_instance
            
            mock_processor_instance = Mock()
            mock_processor_instance.prepare_inputs = Mock(return_value={'prompt': 'test', 'multi_modal_data': {}})
            MockProcessor.return_value = mock_processor_instance
            
            stage = BatchInferenceStage(basic_config)
            # 使用同步接口
            result_batch = stage.process(batch)
            
            assert len(result_batch.items) == 1
            assert 'transcription' in result_batch.items[0]
            assert result_batch.items[0]['transcription'] == "Sync result"
