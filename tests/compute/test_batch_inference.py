"""
BatchInferenceStage 简单单元测试

测试BatchInferenceStage的基本功能
"""

import pytest
import torch
from unittest.mock import Mock, patch

from src.compute.inference import BatchInferenceStage
from src.scheduling.pipeline import DataBatch


class TestBatchInferenceStage:
    """BatchInferenceStage简单测试"""

    def test_init(self):
        """测试初始化"""
        config = {
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
            'batch_size': 4,
            'prompt_template': '请将这段语音转换为纯文本。'
        }
        
        # 模拟VLLMInferenceEngine
        with patch('src.compute.inference.VLLMInferenceEngine'):
            stage = BatchInferenceStage(config)
            assert stage.batch_size == 4
            assert stage.prompt_template == '请将这段语音转换为纯文本。'

    def test_process_normal_batch(self):
        """测试正常批次处理"""
        config = {
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
            'batch_size': 4,
            'prompt_template': '请将这段语音转换为纯文本。'
        }
        
        # 创建测试数据
        items = [
            {
                'file_id': f'test_audio_{i}',
                'audio_features': torch.randn(80, 100),
                'sample_rate': 16000,
                'duration': 5.0 + i
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
        
        with patch('src.compute.inference.VLLMInferenceEngine') as mock_engine:
            stage = BatchInferenceStage(config)
            stage._run_batch_inference = Mock(return_value=mock_results)
            
            result_batch = stage.process(batch)
            
            # 验证结果
            assert len(result_batch.items) == 3
            for i, item in enumerate(result_batch.items):
                assert 'transcription' in item
                assert item['transcription'] == f"Transcription result {i}"

    def test_process_batch_with_errors(self):
        """测试错误处理"""
        config = {
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
            'batch_size': 4,
            'prompt_template': '请将这段语音转换为纯文本。'
        }
        
        # 创建包含错误的数据
        items = [
            {
                'file_id': 'error_audio',
                'audio_features': torch.randn(80, 100),
                'sample_rate': 16000,
                'error': 'Processing error'
            },
            {
                'file_id': 'normal_audio',
                'audio_features': torch.randn(80, 100),
                'sample_rate': 16000
            }
        ]
        
        batch = DataBatch(
            batch_id="error_batch",
            items=items,
            metadata={'stage': 'test'}
        )
        
        with patch('src.compute.inference.VLLMInferenceEngine') as mock_engine:
            stage = BatchInferenceStage(config)
            stage._run_batch_inference = Mock(return_value=["Normal result"])
            
            result_batch = stage.process(batch)
            
            # 验证错误项目保持不变
            assert 'error' in result_batch.items[0]
            # 验证正常项目被处理
            assert 'transcription' in result_batch.items[1]

    def test_empty_batch(self):
        """测试空批次"""
        config = {
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
            'batch_size': 4,
            'prompt_template': '请将这段语音转换为纯文本。'
        }
        
        empty_batch = DataBatch(
            batch_id="empty_batch",
            items=[],
            metadata={'stage': 'test'}
        )
        
        with patch('src.compute.inference.VLLMInferenceEngine') as mock_engine:
            stage = BatchInferenceStage(config)
            result_batch = stage.process(empty_batch)
            
            assert len(result_batch.items) == 0
            assert result_batch.batch_id == "empty_batch"