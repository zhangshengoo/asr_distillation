"""
pytest配置文件和共享fixtures

提供测试所需的基础设施：
- 测试配置
- 模拟数据
- 测试环境设置
- 公共fixtures
"""

import pytest
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, patch

from src.scheduling.pipeline import DataBatch
from src.config.manager import InferenceConfig, AudioConfig


@pytest.fixture(scope="session")
def test_data_dir():
    """创建测试数据目录"""
    temp_dir = tempfile.mkdtemp(prefix="asr_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_audio_config():
    """示例音频配置"""
    return AudioConfig(
        target_sample_rate=16000,
        max_duration=30.0,
        normalize=True,
        remove_silence=False,
        audio_format='wav'
    )


@pytest.fixture
def sample_inference_config():
    """示例推理配置"""
    return InferenceConfig(
        model_name="test-model",
        tensor_parallel_size=1,
        max_num_batched_tokens=1024,
        max_model_len=2048,
        gpu_memory_utilization=0.8,
        trust_remote_code=True,
        dtype="float32",
        temperature=0.1,
        max_tokens=512,
        top_p=0.9,
        repetition_penalty=1.1,
        max_num_seqs=1,
        seed=42
    )


@pytest.fixture
def sample_audio_tensor():
    """生成示例音频张量"""
    # 生成5秒的16kHz音频
    duration = 5.0
    sample_rate = 16000
    num_samples = int(duration * sample_rate)
    
    # 生成正弦波音频
    t = torch.linspace(0, duration, num_samples)
    frequency = 440  # A4音符
    waveform = torch.sin(2 * torch.pi * frequency * t).unsqueeze(0)
    
    return waveform, sample_rate


@pytest.fixture
def sample_audio_features():
    """生成示例音频特征"""
    # 生成Mel频谱图特征
    batch_size = 2
    n_mels = 80
    seq_len = 100
    
    features = torch.randn(batch_size, n_mels, seq_len)
    return features


@pytest.fixture
def sample_data_batch():
    """生成示例数据批次"""
    items = [
        {
            'file_id': f'test_audio_{i}',
            'audio_features': torch.randn(80, 100),
            'sample_rate': 16000,
            'duration': 5.0 + i
        }
        for i in range(3)
    ]
    
    return DataBatch(
        batch_id="test_batch_001",
        items=items,
        metadata={'stage': 'test', 'timestamp': 1234567890}
    )


@pytest.fixture
def mock_vllm_engine():
    """模拟vLLM引擎"""
    mock_engine = Mock()
    
    def mock_generate(inputs, sampling_params, request_id):
        """模拟生成结果"""
        mock_output = Mock()
        mock_output.text = f"Mock transcription for {request_id}"
        
        mock_request_output = Mock()
        mock_request_output.outputs = [mock_output]
        
        return [mock_request_output]
    
    mock_engine.generate = mock_generate
    return mock_engine


@pytest.fixture
def mock_qwen_processor():
    """模拟Qwen3-Omni处理器"""
    mock_processor = Mock()
    
    def mock_apply_chat_template(messages, tokenize=False, add_generation_prompt=True):
        return "Mock chat template: " + str(messages)
    
    mock_processor.apply_chat_template = mock_apply_chat_template
    return mock_processor


@pytest.fixture
def error_data_batch():
    """包含错误的数据批次"""
    items = [
        {
            'file_id': 'error_audio_1',
            'audio_features': torch.randn(80, 100),
            'sample_rate': 16000,
            'error': 'Processing error'
        },
        {
            'file_id': 'normal_audio_1',
            'audio_features': torch.randn(80, 100),
            'sample_rate': 16000
        }
    ]
    
    return DataBatch(
        batch_id="error_batch_001",
        items=items,
        metadata={'stage': 'test', 'has_errors': True}
    )


@pytest.fixture
def temp_config_file(test_data_dir):
    """创建临时配置文件"""
    config_path = test_data_dir / "test_config.yaml"
    
    config_content = """
inference:
  model_name: "test-model"
  tensor_parallel_size: 1
  max_num_batched_tokens: 1024
  max_model_len: 2048
  gpu_memory_utilization: 0.8
  trust_remote_code: true
  dtype: "float32"
  temperature: 0.1
  max_tokens: 512
  top_p: 0.9
  repetition_penalty: 1.1
  max_num_seqs: 1
  seed: 42

audio:
  target_sample_rate: 16000
  max_duration: 30.0
  normalize: true
  remove_silence: false
  audio_format: 'wav'
"""
    
    config_path.write_text(config_content)
    return config_path


# 测试标记
pytest.mark.unit = pytest.mark.unit  # 单元测试
pytest.mark.integration = pytest.mark.integration  # 集成测试
pytest.mark.slow = pytest.mark.slow  # 慢速测试
pytest.mark.gpu = pytest.mark.gpu  # 需要GPU的测试


# 测试配置
def pytest_configure(config):
    """pytest配置"""
    config.addinivalue_line(
        "markers", "unit: 标记单元测试"
    )
    config.addinivalue_line(
        "markers", "integration: 标记集成测试"
    )
    config.addinivalue_line(
        "markers", "slow: 标记慢速测试"
    )
    config.addinivalue_line(
        "markers", "gpu: 标记需要GPU的测试"
    )


# 测试收集钩子
def pytest_collection_modifyitems(config, items):
    """修改测试收集"""
    # 为没有标记的测试添加unit标记
    for item in items:
        if not any(item.iter_markers()):
            item.add_marker(pytest.mark.unit)