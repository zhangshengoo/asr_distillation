"""Configuration manager for ASR Distillation Framework"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
from dataclasses import dataclass, field

from omegaconf import OmegaConf, DictConfig
import yaml


@dataclass
class DataConfig:
    """数据层配置 - 管理音频数据索引、缓存和存储"""
    index_path: str = "./data/index"
    cache_dir: str = "./data/cache"
    cache_size_gb: float = 100.0
    webdataset_output_dir: str = "./data/webdataset"
    shard_size_mb: int = 100
    storage: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeGPUConfig:
    """节点GPU配置"""
    node_id: int
    gpus: List[int]
    workers: List[int]


@dataclass
class GPUAllocationConfig:
    """GPU分配配置"""
    strategy: str = "custom"
    nodes: List[NodeGPUConfig] = field(default_factory=list)


@dataclass
class PipelineConfig:
    """流水线配置 - 控制分布式处理和资源分配"""
    num_cpu_workers: int = 10
    num_gpu_workers: int = 1
    cpu_worker_resources: Dict[str, float] = field(default_factory=lambda: {"num_cpus": 1})
    gpu_worker_resources: Dict[str, float] = field(default_factory=lambda: {"num_cpus": 1, "num_gpus": 1})
    batch_size: int = 32
    max_concurrent_batches: int = 4
    object_store_memory: int = 1024 * 1024 * 1024
    checkpoint_interval: int = 1000
    stage_workers: Dict[str, int] = field(default_factory=lambda: {
        'audio_download': 8,
        'audio_preprocessing': 6,
        'vad_processing': 4,
        'segment_expansion': 4,
        'feature_extraction': 6,
        'batch_inference': 1,
        'segment_aggregation': 2,
        'post_processing': 2,
        'result_writer': 1
    })
    queue_max_size: int = 100
    worker_timeout: Optional[int] = None
    max_retries: int = 3
    checkpoint_dir: str = "./checkpoints"
    enable_streaming: bool = True
    prefetch_batches: int = 10
    gpu_allocation: GPUAllocationConfig = field(default_factory=GPUAllocationConfig)


@dataclass
class MediaConfig:
    """多媒体处理配置 - 控制多种音视频格式的处理参数"""
    target_sample_rate: int = 16000
    target_channels: int = 1
    target_format: str = "wav"
    ffmpeg_num_workers: int = 4
    ffmpeg_timeout: int = 300
    ffmpeg_quality: str = "high"
    cache_enable: bool = True
    cache_max_size_gb: float = 50
    cache_ttl_hours: int = 24
    chunk_size: int = 1024 * 1024
    max_file_size_mb: int = 500


@dataclass
class AudioConfig:
    """音频处理配置 - 控制音频预处理和特征提取参数"""
    target_sample_rate: int = 16000
    max_duration: float = 30.0
    normalize: bool = True
    remove_silence: bool = False
    audio_format: str = 'wav'
    features: Dict[str, Any] = field(default_factory=lambda: {
        'feature_type': 'mel_spectrogram',
        'sample_rate': 16000,
        'n_fft': 400,
        'hop_length': 160,
        'n_mels': 80
    })


@dataclass
class InferenceConfig:
    """推理配置 - 控制Qwen3-Omni模型推理和生成参数"""
    model_name: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    tensor_parallel_size: int = 1
    max_num_batched_tokens: int = 8192
    max_model_len: int = 32768
    gpu_memory_utilization: float = 0.95
    trust_remote_code: bool = True
    dtype: str = "auto"
    temperature: float = 1e-2
    max_tokens: int = 8192
    top_p: float = 0.1
    repetition_penalty: float = 1.1
    max_num_seqs: int = 1
    limit_mm_per_prompt: Optional[Dict[str, int]] = None
    seed: int = 1234
    
    def __post_init__(self):
        if self.limit_mm_per_prompt is None:
            self.limit_mm_per_prompt = {'image': 1, 'video': 3, 'audio': 3}


@dataclass
class WriterConfig:
    """写入配置 - 控制结果写入和存储参数"""
    batch_size: int = 1000
    flush_interval: float = 10.0
    max_file_size_mb: int = 100
    output_format: str = 'jsonl'
    compression: Optional[str] = None
    async_upload: bool = True
    retry_attempts: int = 3
    retry_delay: float = 1.0


@dataclass
class MonitoringConfig:
    """监控配置 - 控制系统监控和告警参数"""
    enable_prometheus: bool = True
    prometheus_port: int = 8000
    metrics_interval: float = 5.0
    enable_gpu_monitoring: bool = True
    enable_ray_monitoring: bool = True
    checkpoint_interval: int = 1000
    checkpoint_dir: str = "./checkpoints"
    alert_rules: list = field(default_factory=list)


@dataclass
class VADConfig:
    """VAD语音活动检测配置 - 控制语音活动检测参数"""
    model_path: str = "silero_vad.onnx"
    sampling_rate: int = 16000
    threshold: float = 0.4
    min_speech_duration_ms: int = 1500
    min_silence_duration_ms: int = 1000
    speech_pad_ms: int = 100
    batch_size: int = 32
    cache_enabled: bool = True
    cache_dir: str = "./cache/vad"
    cache_max_size_gb: float = 10.0
    cache_ttl_hours: int = 24
    parallel_workers: int = 8


@dataclass
class ASRDistillationConfig:
    """ASR蒸馏框架主配置类 - 整合所有子模块配置"""
    data: DataConfig = field(default_factory=DataConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    media: MediaConfig = field(default_factory=MediaConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    writer: WriterConfig = field(default_factory=WriterConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)


class ConfigManager:
    """配置管理器 - 负责配置文件的加载、保存、验证和更新"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = None
        
    def load_config(self, config_path: Optional[str] = None) -> ASRDistillationConfig:
        """从YAML文件加载配置"""
        if config_path:
            self.config_path = config_path
            
        if self.config_path and Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_dict = yaml.safe_load(f)
                
                self.config = ASRDistillationConfig()
                
                if 'data' in config_dict:
                    self._update_dataclass(self.config.data, config_dict['data'])
                if 'pipeline' in config_dict:
                    self._update_dataclass(self.config.pipeline, config_dict['pipeline'])
                    # 处理GPU分配配置
                    if 'gpu_allocation' in config_dict['pipeline']:
                        self._load_gpu_allocation(config_dict['pipeline']['gpu_allocation'])
                if 'media' in config_dict:
                    self._update_dataclass(self.config.media, config_dict['media'])
                if 'vad' in config_dict:
                    self._update_dataclass(self.config.vad, config_dict['vad'])
                if 'audio' in config_dict:
                    self._update_dataclass(self.config.audio, config_dict['audio'])
                if 'inference' in config_dict:
                    self._update_dataclass(self.config.inference, config_dict['inference'])
                if 'writer' in config_dict:
                    self._update_dataclass(self.config.writer, config_dict['writer'])
                if 'monitoring' in config_dict:
                    self._update_dataclass(self.config.monitoring, config_dict['monitoring'])
                
            except Exception as e:
                self.config = ASRDistillationConfig()
        else:
            self.config = ASRDistillationConfig()
            
        return self.config
    
    def _load_gpu_allocation(self, gpu_config: Dict[str, Any]) -> None:
        """加载GPU分配配置"""
        nodes = []
        for node_dict in gpu_config.get('nodes', []):
            nodes.append(NodeGPUConfig(**node_dict))
        
        self.config.pipeline.gpu_allocation = GPUAllocationConfig(
            strategy=gpu_config.get('strategy', 'custom'),
            nodes=nodes
        )
    
    def _update_dataclass(self, dataclass_obj, update_dict):
        """更新数据类对象的属性"""
        for key, value in update_dict.items():
            if hasattr(dataclass_obj, key):
                setattr(dataclass_obj, key, value)
    
    def save_config(self, config_path: str, config: Optional[ASRDistillationConfig] = None) -> None:
        """保存配置到YAML文件"""
        if config is None:
            config = self.config
            
        if config is None:
            raise ValueError("没有可保存的配置")
        
        dict_config = OmegaConf.structured(config)
        OmegaConf.save(dict_config, config_path)
    
    def get_config(self) -> ASRDistillationConfig:
        """获取当前配置"""
        if self.config is None:
            self.load_config()
        return self.config
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """更新配置"""
        if self.config is None:
            self.load_config()
        
        dict_config = OmegaConf.structured(self.config)
        update_config = OmegaConf.create(updates)
        merged_config = OmegaConf.merge(dict_config, update_config)
        self.config = OmegaConf.to_object(merged_config)
    
    def validate_config(self, config: Optional[ASRDistillationConfig] = None) -> bool:
        """验证配置的有效性"""
        if config is None:
            config = self.config
            
        if config is None:
            return False
            
        try:
            if config.pipeline.num_cpu_workers <= 0:
                return False
            if config.pipeline.num_gpu_workers <= 0:
                return False
            if config.audio.target_sample_rate <= 0:
                return False
            if config.inference.model_name == "":
                return False
            return True
        except Exception as e:
            return False
    
    def create_default_config_file(self, config_path: str) -> None:
        """创建默认配置文件"""
        default_config = ASRDistillationConfig()
        self.save_config(config_path, default_config)


def create_sample_config() -> str:
    """创建示例配置文件内容"""
    sample_config = """
# ASR Distillation Configuration

data:
  index_path: "./data/index"
  cache_dir: "./data/cache"
  cache_size_gb: 100.0
  webdataset_output_dir: "./data/webdataset"
  shard_size_mb: 100
  
  storage:
    bucket: "your-bucket-name"
    endpoint: "https://oss-cn-beijing.aliyuncs.com"
    access_key_id: "your-access-key-id"
    access_key_secret: "your-access-key-secret"
    audio_prefix: "audio/"
    result_prefix: "results/"

pipeline:
  num_cpu_workers: 10
  num_gpu_workers: 1
  batch_size: 32
  max_concurrent_batches: 4
  object_store_memory: 1073741824
  checkpoint_interval: 1000

  cpu_worker_resources:
    num_cpus: 1

  gpu_worker_resources:
    num_cpus: 1
    num_gpus: 1

  stage_workers:
    audio_download: 8
    audio_preprocessing: 6
    vad_processing: 4
    segment_expansion: 4
    feature_extraction: 6
    batch_inference: 1
    segment_aggregation: 2
    post_processing: 2

audio:
  target_sample_rate: 16000
  max_duration: 30.0
  normalize: true
  remove_silence: false
  audio_format: "wav"
  
  features:
    feature_type: "mel_spectrogram"
    sample_rate: 16000
    n_fft: 400
    hop_length: 160
    n_mels: 80

inference:
  model_name: "Qwen/Qwen3-Omni-30B-A3B-Instruct"
  tensor_parallel_size: 1
  max_num_batched_tokens: 8192
  max_model_len: 32768
  gpu_memory_utilization: 0.95
  trust_remote_code: true
  dtype: "auto"
  temperature: 0.01
  max_tokens: 8192
  top_p: 0.1
  repetition_penalty: 1.1
  max_num_seqs: 1
  limit_mm_per_prompt:
    image: 1
    video: 3
    audio: 3
  seed: 1234

writer:
  batch_size: 1000
  flush_interval: 10.0
  max_file_size_mb: 100
  output_format: "jsonl"
  compression: null
  async_upload: true
  retry_attempts: 3
  retry_delay: 1.0

monitoring:
  enable_prometheus: true
  prometheus_port: 8000
  metrics_interval: 5.0
  enable_gpu_monitoring: true
  enable_ray_monitoring: true
  checkpoint_interval: 1000
  checkpoint_dir: "./checkpoints"
  
  alert_rules:
    - name: "high_error_rate"
      severity: "warning"
      message: "High error rate detected"
      condition:
        type: "threshold"
        metric: "stages.inference.success_rate"
        threshold: 0.9
        operator: "<"
    
    - name: "high_memory_usage"
      severity: "critical"
      message: "High memory usage"
      condition:
        type: "threshold"
        metric: "system.memory_percent"
        threshold: 90
        operator: ">"
"""
    return sample_config.strip()