"""Configuration management using OmegaConf"""

from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field

from omegaconf import OmegaConf, DictConfig
from loguru import logger


@dataclass
class DataConfig:
    """Data layer configuration"""
    index_path: str = "./data/index"
    cache_dir: str = "./data/cache"
    cache_size_gb: float = 100.0
    webdataset_output_dir: str = "./data/webdataset"
    shard_size_mb: int = 100
    
    # Storage configuration
    storage: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Pipeline configuration"""
    num_cpu_workers: int = 10
    num_gpu_workers: int = 1
    cpu_worker_resources: Dict[str, float] = field(default_factory=lambda: {"CPU": 1})
    gpu_worker_resources: Dict[str, float] = field(default_factory=lambda: {"CPU": 1, "GPU": 1})
    batch_size: int = 32
    max_concurrent_batches: int = 4
    object_store_memory: int = 1024 * 1024 * 1024  # 1GB
    checkpoint_interval: int = 1000


@dataclass
class AudioConfig:
    """Audio processing configuration"""
    target_sample_rate: int = 16000
    max_duration: float = 30.0
    normalize: bool = True
    remove_silence: bool = False
    audio_format: str = 'wav'
    
    # Feature extraction
    features: Dict[str, Any] = field(default_factory=lambda: {
        'feature_type': 'mel_spectrogram',
        'sample_rate': 16000,
        'n_fft': 400,
        'hop_length': 160,
        'n_mels': 80
    })


@dataclass
class InferenceConfig:
    """Inference configuration"""
    model_name: str = "Qwen/Qwen2-Audio-7B-Instruct"
    tensor_parallel_size: int = 1
    max_num_batched_tokens: int = 8192
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.9
    trust_remote_code: bool = True
    dtype: str = "auto"
    temperature: float = 0.0
    max_tokens: int = 512
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    prompt_template: str = "Transcribe the audio to text:"


@dataclass
class WriterConfig:
    """Result writer configuration"""
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
    """Monitoring configuration"""
    enable_prometheus: bool = True
    prometheus_port: int = 8000
    metrics_interval: float = 5.0
    enable_gpu_monitoring: bool = True
    enable_ray_monitoring: bool = True
    checkpoint_interval: int = 1000
    checkpoint_dir: str = "./checkpoints"
    
    # Alert rules
    alert_rules: list = field(default_factory=list)


@dataclass
class ASRDistillationConfig:
    """Main configuration class"""
    data: DataConfig = field(default_factory=DataConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    writer: WriterConfig = field(default_factory=WriterConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)


class ConfigManager:
    """Configuration manager"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = None
        
    def load_config(self, config_path: Optional[str] = None) -> ASRDistillationConfig:
        """Load configuration from file"""
        if config_path:
            self.config_path = config_path
            
        if self.config_path and Path(self.config_path).exists():
            try:
                # Load from YAML file
                dict_config = OmegaConf.load(self.config_path)
                
                # Convert to dataclass
                self.config = OmegaConf.to_object(dict_config)
                
                logger.info(f"Loaded configuration from {self.config_path}")
                
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
                self.config = ASRDistillationConfig()
        else:
            # Use default configuration
            self.config = ASRDistillationConfig()
            logger.info("Using default configuration")
            
        return self.config
    
    def save_config(self, config_path: str, config: Optional[ASRDistillationConfig] = None) -> None:
        """Save configuration to file"""
        if config is None:
            config = self.config
            
        if config is None:
            raise ValueError("No configuration to save")
            
        try:
            # Convert to DictConfig
            dict_config = OmegaConf.structured(config)
            
            # Save to YAML file
            OmegaConf.save(dict_config, config_path)
            
            logger.info(f"Saved configuration to {config_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise
    
    def get_config(self) -> ASRDistillationConfig:
        """Get current configuration"""
        if self.config is None:
            self.load_config()
        return self.config
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values"""
        if self.config is None:
            self.load_config()
            
        try:
            # Convert to DictConfig for merging
            dict_config = OmegaConf.structured(self.config)
            update_config = OmegaConf.create(updates)
            
            # Merge configurations
            merged_config = OmegaConf.merge(dict_config, update_config)
            
            # Convert back to dataclass
            self.config = OmegaConf.to_object(merged_config)
            
            logger.info("Configuration updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
            raise
    
    def validate_config(self, config: Optional[ASRDistillationConfig] = None) -> bool:
        """Validate configuration"""
        if config is None:
            config = self.config
            
        if config is None:
            logger.error("No configuration to validate")
            return False
            
        try:
            # Validate critical parameters
            if config.pipeline.num_cpu_workers <= 0:
                logger.error("num_cpu_workers must be positive")
                return False
                
            if config.pipeline.num_gpu_workers <= 0:
                logger.error("num_gpu_workers must be positive")
                return False
                
            if config.audio.target_sample_rate <= 0:
                logger.error("target_sample_rate must be positive")
                return False
                
            if config.inference.model_name == "":
                logger.error("model_name cannot be empty")
                return False
                
            logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return False
    
    def create_default_config_file(self, config_path: str) -> None:
        """Create a default configuration file"""
        default_config = ASRDistillationConfig()
        self.save_config(config_path, default_config)
        logger.info(f"Created default configuration file: {config_path}")


def create_sample_config() -> str:
    """Create a sample configuration file content"""
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
    endpoint: "https://oss-cn-beijing.aliyuncs.com"  # 阿里云OSS endpoint
    access_key_id: "your-access-key-id"
    access_key_secret: "your-access-key-secret"
    audio_prefix: "audio/"
    result_prefix: "results/"

pipeline:
  num_cpu_workers: 10
  num_gpu_workers: 1
  batch_size: 32
  max_concurrent_batches: 4
  object_store_memory: 1073741824  # 1GB
  checkpoint_interval: 1000
  
  cpu_worker_resources:
    CPU: 1
    
  gpu_worker_resources:
    CPU: 1
    GPU: 1

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
  model_name: "Qwen/Qwen2-Audio-7B-Instruct"
  tensor_parallel_size: 1
  max_num_batched_tokens: 8192
  max_model_len: 8192
  gpu_memory_utilization: 0.9
  trust_remote_code: true
  dtype: "auto"
  temperature: 0.0
  max_tokens: 512
  top_p: 0.9
  repetition_penalty: 1.1
  prompt_template: "Transcribe the audio to text:"

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