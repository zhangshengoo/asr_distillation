"""Configuration management using OmegaConf"""

from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field

from omegaconf import OmegaConf, DictConfig
from loguru import logger


@dataclass
class DataConfig:
    """数据层配置 - 管理音频数据索引、缓存和存储"""
    # 音频索引文件路径，用于存储音频文件元数据信息
    index_path: str = "./data/index"
    
    # 本地缓存目录，存储下载的音频文件以减少重复下载
    cache_dir: str = "./data/cache"
    
    # 本地缓存大小限制(GB)，超出后会自动清理旧文件
    cache_size_gb: float = 100.0
    
    # WebDataset输出目录，用于存储处理后的数据集分片
    webdataset_output_dir: str = "./data/webdataset"
    
    # WebDataset分片大小(MB)，控制每个分片文件的大小
    shard_size_mb: int = 100
    
    # 存储配置，包含OSS等云存储的连接参数
    storage: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """流水线配置 - 控制分布式处理和资源分配"""
    # CPU工作节点数量，负责音频预处理和特征提取
    num_cpu_workers: int = 10
    
    # GPU工作节点数量，负责模型推理
    num_gpu_workers: int = 1
    
    # CPU工作节点资源配置，每个节点的CPU核心数
    cpu_worker_resources: Dict[str, float] = field(default_factory=lambda: {"CPU": 1})
    
    # GPU工作节点资源配置，包含CPU和GPU资源
    gpu_worker_resources: Dict[str, float] = field(default_factory=lambda: {"CPU": 1, "GPU": 1})
    
    # 批处理大小，影响内存使用和处理效率
    batch_size: int = 32
    
    # 最大并发批次数，控制并行处理能力
    max_concurrent_batches: int = 4
    
    # Ray对象存储内存大小(字节)，用于节点间数据传输
    object_store_memory: int = 1024 * 1024 * 1024  # 1GB
    
    # 检查点间隔，每处理多少批次保存一次状态
    checkpoint_interval: int = 1000


@dataclass
class AudioConfig:
    """音频处理配置 - 控制音频预处理和特征提取参数"""
    # 目标采样率(Hz)，所有音频将重采样到此频率
    target_sample_rate: int = 16000
    
    # 最大音频时长(秒)，超过此长度的音频将被截断
    max_duration: float = 30.0
    
    # 是否对音频进行归一化处理，调整音量到标准范围
    normalize: bool = True
    
    # 是否去除音频中的静音片段，可以提高识别准确率
    remove_silence: bool = False
    
    # 音频文件格式，支持wav、mp3、flac等格式
    audio_format: str = 'wav'
    
    # 特征提取配置，定义如何从音频中提取特征
    features: Dict[str, Any] = field(default_factory=lambda: {
        # 特征类型，支持mel_spectrogram、mfcc、spectrogram等
        'feature_type': 'mel_spectrogram',
        # 特征提取的采样率，通常与目标采样率相同
        'sample_rate': 16000,
        # FFT窗口大小，影响频率分辨率
        'n_fft': 400,
        # 跳跃长度，影响时间分辨率
        'hop_length': 160,
        # Mel滤波器数量，影响特征维度
        'n_mels': 80
    })


@dataclass
class InferenceConfig:
    """推理配置 - 控制Qwen3-Omni模型推理和生成参数"""
    # 模型名称或路径，支持HuggingFace模型ID或本地路径
    model_name: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    
    # 张量并行度，将模型分割到多个GPU上
    tensor_parallel_size: int = 1
    
    # 最大批处理token数，影响内存使用和吞吐量
    max_num_batched_tokens: int = 8192
    
    # 最大模型长度(token数)，超过将被截断
    max_model_len: int = 32768
    
    # GPU内存利用率(0-1)，控制vLLM使用的GPU内存比例
    gpu_memory_utilization: float = 0.95
    
    # 是否信任远程代码，用于加载自定义模型
    trust_remote_code: bool = True
    
    # 数据类型，auto表示自动选择最佳类型
    dtype: str = "auto"
    
    # 生成温度，控制输出的随机性，值越小输出越确定
    temperature: float = 1e-2
    
    # 最大生成token数，限制输出长度
    max_tokens: int = 8192
    
    # Top-p采样参数，控制词汇选择的多样性
    top_p: float = 0.1
    
    # 重复惩罚系数，避免生成重复内容
    repetition_penalty: float = 1.1
    
    # 最大序列数，控制并发处理的序列数量
    max_num_seqs: int = 1
    
    # 每个提示的多模态限制，控制图像、视频、音频的数量
    limit_mm_per_prompt: Optional[Dict[str, int]] = None
    
    # 随机种子，确保结果可重现
    seed: int = 1234
    
    def __post_init__(self):
        """初始化后处理，设置默认的多模态限制"""
        if self.limit_mm_per_prompt is None:
            self.limit_mm_per_prompt = {'image': 1, 'video': 3, 'audio': 3}


@dataclass
class WriterConfig:
    """写入配置 - 控制结果写入和存储参数"""
    # 批写入大小，累积多少结果后写入文件
    batch_size: int = 1000
    
    # 刷新间隔(秒)，定期强制写入缓冲区内容
    flush_interval: float = 10.0
    
    # 最大文件大小(MB)，超过将创建新文件
    max_file_size_mb: int = 100
    
    # 输出格式，支持jsonl、parquet、json等
    output_format: str = 'jsonl'
    
    # 压缩格式，支持gzip、bz2等，None表示不压缩
    compression: Optional[str] = None
    
    # 是否异步上传，提高写入性能
    async_upload: bool = True
    
    # 重试次数，写入失败时的重试次数
    retry_attempts: int = 3
    
    # 重试延迟(秒)，每次重试的等待时间
    retry_delay: float = 1.0


@dataclass
class MonitoringConfig:
    """监控配置 - 控制系统监控和告警参数"""
    # 是否启用Prometheus指标收集
    enable_prometheus: bool = True
    
    # Prometheus服务器端口，用于指标暴露
    prometheus_port: int = 8000
    
    # 指标收集间隔(秒)，控制监控数据更新频率
    metrics_interval: float = 5.0
    
    # 是否启用GPU监控，收集GPU利用率和内存信息
    enable_gpu_monitoring: bool = True
    
    # 是否启用Ray监控，收集分布式系统状态
    enable_ray_monitoring: bool = True
    
    # 监控检查点间隔，定期保存监控状态
    checkpoint_interval: int = 1000
    
    # 检查点目录，存储监控快照文件
    checkpoint_dir: str = "./checkpoints"
    
    # 告警规则列表，定义触发告警的条件
    alert_rules: list = field(default_factory=list)


@dataclass
class ASRDistillationConfig:
    """ASR蒸馏框架主配置类 - 整合所有子模块配置"""
    # 数据层配置，管理音频数据索引、缓存和存储
    data: DataConfig = field(default_factory=DataConfig)
    
    # 流水线配置，控制分布式处理和资源分配
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    
    # 音频处理配置，控制音频预处理和特征提取参数
    audio: AudioConfig = field(default_factory=AudioConfig)
    
    # 推理配置，控制模型推理和生成参数
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    # 写入配置，控制结果写入和存储参数
    writer: WriterConfig = field(default_factory=WriterConfig)
    
    # 监控配置，控制系统监控和告警参数
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)


class ConfigManager:
    """配置管理器 - 负责配置文件的加载、保存、验证和更新"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认配置
        """
        self.config_path = config_path
        self.config = None
        
    def load_config(self, config_path: Optional[str] = None) -> ASRDistillationConfig:
        """
        从YAML文件加载配置
        
        Args:
            config_path: 配置文件路径，如果为None则使用初始化时设置的路径
            
        Returns:
            ASRDistillationConfig: 加载的配置对象
        """
        if config_path:
            self.config_path = config_path
            
        if self.config_path and Path(self.config_path).exists():
            try:
                # 从YAML文件加载配置
                dict_config = OmegaConf.load(self.config_path)
                
                # 转换为数据类对象
                self.config = OmegaConf.to_object(dict_config)
                
                logger.info(f"从 {self.config_path} 加载配置成功")
                
            except Exception as e:
                logger.error(f"加载配置失败: {e}")
                self.config = ASRDistillationConfig()
        else:
            # 使用默认配置
            self.config = ASRDistillationConfig()
            logger.info("使用默认配置")
            
        return self.config
    
    def save_config(self, config_path: str, config: Optional[ASRDistillationConfig] = None) -> None:
        """
        保存配置到YAML文件
        
        Args:
            config_path: 保存路径
            config: 要保存的配置对象，如果为None则使用当前配置
        """
        if config is None:
            config = self.config
            
        if config is None:
            raise ValueError("没有可保存的配置")
            
        try:
            # 转换为DictConfig
            dict_config = OmegaConf.structured(config)
            
            # 保存为YAML文件
            OmegaConf.save(dict_config, config_path)
            
            logger.info(f"配置已保存到 {config_path}")
            
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
            raise
    
    def get_config(self) -> ASRDistillationConfig:
        """
        获取当前配置
        
        Returns:
            ASRDistillationConfig: 当前配置对象，如果未加载则自动加载
        """
        if self.config is None:
            self.load_config()
        return self.config
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        更新配置
        
        Args:
            updates: 要更新的配置项字典
        """
        if self.config is None:
            self.load_config()
            
        try:
            # 转换为DictConfig以便合并
            dict_config = OmegaConf.structured(self.config)
            update_config = OmegaConf.create(updates)
            
            # 合并配置
            merged_config = OmegaConf.merge(dict_config, update_config)
            
            # 转换回数据类
            self.config = OmegaConf.to_object(merged_config)
            
            logger.info("配置更新成功")
            
        except Exception as e:
            logger.error(f"更新配置失败: {e}")
            raise
    
    def validate_config(self, config: Optional[ASRDistillationConfig] = None) -> bool:
        """
        验证配置的有效性
        
        Args:
            config: 要验证的配置对象，如果为None则使用当前配置
            
        Returns:
            bool: 验证是否通过
        """
        if config is None:
            config = self.config
            
        if config is None:
            logger.error("没有可验证的配置")
            return False
            
        try:
            # 验证关键参数
            if config.pipeline.num_cpu_workers <= 0:
                logger.error("CPU工作节点数必须为正数")
                return False
                
            if config.pipeline.num_gpu_workers <= 0:
                logger.error("GPU工作节点数必须为正数")
                return False
                
            if config.audio.target_sample_rate <= 0:
                logger.error("目标采样率必须为正数")
                return False
                
            if config.inference.model_name == "":
                logger.error("模型名称不能为空")
                return False
                
            logger.info("配置验证通过")
            return True
            
        except Exception as e:
            logger.error(f"配置验证错误: {e}")
            return False
    
    def create_default_config_file(self, config_path: str) -> None:
        """
        创建默认配置文件
        
        Args:
            config_path: 配置文件保存路径
        """
        default_config = ASRDistillationConfig()
        self.save_config(config_path, default_config)
        logger.info(f"已创建默认配置文件: {config_path}")


def create_sample_config() -> str:
    """
    创建示例配置文件内容
    
    Returns:
        str: 示例配置的YAML格式字符串
    """
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