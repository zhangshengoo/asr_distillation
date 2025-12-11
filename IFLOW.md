# ASR Distillation Framework - iFlow CLI 上下文

## 项目概述

这是一个基于大型多模态模型的ASR蒸馏框架，专为处理大规模音频和视频数据设计。框架采用分布式流水线架构，使用Producer-Consumer模式实现高效的媒体数据处理和转录。通过异构计算架构（CPU预处理+GPU推理）和零拷贝传输，最大化H200等高性能计算资源的利用率。框架支持多媒体处理，能够从视频文件中提取音频并进行ASR处理，同时集成了Silero VAD语音活动检测工具，提供更精确的语音识别能力。

### 最新特性
- **流式处理管道**: 新增`stream_pipeline.py`，支持千万级数据处理的流式管道
- **流式处理主程序**: 新增`main_stream.py`，支持千万级数据的流式处理
- **流式处理配置**: 新增流式处理相关配置选项
- **队列背压控制**: 实现队列背压控制，避免内存溢出
- **容错机制**: 实现检查点、重试、死信队列等容错机制
- **动态调度**: 支持根据负载自动调整
- **VAD处理阶段**: 新增独立的VAD处理流水线阶段，支持语音活动检测和音频切分
- **音频片段处理**: 新增SegmentExpansionStage和SegmentAggregationStage，实现音频片段的展开和聚合
- **增强型运行脚本**: 提供更便捷的命令行操作和参数配置
- **多媒体索引器**: 支持音频和视频文件的统一索引管理
- **Silero VAD集成**: 高精度语音活动检测工具，支持缓存和批量处理
- **WebDataset支持**: 优化大规模数据集处理性能
- **Parquet索引**: 提高大规模数据索引效率
- **批量媒体处理器**: 高效的批量媒体处理，支持并行处理和缓存优化
- **序列化测试**: 新增Ray actor序列化问题测试，提高系统稳定性
- **多媒体存储管理器**: 统一的音频和视频文件存储管理接口
- **Ray初始化诊断测试**: 新增Ray初始化问题诊断和监控系统交互测试
- **问题修复记录**: 新增fixCode.md记录已修复的Ray分布式处理和数据处理问题
- **监控系统优化**: 解决了MonitoringSystem启动导致Ray初始化卡住的问题
- **缓存方法统一**: 统一音频/视频缓存方法，支持多媒体类型
- **配置扩展**: 扩展AudioConfig支持更多音频特征提取参数
- **Ray Worker参数修复**: 修复Ray Worker资源分配参数问题
- **多阶段流水线**: 支持任意数量的顺序处理阶段，每个阶段可独立配置Worker数量
- **批处理推理优化**: 优化GPU利用率，支持连续批处理
- **Qwen3-Omni集成**: 集成最新的Qwen3-Omni多模态模型
- **阶段化Worker配置**: 支持为每个阶段独立配置Worker数量

### 核心技术栈
- **Python 3.9+**
- **Ray Core**: 分布式流水线调度，支持零拷贝对象传输
- **vLLM**: GPU推理优化，支持连续批处理
- **PyTorch & Torchaudio**: 音频处理和特征提取
- **FFmpeg**: 多媒体格式转换和音频提取
- **阿里云OSS**: 存储层，支持大规模媒体数据
- **Prometheus**: 监控系统和指标收集
- **Typer**: 现代化CLI框架
- **Loguru**: 结构化日志记录
- **OmegaConf**: 配置管理
- **WebDataset**: 大规模数据集处理
- **Pydantic**: 数据验证和配置管理
- **Silero VAD**: 语音活动检测工具
- **Parquet**: 大规模数据索引和存储格式
- **Qwen3-Omni**: 最新的多模态模型，支持音频、视频、图像处理

## 项目结构

```
asr_distillation/
├── main.py                    # 主入口文件，CLI命令定义
├── main_stream.py             # 流式处理主程序，支持千万级数据处理
├── config.yaml               # 配置文件模板
├── pyproject.toml            # 项目依赖和配置
├── requirements.txt          # Python依赖列表
├── README.md                 # 项目说明文档
├── Design.md                 # 架构设计文档
├── IFLOW.md                  # iFlow CLI上下文文档
├── Note.md                   # 问题记录和开发笔记
├── fixCode.md                # Bug修复记录
├── src/                      # 源代码目录
│   ├── __init__.py
│   ├── common.py             # 公共模块，包含DataBatch等共享类
│   ├── compute/              # 计算层
│   │   ├── __init__.py
│   │   ├── audio_processor.py    # CPU音频预处理
│   │   ├── inference.py          # GPU推理引擎
│   │   ├── segment_processor.py  # 音频片段处理
│   │   ├── vad_stage.py          # VAD处理阶段
│   │   ├── vad.py                # VAD处理器核心实现
│   │   └── media/                # 多媒体处理模块
│   │       ├── __init__.py
│   │       ├── media_detector.py     # 媒体格式检测器
│   │       ├── media_extractor.py    # 媒体提取器
│   │       └── batch_media_processor.py # 批量媒体处理器
│   ├── config/               # 配置管理
│   │   └── manager.py            # 配置管理器
│   ├── data/                 # 数据层
│   │   ├── __init__.py
│   │   ├── audio_indexer.py      # 音频索引管理
│   │   ├── media_indexer.py      # 多媒体索引管理
│   │   └── storage.py            # OSS存储接口(已更新为多媒体存储管理器)
│   ├── monitoring/           # 监控层
│   │   ├── __init__.py
│   │   └── system.py             # Prometheus监控
│   ├── scheduling/           # 调度层
│   │   ├── __init__.py
│   │   ├── pipeline.py           # Ray分布式流水线
│   │   └── stream_pipeline.py    # 流式分布式流水线，支持千万级数据处理
│   └── storage/              # 存储层
│       ├── __init__.py
│       └── result_writer.py      # 异步结果写入
├── examples/                 # 示例代码
│   ├── model_examples/       # 模型示例代码
│   │   └── qwen3_omni.py     # Qwen3-Omni模型示例
│   └── tool_examples/        # 工具示例代码
│       └── vad.py            # Silero VAD语音活动检测示例
├── scripts/                 # 辅助脚本
│   └── run.sh               # 增强型运行脚本
├── tests/                   # 测试目录
│   ├── compute/             # 计算层测试
│   │   ├── test_audio_processor.py
│   │   ├── test_batch_inference.py
│   │   ├── test_batch_media_processor.py
│   │   ├── test_media_detector.py
│   │   ├── test_media_extractor.py
│   │   └── test_vad.py
│   ├── config/              # 配置管理测试
│   │   └── test_manager.py
│   ├── data/                # 数据层测试
│   │   └── test_audio_indexer.py
│   ├── scheduling/          # 调度层测试
│   │   ├── test_pipeline.py
│   │   ├── test_serialization.py  # 序列化问题测试
│   │   └── test_ray_init.py      # Ray初始化诊断测试
│   ├── conftest.py          # pytest配置
│   ├── run_tests.py         # 测试运行脚本
│   ├── config_test.yaml     # 测试配置文件
│   └── README.md            # 测试说明文档
├── cache/                   # 缓存目录
│   └── vad/                 # VAD缓存目录
├── checkpoints/             # 检查点目录
├── data/                    # 数据目录
├── logs/                    # 日志目录
└── .git/                    # Git版本控制
```

## 核心组件

### 1. 数据层 (src/data/)
- **AudioIndexer**: 支持Parquet格式的大规模音频索引，优化小文件IO
- **MediaIndexer**: 多媒体文件索引管理，支持音频和视频文件的统一索引
- **MediaStorageManager**: 阿里云OSS存储接口，支持批量上传下载（已更新为多媒体存储管理器）
  - 支持音频和视频文件的统一管理
  - 提供向后兼容的AudioStorageManager别名
  - 增强的媒体类型检测和元数据获取
- **MediaCache**: 本地NVMe SSD缓存管理，支持音频和视频文件缓存，减少存储访问延迟
- **WebDatasetBuilder**: 构建WebDataset分片，优化大规模数据集处理
- **MediaDataLoader**: 统一的多媒体数据加载接口，提供索引和缓存管理

### 2. 计算层 (src/compute/)
- **AudioProcessor**: CPU音频预处理（下载、重采样、特征提取）
  - AudioDownloadStage: 音频下载和缓存
  - AudioPreprocessingStage: 音频预处理和格式转换
  - AudioFeatureStage: 特征提取
- **VAD处理**: 语音活动检测和音频切分
  - VADProcessingStage: VAD处理Ray Actor，支持批量处理
  - VADProcessor: VAD核心处理器，集成Silero VAD模型
- **Segment处理**: 音频片段展开和聚合
  - SegmentExpansionStage: 将VAD结果展开为segment级别的items
  - SegmentAggregationStage: 聚合处理后的音频片段
- **Inference**: GPU推理引擎（vLLM集成）
  - AudioInferenceStage: 音频推理
  - BatchInferenceStage: 批量推理
  - PostProcessingStage: 后处理
- **ASR Models**: Qwen3-Omni多模态模型实现
  - 支持模型注册和动态加载
  - 统一的模型接口和配置

### 2.1 多媒体处理模块 (src/compute/media/)
- **MediaDetector**: 媒体格式检测器
  - 支持多种音频格式：mp3, wav, flac, aac, ogg, m4a, wma
  - 支持多种视频格式：mp4, avi, mov, mkv, webm, flv, 3gp
  - 基于文件签名和扩展名的双重检测机制
  - 自动提取媒体元数据（时长、采样率、编码等）
- **MediaExtractor**: 媒体提取器
  - 使用FFmpeg进行音频提取和格式转换
  - 支持从视频中提取音频轨道
  - 可配置的音频质量设置（低/中/高）
  - 异步处理支持，提高批量处理效率
- **BatchMediaProcessor**: 批量媒体处理器
  - 高效的批量处理机制，支持并行处理
  - 内置LRU缓存系统，避免重复处理相同文件
  - 自动处理大文件分块，支持最大500MB文件
  - 详细的处理统计和错误报告
  - 支持多种输入格式和音频质量配置

### 3. 调度层 (src/scheduling/)
- **DistributedPipeline**: Ray分布式流水线核心实现
  - 支持多阶段线性数据流处理
  - 基于Ray Actor的Worker节点管理
  - 零拷贝对象传输优化
  - 轮询负载均衡策略
  - 序列化问题处理和测试支持
- **StreamingPipelineOrchestrator**: 流式Pipeline编排器，支持千万级数据处理
  - 流式处理：生产者-消费者模式，stage间流水线并行
  - 内存管理：队列背压控制，避免OOM
  - 容错机制：检查点、重试、死信队列
  - 动态调度：根据负载自动调整
  - 监控集成：实时进度和性能指标
- **PipelineOrchestrator**: 高级任务调度和管理
  - 多阶段流水线配置和编排
  - CPU/GPU异构资源调度
  - 自动背压和流量控制
  - 断点续传和容错恢复
- **核心调度特性**:
  - **多阶段流水线**: 支持任意数量的顺序处理阶段，每个阶段可独立配置Worker数量
  - **异构计算**: CPU和GPU工作节点独立管理，优化资源利用率
  - **动态负载均衡**: 阶段内采用轮询分配，阶段间通过队列缓冲
  - **容错机制**: Worker失败自动重试，错误信息传播和记录
  - **资源隔离**: 基于Ray的资源调度确保CPU/GPU资源隔离
  - **监控集成**: 实时统计各阶段处理时间和吞吐量
  - **序列化安全**: 通过测试确保Ray actor序列化问题得到妥善处理
  - **Ray初始化优化**: 通过诊断测试确保Ray与监控系统的兼容性
  - **流式处理**: 支持千万级数据的流式处理，避免一次性加载所有数据
  - **队列背压**: 实现队列背压控制，防止内存溢出

### 4. 存储层 (src/storage/)
- **AsyncResultWriter**: 异步结果写入，支持批量聚合
- **BatchResultAggregator**: 批量结果聚合，优化存储IO
- **ResultBuffer**: 线程安全的结果缓冲区
- **WriteConfig**: 写入配置管理

### 5. 监控层 (src/monitoring/)
- **MonitoringSystem**: Prometheus指标收集
  - GPU利用率监控
  - KV Cache使用率
  - 系统资源监控
  - Ray集群状态监控
- **FaultToleranceManager**: 容错管理和自动恢复
- **MetricsCollector**: Prometheus指标收集器
- **MonitoringConfig**: 监控配置管理
- **Ray初始化兼容性**: 确保监控系统与Ray初始化的兼容性

### 6. 配置管理 (src/config/)
- **ConfigManager**: 基于OmegaConf的配置系统
  - 支持配置验证
  - 环境变量覆盖
  - 配置模板生成
- **DataConfig**: 数据层配置
- **PipelineConfig**: 流水线配置，新增流式处理配置选项
- **MediaConfig**: 多媒体处理配置
- **VADConfig**: VAD处理配置
- **AudioConfig**: 音频处理配置
- **InferenceConfig**: 推理配置
- **WriterConfig**: 写入配置
- **MonitoringConfig**: 监控配置

## 流水线架构

### 处理流程
```
[AudioDownload] → [AudioPreprocessing] → [VADProcessing] → [SegmentExpansion] → [AudioFeature] → [BatchInference] → [SegmentAggregation] → [PostProcessing] → [ResultWriter]
```

### 阶段说明
1. **AudioDownloadStage**: 音频/视频文件下载和缓存
2. **AudioPreprocessingStage**: 音频预处理和格式转换
3. **VADProcessingStage**: 语音活动检测和音频切分
4. **SegmentExpansionStage**: 将VAD结果展开为segment级别
5. **AudioFeatureStage**: 音频特征提取
6. **BatchInferenceStage**: 批量GPU推理
7. **SegmentAggregationStage**: 聚合处理后的音频片段
8. **PostProcessingStage**: 结果后处理
9. **ResultWriterStage**: 异步结果写入

## 常用命令

### 1. 创建配置文件
```bash
python main.py create-config --output my_config.yaml
# 或使用流式处理
python main_stream.py create-config --output my_config.yaml
# 或使用脚本
./scripts/run.sh -a create-config -o my_config.yaml
```

### 2. 运行pipeline
```bash
# 标准处理
python main.py run --config my_config.yaml
# 流式处理（推荐用于大规模数据）
python main_stream.py run --config my_config.yaml
# 或使用脚本
./scripts/run.sh -c my_config.yaml
```

### 3. 限制处理批次
```bash
python main.py run --config my_config.yaml --max-batches 100
# 流式处理
python main_stream.py run --config my_config.yaml --max-batches 100
# 或使用脚本
./scripts/run.sh -c my_config.yaml -b 100
```

### 4. 查看流式处理状态
```bash
python main_stream.py status --checkpoint-dir ./checkpoints
```

### 5. 指定日志级别
```bash
python main.py run --config my_config.yaml --log-level DEBUG
# 流式处理
python main_stream.py run --config my_config.yaml --log-level DEBUG
# 或使用脚本
./scripts/run.sh -c my_config.yaml -l DEBUG
```

### 6. 使用增强型运行脚本
```bash
# 显示帮助信息
./scripts/run.sh --help

# 基本运行
./scripts/run.sh

# 完整配置示例
./scripts/run.sh -c production.yaml -b 1000 -l INFO

# 创建配置文件
./scripts/run.sh -a create-config -o new_config.yaml
```

### 7. 运行测试
```bash
# 使用测试运行脚本
python tests/run_tests.py

# 使用pytest直接运行
pytest tests/

# 运行特定测试
pytest tests/compute/test_audio_processor.py
pytest tests/compute/test_vad.py
pytest tests/compute/test_batch_media_processor.py

# 运行序列化测试
pytest tests/scheduling/test_serialization.py

# 运行Ray初始化诊断测试
pytest tests/scheduling/test_ray_init.py -v

# 运行特定Ray初始化测试
python tests/scheduling/test_ray_init.py
```

## 关键配置项

### 数据层配置
- `data.index_path`: 音频索引路径
- `data.cache_dir`: 缓存目录
- `data.cache_size_gb`: 缓存大小(GB)
- `data.webdataset_output_dir`: WebDataset输出目录
- `data.shard_size_mb`: 分片大小(MB)
- `data.storage.bucket`: OSS存储桶
- `data.storage.endpoint`: OSS端点
- `data.storage.access_key_id`: OSS访问密钥ID
- `data.storage.access_key_secret`: OSS访问密钥Secret
- `data.storage.audio_prefix`: 音频文件前缀
- `data.storage.result_prefix`: 结果输出前缀
- `data.storage.video_prefix`: 视频文件前缀（新增）

### 流水线配置
- `pipeline.num_cpu_workers`: CPU工作节点数
- `pipeline.num_gpu_workers`: GPU工作节点数
- `pipeline.batch_size`: 批处理大小
- `pipeline.max_concurrent_batches`: 最大并发批次
- `pipeline.object_store_memory`: Ray对象存储内存
- `pipeline.checkpoint_interval`: 检查点间隔
- `pipeline.cpu_worker_resources`: CPU工作节点资源配置
- `pipeline.gpu_worker_resources`: GPU工作节点资源配置
- `pipeline.stage_workers`: 各阶段的Worker数量配置
  - `audio_download`: 音频下载阶段Worker数量 (默认8)
  - `audio_preprocessing`: 音频预处理阶段Worker数量 (默认6)
  - `vad_processing`: VAD处理阶段Worker数量 (默认4)
  - `segment_expansion`: 片段展开阶段Worker数量 (默认4)
  - `feature_extraction`: 特征提取阶段Worker数量 (默认6)
  - `batch_inference`: 批量推理阶段Worker数量 (默认1)
  - `segment_aggregation`: 片段聚合阶段Worker数量 (默认2)
  - `post_processing`: 后处理阶段Worker数量 (默认2)
- `pipeline.queue_max_size`: 队列最大大小，用于背压控制 (默认100)
- `pipeline.worker_timeout`: Worker超时时间（秒） (默认300)
- `pipeline.max_retries`: 最大重试次数 (默认3)
- `pipeline.checkpoint_dir`: 检查点目录 (默认"./checkpoints")
- `pipeline.enable_streaming`: 启用流式处理 (默认True)
- `pipeline.prefetch_batches`: 预取批次数 (默认10)

### 音频配置
- `audio.target_sample_rate`: 目标采样率(16000)
- `audio.max_duration`: 最大音频时长(30秒)
- `audio.normalize`: 是否归一化
- `audio.remove_silence`: 是否去除静音
- `audio.audio_format`: 音频格式
- `audio.features`: 特征提取配置
  - `feature_type`: 特征类型(mel_spectrogram)
  - `sample_rate`: 采样率
  - `n_fft`: FFT窗口大小
  - `hop_length`: 跳跃长度
  - `n_mels`: Mel滤波器数量

### VAD配置
- `vad.model_path`: VAD模型路径 (silero_vad.onnx)
- `vad.sampling_rate`: 采样率 (16000)
- `vad.threshold`: 检测阈值 (0.0-1.0)
- `vad.min_speech_duration_ms`: 最小语音时长(毫秒)
- `vad.min_silence_duration_ms`: 最小静音时长(毫秒)
- `vad.speech_pad_ms`: 语音填充(毫秒)
- `vad.batch_size`: VAD批处理大小
- `vad.cache_enabled`: 是否启用VAD缓存
- `vad.cache_dir`: VAD缓存目录
- `vad.cache_max_size_gb`: VAD缓存最大大小(GB)
- `vad.cache_ttl_hours`: 缓存生存时间(小时)
- `vad.parallel_workers`: VAD并行工作进程数

### 多媒体处理配置
- `media.target_sample_rate`: 音频转换目标采样率(Hz)
  - 默认: 16000，适用于ASR处理
- `media.target_channels`: 音频转换目标声道数
  - 默认: 1(单声道)，适用于ASR处理
- `media.target_format`: 音频转换目标格式
  - 默认: "wav"，无损格式
- `media.ffmpeg_num_workers`: FFmpeg并行处理进程数
  - 默认: 4，可根据CPU核心数调整
  - 建议设置为CPU核心数的50-80%
- `media.ffmpeg_timeout`: FFmpeg转换超时时间(秒)
  - 默认: 300秒(5分钟)
  - 对于大文件可适当增加
- `media.ffmpeg_quality`: 音频转换质量
  - 选项: "low"(64k), "medium"(128k), "high"(192k)
  - 高质量会增加处理时间和文件大小
- `media.cache_enable`: 是否启用媒体处理缓存
  - 默认: true，建议启用以提高重复处理效率
- `media.cache_max_size_gb`: 媒体缓存最大大小(GB)
  - 默认: 50GB，可根据磁盘空间调整
- `media.cache_ttl_hours`: 缓存生存时间(小时)
  - 默认: 24小时，可根据需求调整
- `media.chunk_size`: 大文件分块处理大小(字节)
  - 默认: 1MB (1048576)
  - 对于大文件可适当增加
- `media.max_file_size_mb`: 最大处理文件大小(MB)
  - 默认: 500MB，可根据内存和处理能力调整
- **支持的格式**: 系统内置支持多种音视频格式
  - 音频格式: mp3, wav, flac, aac, ogg, m4a, wma
  - 视频格式: mp4, avi, mov, mkv, webm, flv, 3gp
  - 格式检测基于文件签名和扩展名双重机制

### 推理配置
- `inference.model_name`: 模型名称 (默认: Qwen/Qwen3-Omni-30B-A3B-Instruct)
- `inference.tensor_parallel_size`: 张量并行度
- `inference.max_num_batched_tokens`: 最大批处理token数
- `inference.max_model_len`: 最大模型长度
- `inference.gpu_memory_utilization`: GPU内存利用率
- `inference.trust_remote_code`: 是否信任远程代码
- `inference.dtype`: 数据类型
- `inference.temperature`: 生成温度
- `inference.max_tokens`: 最大生成token数
- `inference.top_p`: Top-p采样参数
- `inference.top_k`: Top-k采样参数
- `inference.repetition_penalty`: 重复惩罚
- `inference.max_num_seqs`: 最大序列数
- `inference.limit_mm_per_prompt`: 多模态限制
- `inference.seed`: 随机种子
- `inference.prompt_template`: 提示模板

### 写入配置
- `writer.batch_size`: 批写入大小
- `writer.flush_interval`: 刷新间隔
- `writer.max_file_size_mb`: 最大文件大小
- `writer.output_format`: 输出格式(jsonl)
- `writer.compression`: 压缩格式
- `writer.async_upload`: 异步上传
- `writer.retry_attempts`: 重试次数
- `writer.retry_delay`: 重试延迟

### 监控配置
- `monitoring.enable_prometheus`: 启用Prometheus
- `monitoring.prometheus_port`: Prometheus端口
- `monitoring.metrics_interval`: 指标收集间隔
- `monitoring.enable_gpu_monitoring`: 启用GPU监控
- `monitoring.enable_ray_monitoring`: 启用Ray监控
- `monitoring.checkpoint_interval`: 检查点间隔
- `monitoring.checkpoint_dir`: 检查点目录
- `monitoring.alert_rules`: 告警规则配置

## 开发规范

### 代码风格
- 使用 Black 进行代码格式化 (line-length: 88)
- 使用 isort 进行导入排序
- 使用 mypy 进行类型检查
- 遵循PEP 8编码规范

### 测试
- 使用 pytest 进行单元测试
- 测试文件位于 `tests/` 目录，按模块组织
- 支持异步测试 pytest-asyncio
- 开发依赖包含完整的测试工具链
- 使用 `tests/run_tests.py` 运行完整测试套件
- 支持测试覆盖率分析
- 新增序列化测试，确保Ray actor序列化问题得到妥善处理
- 新增Ray初始化诊断测试，确保监控系统与Ray的兼容性

### 添加新模型
1. 在 `examples/model_examples/` 创建新模型文件
2. 参考现有的 `qwen3_omni.py` 实现模式
3. 实现必要的方法：`_load_model_processor`, `run_model`
4. 更新模型配置和依赖

### 添加新工具
1. 在 `examples/tool_examples/` 创建新工具文件
2. 参考现有的 `vad.py` 实现模式
3. 实现必要的工具接口和配置
4. 更新相关依赖和文档

### 添加新的流水线阶段
1. 在 `src/compute/` 创建新阶段文件
2. 继承 `PipelineStage` 类
3. 实现 `process` 方法
4. 在pipeline中注册

### 模型配置最佳实践
- 在模型类中定义DEFAULT_CONFIG
- 支持用户配置覆盖默认配置
- 提供详细的配置文档

### Ray Actor序列化最佳实践
- 避免在actor初始化时创建文件对象或其他不可序列化的资源
- 使用延迟初始化模式，在需要时创建资源
- 确保所有配置和数据都可以序列化
- 使用`tests/scheduling/test_serialization.py`验证actor序列化

### Ray初始化与监控系统交互最佳实践
- 建议先初始化Ray再启动监控系统，避免潜在的冲突
- 如果需要先启动监控系统，确保等待足够时间使系统稳定
- 使用`tests/scheduling/test_ray_init.py`诊断Ray初始化问题
- 在生产环境中，监控系统应配置独立的端口避免冲突

## 性能优化建议

### 1. 资源配置
- CPU/GPU比例建议：1个GPU对应10-15个CPU工作节点
- GPU内存利用率设置为0.95以最大化性能
- 对象存储内存建议设置为1GB或更高
- H200显存优化：单卡多实例或极大Batch Size

### 2. 存储优化
- 使用WebDataset格式减少小文件IO
- 配置本地NVMe SSD缓存
- 考虑使用并行文件系统（JuiceFS/AlluxIO）
- 批量上传下载减少网络开销

### 3. 计算优化
- vLLM连续批处理优化
- PagedAttention减少Padding浪费
- AsyncLLMEngine避免HTTP开销
- 音频预处理流水线化

### 4. 监控指标
- 关注GPU利用率和KV Cache使用率
- 监控Ray对象存储内存使用情况
- 跟踪处理吞吐量和错误率
- 监控各阶段延迟和吞吐量

### 5. 序列化优化
- 确保Ray actor可以正确序列化
- 避免在actor中存储不可序列化的对象
- 使用延迟初始化处理文件句柄等资源
- 定期运行序列化测试验证系统稳定性

### 6. Ray初始化优化
- 确保监控系统与Ray初始化的兼容性
- 在生产环境中使用独立的监控端口
- 定期运行Ray初始化诊断测试
- 监控系统启动后等待足够时间再初始化Ray

### 7. 流式处理优化
- 合理设置队列大小（`queue_max_size`）以平衡内存使用和处理效率
- 调整`prefetch_batches`参数以优化预取策略
- 设置适当的`worker_timeout`以处理长时间运行的任务
- 使用死信队列处理无法处理的错误项目

## 故障排除

### 常见问题
1. **内存不足**: 减少批处理大小，增加对象存储内存
2. **GPU利用率低**: 增加CPU工作节点数，检查数据预处理瓶颈
3. **存储访问慢**: 使用本地缓存，优化存储格式
4. **vLLM启动失败**: 检查CUDA环境和GPU内存
5. **Ray集群问题**: 检查网络配置和端口占用
6. **Qwen3-Omni模型加载失败**: 确保设置了正确的环境变量和依赖
7. **VAD处理失败**: 检查VAD模型文件和缓存目录权限
8. **序列化错误**: 检查Ray actor中是否包含不可序列化的对象，参考`tests/scheduling/test_serialization.py`
9. **Ray初始化卡住**: 检查监控系统是否在Ray之前启动，参考`tests/scheduling/test_ray_init.py`
10. **Prometheus端口冲突**: 确保监控系统使用独立端口，避免与其他服务冲突
11. **Ray装饰器误用**: 确保只有真正的Actor类使用@ray.remote装饰器
12. **Ray Worker参数错误**: 使用`num_cpus`而非`cpu`参数配置CPU资源
13. **Logger序列化问题**: 避免在Ray Worker间传递Logger对象
14. **ONNX模型会话序列化问题**: 在每个Worker中独立加载ONNX模型
15. **缓存方法命名不一致**: 使用统一的缓存方法命名规范
16. **Qwen3-Omni环境变量问题**: 确保设置了VLLM_USE_V1=0, VLLM_WORKER_MULTIPROC_METHOD=spawn, VLLM_LOGGING_LEVEL=ERROR
17. **批处理推理问题**: 检查BatchInferenceStage配置和资源分配
18. **多阶段流水线问题**: 检查PipelineOrchestrator配置和阶段间数据传输
19. **音频片段处理失败**: 检查SegmentExpansionStage和SegmentAggregationStage的配置
20. **Qwen3-Omni多模态输入问题**: 确保音频数据格式符合模型要求
21. **流式处理内存溢出**: 降低队列大小（`queue_max_size`）或增加背压控制
22. **流式处理进度停滞**: 检查阶段间的数据传输和Worker状态
23. **检查点恢复失败**: 验证检查点文件的完整性和格式

### 日志位置
- 应用日志: `logs/asr_distillation_YYYY-MM-DD.log`
- 检查点: `checkpoints/` 目录
- 最终指标: `logs/final_metrics.json`
- Prometheus指标: `http://localhost:8000/metrics`

### 调试技巧
```bash
# 查看实时日志
tail -f logs/asr_distillation_$(date +%Y-%m-%d).log

# 查看错误日志
grep "ERROR" logs/asr_distillation_*.log

# 监控GPU使用
nvidia-smi -l 1

# 检查Ray状态
ray status

# 使用运行脚本调试
./scripts/run.sh -l DEBUG -b 10

# 运行序列化测试
pytest tests/scheduling/test_serialization.py -v

# 运行Ray初始化诊断测试
pytest tests/scheduling/test_ray_init.py -v

# 直接运行Ray初始化诊断
python tests/scheduling/test_ray_init.py

# 检查Qwen3-Omni环境变量
echo $VLLM_USE_V1
echo $VLLM_WORKER_MULTIPROC_METHOD
echo $VLLM_LOGGING_LEVEL

# 运行特定的VAD测试
pytest tests/compute/test_vad.py -v

# 运行音频处理器测试
pytest tests/compute/test_audio_processor.py -v

# 查看流式处理状态
python main_stream.py status --checkpoint-dir ./checkpoints
```

## 扩展性

### 水平扩展
- 增加CPU/GPU工作节点
- 多机分布式部署
- 动态资源调度
- 负载均衡优化

### 垂直扩展
- GPU内存优化
- 批处理大小调整
- 模型并行度配置
- H200显存最大化利用

### 模型扩展
- 支持多种ASR模型
- 模型热加载和切换
- A/B测试支持
- 模型版本管理

## 架构设计原则

### 1. 异构流水线
- CPU负责音频预处理
- GPU负责模型推理
- 通过Ray对象存储实现零拷贝传输
- 自动背压处理

### 2. 容错设计
- Worker节点崩溃自动重启
- 断点续传支持
- 失败任务重试机制
- 检查点定期保存

### 3. 监控可观测性
- 全链路指标监控
- 实时告警系统
- 性能分析工具
- 可视化Dashboard

### 4. 配置管理
- 分层配置系统
- 环境变量覆盖
- 配置验证和模板
- 热更新支持

### 5. 序列化安全
- 确保Ray actor可以正确序列化
- 延迟初始化不可序列化的资源
- 定期测试验证序列化安全性

### 6. 系统兼容性
- 确保各组件间的兼容性
- 监控系统与Ray的协调启动
- 避免资源冲突和端口占用

### 7. 流式处理设计
- 流式数据处理，避免一次性加载所有数据
- 队列背压控制，防止内存溢出
- 检查点和恢复机制，支持断点续传
- 动态资源调度，根据负载自动调整

## 部署建议

### 1. 硬件配置
- H200 GPU节点 (141GB显存)
- 100+ CPU核心
- NVMe SSD本地存储
- 高速网络连接

### 2. 环境准备
- CUDA 11.0+环境
- Python 3.9+环境
- 依赖包版本锁定
- 系统参数调优

### 3. 安全考虑
- OSS访问权限控制
- 网络安全配置
- 数据加密传输
- 审计日志记录

## 注意事项

1. 确保CUDA环境正确配置
2. 检查OSS访问权限和网络连接
3. 监控系统资源使用情况
4. 定期清理缓存和临时文件
5. 备份重要的配置文件和检查点
6. 关注模型许可证和使用条款
7. 定期更新依赖包和安全补丁
8. 对于Qwen3-Omni模型，确保设置了正确的环境变量：
   - `VLLM_USE_V1=0`
   - `VLLM_WORKER_MULTIPROC_METHOD=spawn`
   - `VLLM_LOGGING_LEVEL=ERROR`
   - `CUDA_VISIBLE_DEVICES=0`
9. 注意Ray actor序列化问题，避免在actor中存储不可序列化的对象
10. 使用`tests/scheduling/test_serialization.py`定期验证系统序列化安全性
11. 注意Ray初始化与监控系统的交互，避免潜在的冲突
12. 使用`tests/scheduling/test_ray_init.py`诊断Ray初始化问题
13. 确保监控系统使用独立端口，避免与其他服务冲突
14. 参考fixCode.md了解已修复的问题和解决方案
15. 注意监控系统启动顺序，建议先初始化Ray再启动监控系统
16. 缓存方法已统一，使用`get_cached_media`和`cache_media`方法
17. Ray Worker参数应使用`num_cpus`而非`cpu`参数
18. 确保Qwen3-Omni模型配置正确，特别是limit_mm_per_prompt参数
19. 批处理推理阶段需要特别注意内存管理和资源分配
20. VAD阶段的缓存配置可以显著影响处理性能
21. 多媒体处理阶段需要配置合适的FFmpeg参数和质量设置
22. 音频片段处理阶段需要合理设置时间戳和合并策略
23. 对于大规模数据处理，建议使用流式处理模式
24. 合理配置队列背压参数以防止内存溢出
25. 使用阶段化Worker配置以优化资源利用率
26. 定期检查和清理检查点文件以释放存储空间

## 项目依赖

### 主要依赖
- torch>=2.1.0
- torchaudio>=2.1.0
- ray[data,train,serve]>=2.8.0
- vllm>=0.4.0
- transformers>=4.36.0
- datasets>=2.15.0
- webdataset>=0.2.0
- pyarrow>=14.0.0
- oss2>=2.18.0
- prometheus-client>=0.19.0
- psutil>=5.9.0
- numpy>=1.24.0
- pandas>=2.1.0
- pydantic>=2.5.0
- typer>=0.9.0
- loguru>=0.7.0
- omegaconf>=2.3.0
- ffmpeg-python>=0.2.0
- silero-vad>=1.0.0
- transformers>=4.36.0
- accelerate>=0.25.0

### 开发依赖
- pytest>=7.4.0
- pytest-asyncio>=0.21.0
- black>=23.0.0
- isort>=5.12.0
- flake8>=6.0.0
- mypy>=1.7.0

## 阿里云OSS存储配置

### OSS配置说明
- 使用oss2库进行阿里云OSS操作
- 支持OSS标准端点和自定义端点
- 自动重试和错误处理
- 优化的批量上传下载操作
- 支持音频和视频文件的统一管理

### 配置示例
```yaml
data:
  storage:
    # 阿里云OSS配置
    bucket: "your-oss-bucket"
    endpoint: "https://oss-cn-beijing.aliyuncs.com"
    access_key_id: "your-oss-access-key-id"
    access_key_secret: "your-oss-access-key-secret"
    audio_prefix: "audio/"
    video_prefix: "video/"  # 新增视频文件前缀
    result_prefix: "results/"
```

### OSS端点列表
- 华东1（杭州）：https://oss-cn-hangzhou.aliyuncs.com
- 华东2（上海）：https://oss-cn-shanghai.aliyuncs.com
- 华北1（青岛）：https://oss-cn-qingdao.aliyuncs.com
- 华北2（北京）：https://oss-cn-beijing.aliyuncs.com
- 华北3（张家口）：https://oss-cn-zhangjiakou.aliyuncs.com
- 华南1（深圳）：https://oss-cn-shenzhen.aliyuncs.com
- 西南1（成都）：https://oss-cn-chengdu.aliyuncs.com

## 项目管理

### 安装和设置
```bash
# 克隆仓库
git clone https://github.com/zhangshengoo/asr_distillation.git
cd asr_distillation

# 安装依赖
pip install -r requirements.txt

# 创建配置文件
python main.py create-config --output config.yaml
# 或使用流式处理配置
python main_stream.py create-config --output config.yaml

# 编辑配置文件
vim config.yaml

# 运行pipeline（小规模数据）
python main.py run --config config.yaml
# 运行pipeline（大规模数据，推荐）
python main_stream.py run --config config.yaml
```

### 使用增强型运行脚本
```bash
# 赋予执行权限
chmod +x scripts/run.sh

# 使用脚本运行
./scripts/run.sh

# 查看帮助
./scripts/run.sh --help
```

### 开发环境设置
```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行代码格式化
black src/ tests/ examples/
isort src/ tests/ examples/

# 运行类型检查
mypy src/

# 运行测试
pytest tests/

# 运行特定测试
pytest tests/compute/test_audio_processor.py
pytest tests/compute/test_vad.py
pytest tests/data/test_audio_indexer.py

# 运行序列化测试
pytest tests/scheduling/test_serialization.py -v

# 运行Ray初始化诊断测试
pytest tests/scheduling/test_ray_init.py -v
```

### 项目结构说明
- `main.py`: 主入口文件，定义CLI命令
- `main_stream.py`: 流式处理主程序，支持千万级数据处理
- `config.yaml`: 配置文件模板
- `pyproject.toml`: 项目元数据和依赖配置
- `requirements.txt`: Python依赖列表
- `Note.md`: 问题记录和开发笔记
- `fixCode.md`: Bug修复记录
- `src/`: 源代码目录
  - `compute/`: 计算层，包含音频处理和推理
  - `data/`: 数据层，包含索引和存储
  - `scheduling/`: 调度层，包含流水线管理
  - `storage/`: 存储层，包含结果写入
  - `monitoring/`: 监控层，包含系统监控
  - `config/`: 配置管理
- `examples/`: 示例代码目录
  - `model_examples/`: 模型示例代码
  - `tool_examples/`: 工具示例代码
- `scripts/`: 辅助脚本
- `tests/`: 测试代码
  - `compute/`: 计算层测试
  - `data/`: 数据层测试
  - `scheduling/`: 调度层测试
  - `config/`: 配置管理测试

### 版本信息
- 版本: 0.1.0
- Python要求: >=3.9
- 主要依赖版本见pyproject.toml

### 新增功能
- **流式处理管道**: 新增`stream_pipeline.py`，支持千万级数据处理的流式管道
- **流式处理主程序**: 新增`main_stream.py`，支持千万级数据的流式处理
- **队列背压控制**: 实现队列背压控制，避免内存溢出
- **容错机制**: 实现检查点、重试、死信队列等容错机制
- **动态调度**: 支持根据负载自动调整
- **VAD处理阶段**: 集成Silero VAD语音活动检测，支持音频切分
- **音频片段处理**: 新增SegmentExpansionStage和SegmentAggregationStage
- **多媒体索引器**: 支持音频和视频文件的统一索引管理
- **增强型运行脚本**: 提供更便捷的命令行操作
- **WebDataset支持**: 优化大规模数据集处理性能
- **Parquet索引**: 提高大规模数据索引效率
- **批量媒体处理器**: 高效的批量媒体处理，支持并行处理和缓存优化
- **完整测试套件**: 覆盖所有核心组件的单元测试
- **测试运行脚本**: 提供交互式测试运行界面
- **序列化测试**: 新增Ray actor序列化问题测试，提高系统稳定性
- **多媒体存储管理器**: 统一的音频和视频文件存储管理接口
- **Ray初始化诊断测试**: 新增Ray初始化问题诊断和监控系统交互测试
- **问题记录文档**: 新增Note.md记录开发过程中的问题和解决方案
- **Bug修复记录**: 新增fixCode.md记录已修复的Ray分布式处理和数据处理问题
- **监控系统优化**: 解决了MonitoringSystem启动导致Ray初始化卡住的问题
- **缓存方法统一**: 统一音频/视频缓存方法，支持多媒体类型
- **配置扩展**: 扩展AudioConfig支持更多音频特征提取参数
- **Ray Worker参数修复**: 修复Ray Worker资源分配参数问题
- **多媒体处理**: 新增对多种音视频格式的支持
- **Qwen3-Omni集成**: 集成最新的Qwen3-Omni多模态模型
- **批处理推理优化**: 优化GPU利用率，支持连续批处理
- **多阶段流水线**: 支持任意数量的顺序处理阶段
- **Segment处理**: 音频片段的展开和聚合处理
- **阶段化Worker配置**: 支持为每个阶段独立配置Worker数量