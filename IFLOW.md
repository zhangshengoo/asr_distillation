# ASR Distillation Framework - iFlow CLI 上下文

## 项目概述

这是一个基于大型多模态模型的ASR蒸馏框架，专为处理大规模音频和视频数据设计。框架采用分布式流水线架构，使用Producer-Consumer模式实现高效的媒体数据处理和转录。通过异构计算架构（CPU预处理+GPU推理）和零拷贝传输，最大化H200等高性能计算资源的利用率。框架支持多媒体处理，能够从视频文件中提取音频并进行ASR处理，同时集成了Silero VAD语音活动检测工具，提供更精确的语音识别能力。

### 最新特性
- **VAD处理阶段**: 新增独立的VAD处理流水线阶段，支持语音活动检测和音频切分
- **音频片段处理**: 新增SegmentExpansionStage和SegmentAggregationStage，实现音频片段的展开和聚合
- **增强型运行脚本**: 提供更便捷的命令行操作和参数配置
- **多媒体索引器**: 支持音频和视频文件的统一索引管理
- **Silero VAD集成**: 高精度语音活动检测工具，支持缓存和批量处理
- **WebDataset支持**: 优化大规模数据集处理性能
- **Parquet索引**: 提高大规模数据索引效率
- **批量媒体处理器**: 高效的批量媒体处理，支持并行处理和缓存优化

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

## 项目结构

```
asr_distillation/
├── main.py                    # 主入口文件，CLI命令定义
├── config.yaml               # 配置文件模板
├── pyproject.toml            # 项目依赖和配置
├── requirements.txt          # Python依赖列表
├── README.md                 # 项目说明文档
├── Design.md                 # 架构设计文档
├── IFLOW.md                  # iFlow CLI上下文文档
├── src/                      # 源代码目录
│   ├── compute/              # 计算层
│   │   ├── __init__.py
│   │   ├── audio_processor.py    # CPU音频预处理
│   │   ├── inference.py          # GPU推理引擎
│   │   ├── vad_stage.py          # VAD处理阶段
│   │   ├── vad.py                # VAD处理器核心实现
│   │   ├── segment_processor.py  # 音频片段处理
│   │   └── media/                # 多媒体处理模块
│   │       ├── __init__.py
│   │       ├── media_detector.py     # 媒体格式检测器
│   │       ├── media_extractor.py    # 媒体提取器
│   │       └── batch_media_processor.py # 批量媒体处理器
│   ├── data/                 # 数据层
│   │   ├── __init__.py
│   │   ├── audio_indexer.py      # 音频索引管理
│   │   ├── media_indexer.py      # 多媒体索引管理
│   │   └── storage.py            # OSS存储接口
│   ├── scheduling/           # 调度层
│   │   ├── __init__.py
│   │   └── pipeline.py           # Ray分布式流水线
│   ├── storage/              # 存储层
│   │   ├── __init__.py
│   │   └── result_writer.py      # 异步结果写入
│   ├── monitoring/           # 监控层
│   │   ├── __init__.py
│   │   └── system.py             # Prometheus监控
│   └── config/               # 配置管理
│       └── manager.py            # 配置管理器
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
│   │   └── test_pipeline.py
│   ├── conftest.py          # pytest配置
│   ├── run_tests.py         # 测试运行脚本
│   └── README.md            # 测试说明文档
├── __pycache__/             # Python缓存目录
└── .git/                    # Git版本控制
```

## 核心组件

### 1. 数据层 (src/data/)
- **AudioIndexer**: 支持Parquet格式的大规模音频索引，优化小文件IO
- **MediaIndexer**: 多媒体文件索引管理，支持音频和视频文件的统一索引
- **AudioStorageManager**: 阿里云OSS存储接口，支持批量上传下载
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
- **FaultToleranceManager**: 容错管理和自动恢复
- **MetricsCollector**: Prometheus指标收集器
- **MonitoringConfig**: 监控配置管理

### 6. 配置管理 (src/config/)
- **ConfigManager**: 基于OmegaConf的配置系统
  - 支持配置验证
  - 环境变量覆盖
  - 配置模板生成
- **DataConfig**: 数据层配置
- **PipelineConfig**: 流水线配置
- **AudioConfig**: 音频处理配置
- **MediaConfig**: 多媒体处理配置
- **VADConfig**: VAD处理配置
- **InferenceConfig**: 推理配置
- **WriteConfig**: 写入配置
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
# 或使用脚本
./scripts/run.sh -a create-config -o my_config.yaml
```

### 2. 运行pipeline
```bash
python main.py run --config my_config.yaml
# 或使用脚本
./scripts/run.sh -c my_config.yaml
```

### 3. 限制处理批次
```bash
python main.py run --config my_config.yaml --max-batches 100
# 或使用脚本
./scripts/run.sh -c my_config.yaml -b 100
```

### 4. 指定日志级别
```bash
python main.py run --config my_config.yaml --log-level DEBUG
# 或使用脚本
./scripts/run.sh -c my_config.yaml -l DEBUG
```

### 5. 使用增强型运行脚本
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

### 6. 运行测试
```bash
# 使用测试运行脚本
python tests/run_tests.py

# 使用pytest直接运行
pytest tests/

# 运行特定测试
pytest tests/compute/test_audio_processor.py
pytest tests/compute/test_vad.py
pytest tests/compute/test_batch_media_processor.py
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

### 流水线配置
- `pipeline.num_cpu_workers`: CPU工作节点数
- `pipeline.num_gpu_workers`: GPU工作节点数
- `pipeline.batch_size`: 批处理大小
- `pipeline.max_concurrent_batches`: 最大并发批次
- `pipeline.object_store_memory`: Ray对象存储内存
- `pipeline.checkpoint_interval`: 检查点间隔
- `pipeline.cpu_worker_resources`: CPU工作节点资源配置
- `pipeline.gpu_worker_resources`: GPU工作节点资源配置

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

## 故障排除

### 常见问题
1. **内存不足**: 减少批处理大小，增加对象存储内存
2. **GPU利用率低**: 增加CPU工作节点数，检查数据预处理瓶颈
3. **存储访问慢**: 使用本地缓存，优化存储格式
4. **vLLM启动失败**: 检查CUDA环境和GPU内存
5. **Ray集群问题**: 检查网络配置和端口占用
6. **Qwen3-Omni模型加载失败**: 确保设置了正确的环境变量和依赖
7. **VAD处理失败**: 检查VAD模型文件和缓存目录权限

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

# 编辑配置文件
vim config.yaml

# 运行pipeline
python main.py run --config config.yaml
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
```

### 项目结构说明
- `main.py`: 主入口文件，定义CLI命令
- `config.yaml`: 配置文件模板
- `pyproject.toml`: 项目元数据和依赖配置
- `requirements.txt`: Python依赖列表
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
- **VAD处理阶段**: 集成Silero VAD语音活动检测，支持音频切分
- **音频片段处理**: 新增SegmentExpansionStage和SegmentAggregationStage
- **多媒体索引器**: 支持音频和视频文件的统一索引管理
- **增强型运行脚本**: 提供更便捷的命令行操作
- **WebDataset支持**: 优化大规模数据集处理性能
- **Parquet索引**: 提高大规模数据索引效率
- **批量媒体处理器**: 高效的批量媒体处理，支持并行处理和缓存优化
- **完整测试套件**: 覆盖所有核心组件的单元测试
- **测试运行脚本**: 提供交互式测试运行界面