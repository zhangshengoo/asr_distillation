# ASR Distillation Framework

基于大型多模态模型的ASR蒸馏框架，使用Producer-Consumer流水线模式实现高效的音频数据处理和转录。

## 架构概述

本框架采用分布式流水线架构，专为处理大规模音频数据设计：

- **数据层**: 克服海量小文件IO瓶颈，支持WebDataset和Parquet格式
- **调度层**: 基于Ray Core的分布式流水线，支持零拷贝传输
- **计算层**: CPU音频预处理 + GPU推理(vLLM)的异构流水线
- **存储层**: 异步聚合写入，支持批量上传
- **监控层**: 完整的容错与监控机制

## 系统要求

- Python 3.9+
- CUDA 11.0+ (用于GPU推理)
- Ray 2.8.0+
- PyTorch 2.1.0+
- vLLM 0.4.0+

## 安装

1. 克隆仓库:
```bash
git clone <repository-url>
cd asr_distillation
```

2. 安装依赖:
```bash
pip install -r requirements.txt
```

## 快速开始

1. 创建配置文件:
```bash
python main.py create-config --output my_config.yaml
```

2. 编辑配置文件，设置存储和模型参数

3. 运行pipeline:
```bash
python main.py run --config my_config.yaml
```

## 配置说明

主要配置项：

### 数据层配置
```yaml
data:
  index_path: "./data/index"           # 音频索引路径
  cache_dir: "./data/cache"            # 缓存目录
  cache_size_gb: 100.0                 # 缓存大小(GB)
  storage:
    bucket: "your-bucket-name"         # SS存储桶
    audio_prefix: "audio/"             # 音频文件前缀
    result_prefix: "results/"          # 结果输出前缀
```

### 流水线配置
```yaml
pipeline:
  num_cpu_workers: 10                  # CPU工作节点数
  num_gpu_workers: 1                   # GPU工作节点数
  batch_size: 32                       # 批处理大小
  object_store_memory: 1073741824      # Ray对象存储内存
```

### 推理配置
```yaml
inference:
  model_name: "Qwen/Qwen2-Audio-7B-Instruct"  # 模型名称
  tensor_parallel_size: 1                       # 张量并行度
  gpu_memory_utilization: 0.9                   # GPU内存利用率
```

## 核心组件

### 1. 数据层 (`src/data/`)

- **AudioIndexer**: 音频数据索引管理，支持Parquet格式
- **AudioStorageManager**: OSS存储接口
- **AudioCache**: 本地缓存管理

### 2. 调度层 (`src/scheduling/`)

- **DistributedPipeline**: Ray分布式流水线
- **PipelineOrchestrator**: 高级流水线编排器

### 3. 计算层 (`src/compute/`)

- **AudioProcessor**: CPU音频预处理
  - 音频下载和缓存
  - 重采样和格式转换
  - 特征提取
- **Inference**: GPU推理引擎
  - vLLM集成
  - 批量推理优化

计算层：异构流水线
我们将计算节点划分为两个角色（Actor）：
● A. Audio Pre-processing Actors (CPU Workers)
  ○ 职责：从 OSS 拉取数据 -> FFmpeg/Torchaudio 解码 -> 重采样 (16k) -> 转换为 Tensor。
  ○ 扩缩容：CPU 处理通常是瓶颈。H200 节点通常配备 100+ CPU 核心。我们需要启动大量的 CPU Actors（例如 1个 GPU 对应 10-15 个 CPU Workers）来保证喂得饱 GPU。
● B. Inference Actors (GPU Workers with vLLM)
  ○ 职责：接收 Audio Tensor -> 运行 Qwen2-Audio -> 生成文本。
  ○ 引擎优化 (vLLM)：
    ■ AsyncLLMEngine：使用 vLLM 的异步引擎，不使用 HTTP Server 模式，减少网络开销。
    ■ Continuous Batching：这是关键。由于音频转换后的 Token 长度不一，vLLM 的 PagedAttention 可以完美解决 Padding 浪费问题。
  ○ H200 策略：H200 显存极大（141GB）。如果模型较小（如 7B），推荐单卡运行多个 vLLM 实例（MP=1），或者单卡极大 Batch Size。如果模型较大，使用 Tensor Parallel (TP)。

### 4. 存储层 (`src/storage/`)

- **AsyncResultWriter**: 异步结果写入
- **BatchResultAggregator**: 批量结果聚合

### 5. 监控层 (`src/monitoring/`)

- **MonitoringSystem**: Prometheus指标收集
- **FaultToleranceManager**: 容错管理
- **CheckpointManager**: 检查点管理

## 性能优化

### 1. 存储优化
- 使用WebDataset格式减少小文件IO
- 本地NVMe SSD缓存
- 并行文件系统(JuiceFS/AlluxIO)

### 2. 计算优化
- CPU/GPU比例调优(建议1个GPU对应10-15个CPU)
- vLLM连续批处理优化
- 显存利用率最大化

### 3. 网络优化
- Ray对象存储零拷贝传输
- 自动背压处理
- 异步结果上传

## 监控指标

### 系统指标
- CPU/GPU利用率
- 内存使用情况
- Ray对象存储使用率

### 业务指标
- 处理吞吐量(Items/秒)
- 成功率
- 各阶段延迟

### 告警规则
- 高错误率(>10%)
- 高内存使用(>90%)
- GPU异常

## 故障处理

### 自动恢复
- Worker节点崩溃自动重启
- 断点续传支持
- 失败任务重试机制

### 手动干预
- 检查点恢复
- 配置热更新
- 优雅关闭

## 扩展性

### 水平扩展
- 增加CPU/GPU工作节点
- 多机分布式部署
- 动态资源调度

### 垂直扩展
- GPU内存优化
- 批处理大小调整
- 模型并行度配置

## 开发指南

### 添加新的处理阶段
1. 继承`PipelineStage`类
2. 实现`process`方法
3. 在pipeline中注册

### 自定义监控指标
1. 在`MetricsCollector`中添加新指标
2. 在处理阶段记录指标
3. 配置告警规则

### 扩展存储后端
1. 实现`StorageManager`接口
2. 添加配置选项
3. 更新依赖

## 故障排除

### 常见问题

1. **内存不足**
   - 减少批处理大小
   - 增加对象存储内存
   - 优化缓存策略

2. **GPU利用率低**
   - 增加CPU工作节点
   - 检查数据预处理瓶颈
   - 调整批处理参数

3. **存储访问慢**
   - 使用本地缓存
   - 优化存储格式
   - 增加并发度

### 日志分析
```bash
# 查看实时日志
tail -f logs/asr_distillation_$(date +%Y-%m-%d).log

# 查看错误日志
grep "ERROR" logs/asr_distillation_*.log
```

## 许可证

[License Information]

## 贡献

[Contributing Guidelines]