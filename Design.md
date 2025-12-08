# Context
我目前有一个大规模音频数据处理任务，具体情况如下：
1. **任务目标**：ASR蒸馏（Distillation）。即使用强大的多模态大模型（Teacher Model）对海量无标注音频进行识别，生成文本，用于训练更小的ASR模型。
2. **数据规模**：超过 1,000 万条音频文件（时长不等，总时长约数百万小时）。
3. **核心模型**：计划部署类似 Qwen2-Audio或Qwen3-Omni）这样的大型多模态模型作为Teacher。

# Task
请你根据上述背景，帮忙实现基础的架构代码
1. 数据层 (src/data/)
- 音频索引器: 支持Parquet格式的大规模音频索引
- 存储管理: S3/OSS接口，支持文件上传下载
- 缓存系统: 本地NVMe SSD缓存管理

2. 调度层 (src/scheduling/)
- Ray分布式流水线: 支持零拷贝对象传输
- 流水线编排器: 高级任务调度和管理

3. 计算层 (src/compute/)
- CPU音频预处理: 音频下载、重采样、特征提取
- GPU推理引擎: vLLM集成，支持连续批处理

4. 存储层 (src/storage/)
- 异步结果写入: 批量聚合写入和上传
- 结果聚合器: 处理结果统计和汇总

5. 监控层 (src/monitoring/)
- Prometheus指标: 系统和业务指标收集
- 容错管理: 自动重试和检查点恢复
- 告警系统: 自定义告警规则

6. 配置和入口
- 配置管理: 基于OmegaConf的配置系统
- CLI工具: 完整的命令行界面
- 启动脚本: 便捷的运行脚本

# 架构
采用 "Producer-Consumer" 流水线模式，通过全内存的对象传输来连接 CPU 和 GPU 阶段，最大化 H200 的利用率。
数据层：克服海量小文件 IO
● 问题：直接从 OSS 读取 1000 万个小文件会导致极高的 Latency 和 Metadata 请求费用。
● 优化策略：
  ○ 存储格式：不建议直接读原始 Wav/MP3。建议预先将音频列表及其元数据（S3 Path, Duration）构建索引。如果是超大规模，推荐使用 WebDataset 或 Parquet 格式，将音频打包成 Shard（如 100MB 一个包），减少 S3 请求次数。
  ○ 缓存层：鉴于 H200 的极高算力，建议挂载高性能并行文件系统（如 JuiceFS 或 Alluxio）作为 OSS 的缓存层，或者利用计算节点的本地 NVMe SSD 做缓存。
调度层：分布式流水线 (Ray Core)
● 选型：放弃传统的 Kafka/RabbitMQ，它们对于传输“大对象”（解码后的音频 Tensor）效率较低，且难以精细控制 GPU 显存。
● 方案：使用 Ray Data (Ray Datasets) 或 Ray Core。
  ○ Ray 的 Object Store (Plasma) 允许 CPU Worker 解码后的 Tensor 零拷贝（Zero-copy）传输给 GPU Worker。
  ○ Ray 支持 Pipelining，可以实现“边读取、边解码、边推理、边写入”，自动处理 Backpressure（背压）。
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
存储层：异步聚合写入
● Writer Actor：推理结果不要直接写回 OSS。结果是一个文本字符串。
● 机制：GPU Worker 将结果 yield 给 Writer Actor。Writer 在内存中聚合（Buffer）满一定数量（如 1000 条）或时间（如 10秒）后，批量写入一个 .jsonl 文件到 OSS。

容错与监控
● 容错 (Fault Tolerance)：
  ○ Task Level：Ray 自动处理 Worker 崩溃重启。
  ○ Data Level：记录已处理的 File_ID。如果任务中断，重新提交任务时通过 Checkpoint 过滤已处理文件（类似断点续传）。
● 监控 (Observability)：
  ○ Metrics：通过 Ray Dashboard 监控 Object Store 内存使用率。通过 Prometheus 采集 DCGM (GPU Utilization, Temperature) 和 vLLM Metrics (Token Throughput, Queue Length)。
  ○ 关键指标：关注 GPU KV Cache Usage。如果常年 90% 以上，说明 GPU 跑满了；如果低，说明 CPU 预处理跟不上。