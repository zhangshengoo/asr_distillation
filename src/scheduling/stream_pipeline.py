"""高效的流式分布式调度系统 - 支持千万级数据处理

主要特性：
1. 流式处理：生产者-消费者模式，stage间流水线并行
2. 内存管理：队列背压控制，避免OOM
3. 容错机制：检查点、重试、死信队列
4. 动态调度：根据负载自动调整
5. 监控集成：实时进度和性能指标
"""

import time
import pickle
import asyncio
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pathlib import Path
from collections import defaultdict
import threading

import ray
from ray.util.queue import Queue, Empty, Full
from loguru import logger

# 从配置管理器导入PipelineConfig
from src.config.manager import PipelineConfig
from src.common import BatchData, SourceItem


# Pipeline控制信号
END_OF_STREAM = "END_OF_STREAM"

@dataclass
class PipelineSignal:
    """Pipeline控制信号 - 用于明确的流程控制，替代None"""
    signal_type: str  # END_OF_STREAM, SHUTDOWN, PAUSE
    source: str  # 发送者标识 (producer, worker_id)
    target_worker_count: int = 1  # 下游需要接收的worker数量
    timestamp: float = field(default_factory=time.time)
    
    def __repr__(self) -> str:
        return f"PipelineSignal({self.signal_type}, from={self.source}, targets={self.target_worker_count})"


@ray.remote
class TerminationBarrier:
    """终止信号屏障 - 解决多Worker场景下的终止竞争问题"""
    
    def __init__(self, 
                 upstream_worker_count: int, 
                 downstream_worker_count: int,
                 output_queue: Optional[Queue],
                 stage_name: str = "unknown"):
        self.upstream_worker_count = upstream_worker_count
        self.downstream_worker_count = downstream_worker_count
        self.output_queue = output_queue
        self.stage_name = stage_name
        self.signals_received = 0
        self.finished = False
        
    def signal(self, source: str) -> None:
        """接收上游Worker的终止信号"""
        if self.finished:
            return
            
        self.signals_received += 1
        if self.signals_received >= self.upstream_worker_count:
            # 如果没有输出队列（最后一个stage），直接标记完成
            if self.output_queue is None:
                logger.info(f"[BARRIER:{self.stage_name}] All upstream workers finished. Final stage - no downstream signals.")
                self.finished = True
                return
            
            logger.info(f"[BARRIER:{self.stage_name}] All upstream workers finished. Sending {self.downstream_worker_count} END_OF_STREAM signals downstream.")
            # 向下游发送指定数量的结束信号
            for i in range(self.downstream_worker_count):
                signal = PipelineSignal(
                    signal_type=END_OF_STREAM,
                    source=f"barrier_{self.stage_name}",
                    target_worker_count=self.downstream_worker_count
                )
                try:
                    self.output_queue.put(signal, block=True, timeout=30)
                except Full:
                    logger.error(f"[BARRIER:{self.stage_name}] Failed to put END_OF_STREAM signal (Queue Full)")
                except Exception as e:
                    logger.error(f"[BARRIER:{self.stage_name}] Error putting signal: {e}")
            
            self.finished = True
        else:
            logger.debug(f"[BARRIER:{self.stage_name}] Received signal from {source} ({self.signals_received}/{self.upstream_worker_count})")


class PipelineStage(ABC):
    """Abstract base class for pipeline stages"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    @abstractmethod
    def process(self, batch: BatchData) -> BatchData:
        """Process a batch of data"""
        pass


@ray.remote
class StreamingDataProducer:
    """流式数据生产者 - 避免一次性加载所有数据"""
    
    def __init__(self, 
                 data_loader_config: Dict[str, Any],
                 batch_size: int = 32,
                 checkpoint_dir: str = "./checkpoints"):
        from src.data.media_indexer import MediaDataLoader
        from src.data.storage import MediaStorageManager
        
        self.data_loader = MediaDataLoader(data_loader_config)
        
        # 使用分离的输入和输出存储配置
        self.storage_manager = MediaStorageManager(
            input_config=data_loader_config['input_storage'],
            output_config=data_loader_config['output_storage']
        )
        self.batch_size = batch_size
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 处理状态
        self.processed_file_ids: Set[str] = set()
        self.current_batch_idx = 0
        self.total_produced = 0
        
        # 加载检查点
        self._load_checkpoint()
        
    def _load_checkpoint(self) -> None:
        """加载生产者检查点"""
        checkpoint_file = self.checkpoint_dir / "producer_checkpoint.pkl"
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'rb') as f:
                    checkpoint = pickle.load(f)
                    self.processed_file_ids = checkpoint['processed_file_ids']
                    self.current_batch_idx = checkpoint['current_batch_idx']
                    self.total_produced = checkpoint['total_produced']
                logger.info(f"Loaded producer checkpoint: {len(self.processed_file_ids)} files processed")
            except Exception as e:
                logger.error(f"Failed to load producer checkpoint: {e}")
    
    def _save_checkpoint(self) -> None:
        """保存生产者检查点"""
        checkpoint_file = self.checkpoint_dir / "producer_checkpoint.pkl"
        try:
            checkpoint = {
                'processed_file_ids': self.processed_file_ids,
                'current_batch_idx': self.current_batch_idx,
                'total_produced': self.total_produced,
                'timestamp': time.time()
            }
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint, f)
        except Exception as e:
            logger.error(f"Failed to save producer checkpoint: {e}")
    
    def load_index(self) -> List[Dict[str, Any]]:
        """Load media index"""
        df = self.data_loader.load_index()
        if df.empty:
            # Build index from storage if not exists
            media_files = self.storage_manager.list_media_files()
            if media_files:
                df = self.data_loader.create_index(media_files)
        return df.to_dict('records')
    
    def stream_batches(self, output_queue: Queue, max_batches: Optional[int] = None,
                       num_downstream_workers: int = 1) -> None:
        """流式产生数据批次到队列
        
        Args:
            output_queue: 输出队列
            max_batches: 最大批次数（用于测试）
            num_downstream_workers: 下游stage的worker数量，用于发送正确数量的结束信号
        """
        try:
            audio_records = self.load_index()
            logger.info(f"[PRODUCER] Total records in index: {len(audio_records)}")
            
            # 过滤已处理的文件
            remaining_records = [
                record for record in audio_records 
                if record['file_id'] not in self.processed_file_ids
            ]
            logger.info(f"[PRODUCER] Remaining records to process: {len(remaining_records)}")
            logger.info(f"[PRODUCER] Will send {num_downstream_workers} END_OF_STREAM signals when done")
            logger.info(f"[PRODUCER] current_batch_idx={self.current_batch_idx}, remaining_records={len(remaining_records)}, batch_size={self.batch_size}")
            logger.info(f"[PRODUCER] range({self.current_batch_idx}, {len(remaining_records)}, {self.batch_size})")
            batch_count = 0
            checkpoint_interval = 100  # 每100个batch保存一次检查点
            
            # 流式产生批次
            for i in range(self.current_batch_idx, len(remaining_records), self.batch_size):
                if max_batches and batch_count >= max_batches:
                    break
                
                batch_records = remaining_records[i:i + self.batch_size]
                # Convert records to SourceItem objects
                items = [
                    SourceItem(
                        file_id=r['file_id'],
                        oss_path=r['oss_path'],
                        format=r.get('format', 'wav'),
                        duration=r.get('duration', 0.0),
                        metadata={k: v for k, v in r.items() 
                                 if k not in ['file_id', 'oss_path', 'format', 'duration']}
                    ) for r in batch_records
                ]

                batch = BatchData(
                    batch_id=f"batch_{self.total_produced}",
                    items=items,
                    metadata={'stage': 'producer', 'batch_index': self.total_produced}
                )
                
                logger.debug(f"[PRODUCER] Created batch '{batch.batch_id}' with {len(items)} SourceItems")
                
                # Rate limit to prevent object store flooding
                time.sleep(0.01)
                
                # 将batch放入队列（会阻塞直到队列有空间）
                try:
                    output_queue.put(batch, block=True, timeout=60)
                    self.total_produced += 1
                    self.current_batch_idx = i + self.batch_size
                    batch_count += 1
                    
                    # 定期保存检查点
                    if batch_count % checkpoint_interval == 0:
                        self._save_checkpoint()
                        logger.info(f"[PRODUCER] Checkpoint saved: {batch_count} batches produced")
                    
                except Full:
                    logger.warning("[PRODUCER] Output queue full, retrying...")
                    time.sleep(1)
            
            # 发送多个结束信号（每个下游worker一个）
            for i in range(num_downstream_workers):
                end_signal = PipelineSignal(
                    signal_type=END_OF_STREAM,
                    source="producer",
                    target_worker_count=num_downstream_workers
                )
                output_queue.put(end_signal, block=True)
                logger.info(f"[PRODUCER] Sent END_OF_STREAM signal {i+1}/{num_downstream_workers}")
            
            # 最终保存检查点
            self._save_checkpoint()
            
            logger.info(f"[PRODUCER] Completed: {batch_count} batches produced, {num_downstream_workers} end signals sent")
            
        except Exception as e:
            import traceback
            logger.error(f"[PRODUCER] Error: {e}")
            logger.error(f"[PRODUCER] Traceback:\n{traceback.format_exc()}")
            # 发送结束信号以避免下游worker无限等待
            for i in range(num_downstream_workers):
                end_signal = PipelineSignal(
                    signal_type=END_OF_STREAM,
                    source="producer_error",
                    target_worker_count=num_downstream_workers
                )
                output_queue.put(end_signal, block=True)
            raise
    
    def mark_batch_processed(self, file_ids: List[str]) -> None:
        """标记批次已处理"""
        self.processed_file_ids.update(file_ids)
        self._save_checkpoint()


@ray.remote
class StreamingPipelineWorker:
    """流式Pipeline Worker - 支持同步和异步Stage"""
    
    def __init__(self,
                 worker_id: str,
                 stage_name: str,
                 stage_class: type,
                 stage_config: Dict[str, Any],
                 max_retries: int = 3):
        self.worker_id = worker_id
        self.stage_name = stage_name
        self.stage = stage_class(stage_config)
        self.max_retries = max_retries
        
        # 检测是否为异步Stage
        self.is_async_stage = hasattr(self.stage, 'process_async')
        
        # 如果是异步Stage，创建事件循环
        self.loop = None
        if self.is_async_stage:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            logger.info(f"Worker {self.worker_id} initialized with async event loop")
        
        # 统计信息
        self.processed_count = 0
        self.error_count = 0
        self.total_processing_time = 0.0
    
    def _count_item_types(self, items: List) -> Dict[str, int]:
        """统计items中各类型的数量"""
        from collections import Counter
        type_counts = Counter(type(item).__name__ for item in items)
        return dict(type_counts)
        
    def process_stream(self,
                      input_queue: Queue,
                      output_queue: Optional[Queue],
                      dead_letter_queue: Queue,
                      num_downstream_workers: int = 1,
                      barrier_actor: Optional[Any] = None,
                      is_final_stage: bool = False) -> Dict[str, Any]:
        """从输入队列流式处理数据
        
        Args:
            input_queue: 输入队列
            output_queue: 输出队列（如果是最后stage则为None）
            dead_letter_queue: 死信队列
            num_downstream_workers: 下游stage的worker数量，用于发送正确数量的结束信号
            barrier_actor: 终止屏障Actor (TerminationBarrier)
            is_final_stage: 是否为最后一个stage
        """
        logger.info(f"[STAGE:{self.stage_name}][WORKER:{self.worker_id}] Started, is_final_stage={is_final_stage}, downstream_workers={num_downstream_workers}")
        
        try:
            while True:
                try:
                    # 从输入队列获取批次（带超时）
                    batch = input_queue.get(block=True, timeout=10)
                    
                    # 检查是否为PipelineSignal结束信号
                    if isinstance(batch, PipelineSignal):
                        logger.info(f"[STAGE:{self.stage_name}][WORKER:{self.worker_id}] Received {batch}")
                        
                        if barrier_actor:
                            # 使用屏障协调终止
                            logger.info(f"[STAGE:{self.stage_name}][WORKER:{self.worker_id}] Signaling termination barrier...")
                            barrier_actor.signal.remote(self.worker_id)
                        elif not is_final_stage and output_queue:
                            # 非最后stage：传统模式，向下游发送对应数量的结束信号
                            for i in range(num_downstream_workers):
                                downstream_signal = PipelineSignal(
                                    signal_type=END_OF_STREAM,
                                    source=self.worker_id,
                                    target_worker_count=num_downstream_workers
                                )
                                output_queue.put(downstream_signal, block=True)
                            logger.info(f"[STAGE:{self.stage_name}][WORKER:{self.worker_id}] Forwarded {num_downstream_workers} END_OF_STREAM signals to downstream")
                        else:
                            # 最后stage：直接退出
                            logger.info(f"[STAGE:{self.stage_name}][WORKER:{self.worker_id}] Final stage received END_OF_STREAM, exiting...")
                        break
                    
                    # 向后兼容：处理 None 信号（旧版本）
                    if batch is None:
                        logger.warning(f"[STAGE:{self.stage_name}][WORKER:{self.worker_id}] Received legacy None signal")
                        if not is_final_stage and output_queue:
                            output_queue.put(None, block=True)
                        break
                    
                    # 处理批次
                    start_time = time.time()
                    # 详细日志: 输入batch信息(DEBUG)
                    item_types = self._count_item_types(batch.items)
                    logger.debug(f"[STAGE:{self.stage_name}][WORKER:{self.worker_id}] INPUT batch '{batch.batch_id}' | items={len(batch.items)} | types={item_types}")
                    
                    try:
                        # 根据Stage类型选择处理方式
                        if self.is_async_stage:
                            # 异步Stage：在事件循环中执行
                            result = self.loop.run_until_complete(
                                self.stage.process_async(batch)
                            )
                        else:
                            # 同步Stage：直接调用
                            result = self.stage.process(batch)
                        
                        result.metadata['worker_id'] = self.worker_id
                        result.metadata['stage'] = self.stage_name
                        result.metadata['processed_at'] = time.time()
                        # 详细日志: 输出batch信息(DEBUG)
                        output_item_types = self._count_item_types(result.items)
                        processing_time = time.time() - start_time
                        logger.debug(f"[STAGE:{self.stage_name}][WORKER:{self.worker_id}] OUTPUT batch '{batch.batch_id}' | input={len(batch.items)} -> output={len(result.items)} | types={output_item_types} | time={processing_time:.2f}s")
                        
                        # 如果item数量变化明显，额外输出警告
                        if len(result.items) == 0 and len(batch.items) > 0:
                            logger.warning(f"[STAGE:{self.stage_name}] Batch '{batch.batch_id}' produced ZERO output items from {len(batch.items)} inputs!")
                        elif len(result.items) > len(batch.items) * 10:
                            logger.debug(f"[STAGE:{self.stage_name}] Batch '{batch.batch_id}' EXPANDED: {len(batch.items)} -> {len(result.items)} items (expansion stage)")
                        
                        # 放入输出队列（如果不是最后stage）
                        if not is_final_stage and output_queue is not None:
                            try:
                                output_queue.put(result, block=True, timeout=300)
                            except Full:
                                logger.error(f"[STAGE:{self.stage_name}][WORKER:{self.worker_id}] CRITICAL: Output queue FULL after 300s wait. Deadlock potential!")
                                raise
                        # 最后stage直接完成，不输出
                        
                        self.processed_count += 1
                        self.total_processing_time += time.time() - start_time
                        
                        # 定期打印状态 (每100个batch)
                        if self.processed_count % 100 == 0:
                            avg_time = self.total_processing_time / self.processed_count
                            logger.info(f"[STAGE:{self.stage_name}][WORKER:{self.worker_id}] Processed {self.processed_count} batches | Avg time: {avg_time:.3f}s")
                        
                    except Exception as e:
                        import traceback
                        logger.error(f"[STAGE:{self.stage_name}] ERROR processing batch '{batch.batch_id}': {e}")
                        logger.error(f"[STAGE:{self.stage_name}] Traceback:\n{traceback.format_exc()}")
                        
                        # 输出batch中items的详细信息以便排查
                        logger.error(f"[STAGE:{self.stage_name}] Failed batch details: items={len(batch.items)}, types={self._count_item_types(batch.items)}")
                        if batch.items:
                            first_item = batch.items[0]
                            logger.error(f"[STAGE:{self.stage_name}] First item type: {type(first_item).__name__}, has metadata: {hasattr(first_item, 'metadata')}")
                        
                        # 重试逻辑
                        batch.retry_count += 1
                        if batch.retry_count <= self.max_retries:
                            logger.warning(f"[STAGE:{self.stage_name}] RETRY batch '{batch.batch_id}' (attempt {batch.retry_count}/{self.max_retries})")
                            input_queue.put(batch, block=True)
                        else:
                            logger.error(f"[STAGE:{self.stage_name}] DEAD LETTER: batch '{batch.batch_id}' failed after {self.max_retries} retries")
                            batch.metadata['error'] = str(e)
                            batch.metadata['error_traceback'] = traceback.format_exc()
                            batch.metadata['failed_worker'] = self.worker_id
                            batch.metadata['failed_stage'] = self.stage_name
                            dead_letter_queue.put(batch, block=True)
                            
                        self.error_count += 1
                        
                except Empty:
                    # 队列为空，继续等待
                    continue
                except Exception as e:
                    import traceback
                    logger.error(f"Worker {self.worker_id} unexpected error in stage '{self.stage_name}': {e}")
                    logger.error(f"Traceback:\n{traceback.format_exc()}")
                    break
            
            # 返回统计信息
            stats = {
                'worker_id': self.worker_id,
                'stage': self.stage_name,
                'processed_count': self.processed_count,
                'error_count': self.error_count,
                'avg_processing_time': (self.total_processing_time / self.processed_count 
                                       if self.processed_count > 0 else 0)
            }
            
            logger.info(f"[STAGE:{self.stage_name}] Worker '{self.worker_id}' COMPLETED | processed={self.processed_count} | errors={self.error_count} | avg_time={stats['avg_processing_time']:.2f}s")
            return stats
            
        finally:
            # 清理事件循环
            if self.loop is not None:
                self.loop.close()
                logger.info(f"Worker {self.worker_id} event loop closed")


class StreamingPipelineOrchestrator:
    """流式Pipeline编排器 - 支持千万级数据处理"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pipeline_config = PipelineConfig(**config.get('pipeline', {}))
        self.data_config = config['data']
        
        # Pipeline组件
        self.producer = None
        self.stage_workers: Dict[str, List] = {}  # stage_name -> [workers]
        self.stage_queues: Dict[str, Queue] = {}  # stage_name -> queue
        self.dead_letter_queue = None
        
        # 监控
        self.monitoring_system = None
        self.stats = defaultdict(int)
        self.start_time = None
        
        # 初始化Ray
        if not ray.is_initialized():
            ray.init(
                object_store_memory=self.pipeline_config.object_store_memory,
                ignore_reinit_error=True
            )
            logger.info("Ray initialized")
    
    def setup_multi_stage_pipeline(self, stages_config: List[Dict[str, Any]]) -> None:
        """设置多阶段流水线
        
        Args:
            stages_config: List of stage configurations, each containing:
                - type: 'cpu' or 'gpu'
                - class: Stage class to instantiate
                - config: Configuration dictionary for the stage
                - name: Stage name for logging
                - num_workers: Number of workers for this stage (optional)
        """
        logger.info(f"Setting up streaming pipeline with {len(stages_config)} stages")
        
        # 创建生产者
        self.producer = StreamingDataProducer.remote(
            self.data_config,
            self.pipeline_config.batch_size,
            self.pipeline_config.checkpoint_dir
        )
        
        # 创建死信队列
        self.dead_letter_queue = Queue(maxsize=1000)
        
        # 创建各阶段的队列和workers
        for stage_idx, stage_config in enumerate(stages_config):
            stage_name = stage_config.get('name', f"stage_{stage_idx}")
            stage_type = stage_config['type']
            stage_class = stage_config['class']
            stage_params = stage_config['config']
            
            # 确定worker数量
            if 'num_workers' in stage_config:
                num_workers = stage_config['num_workers']
            else:
                num_workers = (self.pipeline_config.num_cpu_workers 
                              if stage_type == 'cpu' 
                              else self.pipeline_config.num_gpu_workers)
            
            # 创建阶段队列（背压控制）
            queue_size = self.pipeline_config.queue_max_size
            self.stage_queues[stage_name] = Queue(maxsize=queue_size)
            
            # 创建workers
            workers = []
            resource_config = (self.pipeline_config.cpu_worker_resources 
                             if stage_type == 'cpu' 
                             else self.pipeline_config.gpu_worker_resources)
            
            if resource_config is None:
                resource_config = {"num_cpus": 1} if stage_type == 'cpu' else {"num_cpus": 1, "num_gpus": 1}
            
            for worker_idx in range(num_workers):
                worker = StreamingPipelineWorker.options(**resource_config).remote(
                    f"{stage_name}_worker_{worker_idx}",
                    stage_name,
                    stage_class,
                    stage_params,
                    self.pipeline_config.max_retries
                )
                workers.append(worker)
            
            self.stage_workers[stage_name] = workers
            
            logger.info(f"Setup stage '{stage_name}': {num_workers} {stage_type} workers, queue_size={queue_size}")
        
        # 输出pipeline拓扑结构
        stage_names = list(self.stage_queues.keys())
        topology = " -> ".join(stage_names)
        logger.info(f"[PIPELINE] Topology: PRODUCER -> {topology}")
        logger.info(f"[PIPELINE] Total workers: {sum(len(w) for w in self.stage_workers.values())}")
        
        logger.info("Streaming pipeline setup completed")
    
    def run(self,
            max_batches: Optional[int] = None,
            progress_callback: Optional[Callable] = None,
            monitoring_system: Optional[Any] = None) -> Dict[str, Any]:
        """运行流式Pipeline
        
        Returns:
            Pipeline execution statistics
        """
        self.start_time = time.time()
        self.monitoring_system = monitoring_system
        
        if not self.producer:
            raise ValueError("Pipeline not setup. Call setup_multi_stage_pipeline() first.")
        
        logger.info("Starting streaming pipeline execution")
        
        try:
            # 获取所有阶段名称（按顺序）
            stage_names = list(self.stage_queues.keys())
            
            # 计算第一阶段的worker数量
            first_stage_worker_count = len(self.stage_workers[stage_names[0]])
            
            # 启动生产者（异步）- 传入第一阶段的worker数量
            producer_queue = self.stage_queues[stage_names[0]]
            producer_task = self.producer.stream_batches.remote(
                producer_queue,
                max_batches,
                first_stage_worker_count  # 发送对应数量的结束信号
            )
            logger.info(f"[PIPELINE] Producer started, will send {first_stage_worker_count} END_OF_STREAM signals to {stage_names[0]}")
            
            # 启动所有阶段的workers
            worker_tasks = []
            for stage_idx, stage_name in enumerate(stage_names):
                input_queue = self.stage_queues[stage_name]
                
                # 确定输出队列和下游worker数量
                if stage_idx < len(stage_names) - 1:
                    output_queue = self.stage_queues[stage_names[stage_idx + 1]]
                    next_stage_name = stage_names[stage_idx + 1]
                    num_downstream_workers = len(self.stage_workers[next_stage_name])
                    is_final_stage = False
                else:
                    # 最后一个阶段，不需要输出队列
                    output_queue = None
                    num_downstream_workers = 0
                    is_final_stage = True
                
                # 创建终止屏障 (Termination Barrier)
                # upstream_count = 当前stage worker数量
                # downstream_count = 下游需要接收的信号数量
                current_stage_worker_count = len(self.stage_workers[stage_name])
                barrier = TerminationBarrier.remote(
                    current_stage_worker_count,
                    num_downstream_workers,
                    output_queue,
                    stage_name
                )
                
                # 启动该阶段的所有workers
                for worker in self.stage_workers[stage_name]:
                    task = worker.process_stream.remote(
                        input_queue,
                        output_queue,
                        self.dead_letter_queue,
                        num_downstream_workers,
                        barrier,
                        is_final_stage  # 传递is_final_stage标志
                    )
                    worker_tasks.append((stage_name, task))
            
            # 监控进度
            progress_thread = threading.Thread(
                target=self._monitor_progress,
                args=(progress_callback,),
                daemon=True
            )
            progress_thread.start()
            
            # 等待所有workers完成
            logger.info("Waiting for pipeline workers to complete...")
            
            worker_stats = defaultdict(list)
            for stage_name, task in worker_tasks:
                try:
                    stats = ray.get(task, timeout=self.pipeline_config.worker_timeout)
                    worker_stats[stage_name].append(stats)
                except Exception as e:
                    import traceback
                    logger.error(f"Worker task failed in stage '{stage_name}': {e}")
                    logger.error(f"Stage '{stage_name}' traceback:\n{traceback.format_exc()}")
            
            # 等待生产者完成
            ray.get(producer_task)
            
            # 计算统计信息（不需要results列表）
            execution_stats = self._compute_stats(worker_stats, [])
            
            logger.info("=" * 60)
            logger.info("Pipeline Execution Completed")
            logger.info("=" * 60)
            logger.info(f"Total Duration:   {execution_stats['total_duration']:.2f}s")
            logger.info(f"Dead Letter:      {execution_stats['dead_letter_count']}")
            logger.info("-" * 60)
            logger.info("Stage Statistics:")
            for stage, s in execution_stats['stage_stats'].items():
                logger.info(f"  {stage:<20} | Processed: {s['processed']:<8} | Errors: {s['errors']:<6} | Avg Time: {s['avg_processing_time']:.3f}s")
            logger.info("=" * 60)
            
            return execution_stats
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise
        finally:
            self._cleanup()

    def _monitor_progress(self, progress_callback: Optional[Callable]) -> None:
        """监控Pipeline进度"""
        last_update = time.time()
        update_interval = 5.0  # 每5秒更新一次
        
        while True:
            try:
                current_time = time.time()
                if current_time - last_update < update_interval:
                    time.sleep(1)
                    continue
                
                # 收集队列状态
                queue_stats = {}
                queue_summary = []
                for stage_name, queue in self.stage_queues.items():
                    size = queue.qsize()
                    maxsize = queue.maxsize
                    usage_pct = (size / maxsize * 100) if maxsize > 0 else 0
                    queue_stats[stage_name] = {
                        'size': size,
                        'maxsize': maxsize,
                        'usage_pct': usage_pct
                    }
                    # 创建状态指示符
                    if usage_pct > 80:
                        indicator = '🔴'  # 队列接近满
                    elif usage_pct > 50:
                        indicator = '🟡'  # 中等负载
                    elif usage_pct > 10:
                        indicator = '🟢'  # 正常
                    else:
                        indicator = '⚪'  # 空闲
                    queue_summary.append(f"{stage_name}:{size}/{maxsize}({usage_pct:.0f}%){indicator}")
                
                # 输出队列状态汇总
                elapsed = current_time - self.start_time if self.start_time else 0
                logger.info(f"[PIPELINE] Elapsed: {elapsed:.1f}s | Queue Status: {' | '.join(queue_summary)}")
                
                # 检查潜在瓶颈
                for stage_name, stats in queue_stats.items():
                    if stats['usage_pct'] > 90:
                        logger.warning(f"[PIPELINE] BACKPRESSURE: Queue '{stage_name}' is {stats['usage_pct']:.0f}% full!")
                
                # 调用进度回调
                if progress_callback:
                    progress_callback(0, queue_stats)
                
                # 集成监控系统
                if self.monitoring_system:
                    for stage_name, stats in queue_stats.items():
                        self.monitoring_system.metrics_collector.update_queue_size(
                            stage_name, stats['size']
                        )
                
                last_update = current_time
                
            except Exception as e:
                logger.error(f"Error in progress monitoring: {e}")
                break
    
    def _compute_stats(self,
                    worker_stats: Dict[str, List[Dict]],
                    results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算Pipeline统计信息
        
        Args:
            worker_stats: 各Worker的处理统计
            results: 结果元数据列表（不是完整批次）
        """
        total_duration = time.time() - self.start_time
        
        # 统计每个阶段
        stage_stats = {}
        for stage_name, stats_list in worker_stats.items():
            total_processed = sum(s['processed_count'] for s in stats_list)
            total_errors = sum(s['error_count'] for s in stats_list)
            avg_time = sum(s['avg_processing_time'] for s in stats_list) / len(stats_list) if stats_list else 0
            
            stage_stats[stage_name] = {
                'processed': total_processed,
                'errors': total_errors,
                'avg_processing_time': avg_time,
                'num_workers': len(stats_list)
            }
        
        # 死信队列
        dead_letter_count = self.dead_letter_queue.qsize()
        
        return {
            'total_duration': total_duration,
            'dead_letter_count': dead_letter_count,
            'stage_stats': stage_stats
        }
    
    
    def _cleanup(self) -> None:
        """清理资源"""
        logger.info("Cleaning up pipeline resources...")
        
        # 清空队列
        for queue in self.stage_queues.values():
            while not queue.empty():
                try:
                    queue.get_nowait()
                except Empty:
                    break
    
    def get_checkpoint_status(self) -> Dict[str, Any]:
        """获取检查点状态"""
        checkpoint_dir = Path(self.pipeline_config.checkpoint_dir)
        
        checkpoint_files = list(checkpoint_dir.glob("*.pkl"))
        
        status = {
            'checkpoint_dir': str(checkpoint_dir),
            'num_checkpoints': len(checkpoint_files),
            'checkpoints': []
        }
        
        for ckpt_file in checkpoint_files:
            status['checkpoints'].append({
                'name': ckpt_file.name,
                'size': ckpt_file.stat().st_size,
                'modified': ckpt_file.stat().st_mtime
            })
        
        return status
    
    def cleanup(self) -> None:
        """清理Pipeline资源"""
        self._cleanup()


# 向后兼容的包装器
class PipelineOrchestrator:
    """高级Pipeline编排器（向后兼容）"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.orchestrator = StreamingPipelineOrchestrator(config)
    
    def setup_multi_stage_pipeline(self, stages_config: List[Dict[str, Any]]) -> None:
        """设置多阶段Pipeline"""
        self.orchestrator.setup_multi_stage_pipeline(stages_config)
    
    def run(self,
            max_batches: Optional[int] = None,
            progress_callback: Optional[Callable] = None) -> List[BatchData]:
        """运行Pipeline（向后兼容接口）"""
        stats = self.orchestrator.run(max_batches, progress_callback)
        
        # 返回空列表以保持向后兼容（实际结果通过ResultWriter处理）
        return []
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.orchestrator.get_checkpoint_status()
    
    def cleanup(self) -> None:
        """清理资源"""
        self.orchestrator.cleanup()