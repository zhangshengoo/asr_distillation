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
@dataclass
class DataBatch:
    """Data batch for pipeline processing"""
    batch_id: str
    items: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    
    def __post_init__(self):
        if not self.metadata:
            self.metadata = {}


class PipelineStage(ABC):
    """Abstract base class for pipeline stages"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    @abstractmethod
    def process(self, batch: DataBatch) -> DataBatch:
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
        self.storage_manager = MediaStorageManager(data_loader_config['storage'])
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
    
    def stream_batches(self, output_queue: Queue, max_batches: Optional[int] = None) -> None:
        """流式产生数据批次到队列"""
        try:
            audio_records = self.load_index()
            logger.info(f"Total records in index: {len(audio_records)}")
            
            # 过滤已处理的文件
            remaining_records = [
                record for record in audio_records 
                if record['file_id'] not in self.processed_file_ids
            ]
            logger.info(f"Remaining records to process: {len(remaining_records)}")
            
            batch_count = 0
            checkpoint_interval = 100  # 每100个batch保存一次检查点
            
            # 流式产生批次
            for i in range(self.current_batch_idx, len(remaining_records), self.batch_size):
                if max_batches and batch_count >= max_batches:
                    break
                
                batch_records = remaining_records[i:i + self.batch_size]
                batch = DataBatch(
                    batch_id=f"batch_{self.total_produced}",
                    items=batch_records,
                    metadata={'stage': 'producer', 'batch_index': self.total_produced}
                )
                
                # 将batch放入队列（会阻塞直到队列有空间）
                try:
                    output_queue.put(batch, block=True, timeout=60)
                    self.total_produced += 1
                    self.current_batch_idx = i + self.batch_size
                    batch_count += 1
                    
                    # 定期保存检查点
                    if batch_count % checkpoint_interval == 0:
                        self._save_checkpoint()
                        logger.info(f"Producer checkpoint saved: {batch_count} batches produced")
                    
                except Full:
                    logger.warning("Output queue full, retrying...")
                    time.sleep(1)
            
            # 发送结束信号
            output_queue.put(None, block=True)
            
            # 最终保存检查点
            self._save_checkpoint()
            
            logger.info(f"Producer completed: {batch_count} batches produced")
            
        except Exception as e:
            logger.error(f"Error in streaming producer: {e}")
            output_queue.put(None, block=True)  # 发送结束信号
            raise
    
    def mark_batch_processed(self, file_ids: List[str]) -> None:
        """标记批次已处理"""
        self.processed_file_ids.update(file_ids)
        self._save_checkpoint()


@ray.remote
class StreamingPipelineWorker:
    """流式Pipeline Worker - 从输入队列读取，处理后写入输出队列"""
    
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
        
        # 统计信息
        self.processed_count = 0
        self.error_count = 0
        self.total_processing_time = 0.0
        
    def process_stream(self,
                      input_queue: Queue,
                      output_queue: Queue,
                      dead_letter_queue: Queue) -> Dict[str, Any]:
        """从输入队列流式处理数据"""
        logger.info(f"Worker {self.worker_id} started processing stream")
        
        while True:
            try:
                # 从输入队列获取批次（带超时）
                batch = input_queue.get(block=True, timeout=10)
                
                # None 表示流结束
                if batch is None:
                    logger.info(f"Worker {self.worker_id} received end signal")
                    # 传递结束信号到下游
                    output_queue.put(None, block=True)
                    break
                
                # 处理批次
                start_time = time.time()
                try:
                    result = self.stage.process(batch)
                    result.metadata['worker_id'] = self.worker_id
                    result.metadata['stage'] = self.stage_name
                    result.metadata['processed_at'] = time.time()
                    result.metadata['processing_time'] = time.time() - start_time
                    
                    # 放入输出队列
                    output_queue.put(result, block=True, timeout=60)
                    
                    self.processed_count += 1
                    self.total_processing_time += time.time() - start_time
                    
                except Exception as e:
                    logger.error(f"Worker {self.worker_id} processing error: {e}")
                    
                    # 重试逻辑
                    batch.retry_count += 1
                    if batch.retry_count <= self.max_retries:
                        logger.info(f"Retrying batch {batch.batch_id} (attempt {batch.retry_count})")
                        input_queue.put(batch, block=True)
                    else:
                        logger.error(f"Batch {batch.batch_id} failed after {self.max_retries} retries")
                        batch.metadata['error'] = str(e)
                        batch.metadata['failed_worker'] = self.worker_id
                        dead_letter_queue.put(batch, block=True)
                        
                    self.error_count += 1
                    
            except Empty:
                # 队列为空，继续等待
                continue
            except Exception as e:
                logger.error(f"Worker {self.worker_id} unexpected error: {e}")
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
        
        logger.info(f"Worker {self.worker_id} completed: {stats}")
        return stats


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
                resource_config = {"CPU": 1} if stage_type == 'cpu' else {"CPU": 1, "GPU": 1}
            
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
            
            logger.info(f"Setup stage '{stage_name}': {num_workers} {stage_type} workers")
        
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
            
            # 启动生产者（异步）
            producer_queue = self.stage_queues[stage_names[0]]
            producer_task = self.producer.stream_batches.remote(
                producer_queue,
                max_batches
            )
            
            # 启动所有阶段的workers
            worker_tasks = []
            for stage_idx, stage_name in enumerate(stage_names):
                input_queue = self.stage_queues[stage_name]
                
                # 确定输出队列
                if stage_idx < len(stage_names) - 1:
                    output_queue = self.stage_queues[stage_names[stage_idx + 1]]
                else:
                    # 最后一个阶段，输出到结果队列
                    output_queue = Queue(maxsize=self.pipeline_config.queue_max_size)
                    self.stage_queues['results'] = output_queue
                
                # 启动该阶段的所有workers
                for worker in self.stage_workers[stage_name]:
                    task = worker.process_stream.remote(
                        input_queue,
                        output_queue,
                        self.dead_letter_queue
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
                    logger.error(f"Worker task failed: {e}")
            
            # 等待生产者完成
            ray.get(producer_task)
            
            # 收集结果
            results = self._collect_results()
            
            # 计算统计信息
            execution_stats = self._compute_stats(worker_stats, results)
            
            logger.info(f"Pipeline execution completed: {execution_stats}")
            
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
                for stage_name, queue in self.stage_queues.items():
                    queue_stats[stage_name] = {
                        'size': queue.qsize(),
                        'maxsize': queue.maxsize
                    }
                
                # 计算总体进度
                total_processed = sum(
                    len(workers) for workers in self.stage_workers.values()
                )
                
                # 调用进度回调
                if progress_callback:
                    progress_callback(total_processed, queue_stats)
                
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
    
    def _collect_results(self) -> List[DataBatch]:
        """收集Pipeline结果"""
        results = []
        result_queue = self.stage_queues.get('results')
        
        if result_queue:
            while True:
                try:
                    batch = result_queue.get(block=True, timeout=10)
                    if batch is None:
                        break
                    results.append(batch)
                except Empty:
                    break
        
        logger.info(f"Collected {len(results)} result batches")
        return results
    
    def _compute_stats(self,
                      worker_stats: Dict[str, List[Dict]],
                      results: List[DataBatch]) -> Dict[str, Any]:
        """计算Pipeline统计信息"""
        total_duration = time.time() - self.start_time
        
        # 统计每个阶段
        stage_stats = {}
        for stage_name, stats_list in worker_stats.items():
            total_processed = sum(s['processed_count'] for s in stats_list)
            total_errors = sum(s['error_count'] for s in stats_list)
            avg_time = sum(s['avg_processing_time'] for s in stats_list) / len(stats_list)
            
            stage_stats[stage_name] = {
                'processed': total_processed,
                'errors': total_errors,
                'avg_processing_time': avg_time,
                'num_workers': len(stats_list)
            }
        
        # 统计结果
        total_items = sum(len(batch.items) for batch in results)
        successful_items = sum(
            len([item for item in batch.items if 'error' not in item])
            for batch in results
        )
        
        # 死信队列
        dead_letter_count = self.dead_letter_queue.qsize()
        
        return {
            'total_duration': total_duration,
            'total_batches': len(results),
            'total_items': total_items,
            'successful_items': successful_items,
            'failed_items': total_items - successful_items,
            'success_rate': successful_items / total_items if total_items > 0 else 0,
            'throughput': total_items / total_duration if total_duration > 0 else 0,
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
            progress_callback: Optional[Callable] = None) -> List[DataBatch]:
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