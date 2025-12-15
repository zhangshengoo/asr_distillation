"""简化的Ray Pipeline调度器 - 清晰、高效、易调试

设计原则：
1. 批处理循环：简单的for循环，易于理解和调试
2. Actor Pool：使用Ray的ActorPool自动负载均衡
3. 明确类型：每个stage输入输出类型清晰
4. 容错机制：失败重试、跳过、记录

架构：
Producer -> Stage1 Pool -> Stage2 Pool -> Stage3 Pool -> Stage4 Pool -> Consumer
"""

import time
import logging
from typing import Dict, List, Any, Optional, Callable, Type
from dataclasses import dataclass, field
from pathlib import Path
import pickle

import ray
from ray.util.actor_pool import ActorPool

logger = logging.getLogger(__name__)


# ==================== 数据结构 ====================

@dataclass
class ProcessBatch:
    """统一的批次数据结构 - 简化版"""
    batch_id: str
    data: Any  # 实际数据，可以是任何类型
    metadata: Dict[str, Any] = field(default_factory=dict)
    stage: str = "unknown"
    retry_count: int = 0
    error: Optional[str] = None
    
    def mark_stage(self, stage_name: str):
        """标记当前所在stage"""
        self.stage = stage_name
        self.metadata['current_stage'] = stage_name
        self.metadata['stage_timestamp'] = time.time()
    
    def mark_error(self, error: str):
        """标记错误"""
        self.error = error
        self.metadata['error'] = error
        self.metadata['error_timestamp'] = time.time()


# ==================== Stage接口 ====================

class StageProcessor:
    """Stage处理器基类 - 所有stage必须继承"""
    
    def __init__(self, config: Dict[str, Any], stage_name: str):
        self.config = config
        self.stage_name = stage_name
        self.processed_count = 0
        self.error_count = 0
        
    def process(self, batch: ProcessBatch) -> ProcessBatch:
        """处理批次 - 子类必须实现
        
        Args:
            batch: 输入批次
            
        Returns:
            处理后的批次
        """
        raise NotImplementedError
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'stage': self.stage_name,
            'processed': self.processed_count,
            'errors': self.error_count
        }


# ==================== Producer ====================

@ray.remote
class DataProducer:
    """数据生产者 - 从存储加载数据生成批次"""
    
    def __init__(self, config: Dict[str, Any]):
        from src.data.media_indexer import MediaDataLoader
        from src.data.storage import MediaStorageManager
        
        self.config = config
        self.batch_size = config.get('batch_size', 32)
        
        # 初始化数据加载器
        data_config = config['data']
        self.data_loader = MediaDataLoader(data_config)
        self.storage_manager = MediaStorageManager(
            input_config=data_config['input_storage'],
            output_config=data_config['output_storage']
        )
        
        # 已处理文件集合
        self.processed_files = self._load_checkpoint()
        
    def _load_checkpoint(self) -> set:
        """加载checkpoint"""
        checkpoint_file = Path("./checkpoints/producer_checkpoint.pkl")
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'rb') as f:
                    return pickle.load(f)
            except:
                return set()
        return set()
    
    def _save_checkpoint(self):
        """保存checkpoint"""
        checkpoint_file = Path("./checkpoints/producer_checkpoint.pkl")
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(self.processed_files, f)
    
    def load_batches(self, max_batches: Optional[int] = None) -> List[ProcessBatch]:
        """加载所有批次到内存（适合中小规模数据）"""
        # 加载索引
        df = self.data_loader.load_index()
        if df.empty:
            media_files = self.storage_manager.list_media_files()
            if media_files:
                df = self.data_loader.create_index(media_files)
        
        records = df.to_dict('records')
        
        # 过滤已处理文件
        remaining = [r for r in records if r['file_id'] not in self.processed_files]
        
        logger.info(f"总文件数: {len(records)}, 剩余: {len(remaining)}")
        
        # 分批
        batches = []
        for i in range(0, len(remaining), self.batch_size):
            if max_batches and len(batches) >= max_batches:
                break
                
            batch_records = remaining[i:i + self.batch_size]
            batch = ProcessBatch(
                batch_id=f"batch_{len(batches)}",
                data=batch_records,
                metadata={'record_count': len(batch_records)},
                stage='producer'
            )
            batches.append(batch)
        
        logger.info(f"生成 {len(batches)} 个批次")
        return batches
    
    def mark_completed(self, file_ids: List[str]):
        """标记文件已完成"""
        self.processed_files.update(file_ids)
        self._save_checkpoint()


# ==================== Stage Actor Pool ====================

@ray.remote
class StageActor:
    """Stage执行器 - 封装实际的Stage处理逻辑"""
    
    def __init__(self, 
                 stage_class: Type[StageProcessor],
                 config: Dict[str, Any],
                 stage_name: str,
                 worker_id: int):
        self.processor = stage_class(config, stage_name)
        self.worker_id = worker_id
        self.stage_name = stage_name
        
    def process(self, batch: ProcessBatch) -> ProcessBatch:
        """处理批次"""
        try:
            start_time = time.time()
            
            # 标记stage
            batch.mark_stage(self.stage_name)
            
            # 处理
            result = self.processor.process(batch)
            
            # 更新统计
            self.processor.processed_count += 1
            result.metadata[f'{self.stage_name}_duration'] = time.time() - start_time
            result.metadata['worker_id'] = self.worker_id
            
            return result
            
        except Exception as e:
            logger.error(f"[{self.stage_name}] Worker {self.worker_id} 处理失败: {e}")
            batch.mark_error(str(e))
            self.processor.error_count += 1
            return batch
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.processor.get_stats()
        stats['worker_id'] = self.worker_id
        return stats


# ==================== Pipeline调度器 ====================

class SimplifiedPipeline:
    """简化的Pipeline调度器 - 基于ActorPool"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.producer = None
        self.stage_pools: Dict[str, ActorPool] = {}
        self.stage_configs: List[Dict[str, Any]] = []
        
        # 统计
        self.stats = {
            'total_batches': 0,
            'successful_batches': 0,
            'failed_batches': 0,
            'start_time': None,
            'end_time': None,
            'stage_stats': {}
        }
        
        # 初始化Ray
        if not ray.is_initialized():
            ray.init(
                object_store_memory=config.get('pipeline', {}).get('object_store_memory', 1024*1024*1024),
                ignore_reinit_error=True
            )
    
    def setup_producer(self):
        """设置生产者"""
        self.producer = DataProducer.remote(self.config)
        logger.info("Producer已创建")
    
    def add_stage(self,
                  stage_class: Type[StageProcessor],
                  stage_config: Dict[str, Any],
                  stage_name: str,
                  num_workers: int = 4,
                  resources: Optional[Dict[str, float]] = None):
        """添加处理阶段
        
        Args:
            stage_class: Stage处理器类
            stage_config: Stage配置
            stage_name: Stage名称
            num_workers: Worker数量
            resources: Ray资源配置
        """
        if resources is None:
            resources = {"num_cpus": 1}
        
        # 创建Actors
        actors = []
        for i in range(num_workers):
            actor = StageActor.options(**resources).remote(
                stage_class, stage_config, stage_name, i
            )
            actors.append(actor)
        
        # 创建ActorPool
        pool = ActorPool(actors)
        self.stage_pools[stage_name] = pool
        
        # 保存配置
        self.stage_configs.append({
            'name': stage_name,
            'num_workers': num_workers,
            'class': stage_class.__name__
        })
        
        logger.info(f"Stage '{stage_name}' 已创建: {num_workers} workers")
    
    def run(self,
            max_batches: Optional[int] = None,
            max_retries: int = 3,
            progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """运行Pipeline
        
        Args:
            max_batches: 最大批次数
            max_retries: 最大重试次数
            progress_callback: 进度回调
            
        Returns:
            执行统计信息
        """
        if not self.producer:
            raise ValueError("Producer未设置，请先调用setup_producer()")
        
        if not self.stage_pools:
            raise ValueError("没有Stage，请先添加Stage")
        
        self.stats['start_time'] = time.time()
        
        # ==================== 1. 加载所有批次 ====================
        logger.info("加载批次数据...")
        batches = ray.get(self.producer.load_batches.remote(max_batches))
        self.stats['total_batches'] = len(batches)
        
        if not batches:
            logger.warning("没有数据需要处理")
            return self.stats
        
        logger.info(f"共 {len(batches)} 个批次待处理")
        
        # ==================== 2. 逐Stage处理 ====================
        current_batches = batches
        stage_names = [s['name'] for s in self.stage_configs]
        
        for stage_idx, stage_name in enumerate(stage_names):
            stage_start = time.time()
            logger.info("=" * 60)
            logger.info(f"Stage {stage_idx + 1}/{len(stage_names)}: {stage_name}")
            logger.info(f"输入批次数: {len(current_batches)}")
            logger.info("-" * 60)
            
            # 获取ActorPool
            pool = self.stage_pools[stage_name]
            
            # 提交所有任务
            for batch in current_batches:
                pool.submit(lambda a, b: a.process.remote(b), batch)
            
            # 收集结果
            processed_batches = []
            failed_batches = []
            
            while pool.has_next():
                try:
                    result = pool.get_next(timeout=300)  # 5分钟超时
                    
                    if result.error and result.retry_count < max_retries:
                        # 重试
                        result.retry_count += 1
                        logger.warning(f"批次 {result.batch_id} 失败，重试 {result.retry_count}/{max_retries}")
                        pool.submit(lambda a, b: a.process.remote(b), result)
                    elif result.error:
                        # 达到最大重试次数
                        logger.error(f"批次 {result.batch_id} 失败: {result.error}")
                        failed_batches.append(result)
                        self.stats['failed_batches'] += 1
                    else:
                        # 成功
                        processed_batches.append(result)
                    
                    # 进度回调
                    if progress_callback and len(processed_batches) % 10 == 0:
                        progress_callback(len(processed_batches), len(current_batches), stage_name)
                
                except TimeoutError:
                    logger.error(f"Stage {stage_name} 处理超时")
                    break
            
            # 更新当前批次列表（只保留成功的）
            current_batches = processed_batches
            
            # Stage统计
            stage_duration = time.time() - stage_start
            logger.info("-" * 60)
            logger.info(f"Stage完成: {stage_name}")
            logger.info(f"  成功: {len(processed_batches)}")
            logger.info(f"  失败: {len(failed_batches)}")
            logger.info(f"  耗时: {stage_duration:.2f}秒")
            logger.info(f"  吞吐: {len(processed_batches)/stage_duration:.2f} 批次/秒")
            
            if not current_batches:
                logger.error("所有批次都失败了，Pipeline终止")
                break
        
        # ==================== 3. 标记完成 ====================
        successful_file_ids = []
        for batch in current_batches:
            if isinstance(batch.data, list):
                file_ids = [r.get('file_id') or r.get('parent_file_id') 
                           for r in batch.data if isinstance(r, dict)]
                successful_file_ids.extend([fid for fid in file_ids if fid])
        
        if successful_file_ids:
            ray.get(self.producer.mark_completed.remote(successful_file_ids))
        
        # ==================== 4. 收集统计 ====================
        self.stats['end_time'] = time.time()
        self.stats['successful_batches'] = len(current_batches)
        self.stats['duration'] = self.stats['end_time'] - self.stats['start_time']
        
        # 收集各stage统计
        for stage_name, pool in self.stage_pools.items():
            stage_stats_futures = []
            for actor in pool._idle_actors:
                stage_stats_futures.append(actor.get_stats.remote())
            
            stage_stats_list = ray.get(stage_stats_futures)
            
            # 聚合统计
            total_processed = sum(s['processed'] for s in stage_stats_list)
            total_errors = sum(s['errors'] for s in stage_stats_list)
            
            self.stats['stage_stats'][stage_name] = {
                'workers': len(stage_stats_list),
                'processed': total_processed,
                'errors': total_errors,
                'success_rate': (total_processed - total_errors) / total_processed if total_processed > 0 else 0
            }
        
        return self.stats
    
    def shutdown(self):
        """关闭Pipeline"""
        logger.info("关闭Pipeline...")
        for pool in self.stage_pools.values():
            pool.__del__()  # 清理ActorPool
        
        if ray.is_initialized():
            ray.shutdown()


# ==================== 工厂函数 ====================

def create_audio_vad_pipeline(config_dict: Dict[str, Any]) -> SimplifiedPipeline:
    """创建音频VAD Pipeline的快捷函数
    
    Args:
        config_dict: 配置字典
        
    Returns:
        配置好的Pipeline实例
    """
    pipeline = SimplifiedPipeline(config_dict)
    
    # 设置Producer
    pipeline.setup_producer()
    
    # 获取worker配置
    stage_workers = config_dict.get('pipeline', {}).get('stage_workers', {})
    
    # TODO: 添加4个stage
    # 需要创建对应的StageProcessor实现
    
    return pipeline