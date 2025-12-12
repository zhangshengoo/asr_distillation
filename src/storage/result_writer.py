"""Async result writing and storage management"""

import json
import asyncio
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, is_dataclass
from concurrent.futures import ThreadPoolExecutor
import threading
from datetime import datetime
from enum import Enum

import numpy as np

from ..data.storage import MediaStorageManager
from ..common import BatchData, FileResultItem

logger = logging.getLogger(__name__)


class EnhancedJSONEncoder(json.JSONEncoder):
    """增强的JSON编码器 - 支持dataclass、Enum、numpy等类型
    
    自动处理常见的不可序列化对象：
    - dataclass → dict
    - Enum → value
    - numpy类型 → Python原生类型
    - datetime → ISO格式字符串
    - bytes → base64字符串
    """
    
    def default(self, obj):
        # 处理 dataclass
        if is_dataclass(obj):
            return asdict(obj)
        
        # 处理 Enum
        if isinstance(obj, Enum):
            return obj.value
        
        # 处理 numpy 类型
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        # 处理 datetime
        if isinstance(obj, datetime):
            return obj.isoformat()
        
        # 处理 bytes
        if isinstance(obj, bytes):
            import base64
            return base64.b64encode(obj).decode('utf-8')
        
        # 处理 Path
        if isinstance(obj, Path):
            return str(obj)
        
        # 默认处理
        try:
            return super().default(obj)
        except TypeError:
            # 最后的兜底：尝试转换为字符串
            return str(obj)


def safe_serialize(obj: Any) -> Any:
    """安全的序列化预处理
    
    递归处理复杂对象，确保所有内容都可JSON序列化
    
    Args:
        obj: 要序列化的对象
        
    Returns:
        可序列化的对象
    """
    # 处理字典
    if isinstance(obj, dict):
        return {k: safe_serialize(v) for k, v in obj.items()}
    
    # 处理列表
    if isinstance(obj, (list, tuple)):
        return [safe_serialize(item) for item in obj]
    
    # 处理 dataclass
    if is_dataclass(obj) and not isinstance(obj, type):
        return safe_serialize(asdict(obj))
    
    # 处理 Enum
    if isinstance(obj, Enum):
        return obj.value
    
    # 处理 numpy 类型
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    
    # 处理 datetime
    if isinstance(obj, datetime):
        return obj.isoformat()
    
    # 处理 bytes
    if isinstance(obj, bytes):
        import base64
        return base64.b64encode(obj).decode('utf-8')
    
    # 处理 Path
    if isinstance(obj, Path):
        return str(obj)
    
    # 基本类型直接返回
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    
    # 其他类型尝试转字符串
    try:
        json.dumps(obj)  # 测试是否可序列化
        return obj
    except (TypeError, ValueError):
        return str(obj)


@dataclass
class WriteConfig:
    """Configuration for result writing"""
    batch_size: int = 1000
    flush_interval: float = 10.0
    max_file_size_mb: int = 100
    output_format: str = 'jsonl'
    compression: Optional[str] = None
    async_upload: bool = True
    retry_attempts: int = 3
    retry_delay: float = 1.0
    sync_mode: bool = True


class ResultBuffer:
    """Thread-safe result buffer"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer = []
        self.lock = threading.Lock()
        self.not_empty = threading.Condition(self.lock)
        self.not_full = threading.Condition(self.lock)
        
    def put(self, item: Dict[str, Any]) -> None:
        """Add item to buffer"""
        with self.not_full:
            while len(self.buffer) >= self.max_size:
                self.not_full.wait()
            
            self.buffer.append(item)
            self.not_empty.notify()
    
    def get_batch(self, max_items: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get batch of items from buffer"""
        with self.not_empty:
            while not self.buffer:
                self.not_empty.wait()
            
            if max_items is None:
                batch = self.buffer.copy()
                self.buffer.clear()
            else:
                batch = self.buffer[:max_items]
                self.buffer = self.buffer[max_items:]
            
            self.not_full.notify()
            return batch
    
    def size(self) -> int:
        """Get current buffer size"""
        with self.lock:
            return len(self.buffer)
    
    def flush(self) -> List[Dict[str, Any]]:
        """Flush all items from buffer"""
        with self.lock:
            batch = self.buffer.copy()
            self.buffer.clear()
            self.not_full.notify()
            return batch


class FileWriter:
    """File writing utilities with enhanced serialization"""
    
    def __init__(self, config: WriteConfig):
        self.config = config
        
    def write_jsonl(self, data: List[Dict[str, Any]], file_path: str) -> bool:
        """Write data as JSONL with safe serialization"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                for item in data:
                    # ✅ 使用增强编码器和安全序列化
                    safe_item = safe_serialize(item)
                    json_str = json.dumps(safe_item, ensure_ascii=False, cls=EnhancedJSONEncoder)
                    f.write(json_str + '\n')
            return True
        except Exception as e:
            logger.error(f"Error writing JSONL: {e}", exc_info=True)
            return False
    
    def write_json(self, data: List[Dict[str, Any]], file_path: str) -> bool:
        """Write data as JSON array with safe serialization"""
        try:
            # ✅ 使用增强编码器和安全序列化
            safe_data = safe_serialize(data)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(safe_data, f, ensure_ascii=False, indent=2, cls=EnhancedJSONEncoder)
            return True
        except Exception as e:
            logger.error(f"Error writing JSON: {e}", exc_info=True)
            return False
    
    def write_parquet(self, data: List[Dict[str, Any]], file_path: str) -> bool:
        """Write data as Parquet"""
        try:
            import pandas as pd
            
            # ✅ Parquet也需要安全序列化
            safe_data = safe_serialize(data)
            df = pd.DataFrame(safe_data)
            df.to_parquet(file_path, index=False)
            return True
        except Exception as e:
            logger.error(f"Error writing Parquet: {e}", exc_info=True)
            return False
    
    def write_data(self, data: List[Dict[str, Any]], file_path: str) -> bool:
        """Write data in specified format"""
        if self.config.output_format == 'jsonl':
            return self.write_jsonl(data, file_path)
        elif self.config.output_format == 'json':
            return self.write_json(data, file_path)
        elif self.config.output_format == 'parquet':
            return self.write_parquet(data, file_path)
        else:
            logger.error(f"Unsupported format: {self.config.output_format}")
            return False


class AsyncResultWriter:
    """Async result writer with buffering"""
    
    def __init__(self, 
                 config: WriteConfig,
                 storage_manager: Optional[MediaStorageManager] = None):
        """初始化异步结果写入器
        
        Args:
            config: 写入配置
            storage_manager: 媒体存储管理器（支持OSS等云存储）
        """
        self.config = config
        self.storage_manager = storage_manager
        self.buffer = ResultBuffer(config.batch_size)
        self.file_writer = FileWriter(config)
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.write_queue = asyncio.Queue()
        self.running = False
        self.writer_task = None
        
        # Statistics
        self.stats = {
            'items_written': 0,
            'files_written': 0,
            'errors': 0,
            'last_flush': time.time()
        }
        
    async def start(self) -> None:
        """Start the async writer"""
        if self.running:
            return
            
        self.running = True
        self.writer_task = asyncio.create_task(self._writer_loop())
        logger.info("AsyncResultWriter started")
        
    async def stop(self) -> None:
        """Stop the async writer"""
        if not self.running:
            return
            
        self.running = False
        
        # Flush remaining items
        remaining = self.buffer.flush()
        if remaining:
            await self._write_batch(remaining)
        
        # Wait for writer task to complete
        if self.writer_task:
            await self.writer_task
            
        self.executor.shutdown(wait=True)
        logger.info("AsyncResultWriter stopped")
        
    async def write_item(self, item: Dict[str, Any]) -> None:
        """Write a single item"""
        if not self.running:
            raise RuntimeError("Writer not started")
            
        self.buffer.put(item)
        
        # Trigger write if buffer is full
        if self.buffer.size() >= self.config.batch_size:
            batch = self.buffer.get_batch()
            await self._write_batch(batch)
    
    async def write_batch(self, items: List[Dict[str, Any]]) -> None:
        """Write a batch of items"""
        if not self.running:
            raise RuntimeError("Writer not started")
            
        for item in items:
            self.buffer.put(item)
            
        # Trigger write
        batch = self.buffer.get_batch()
        await self._write_batch(batch)
    
    async def _writer_loop(self) -> None:
        """Main writer loop"""
        last_flush = time.time()
        
        while self.running:
            try:
                # Check if it's time to flush
                current_time = time.time()
                time_since_flush = current_time - last_flush
                
                if time_since_flush >= self.config.flush_interval:
                    # Flush buffer regardless of size
                    batch = self.buffer.flush()
                    if batch:
                        await self._write_batch(batch)
                    last_flush = current_time
                
                # Sleep a bit to avoid busy waiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in writer loop: {e}", exc_info=True)
                self.stats['errors'] += 1
                await asyncio.sleep(1)
    
    async def _write_batch(self, batch: List[Dict[str, Any]]) -> None:
        """Write a batch of data"""
        if not batch:
            return
            
        try:
            # Generate filename with timestamp
            timestamp = int(time.time())
            filename = f"results_{timestamp}.{self.config.output_format}"
            
            # Write to temporary file first
            temp_file = f"/tmp/{filename}"
            success = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.file_writer.write_data,
                batch,
                temp_file
            )
            
            if not success:
                self.stats['errors'] += 1
                logger.error(f"Failed to write batch to {temp_file}")
                return
            
            # Upload to storage if configured
            if self.storage_manager and self.config.async_upload:
                upload_success = await self._upload_to_storage(temp_file, filename)
                if upload_success:
                    # Remove temp file after successful upload
                    Path(temp_file).unlink(missing_ok=True)
                    logger.debug(f"Uploaded and cleaned up {temp_file}")
                else:
                    self.stats['errors'] += 1
                    logger.warning(f"Upload failed, keeping local file: {temp_file}")
            
            # Update statistics
            self.stats['items_written'] += len(batch)
            self.stats['files_written'] += 1
            self.stats['last_flush'] = time.time()
            
            logger.info(f"Wrote {len(batch)} items to {filename}")
            
        except Exception as e:
            logger.error(f"Error writing batch: {e}", exc_info=True)
            self.stats['errors'] += 1
    
    async def _upload_to_storage(self, local_file: str, remote_filename: str) -> bool:
        """Upload file to storage"""
        if not self.storage_manager:
            return True
            
        try:
            # Upload with retry logic
            for attempt in range(self.config.retry_attempts):
                batch_id = Path(remote_filename).stem
                success = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self.storage_manager.upload_batch_results,
                    local_file,
                    batch_id
                )
                
                if success:
                    logger.info(f"Uploaded {remote_filename} to storage")
                    return True
                    
                if attempt < self.config.retry_attempts - 1:
                    delay = self.config.retry_delay * (2 ** attempt)
                    logger.warning(f"Upload failed, retrying in {delay}s... (attempt {attempt + 1}/{self.config.retry_attempts})")
                    await asyncio.sleep(delay)
            
            logger.error(f"Upload failed after {self.config.retry_attempts} attempts")
            return False
            
        except Exception as e:
            logger.error(f"Error uploading to storage: {e}", exc_info=True)
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get writer statistics"""
        return {
            **self.stats,
            'buffer_size': self.buffer.size(),
            'running': self.running
        }


class ResultWriterStage:
    """Result writer stage for pipeline integration"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        write_config = WriteConfig(**config.get('writer', {}))
        
        storage_manager = None
        if 'storage' in config:
            storage_manager = MediaStorageManager(config['storage'])
        
        self.async_writer = AsyncResultWriter(write_config, storage_manager)
        
        # 为同步环境创建事件循环
        self._loop = None
        self._started = False
        
        self.logger = logging.getLogger("ResultWriterStage")
    
    def _ensure_started(self):
        """确保异步写入器已启动"""
        if not self._started:
            if self._loop is None:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
            
            # 启动异步写入器
            self._loop.run_until_complete(self.async_writer.start())
            self._started = True
    
    def process(self, batch: BatchData) -> BatchData:
        """同步处理接口"""
        self._ensure_started()
        self._loop.run_until_complete(self.process_async(batch))
        return batch
    
    async def process_async(self, batch: BatchData) -> BatchData:
        """异步处理批次并写入结果"""
        try:
            # 确保写入器已启动
            if not self._started:
                await self.async_writer.start()
                self._started = True
            
            # 提取结果并转换为dict
            results = []
            for item in batch.items:
                if hasattr(item, 'file_id'):
                    result_dict = {
                        'file_id': item.file_id,
                        'transcription': getattr(item, 'transcription', ''),
                        'segments': getattr(item, 'segments', []),
                        #'stats': getattr(item, 'stats', {}),
                        #'metadata': getattr(item, 'metadata', {})
                    }
                    
                    # ✅ 在写入前进行安全序列化预处理
                    result_dict = safe_serialize(result_dict)
                    
                    results.append(result_dict)

            # 异步批量写入
            if results:
                await self.async_writer.write_batch(results)
                self.logger.debug(f"Wrote {len(results)} results from batch {batch.batch_id}")
            else:
                self.logger.warning(f"No valid results in batch {batch.batch_id}")
                
            # 更新batch元数据
            batch.metadata['stage'] = 'result_writer'
            batch.metadata['results_written'] = len(results)
            batch.metadata['write_completed'] = True
            
            return batch
            
        except Exception as e:
            import traceback
            self.logger.error(f"Error writing batch {batch.batch_id}: {e}")
            self.logger.error(f"Traceback:\n{traceback.format_exc()}")
            batch.metadata['error'] = str(e)
            batch.metadata['write_completed'] = False
            return batch
    
    def cleanup(self):
        """清理资源"""
        if self._started and self._loop:
            try:
                self._loop.run_until_complete(self.async_writer.stop())
                self.logger.info("AsyncResultWriter stopped")
            except Exception as e:
                self.logger.error(f"Error stopping writer: {e}")
            finally:
                if self._loop and not self._loop.is_closed():
                    self._loop.close()
                self._loop = None
                self._started = False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取写入统计"""
        return self.async_writer.get_stats()


class BatchResultAggregator:
    """Aggregate results from multiple batches"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.aggregated_results = []
        self.start_time = time.time()
        
    def add_batch_results(self, batch: BatchData[FileResultItem]) -> None:
        """Add results from a batch"""
        for item in batch.items:
            if isinstance(item, FileResultItem):
                self.aggregated_results.append({
                    'file_id': item.file_id,
                    'transcription': item.transcription,
                    'stats': item.stats
                })
                
    def get_summary(self) -> Dict[str, Any]:
        """Get aggregation summary"""
        end_time = time.time()
        duration = end_time - self.start_time
        
        total_items = len(self.aggregated_results)
        successful_items = sum(1 for item in self.aggregated_results if 'error' not in item)
        error_items = total_items - successful_items
        
        throughput = total_items / duration if duration > 0 else 0
        
        return {
            'total_items': total_items,
            'successful_items': successful_items,
            'error_items': error_items,
            'success_rate': successful_items / total_items if total_items > 0 else 0,
            'duration_seconds': duration,
            'throughput_items_per_second': throughput,
            'start_time': self.start_time,
            'end_time': end_time
        }
    
    def save_aggregated_results(self, output_path: str) -> bool:
        """Save aggregated results to file"""
        try:
            config = WriteConfig(**self.config.get('writer', {}))
            file_writer = FileWriter(config)
            return file_writer.write_data(self.aggregated_results, output_path)
        except Exception as e:
            logger.error(f"Error saving aggregated results: {e}", exc_info=True)
            return False


class SyncResultWriter:
    """同步结果写入器 - 与AsyncResultWriter功能对齐的同步版本"""
    
    def __init__(self, 
                 config: WriteConfig,
                 storage_manager: Optional[MediaStorageManager] = None):
        """初始化同步结果写入器
        
        Args:
            config: 写入配置
            storage_manager: 媒体存储管理器（支持OSS等云存储）
        """
        self.config = config
        self.storage_manager = storage_manager
        self.file_writer = FileWriter(config)  # 复用现有的FileWriter
        self.executor = ThreadPoolExecutor(max_workers=4)  # 用于文件I/O操作
        
        # Statistics
        self.stats = {
            'items_written': 0,
            'files_written': 0,
            'errors': 0,
            'last_flush': time.time()
        }
        
    def write_item(self, item: Dict[str, Any]) -> None:
        """写入单个项目"""
        self.write_batch([item])
    
    def write_batch(self, items: List[Dict[str, Any]]) -> bool:
        """写入批次数据
        
        Args:
            items: 要写入的数据项列表
            
        Returns:
            bool: 写入是否成功
        """
        if not items:
            return True
            
        try:
            # 生成带时间戳的文件名
            timestamp = int(time.time())
            filename = f"results_{timestamp}.{self.config.output_format}"
            
            # 写入临时文件
            temp_file = f"/tmp/{filename}"
            success = self.file_writer.write_data(items, temp_file)
            
            if not success:
                self.stats['errors'] += 1
                logger.error(f"Failed to write batch to {temp_file}")
                return False
            
            # 上传到存储（如果配置了存储管理器）
            upload_success = True
            if self.storage_manager and self.config.async_upload:
                upload_success = self._upload_to_storage(temp_file, filename)
                if upload_success:
                    # 上传成功后删除临时文件
                    Path(temp_file).unlink(missing_ok=True)
                    logger.debug(f"Uploaded and cleaned up {temp_file}")
                else:
                    self.stats['errors'] += 1
                    logger.warning(f"Upload failed, keeping local file: {temp_file}")
            
            # 只有在写入和上传都成功时才更新统计
            if upload_success:
                self.stats['items_written'] += len(items)
                self.stats['files_written'] += 1
                self.stats['last_flush'] = time.time()
            
            logger.info(f"Wrote {len(items)} items to {filename}, upload_success={upload_success}")
            return upload_success
            
        except Exception as e:
            logger.error(f"Error writing batch: {e}", exc_info=True)
            self.stats['errors'] += 1
            return False
    
    def _upload_to_storage(self, local_file: str, remote_filename: str) -> bool:
        """上传文件到存储（同步版本）"""
        if not self.storage_manager:
            return True
            
        try:
            # 上传带重试逻辑
            for attempt in range(self.config.retry_attempts):
                batch_id = Path(remote_filename).stem
                success = self.storage_manager.upload_batch_results(local_file, batch_id)
                
                if success:
                    logger.info(f"Uploaded {remote_filename} to storage")
                    return True
                    
                if attempt < self.config.retry_attempts - 1:
                    delay = self.config.retry_delay * (2 ** attempt)
                    logger.warning(f"Upload failed, retrying in {delay}s... (attempt {attempt + 1}/{self.config.retry_attempts})")
                    time.sleep(delay)
            
            logger.error(f"Upload failed after {self.config.retry_attempts} attempts")
            return False
            
        except Exception as e:
            logger.error(f"Error uploading to storage: {e}", exc_info=True)
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取写入统计"""
        return self.stats.copy()
    
    def flush(self) -> bool:
        """刷新缓冲区（同步版本中不需要缓冲，所以只是返回状态）"""
        return True
    
    def close(self) -> None:
        """关闭写入器，清理资源"""
        self.executor.shutdown(wait=True)
        logger.info("SyncResultWriter closed")


class SyncResultWriterStage:
    """同步结果写入阶段 - 与ResultWriterStage功能对齐的同步版本"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        write_config = WriteConfig(**config.get('writer', {}))
        
        storage_manager = None
        if 'storage' in config:
            storage_manager = MediaStorageManager(config['storage'])
        
        self.sync_writer = SyncResultWriter(write_config, storage_manager)
        self.logger = logging.getLogger("SyncResultWriterStage")
    
    def process(self, batch: BatchData) -> BatchData:
        """同步处理批次并写入结果"""
        try:
            # 提取结果并转换为字典
            results = []
            for item in batch.items:
                if hasattr(item, 'file_id'):
                    result_dict = {
                        'file_id': item.file_id,
                        'transcription': getattr(item, 'transcription', ''),
                        'segments': getattr(item, 'segments', []),
                        #'stats': getattr(item, 'stats', {}),
                        #'metadata': getattr(item, 'metadata', {})
                    }
                    
                    # 在写入前进行安全序列化预处理
                    result_dict = safe_serialize(result_dict)
                    
                    results.append(result_dict)

            # 同步批量写入
            if results:
                success = self.sync_writer.write_batch(results)
                if success:
                    self.logger.debug(f"Wrote {len(results)} results from batch {batch.batch_id}")
                else:
                    self.logger.error(f"Failed to write results from batch {batch.batch_id}")
            else:
                self.logger.warning(f"No valid results in batch {batch.batch_id}")
                
            # 更新batch元数据
            batch.metadata['stage'] = 'sync_result_writer'
            batch.metadata['results_written'] = len(results)
            batch.metadata['write_completed'] = True
            
            return batch
            
        except Exception as e:
            import traceback
            self.logger.error(f"Error writing batch {batch.batch_id}: {e}")
            self.logger.error(f"Traceback:\n{traceback.format_exc()}")
            batch.metadata['error'] = str(e)
            batch.metadata['write_completed'] = False
            return batch
    
    def cleanup(self):
        """清理资源"""
        try:
            self.sync_writer.close()
            self.logger.info("SyncResultWriter closed")
        except Exception as e:
            self.logger.error(f"Error closing sync writer: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取写入统计"""
        return self.sync_writer.get_stats()
    


@dataclass
class OptimizedWriteConfig:
    """优化写入配置"""
    batch_size: int = 1000
    buffer_size_mb: float = 10.0
    max_buffer_size_mb: float = 50.0
    min_buffer_size_mb: float = 5.0
    multipart_threshold_mb: int = 100
    part_size_mb: int = 10
    max_concurrent_parts: int = 4
    output_format: str = 'jsonl'
    retry_attempts: int = 3
    retry_delay: float = 1.0

class OptimizedResultWriter:
    """优化的结果写入器 - 内存直传 + 批量聚合 + 分片并发"""
    
    def __init__(self, 
                 config: OptimizedWriteConfig,
                 storage_manager: Optional[MediaStorageManager] = None,
                 worker_id: str = "default"):
        self.config = config
        self.storage_manager = storage_manager
        self.worker_id = worker_id
        
        self.buffer: List[Dict[str, Any]] = []
        self.buffer_size_bytes = 0
        self.current_buffer_limit = int(config.buffer_size_mb * 1024 * 1024)
        
        self.upload_speeds: List[float] = []
        self.last_adjust_time = time.time()
        
        self.executor = ThreadPoolExecutor(max_workers=config.max_concurrent_parts)
        self.lock = threading.Lock()
        
        self.stats = {
            'items_written': 0,
            'bytes_uploaded': 0,
            'files_uploaded': 0,
            'multipart_uploads': 0,
            'avg_upload_speed_mbps': 0.0
        }
        
    def write_batch(self, items: List[Dict[str, Any]]) -> bool:
        """批量写入"""
        with self.lock:
            for item in items:
                serialized = self._serialize_item(item)
                item_size = len(serialized.encode('utf-8'))
                
                self.buffer.append(item)
                self.buffer_size_bytes += item_size
                
                if self.buffer_size_bytes >= self.current_buffer_limit:
                    self._flush_buffer()
            
            return True
    
    def _serialize_item(self, item: Dict[str, Any]) -> str:
        """序列化"""
        return json.dumps(safe_serialize(item), ensure_ascii=False, cls=EnhancedJSONEncoder) + '\n'
    
    def _flush_buffer(self) -> bool:
        """刷新buffer"""
        if not self.buffer:
            return True
            
        try:
            content = io.BytesIO()
            for item in self.buffer:
                line = self._serialize_item(item)
                content.write(line.encode('utf-8'))
            
            content_size = content.tell()
            content.seek(0)
            
            timestamp = int(time.time() * 1000)
            filename = f"results_{self.worker_id}_{timestamp}.{self.config.output_format}"
            
            start_time = time.time()
            success = self._upload_content(content, filename, content_size)
            upload_time = time.time() - start_time
            
            if success:
                self.stats['items_written'] += len(self.buffer)
                self.stats['bytes_uploaded'] += content_size
                self.stats['files_uploaded'] += 1
                
                speed_mbps = (content_size / 1024 / 1024) / upload_time if upload_time > 0 else 0
                self._adjust_buffer_size(speed_mbps)
                
                self.buffer.clear()
                self.buffer_size_bytes = 0
                
                logger.info(f"Uploaded {filename}: {content_size/1024/1024:.2f}MB in {upload_time:.2f}s ({speed_mbps:.2f}MB/s)")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Flush failed: {e}", exc_info=True)
            return False
    
    def _upload_content(self, content: io.BytesIO, filename: str, size: int) -> bool:
        """上传内容"""
        if not self.storage_manager:
            return True
        
        threshold = self.config.multipart_threshold_mb * 1024 * 1024
        
        if size > threshold:
            return self._multipart_upload(content, filename, size)
        else:
            return self._simple_upload(content, filename)
    
    def _simple_upload(self, content: io.BytesIO, filename: str) -> bool:
        """普通上传"""
        try:
            oss_client = self.storage_manager.oss_client
            key = f"{self.storage_manager.result_prefix}{filename}"
            
            for attempt in range(self.config.retry_attempts):
                try:
                    content.seek(0)
                    oss_client.bucket.put_object(key, content)
                    return True
                except Exception as e:
                    if attempt < self.config.retry_attempts - 1:
                        time.sleep(self.config.retry_delay * (2 ** attempt))
                    else:
                        raise
            
        except Exception as e:
            logger.error(f"Simple upload failed: {e}")
            return False
    
    def _multipart_upload(self, content: io.BytesIO, filename: str, total_size: int) -> bool:
        """分片上传"""
        try:
            oss_client = self.storage_manager.oss_client
            key = f"{self.storage_manager.result_prefix}{filename}"
            
            upload_id = oss_client.bucket.init_multipart_upload(key).upload_id
            self.stats['multipart_uploads'] += 1
            
            part_size = self.config.part_size_mb * 1024 * 1024
            parts = []
            
            content.seek(0)
            part_number = 1
            futures = []
            
            while True:
                chunk = content.read(part_size)
                if not chunk:
                    break
                
                future = self.executor.submit(
                    self._upload_part,
                    oss_client.bucket,
                    key,
                    upload_id,
                    part_number,
                    chunk
                )
                futures.append((part_number, future))
                part_number += 1
            
            for part_num, future in futures:
                result = future.result()
                if result:
                    parts.append(oss2.models.PartInfo(part_num, result))
                else:
                    oss_client.bucket.abort_multipart_upload(key, upload_id)
                    return False
            
            oss_client.bucket.complete_multipart_upload(key, upload_id, parts)
            logger.info(f"Multipart upload: {len(parts)} parts")
            return True
            
        except Exception as e:
            logger.error(f"Multipart upload failed: {e}", exc_info=True)
            return False
    
    def _upload_part(self, bucket, key: str, upload_id: str, part_number: int, data: bytes) -> Optional[str]:
        """上传分片"""
        try:
            result = bucket.upload_part(key, upload_id, part_number, data)
            return result.etag
        except Exception as e:
            logger.error(f"Part {part_number} failed: {e}")
            return None
    
    def _adjust_buffer_size(self, speed_mbps: float) -> None:
        """动态调整buffer"""
        self.upload_speeds.append(speed_mbps)
        if len(self.upload_speeds) > 10:
            self.upload_speeds.pop(0)
        
        if time.time() - self.last_adjust_time < 30:
            return
        
        avg_speed = sum(self.upload_speeds) / len(self.upload_speeds)
        self.stats['avg_upload_speed_mbps'] = avg_speed
        
        if avg_speed > 20:
            new_limit = min(
                self.current_buffer_limit * 1.5,
                self.config.max_buffer_size_mb * 1024 * 1024
            )
        elif avg_speed < 5:
            new_limit = max(
                self.current_buffer_limit * 0.7,
                self.config.min_buffer_size_mb * 1024 * 1024
            )
        else:
            new_limit = self.current_buffer_limit
        
        if new_limit != self.current_buffer_limit:
            logger.info(f"Buffer: {self.current_buffer_limit/1024/1024:.1f}MB → {new_limit/1024/1024:.1f}MB (speed: {avg_speed:.1f}MB/s)")
            self.current_buffer_limit = int(new_limit)
        
        self.last_adjust_time = time.time()
    
    def flush(self) -> bool:
        """手动刷新"""
        with self.lock:
            return self._flush_buffer()
    
    def close(self) -> None:
        """关闭"""
        self.flush()
        self.executor.shutdown(wait=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """统计"""
        return self.stats.copy()


class OptimizedResultWriterStage:
    """优化的结果写入Stage"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        writer_config = config.get('writer', {})
        
        # 构建优化配置
        opt_config = OptimizedWriteConfig(
            batch_size=writer_config.get('batch_size', 1000),
            buffer_size_mb=writer_config.get('buffer_size_mb', 10.0),
            max_buffer_size_mb=writer_config.get('max_buffer_size_mb', 50.0),
            min_buffer_size_mb=writer_config.get('min_buffer_size_mb', 5.0),
            multipart_threshold_mb=writer_config.get('multipart_threshold_mb', 100),
            part_size_mb=writer_config.get('part_size_mb', 10),
            max_concurrent_parts=writer_config.get('max_concurrent_parts', 4),
            output_format=writer_config.get('output_format', 'jsonl'),
            retry_attempts=writer_config.get('retry_attempts', 3),
            retry_delay=writer_config.get('retry_delay', 1.0)
        )
        
        storage_manager = None
        if 'storage' in config:
            storage_manager = MediaStorageManager(config['storage'])
        
        worker_id = str(uuid.uuid4())[:8]
        
        self.writer = OptimizedResultWriter(
            opt_config,
            storage_manager,
            worker_id=worker_id
        )
        
        self.logger = logging.getLogger(f"OptimizedResultWriterStage-{worker_id}")
    
    def process(self, batch: BatchData) -> BatchData:
        """处理batch"""
        try:
            results = []
            for item in batch.items:
                if hasattr(item, 'file_id'):
                    result_dict = {
                        'file_id': item.file_id,
                        'transcription': getattr(item, 'transcription', ''),
                        'segments': getattr(item, 'segments', [])
                    }
                    results.append(result_dict)
            
            if results:
                self.writer.write_batch(results)
                self.logger.debug(f"Wrote {len(results)} results")
            
            batch.metadata['stage'] = 'optimized_result_writer'
            batch.metadata['results_written'] = len(results)
            return batch
            
        except Exception as e:
            self.logger.error(f"Process failed: {e}", exc_info=True)
            batch.metadata['error'] = str(e)
            return batch
    
    def cleanup(self):
        """清理"""
        try:
            self.writer.close()
            self.logger.info("Writer closed")
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """统计"""
        return self.writer.get_stats()