"""Async result writing and storage management"""

import json
import asyncio
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, AsyncGenerator
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import threading

# ✅ 修改：导入正确的类名
from ..data.storage import MediaStorageManager
from ..common import BatchData, FileResultItem

logger = logging.getLogger(__name__)


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
    """File writing utilities"""
    
    def __init__(self, config: WriteConfig):
        self.config = config
        
    def write_jsonl(self, data: List[Dict[str, Any]], file_path: str) -> bool:
        """Write data as JSONL"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            return True
        except Exception as e:
            logger.error(f"Error writing JSONL: {e}")
            return False
    
    def write_json(self, data: List[Dict[str, Any]], file_path: str) -> bool:
        """Write data as JSON array"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error writing JSON: {e}")
            return False
    
    def write_parquet(self, data: List[Dict[str, Any]], file_path: str) -> bool:
        """Write data as Parquet"""
        try:
            import pandas as pd
            df = pd.DataFrame(data)
            df.to_parquet(file_path, index=False)
            return True
        except Exception as e:
            logger.error(f"Error writing Parquet: {e}")
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
    """Async result writer with buffering
    
    支持多种输出格式和存储后端（本地文件、OSS等）
    """
    
    def __init__(self, 
                 config: WriteConfig,
                 storage_manager: Optional[MediaStorageManager] = None):  # ✅ 修改类型
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
                logger.error(f"Error in writer loop: {e}")
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
            logger.error(f"Error writing batch: {e}")
            self.stats['errors'] += 1
    
    async def _upload_to_storage(self, local_file: str, remote_filename: str) -> bool:
        """Upload file to storage (OSS/云存储)
        
        Args:
            local_file: 本地文件路径
            remote_filename: 远程文件名
            
        Returns:
            上传是否成功
        """
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
            logger.error(f"Error uploading to storage: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get writer statistics"""
        return {
            **self.stats,
            'buffer_size': self.buffer.size(),
            'running': self.running
        }


class ResultWriterStage:
    """Result writer stage for pipeline integration
    
    作为Pipeline的最后一个Stage，负责将处理结果写入文件/OSS
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        write_config = WriteConfig(**config.get('writer', {}))
        
        # ✅ 使用正确的类名
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
        """同步处理接口（供StreamingPipelineWorker调用）
        
        注意：StreamingPipelineWorker会自动检测并使用process_async，
        但保留此方法以防万一
        """
        self._ensure_started()
        
        # 在事件循环中运行异步方法
        self._loop.run_until_complete(self.process_async(batch))
        
        return batch
    
    async def process_async(self, batch: BatchData) -> BatchData:
        """异步处理批次并写入结果
        
        这是主要的处理方法，会被StreamingPipelineWorker的异步机制调用
        """
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
                        'stats': getattr(item, 'stats', {}),
                        'metadata': getattr(item, 'metadata', {})
                    }
                    results.append(result_dict)

            # 异步批量写入
            if results:
                await self.async_writer.write_batch(results)
                self.logger.debug(f"Wrote {len(results)} results from batch {batch.batch_id}")
            else:
                self.logger.warning(f"No valid results in batch {batch.batch_id}")
                
            # 更新batch元数据（传递给下游统计队列）
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
                # 停止异步写入器
                self._loop.run_until_complete(self.async_writer.stop())
                self.logger.info("AsyncResultWriter stopped")
            except Exception as e:
                self.logger.error(f"Error stopping writer: {e}")
            finally:
                # 关闭事件循环
                if self._loop and not self._loop.is_closed():
                    self._loop.close()
                self._loop = None
                self._started = False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取写入统计"""
        return self.async_writer.get_stats()


class BatchResultAggregator:
    """Aggregate results from multiple batches (向后兼容)"""
    
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
        
        # Calculate statistics
        total_items = len(self.aggregated_results)
        successful_items = sum(1 for item in self.aggregated_results if 'error' not in item)
        error_items = total_items - successful_items
        
        # Calculate throughput
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
            logger.error(f"Error saving aggregated results: {e}")
            return False