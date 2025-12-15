# src/storage/result_writer.py 末尾添加优化实现

"""Optimized result writing with memory streaming and batch uploads (sync version)"""

import io
import json
import time
import logging
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import oss2

from ..data.storage import MediaStorageManager
from ..common import BatchData

logger = logging.getLogger(__name__)


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
    """优化的结果写入器 - 内存直传 + 批量聚合（同步版本，避免Ray中的多线程冲突）"""
    
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
        
        # 移除多线程部分，使用同步上传
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
            oss_client = self.storage_manager.get_output_client()
            key = f"{self.storage_manager.output_result_prefix}{filename}"
            
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
        """分片上传 - 同步版本"""
        try:
            oss_client = self.storage_manager.get_output_client()
            key = f"{self.storage_manager.output_result_prefix}{filename}"
            
            upload_id = oss_client.bucket.init_multipart_upload(key).upload_id
            self.stats['multipart_uploads'] += 1
            
            part_size = self.config.part_size_mb * 1024 * 1024
            parts = []
            
            content.seek(0)
            part_number = 1
            
            while True:
                chunk = content.read(part_size)
                if not chunk:
                    break
                
                # 同步上传分片
                result = self._upload_part_sync(
                    oss_client.bucket,
                    key,
                    upload_id,
                    part_number,
                    chunk
                )
                
                if result:
                    parts.append(oss2.models.PartInfo(part_number, result))
                else:
                    oss_client.bucket.abort_multipart_upload(key, upload_id)
                    return False
                
                part_number += 1
            
            oss_client.bucket.complete_multipart_upload(key, upload_id, parts)
            logger.info(f"Multipart upload: {len(parts)} parts")
            return True
            
        except Exception as e:
            logger.error(f"Multipart upload failed: {e}", exc_info=True)
            return False
    
    def _upload_part_sync(self, bucket, key: str, upload_id: str, part_number: int, data: bytes) -> Optional[str]:
        """上传分片 - 同步版本"""
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
        return self._flush_buffer()
    
    def close(self) -> None:
        """关闭"""
        self.flush()
        # 无需关闭线程池，因为现在使用同步上传
    
    def get_stats(self) -> Dict[str, Any]:
        """统计"""
        return self.stats.copy()


class OptimizedResultWriterStage:
    """优化的结果写入Stage - 与流式管道兼容"""
    
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
        if 'output_storage' in config and 'input_storage' in config:
            # 结果写入使用输出存储
            storage_manager = MediaStorageManager(
                input_config=config['input_storage'],
                output_config=config['output_storage']
            )
        
        worker_id = str(uuid.uuid4())[:8]
        
        self.writer = OptimizedResultWriter(
            opt_config,
            storage_manager,
            worker_id=worker_id
        )
        
        self.logger = logging.getLogger(f"OptimizedResultWriterStage-{worker_id}")
        self._closed = False  # 添加关闭标志
    
    def process(self, batch: BatchData) -> BatchData:
        """处理batch"""
        import psutil
        import threading
        current_process = psutil.Process()
        self.logger.info(f"OptimizedResultWriterStage processing batch {batch.batch_id}, Items: {len(batch.items)}, Threads: {current_process.num_threads()}, Active: {threading.active_count()}")
        
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
            
            current_process = psutil.Process()
            self.logger.info(f"OptimizedResultWriterStage completed batch {batch.batch_id}, Threads: {current_process.num_threads()}, Active: {threading.active_count()}")
            return batch
            
        except Exception as e:
            self.logger.error(f"Process failed: {e}", exc_info=True)
            batch.metadata['error'] = str(e)
            return batch
    
    def cleanup(self):
        """清理 - 确保只清理一次"""
        if not self._closed:
            try:
                self.writer.flush()  # 确保所有缓冲数据都写入
                self.writer.close()
                self.logger.info("Writer closed")
                self._closed = True
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """统计"""
        return self.writer.get_stats()