"""Async result writing and storage management"""

import json
import asyncio
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, AsyncGenerator
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import threading

from ..data.storage import AudioStorageManager
from ..scheduling.pipeline import DataBatch


@dataclass
class WriteConfig:
    """Configuration for result writing"""
    batch_size: int = 1000  # Number of results to buffer before writing
    flush_interval: float = 10.0  # Seconds between flushes
    max_file_size_mb: int = 100  # Max file size in MB
    output_format: str = 'jsonl'  # jsonl, parquet, or json
    compression: Optional[str] = None  # gzip, etc.
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
            return False
    
    def write_json(self, data: List[Dict[str, Any]], file_path: str) -> bool:
        """Write data as JSON array"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            return False
    
    def write_parquet(self, data: List[Dict[str, Any]], file_path: str) -> bool:
        """Write data as Parquet"""
        try:
            import pandas as pd
            df = pd.DataFrame(data)
            df.to_parquet(file_path, index=False)
            return True
        except Exception as e:
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
            return False


class AsyncResultWriter:
    """Async result writer with buffering"""
    
    def __init__(self, 
                 config: WriteConfig,
                 storage_manager: Optional[AudioStorageManager] = None):
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
                return
            
            # Upload to storage if configured
            if self.storage_manager and self.config.async_upload:
                upload_success = await self._upload_to_storage(temp_file, filename)
                if upload_success:
                    # Remove temp file after successful upload
                    Path(temp_file).unlink(missing_ok=True)
                else:
                    self.stats['errors'] += 1
            
            # Update statistics
            self.stats['items_written'] += len(batch)
            self.stats['files_written'] += 1
            self.stats['last_flush'] = time.time()
            
        except Exception as e:
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
                    return True
                    
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
            
            return False
            
        except Exception as e:
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
        
        # Setup storage manager if configured
        storage_manager = None
        if 'storage' in config:
            storage_manager = AudioStorageManager(config['storage'])
        
        self.async_writer = AsyncResultWriter(write_config, storage_manager)
        
    async def process_batch(self, batch: DataBatch) -> DataBatch:
        """Process batch and write results"""
        try:
            # Extract results from batch
            results = []
            for item in batch.items:
                if 'error' not in item and 'output' in item:
                    results.append(item['output'])
                elif 'error' in item:
                    # Write error items too
                    error_result = {
                        'file_id': item['file_id'],
                        'error': item['error'],
                        'timestamp': time.time()
                    }
                    results.append(error_result)
            
            # Write results asynchronously
            if results:
                await self.async_writer.write_batch(results)
            
            # Update batch metadata
            batch.metadata['results_written'] = len(results)
            batch.metadata['stage'] = 'result_writer'
            
            return batch
            
        except Exception as e:
            batch.metadata['writer_error'] = str(e)
            return batch
    
    async def start(self) -> None:
        """Start the async writer"""
        await self.async_writer.start()
        
    async def stop(self) -> None:
        """Stop the async writer"""
        await self.async_writer.stop()
        
    def get_stats(self) -> Dict[str, Any]:
        """Get writer statistics"""
        return self.async_writer.get_stats()


class BatchResultAggregator:
    """Aggregate results from multiple batches"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.aggregated_results = []
        self.start_time = time.time()
        
    def add_batch_results(self, batch: DataBatch) -> None:
        """Add results from a batch"""
        for item in batch.items:
            if 'output' in item:
                self.aggregated_results.append(item['output'])
                
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
            return False