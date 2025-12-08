"""Ray-based distributed pipeline scheduling"""

import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

import ray
from ray import data
from ray.util.queue import Queue
from loguru import logger


@dataclass
class PipelineConfig:
    """Pipeline configuration"""
    num_cpu_workers: int = 10
    num_gpu_workers: int = 1
    cpu_worker_resources: Dict[str, float] = None
    gpu_worker_resources: Dict[str, float] = None
    batch_size: int = 32
    max_concurrent_batches: int = 4
    object_store_memory: int = 1024 * 1024 * 1024  # 1GB
    checkpoint_interval: int = 1000


@dataclass
class DataBatch:
    """Data batch for pipeline processing"""
    batch_id: str
    items: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class PipelineStage(ABC):
    """Abstract base class for pipeline stages"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    @abstractmethod
    def process(self, batch: DataBatch) -> DataBatch:
        """Process a batch of data"""
        pass


@ray.remote
class DataProducer:
    """Data producer stage"""
    
    def __init__(self, 
                 data_loader_config: Dict[str, Any],
                 batch_size: int = 32):
        from src.data.audio_indexer import DataLoader
        from src.data.storage import AudioStorageManager
        
        self.data_loader = DataLoader(data_loader_config)
        self.storage_manager = AudioStorageManager(data_loader_config['storage'])
        self.batch_size = batch_size
        self.processed_files = set()
        
    def load_index(self) -> List[Dict[str, Any]]:
        """Load audio index"""
        df = self.data_loader.load_index()
        if df.empty:
            # Build index from storage if not exists
            audio_files = self.storage_manager.list_audio_files()
            df = self.data_loader.build_index(audio_files)
        return df.to_dict('records')
    
    def produce_batches(self, 
                       max_batches: Optional[int] = None) -> List[DataBatch]:
        """Produce data batches"""
        audio_records = self.load_index()
        
        # Filter already processed files
        remaining_records = [
            record for record in audio_records 
            if record['file_id'] not in self.processed_files
        ]
        
        batches = []
        for i in range(0, len(remaining_records), self.batch_size):
            batch_records = remaining_records[i:i + self.batch_size]
            batch = DataBatch(
                batch_id=f"batch_{i // self.batch_size}",
                items=batch_records,
                metadata={'stage': 'producer'}
            )
            batches.append(batch)
            
            if max_batches and len(batches) >= max_batches:
                break
                
        logger.info(f"Produced {len(batches)} batches")
        return batches
    
    def mark_processed(self, file_ids: List[str]) -> None:
        """Mark files as processed"""
        self.processed_files.update(file_ids)


@ray.remote
class PipelineWorker:
    """Generic pipeline worker"""
    
    def __init__(self, 
                 worker_id: str,
                 stage_class: type,
                 stage_config: Dict[str, Any]):
        self.worker_id = worker_id
        self.stage = stage_class(stage_config)
        
    def process_batch(self, batch: DataBatch) -> DataBatch:
        """Process a single batch"""
        try:
            result = self.stage.process(batch)
            result.metadata['worker_id'] = self.worker_id
            result.metadata['processed_at'] = time.time()
            return result
        except Exception as e:
            logger.error(f"Worker {self.worker_id} failed to process batch {batch.batch_id}: {e}")
            batch.metadata['error'] = str(e)
            batch.metadata['failed_worker'] = self.worker_id
            return batch


class DistributedPipeline:
    """Distributed pipeline using Ray"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.producer = None
        self.cpu_workers = []
        self.gpu_workers = []
        self.result_queue = Queue()
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(
                object_store_memory=config.object_store_memory,
                ignore_reinit_error=True
            )
    
    def setup_producer(self, data_loader_config: Dict[str, Any]) -> None:
        """Setup data producer"""
        self.producer = DataProducer.remote(
            data_loader_config,
            self.config.batch_size
        )
        logger.info("Data producer setup complete")
    
    def setup_cpu_workers(self, 
                         stage_class: type,
                         stage_config: Dict[str, Any]) -> None:
        """Setup CPU workers"""
        resources = self.config.cpu_worker_resources or {"CPU": 1}
        
        for i in range(self.config.num_cpu_workers):
            worker = PipelineWorker.options(**resources).remote(
                f"cpu_worker_{i}",
                stage_class,
                stage_config
            )
            self.cpu_workers.append(worker)
            
        logger.info(f"Setup {len(self.cpu_workers)} CPU workers")
    
    def setup_gpu_workers(self,
                         stage_class: type, 
                         stage_config: Dict[str, Any]) -> None:
        """Setup GPU workers"""
        resources = self.config.gpu_worker_resources or {"CPU": 1, "GPU": 1}
        
        for i in range(self.config.num_gpu_workers):
            worker = PipelineWorker.options(**resources).remote(
                f"gpu_worker_{i}",
                stage_class,
                stage_config
            )
            self.gpu_workers.append(worker)
            
        logger.info(f"Setup {len(self.gpu_workers)} GPU workers")
    
    def run_pipeline(self, 
                    max_batches: Optional[int] = None,
                    progress_callback: Optional[Callable] = None) -> List[DataBatch]:
        """Run the distributed pipeline"""
        if not self.producer:
            raise ValueError("Producer not setup. Call setup_producer() first.")
            
        if not self.cpu_workers and not self.gpu_workers:
            raise ValueError("No workers setup. Call setup_cpu_workers() or setup_gpu_workers() first.")
        
        # Produce batches
        logger.info("Starting data production...")
        batch_futures = self.producer.produce_batches.remote(max_batches)
        batches = ray.get(batch_futures)
        
        if not batches:
            logger.warning("No batches to process")
            return []
        
        # Process batches through pipeline stages
        results = []
        available_workers = self.cpu_workers + self.gpu_workers
        
        for i, batch in enumerate(batches):
            # Select worker (round-robin)
            worker = available_workers[i % len(available_workers)]
            
            # Submit batch for processing
            result_future = worker.process_batch.remote(batch)
            results.append(result_future)
            
            # Progress callback
            if progress_callback:
                progress_callback(i + 1, len(batches))
        
        # Wait for all results
        logger.info("Waiting for batch processing to complete...")
        processed_batches = ray.get(results)
        
        # Mark processed files
        processed_file_ids = []
        for batch in processed_batches:
            if 'error' not in batch.metadata:
                processed_file_ids.extend([item['file_id'] for item in batch.items])
        
        if processed_file_ids:
            self.producer.mark_processed.remote(processed_file_ids)
        
        logger.info(f"Pipeline completed. Processed {len(processed_batches)} batches")
        return processed_batches
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        stats = {
            'num_cpu_workers': len(self.cpu_workers),
            'num_gpu_workers': len(self.gpu_workers),
            'ray_cluster_resources': ray.cluster_resources(),
            'available_resources': ray.available_resources()
        }
        return stats
    
    def shutdown(self) -> None:
        """Shutdown the pipeline"""
        if ray.is_initialized():
            ray.shutdown()
        logger.info("Pipeline shutdown complete")


class PipelineOrchestrator:
    """High-level pipeline orchestrator"""
    
    def __init__(self, config: Dict[str, Any]):
        self.pipeline_config = PipelineConfig(**config.get('pipeline', {}))
        self.pipeline = DistributedPipeline(self.pipeline_config)
        self.data_config = config['data']
        
    def setup_pipeline(self, 
                      cpu_stage_class: type,
                      cpu_stage_config: Dict[str, Any],
                      gpu_stage_class: type,
                      gpu_stage_config: Dict[str, Any]) -> None:
        """Setup complete pipeline"""
        # Setup producer
        self.pipeline.setup_producer(self.data_config)
        
        # Setup CPU workers (audio preprocessing)
        self.pipeline.setup_cpu_workers(cpu_stage_class, cpu_stage_config)
        
        # Setup GPU workers (inference)
        self.pipeline.setup_gpu_workers(gpu_stage_class, gpu_stage_config)
        
        logger.info("Pipeline setup complete")
    
    def run(self, 
            max_batches: Optional[int] = None,
            progress_callback: Optional[Callable] = None) -> List[DataBatch]:
        """Run the complete pipeline"""
        return self.pipeline.run_pipeline(max_batches, progress_callback)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return self.pipeline.get_pipeline_stats()
    
    def cleanup(self) -> None:
        """Cleanup pipeline resources"""
        self.pipeline.shutdown()