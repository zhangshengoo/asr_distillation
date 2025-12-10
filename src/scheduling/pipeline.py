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
        from src.data.media_indexer import DataLoader
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
            df = self.data_loader.create_index(audio_files)
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
    """Distributed pipeline using Ray with multi-stage support"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.producer = None
        self.stage_workers = {}  # Dict: stage_name -> List[workers]
        self.stage_queues = {}   # Dict: stage_name -> Queue
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
                         stage_config: Dict[str, Any],
                         num_workers: Optional[int] = None,
                         stage_name: str = "cpu_stage") -> None:
        """Setup CPU workers for specific stage"""
        if num_workers is None:
            num_workers = self.config.num_cpu_workers
            
        resources = self.config.cpu_worker_resources or {"num_cpus": 1}
        workers = []
        
        for i in range(num_workers):
            worker = PipelineWorker.options(**resources).remote(
                f"{stage_name}_worker_{i}",
                stage_class,
                stage_config
            )
            workers.append(worker)
            
        self.stage_workers[stage_name] = workers
        self.stage_queues[stage_name] = Queue()
        logger.info(f"Setup {len(workers)} CPU workers for stage {stage_name}")
    
    def setup_gpu_workers(self,
                         stage_class: type, 
                         stage_config: Dict[str, Any],
                         num_workers: Optional[int] = None,
                         stage_name: str = "gpu_stage") -> None:
        """Setup GPU workers for specific stage"""
        if num_workers is None:
            num_workers = self.config.num_gpu_workers
            
        resources = self.config.gpu_worker_resources or {"num_cpus": 1, "num_gpus": 1}
        workers = []
        
        for i in range(num_workers):
            worker = PipelineWorker.options(**resources).remote(
                f"{stage_name}_worker_{i}",
                stage_class,
                stage_config
            )
            workers.append(worker)
            
        self.stage_workers[stage_name] = workers
        self.stage_queues[stage_name] = Queue()
        logger.info(f"Setup {len(workers)} GPU workers for stage {stage_name}")
    
    def run_pipeline(self, 
                    max_batches: Optional[int] = None,
                    progress_callback: Optional[Callable] = None) -> List[DataBatch]:
        """Run the distributed pipeline (legacy method)"""
        if not self.producer:
            raise ValueError("Producer not setup. Call setup_producer() first.")
            
        if not self.stage_workers:
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
        all_workers = []
        for workers in self.stage_workers.values():
            all_workers.extend(workers)
        
        for i, batch in enumerate(batches):
            # Select worker (round-robin)
            worker = all_workers[i % len(all_workers)]
            
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
    
    def run_multi_stage_pipeline(self, 
                                stages_config: List[Dict[str, Any]],
                                max_batches: Optional[int] = None,
                                progress_callback: Optional[Callable] = None) -> List[DataBatch]:
        """Run multi-stage pipeline with linear data flow"""
        if not self.producer:
            raise ValueError("Producer not setup. Call setup_producer() first.")
            
        if not self.stage_workers:
            raise ValueError("No workers setup. Call setup_cpu_workers() or setup_gpu_workers() first.")
        
        if len(stages_config) < 1:
            raise ValueError("At least one stage must be configured.")
        
        logger.info(f"Starting multi-stage pipeline with {len(stages_config)} stages")
        
        # Produce batches
        batch_futures = self.producer.produce_batches.remote(max_batches)
        batches = ray.get(batch_futures)
        
        if not batches:
            logger.warning("No batches to process")
            return []
        
        # Process batches through all stages sequentially
        current_batches = batches
        stage_names = [stage['name'] for stage in stages_config]
        
        for stage_idx, stage_config in enumerate(stages_config):
            stage_name = stage_config['name']
            workers = self.stage_workers.get(stage_name, [])
            
            if not workers:
                logger.error(f"No workers found for stage {stage_name}")
                continue
            
            logger.info(f"Processing stage {stage_idx + 1}/{len(stages_config)}: {stage_name}")
            stage_start_time = time.time()
            
            # Process current batches through this stage
            stage_results = []
            for batch_idx, batch in enumerate(current_batches):
                # Select worker (round-robin within stage)
                worker = workers[batch_idx % len(workers)]
                
                # Submit batch for processing
                result_future = worker.process_batch.remote(batch)
                stage_results.append(result_future)
            
            # Wait for all batches in this stage to complete
            current_batches = ray.get(stage_results)
            
            stage_duration = time.time() - stage_start_time
            logger.info(f"Stage {stage_name} completed in {stage_duration:.2f}s")
            
            # Update progress callback
            if progress_callback:
                overall_progress = (stage_idx + 1) / len(stages_config) * 100
                progress_callback(int(overall_progress), 100)
        
        # Mark processed files
        processed_file_ids = []
        for batch in current_batches:
            if 'error' not in batch.metadata:
                processed_file_ids.extend([item['file_id'] for item in batch.items])
        
        if processed_file_ids:
            self.producer.mark_processed.remote(processed_file_ids)
        
        logger.info(f"Multi-stage pipeline completed. Processed {len(current_batches)} batches")
        return current_batches
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        total_cpu_workers = sum(len(workers) for name, workers in self.stage_workers.items() 
                               if 'cpu' in name.lower())
        total_gpu_workers = sum(len(workers) for name, workers in self.stage_workers.items() 
                               if 'gpu' in name.lower())
        
        stats = {
            'stages': list(self.stage_workers.keys()),
            'num_cpu_workers': total_cpu_workers,
            'num_gpu_workers': total_gpu_workers,
            'total_workers': sum(len(workers) for workers in self.stage_workers.values()),
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
    """High-level pipeline orchestrator with multi-stage support"""
    
    def __init__(self, config: Dict[str, Any]):
        self.pipeline_config = PipelineConfig(**config.get('pipeline', {}))
        self.pipeline = DistributedPipeline(self.pipeline_config)
        self.data_config = config['data']
        self.stages_config = []
        
    def setup_pipeline(self, 
                      cpu_stage_class: type,
                      cpu_stage_config: Dict[str, Any],
                      gpu_stage_class: type,
                      gpu_stage_config: Dict[str, Any]) -> None:
        """Setup complete pipeline (legacy method for backward compatibility)"""
        self.stages_config = [
            {
                'type': 'cpu',
                'class': cpu_stage_class,
                'config': cpu_stage_config,
                'name': 'audio_preprocessing'
            },
            {
                'type': 'gpu',
                'class': gpu_stage_class,
                'config': gpu_stage_config,
                'name': 'inference'
            }
        ]
        self._setup_stages()
        
    def setup_multi_stage_pipeline(self, stages_config: List[Dict[str, Any]]) -> None:
        """Setup multi-stage pipeline with detailed stage configuration
        
        Args:
            stages_config: List of stage configurations, each containing:
                - type: 'cpu' or 'gpu'
                - class: Stage class to instantiate
                - config: Configuration dictionary for the stage
                - name: Stage name for logging
                - num_workers: Number of workers for this stage (optional)
        """
        self.stages_config = stages_config
        self._setup_stages()
        
    def _setup_stages(self) -> None:
        """Setup all pipeline stages"""
        # Setup producer
        self.pipeline.setup_producer(self.data_config)
        
        # Setup each stage
        for stage_config in self.stages_config:
            stage_type = stage_config['type']
            stage_class = stage_config['class']
            stage_params = stage_config['config']
            stage_name = stage_config.get('name', f"{stage_type}_stage")
            
            # Determine number of workers for this stage
            if 'num_workers' in stage_config:
                num_workers = stage_config['num_workers']
            else:
                num_workers = (self.pipeline_config.num_cpu_workers 
                              if stage_type == 'cpu' 
                              else self.pipeline_config.num_gpu_workers)
            
            logger.info(f"Setting up {stage_name} with {num_workers} {stage_type} workers")
            
            if stage_type == 'cpu':
                self.pipeline.setup_cpu_workers(
                    stage_class, stage_params, num_workers=num_workers, stage_name=stage_name
                )
            elif stage_type == 'gpu':
                self.pipeline.setup_gpu_workers(
                    stage_class, stage_params, num_workers=num_workers, stage_name=stage_name
                )
            else:
                raise ValueError(f"Unknown stage type: {stage_type}")
        
        logger.info(f"Multi-stage pipeline setup complete with {len(self.stages_config)} stages")
    
    def run(self, 
            max_batches: Optional[int] = None,
            progress_callback: Optional[Callable] = None) -> List[DataBatch]:
        """Run the complete pipeline"""
        return self.pipeline.run_multi_stage_pipeline(
            self.stages_config, max_batches, progress_callback
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        stats = self.pipeline.get_pipeline_stats()
        stats['stages'] = [
            {
                'name': stage.get('name', f"{stage['type']}_stage"),
                'type': stage['type'],
                'class': stage['class'].__name__
            }
            for stage in self.stages_config
        ]
        return stats
    
    def cleanup(self) -> None:
        """Cleanup pipeline resources"""
        self.pipeline.shutdown()