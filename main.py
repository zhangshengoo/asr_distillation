"""Main entry point for ASR Distillation Framework"""

import asyncio
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import typer
import ray

from src.config.manager import ConfigManager, create_sample_config
from src.scheduling.pipeline import PipelineOrchestrator
from src.compute.audio_processor import (
    AudioDownloadStage, 
    AudioPreprocessingStage,
    AudioFeatureStage
)
from src.compute.vad_stage import VADProcessingStage
from src.compute.segment_processor import (
    SegmentExpansionStage,
    SegmentAggregationStage
)
from src.compute.inference import (
    AudioInferenceStage,
    BatchInferenceStage,
    PostProcessingStage
)
from src.storage.result_writer import ResultWriterStage
from src.monitoring.system import MonitoringSystem
from src.config.manager import MonitoringConfig


# Global variables for graceful shutdown
shutdown_event = asyncio.Event()
pipeline_orchestrator = None
monitoring_system = None


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    shutdown_event.set()


async def run_pipeline(config_path: str, 
                      max_batches: Optional[int] = None,
                      log_level: str = "INFO") -> None:
    """Run the ASR distillation pipeline"""
    global pipeline_orchestrator, monitoring_system
    
    try:
        # Load configuration
        config_manager = ConfigManager(config_path)
        config = config_manager.load_config()
        
        # Validate configuration
        if not config_manager.validate_config(config):
            return
        
        # Setup Ray first (before monitoring system to avoid conflicts)
        if not ray.is_initialized():
            ray.init(
                object_store_memory=config.pipeline.object_store_memory,
                ignore_reinit_error=True
            )
        
        # Initialize monitoring system after Ray is ready
        monitoring_config = MonitoringConfig(**config.monitoring.__dict__)
        monitoring_system = MonitoringSystem(monitoring_config)
        monitoring_system.start()
        
        # Initialize pipeline orchestrator
        pipeline_config = {
            'pipeline': config.pipeline.__dict__,
            'data': config.data.__dict__
        }
        pipeline_orchestrator = PipelineOrchestrator(pipeline_config)
        
        # Setup pipeline stages
        cpu_stage_config = {
            'data': config.data.__dict__,
            'audio': config.audio.__dict__,
            'input_storage': config.data.input_storage,
            'output_storage': config.data.output_storage
        }
        
        gpu_stage_config = {
            'inference': config.inference.__dict__,
            'writer': config.writer.__dict__,
            'input_storage': config.data.input_storage,
            'output_storage': config.data.output_storage
        }
        
        # Setup multi-stage pipeline with all stages
        # Enable multimedia processing if configured
        enable_multimedia = config.media is not None
        
        # 获取stage_workers配置
        stage_workers_config = config.pipeline.stage_workers
        
        stages_config = [
            {
                'type': 'cpu',
                'class': AudioDownloadStage,
                'name': 'audio_download',
                'num_workers': stage_workers_config.get('audio_download', 8),  # IO密集型
                'config': {
                    **cpu_stage_config,
                    'enable_multimedia': enable_multimedia,
                    'media': config.media.__dict__ if enable_multimedia else {}
                }
            },
            {
                'type': 'cpu',
                'class': AudioPreprocessingStage,
                'name': 'audio_preprocessing',
                'num_workers': stage_workers_config.get('audio_preprocessing', 6),  # CPU密集型
                'config': cpu_stage_config
            },
            {
                'type': 'cpu',
                'class': VADProcessingStage,
                'name': 'vad_processing',
                'num_workers': stage_workers_config.get('vad_processing', 4),  # 并行处理
                'config': {
                    'vad': config.vad.__dict__,
                    'batch_size': config.pipeline.batch_size
                }
            },
            {
                'type': 'cpu',
                'class': SegmentExpansionStage,
                'name': 'segment_expansion',
                'num_workers': stage_workers_config.get('segment_expansion', 4),  # 中等CPU需求
                'config': {
                    'min_segment_duration': config.segment_expansion.min_segment_duration,
                    'max_segment_duration': config.segment_expansion.max_segment_duration,
                    'segment_threshold': config.segment_expansion.segment_threshold,
                    'preserve_order': config.segment_expansion.preserve_order,
                    'sampling_rate': config.segment_expansion.sampling_rate,
                    'resplit_min_speech_ms': config.segment_expansion.resplit_min_speech_ms,
                    'resplit_min_silence_ms': config.segment_expansion.resplit_min_silence_ms,
                    'resplit_threshold': config.segment_expansion.resplit_threshold,
                    'resplit_neg_threshold': config.segment_expansion.resplit_neg_threshold,
                    'resplit_speech_pad_ms': config.segment_expansion.resplit_speech_pad_ms,
                    'segment_upload': config.segment_upload.__dict__ if hasattr(config, 'segment_upload') else {},
                    'input_storage': config.data.input_storage,
                    'output_storage': config.data.output_storage
                }
            },
            {
                'type': 'cpu',
                'class': AudioFeatureStage,
                'name': 'feature_extraction',
                'num_workers': stage_workers_config.get('feature_extraction', 6),  # 中等CPU需求
                'config': cpu_stage_config
            },
            {
                'type': 'gpu',
                'class': BatchInferenceStage,  # 使用批处理优化
                'name': 'batch_inference',
                'num_workers': stage_workers_config.get('batch_inference', 1),  # GPU密集型
                'config': gpu_stage_config
            },
            {
                'type': 'cpu',  # 使用CPU资源即可，聚合是轻量级处理
                'class': SegmentAggregationStage,
                'name': 'segment_aggregation',
                'num_workers': stage_workers_config.get('segment_aggregation', 2),  # 轻量级处理
                'config': {
                    'sort_by_timestamp': True,
                    'include_segment_details': True,
                    'calculate_file_stats': True
                }
            },
            {
                'type': 'cpu',  # 使用CPU资源即可，后处理是轻量级处理
                'class': PostProcessingStage,
                'name': 'post_processing',
                'num_workers': stage_workers_config.get('post_processing', 2),  # 轻量级处理
                'config': gpu_stage_config
            }
        ]
        
        pipeline_orchestrator.setup_multi_stage_pipeline(stages_config)
        
        # Setup result writer
        result_writer = ResultWriterStage({
            'writer': config.writer.__dict__,
            'input_storage': config.data.input_storage,
            'output_storage': config.data.output_storage
        })
        await result_writer.start()
        
        # Progress callback
        def progress_callback(current: int, total: int):
            progress = (current / total) * 100
        
        # Run pipeline
        start_time = time.time()
        
        results = await asyncio.get_event_loop().run_in_executor(
            None,
            pipeline_orchestrator.run,
            max_batches,
            progress_callback
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Process results through writer
        for batch in results:
            await result_writer.process_batch(batch)
        
        # Stop result writer
        await result_writer.stop()
        
        # Print summary
        total_items = sum(len(batch.items) for batch in results)
        successful_items = sum(
            len([item for item in batch.items if 'error' not in item])
            for batch in results
        )
        
        # Export final metrics
        if monitoring_system:
            stats = monitoring_system.get_system_stats()
            monitoring_system.export_metrics("logs/final_metrics.json")
        
    except Exception as e:
        raise
    finally:
        await cleanup()


async def cleanup():
    """Cleanup resources"""
    global pipeline_orchestrator, monitoring_system

    if pipeline_orchestrator:
        pipeline_orchestrator.cleanup()
    
    if monitoring_system:
        monitoring_system.stop()
    
    if ray.is_initialized():
        ray.shutdown()
            


# CLI commands
app = typer.Typer(help="ASR Distillation Framework")


@app.command()
def run(
    config: str = typer.Option(
        "config.yaml",
        "--config", "-c",
        help="Configuration file path"
    ),
    max_batches: Optional[int] = typer.Option(
        None,
        "--max-batches",
        help="Maximum number of batches to process"
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging level"
    )
):
    """Run the ASR distillation pipeline"""
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create necessary directories
    Path("logs").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)
    
    # Run pipeline
    asyncio.run(run_pipeline(config, max_batches, log_level))


@app.command()
def create_config(
    output: str = typer.Option(
        "config.yaml",
        "--output", "-o",
        help="Output configuration file path"
    )
):
    """Create a sample configuration file"""
    
    config_content = create_sample_config()
    
    with open(output, 'w') as f:
        f.write(config_content)
    
    typer.echo(f"Sample configuration created: {output}")
    typer.echo("Please edit the configuration file with your settings before running the pipeline.")




if __name__ == "__main__":
    app()