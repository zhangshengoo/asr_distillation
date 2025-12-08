"""Main entry point for ASR Distillation Framework"""

import asyncio
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import typer
import ray
from loguru import logger

from src.config.manager import ConfigManager, create_sample_config
from src.scheduling.pipeline import PipelineOrchestrator
from src.compute.audio_processor import (
    AudioDownloadStage, 
    AudioPreprocessingStage,
    AudioFeatureStage
)
from src.compute.inference import (
    AudioInferenceStage,
    BatchInferenceStage,
    PostProcessingStage
)
from src.storage.result_writer import ResultWriterStage
from src.monitoring.system import MonitoringSystem, MonitoringConfig


# Global variables for graceful shutdown
shutdown_event = asyncio.Event()
pipeline_orchestrator = None
monitoring_system = None


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration"""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    logger.add(
        "logs/asr_distillation_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation="100 MB",
        retention="7 days"
    )


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    shutdown_event.set()


async def run_pipeline(config_path: str, 
                      max_batches: Optional[int] = None,
                      log_level: str = "INFO") -> None:
    """Run the ASR distillation pipeline"""
    global pipeline_orchestrator, monitoring_system
    
    try:
        # Setup logging
        setup_logging(log_level)
        
        # Load configuration
        config_manager = ConfigManager(config_path)
        config = config_manager.load_config()
        
        # Validate configuration
        if not config_manager.validate_config(config):
            logger.error("Configuration validation failed")
            return
        
        logger.info("Starting ASR Distillation Pipeline")
        logger.info(f"Configuration loaded from: {config_path}")
        
        # Initialize monitoring system
        monitoring_config = MonitoringConfig(**config.monitoring.__dict__)
        monitoring_system = MonitoringSystem(monitoring_config)
        monitoring_system.start()
        
        # Setup Ray
        if not ray.is_initialized():
            ray.init(
                object_store_memory=config.pipeline.object_store_memory,
                ignore_reinit_error=True
            )
            logger.info("Ray cluster initialized")
        
        # Initialize pipeline orchestrator
        pipeline_config = {
            'pipeline': config.pipeline.__dict__,
            'data': config.data.__dict__
        }
        pipeline_orchestrator = PipelineOrchestrator(pipeline_config)
        
        # Setup pipeline stages
        cpu_stage_config = {
            'data': config.data.__dict__,
            'audio': config.audio.__dict__
        }
        
        gpu_stage_config = {
            'inference': config.inference.__dict__,
            'writer': config.writer.__dict__,
            'storage': config.data.storage
        }
        
        # Setup complete pipeline
        pipeline_orchestrator.setup_pipeline(
            cpu_stage_class=AudioDownloadStage,
            cpu_stage_config=cpu_stage_config,
            gpu_stage_class=AudioInferenceStage,
            gpu_stage_config=gpu_stage_config
        )
        
        # Setup result writer
        result_writer = ResultWriterStage({
            'writer': config.writer.__dict__,
            'storage': config.data.storage
        })
        await result_writer.start()
        
        # Progress callback
        def progress_callback(current: int, total: int):
            progress = (current / total) * 100
            logger.info(f"Pipeline progress: {current}/{total} ({progress:.1f}%)")
        
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
        
        logger.info(f"Pipeline completed in {duration:.2f} seconds")
        logger.info(f"Processed {total_items} items, {successful_items} successful")
        logger.info(f"Throughput: {total_items / duration:.2f} items/second")
        
        # Export final metrics
        if monitoring_system:
            stats = monitoring_system.get_system_stats()
            monitoring_system.export_metrics("logs/final_metrics.json")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise
    finally:
        await cleanup()


async def cleanup():
    """Cleanup resources"""
    global pipeline_orchestrator, monitoring_system
    
    logger.info("Cleaning up resources...")
    
    try:
        if pipeline_orchestrator:
            pipeline_orchestrator.cleanup()
        
        if monitoring_system:
            monitoring_system.stop()
        
        if ray.is_initialized():
            ray.shutdown()
            
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")


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
    try:
        asyncio.run(run_pipeline(config, max_batches, log_level))
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


@app.command()
def create_config(
    output: str = typer.Option(
        "config.yaml",
        "--output", "-o",
        help="Output configuration file path"
    )
):
    """Create a sample configuration file"""
    
    try:
        config_content = create_sample_config()
        
        with open(output, 'w') as f:
            f.write(config_content)
        
        typer.echo(f"Sample configuration created: {output}")
        typer.echo("Please edit the configuration file with your settings before running the pipeline.")
        
    except Exception as e:
        typer.echo(f"Error creating configuration: {e}")
        raise typer.Exit(1)




if __name__ == "__main__":
    app()