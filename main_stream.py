"""流式ASR蒸馏框架主程序 - 支持千万级数据处理"""

import asyncio
import signal
import sys
import time
import logging
from pathlib import Path
from typing import Optional

import typer
import ray

from src.config.manager import ConfigManager, create_sample_config
from src.scheduling.stream_pipeline import StreamingPipelineOrchestrator
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
    BatchInferenceStage,
    PostProcessingStage
)
from src.storage.result_writer import ResultWriterStage, SyncResultWriterStage  # 保留导入
from src.monitoring.system import MonitoringSystem


# 全局变量
pipeline_orchestrator = None
monitoring_system = None
shutdown_requested = False


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """配置简洁的日志系统"""
    logger = logging.getLogger("asr_distillation")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 只输出到控制台
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(handler)
    
    return logger


def signal_handler(signum, frame):
    """处理中断信号"""
    global shutdown_requested
    shutdown_requested = True


async def run_pipeline(config_path: str, 
                      max_batches: Optional[int] = None,
                      log_level: str = "INFO") -> None:
    """运行流式ASR蒸馏Pipeline"""
    global pipeline_orchestrator, monitoring_system
    
    logger = setup_logging(log_level)
    
    try:
        # 1. 加载配置
        logger.info("Loading configuration...")
        config_manager = ConfigManager(config_path)
        config = config_manager.load_config()
        
        if not config_manager.validate_config(config):
            logger.error("Configuration validation failed")
            return
        
        # 2. 初始化Ray
        if not ray.is_initialized():
            logger.info("Initializing Ray cluster...")
            ray.init(
                object_store_memory=config.pipeline.object_store_memory,
                ignore_reinit_error=True,
                logging_level=logging.ERROR  # 抑制Ray日志
            )
        
        # 3. 初始化监控系统
        logger.info("Starting monitoring system...")
        from src.config.manager import MonitoringConfig
        monitoring_config = MonitoringConfig(**config.monitoring.__dict__)
        monitoring_system = MonitoringSystem(monitoring_config)
        monitoring_system.start()
        
        # 4. 初始化Pipeline
        logger.info("Setting up streaming pipeline...")
        pipeline_config = {
            'pipeline': config.pipeline.__dict__,
            'data': config.data.__dict__
        }
        pipeline_orchestrator = StreamingPipelineOrchestrator(pipeline_config)
        
        # 5. 配置Pipeline stages (新增第9个Stage)
        enable_multimedia = config.media is not None
        stage_workers_config = config.pipeline.stage_workers
        
        stages_config = [
            {
                'type': 'cpu',
                'class': AudioDownloadStage,
                'name': 'audio_download',
                'num_workers': stage_workers_config.get('audio_download', 8),
                'config': {
                    'data': config.data.__dict__,
                    'audio': config.audio.__dict__,
                    'storage': config.data.storage,
                    'enable_multimedia': enable_multimedia,
                    'media': config.media.__dict__ if enable_multimedia else {}
                }
            },
            {
                'type': 'cpu',
                'class': AudioPreprocessingStage,
                'name': 'audio_preprocessing',
                'num_workers': stage_workers_config.get('audio_preprocessing', 6),
                'config': {
                    'data': config.data.__dict__,
                    'audio': config.audio.__dict__,
                    'storage': config.data.storage
                }
            },
            {
                'type': 'cpu',
                'class': VADProcessingStage,
                'name': 'vad_processing',
                'num_workers': stage_workers_config.get('vad_processing', 4),
                'config': {
                    'vad': config.vad.__dict__,
                    'batch_size': config.pipeline.batch_size
                }
            },
            {
                'type': 'cpu',
                'class': SegmentExpansionStage,
                'name': 'segment_expansion',
                'num_workers': stage_workers_config.get('segment_expansion', 4),
                'config': {
                    'min_segment_duration': 0.1,
                    'preserve_order': True
                }
            },
            {
                'type': 'cpu',
                'class': AudioFeatureStage,
                'name': 'feature_extraction',
                'num_workers': stage_workers_config.get('feature_extraction', 6),
                'config': {
                    'data': config.data.__dict__,
                    'audio': config.audio.__dict__,
                    'storage': config.data.storage
                }
            },
            {
                'type': 'gpu',
                'class': BatchInferenceStage,
                'name': 'batch_inference',
                'num_workers': stage_workers_config.get('batch_inference', 1),
                'config': {
                    'inference': config.inference.__dict__,
                    'writer': config.writer.__dict__,
                    'storage': config.data.storage
                }
            },
            {
                'type': 'cpu',
                'class': SegmentAggregationStage,
                'name': 'segment_aggregation',
                'num_workers': stage_workers_config.get('segment_aggregation', 2),
                'config': {
                    'sort_by_timestamp': True,
                    'include_segment_details': True,
                    'calculate_file_stats': True
                }
            },
            {
                'type': 'cpu',
                'class': PostProcessingStage,
                'name': 'post_processing',
                'num_workers': stage_workers_config.get('post_processing', 2),
                'config': {
                    'inference': config.inference.__dict__,
                    'writer': config.writer.__dict__,
                    'storage': config.data.storage
                }
            },
            # ✅ 新增：第9个Stage - 结果写入
            # 根据配置选择同步或异步版本
            {
                'type': 'cpu',  # 写入是IO密集型，用CPU即可
                'class': SyncResultWriterStage if config.writer.get('sync_mode', False) else ResultWriterStage,
                'name': 'result_writer',
                'num_workers': stage_workers_config.get('result_writer', 1),  # 通常1个即可
                'config': {
                    'writer': config.writer.__dict__,
                    'storage': config.data.storage
                }
            }
        ]
        
        pipeline_orchestrator.setup_multi_stage_pipeline(stages_config)
        
        # 6. ❌ 移除：不再单独初始化 ResultWriter
        # result_writer = ResultWriterStage({...})
        # await result_writer.start()
        
        # 7. 进度回调
        def progress_callback(current: int, queue_stats: dict):
            pass
            #if current % 100 == 0:  # 每100次更新打印一次
                #logger.info(f"Progress: {current} items processed")
        
        # 8. 运行Pipeline
        logger.info("Starting pipeline execution...")
        start_time = time.time()
        
        stats = pipeline_orchestrator.run(
            max_batches=max_batches,
            progress_callback=progress_callback,
            monitoring_system=monitoring_system
        )
        
        duration = time.time() - start_time
        
        # 9. 打印执行摘要
        logger.info("=" * 60)
        logger.info("Pipeline Execution Summary")
        logger.info("=" * 60)
        logger.info(f"Total Duration: {duration:.2f}s")
        logger.info(f"Total Batches: {stats.get('total_batches', 0)}")
        logger.info(f"Total Items: {stats.get('total_items', 0)}")
        logger.info(f"Successful Items: {stats.get('successful_items', 0)}")
        logger.info(f"Success Rate: {stats.get('success_rate', 0):.2%}")
        logger.info(f"Throughput: {stats.get('throughput', 0):.2f} items/s")
        logger.info("=" * 60)
        
        # 10. 导出最终指标
        if monitoring_system:
            metrics_file = "logs/final_metrics.json"
            monitoring_system.export_metrics(metrics_file)
            logger.info(f"Metrics exported to {metrics_file}")
        
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise
    finally:
        await cleanup(logger)


async def cleanup(logger: logging.Logger):
    """清理资源"""
    global pipeline_orchestrator, monitoring_system
    
    logger.info("Cleaning up resources...")
    
    # ❌ 移除：result_writer 清理
    # if result_writer:
    #     await result_writer.stop()
    
    if pipeline_orchestrator:
        try:
            pipeline_orchestrator.cleanup()
        except Exception as e:
            logger.error(f"Error cleaning up pipeline: {e}")
    
    if monitoring_system:
        try:
            monitoring_system.stop()
        except Exception as e:
            logger.error(f"Error stopping monitoring: {e}")
    
    if ray.is_initialized():
        try:
            ray.shutdown()
        except Exception as e:
            logger.error(f"Error shutting down Ray: {e}")
    
    logger.info("Cleanup completed")


# CLI应用
app = typer.Typer(help="流式ASR蒸馏框架")


@app.command()
def run(
    config: str = typer.Option(
        "config.yaml",
        "--config", "-c",
        help="配置文件路径"
    ),
    max_batches: Optional[int] = typer.Option(
        None,
        "--max-batches",
        help="最大处理批次数（用于测试）"
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="日志级别 (DEBUG/INFO/WARNING/ERROR)"
    )
):
    """运行流式ASR蒸馏Pipeline"""
    
    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 创建必要目录
    Path("logs").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)
    
    # 运行Pipeline
    asyncio.run(run_pipeline(config, max_batches, log_level))


@app.command()
def create_config(
    output: str = typer.Option(
        "config.yaml",
        "--output", "-o",
        help="输出配置文件路径"
    )
):
    """创建示例配置文件"""
    config_content = create_sample_config()
    
    with open(output, 'w') as f:
        f.write(config_content)
    
    print(f"✓ 配置文件已创建: {output}")
    print("请编辑配置文件后再运行Pipeline")


@app.command()
def status(
    checkpoint_dir: str = typer.Option(
        "./checkpoints",
        "--checkpoint-dir",
        help="检查点目录"
    )
):
    """查看Pipeline状态和检查点"""
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        print(f"检查点目录不存在: {checkpoint_dir}")
        return
    
    # 列出检查点文件
    checkpoints = list(checkpoint_path.glob("*.pkl"))
    
    print("=" * 60)
    print("Pipeline Status")
    print("=" * 60)
    print(f"Checkpoint Directory: {checkpoint_dir}")
    print(f"Total Checkpoints: {len(checkpoints)}")
    
    if checkpoints:
        print("\nRecent Checkpoints:")
        for ckpt in sorted(checkpoints, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
            size_mb = ckpt.stat().st_size / 1024 / 1024
            mtime = time.strftime('%Y-%m-%d %H:%M:%S', 
                                 time.localtime(ckpt.stat().st_mtime))
            print(f"  - {ckpt.name} ({size_mb:.2f} MB, {mtime})")
    
    print("=" * 60)


if __name__ == "__main__":
    app()