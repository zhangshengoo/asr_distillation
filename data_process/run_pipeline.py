"""音频VAD处理Pipeline - 简化版执行脚本

使用方法:
    python run_simple_pipeline.py --config config.yaml --max-batches 10

特点:
- 简单清晰的批处理流程
- 基于Ray ActorPool的高效并行
- 明确的错误处理和重试
- 实时进度监控
"""

import sys
import time
import logging
from pathlib import Path
from typing import Optional

import typer
import ray

from data_process.simple_ray_pipeline import SimplifiedPipeline
from data_process.audio_stage_processors import (
    AudioDownloadProcessor,
    AudioPreprocessingProcessor,
    VADProcessor,
    SegmentExpansionProcessor
)
from src.config.manager import ConfigManager


app = typer.Typer()


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """配置日志"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


@app.command()
def run(
    config: str = typer.Option("config.yaml", "--config", "-c", help="配置文件路径"),
    max_batches: Optional[int] = typer.Option(None, "--max-batches", "-n", help="最大批次数(测试用)"),
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="日志级别"),
    checkpoint_dir: str = typer.Option("./checkpoints", "--checkpoint-dir", help="检查点目录")
):
    """运行音频VAD处理Pipeline"""
    
    logger = setup_logging(log_level)
    
    try:
        # ==================== 1. 加载配置 ====================
        logger.info("=" * 70)
        logger.info(" 音频VAD处理Pipeline - 简化版 ")
        logger.info("=" * 70)
        logger.info(f"配置文件: {config}")
        
        config_manager = ConfigManager(config)
        cfg = config_manager.load_config()
        
        if not config_manager.validate_config(cfg):
            logger.error("❌ 配置验证失败")
            sys.exit(1)
        
        # 创建必要目录
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        
        # ==================== 2. 初始化Pipeline ====================
        logger.info("\n初始化Pipeline...")
        
        # 构建配置字典
        config_dict = {
            'data': cfg.data.__dict__,
            'pipeline': cfg.pipeline.__dict__,
            'batch_size': cfg.pipeline.batch_size
        }
        
        # 创建Pipeline
        pipeline = SimplifiedPipeline(config_dict)
        
        # 设置Producer
        pipeline.setup_producer()
        
        # ==================== 3. 添加4个Stage ====================
        logger.info("\n配置处理阶段...")
        
        stage_workers = cfg.pipeline.stage_workers
        enable_multimedia = cfg.media is not None
        
        # Stage 1: 音频下载
        stage1_config = {
            'data': cfg.data.__dict__,
            'input_storage': cfg.data.input_storage,
            'output_storage': cfg.data.output_storage,
            'enable_multimedia': enable_multimedia,
            'media': cfg.media.__dict__ if enable_multimedia else {}
        }
        pipeline.add_stage(
            stage_class=AudioDownloadProcessor,
            stage_config=stage1_config,
            stage_name='audio_download',
            num_workers=stage_workers.get('audio_download', 8),
            resources={'num_cpus': 1}
        )
        
        # Stage 2: 音频预处理
        stage2_config = {
            'audio': cfg.audio.__dict__
        }
        pipeline.add_stage(
            stage_class=AudioPreprocessingProcessor,
            stage_config=stage2_config,
            stage_name='audio_preprocessing',
            num_workers=stage_workers.get('audio_preprocessing', 6),
            resources={'num_cpus': 1}
        )
        
        # Stage 3: VAD处理
        stage3_config = {
            'vad': cfg.vad.__dict__
        }
        pipeline.add_stage(
            stage_class=VADProcessor,
            stage_config=stage3_config,
            stage_name='vad_processing',
            num_workers=stage_workers.get('vad_processing', 4),
            resources={'num_cpus': 1}
        )
        
        # Stage 4: 片段展开
        stage4_config = {
            'min_segment_duration': cfg.segment_expansion.min_segment_duration,
            'max_segment_duration': cfg.segment_expansion.max_segment_duration,
            'segment_threshold': cfg.segment_expansion.segment_threshold,
            'sampling_rate': cfg.segment_expansion.sampling_rate,
            'resplit_min_speech_ms': cfg.segment_expansion.resplit_min_speech_ms,
            'resplit_min_silence_ms': cfg.segment_expansion.resplit_min_silence_ms,
            'resplit_threshold': cfg.segment_expansion.resplit_threshold,
            'resplit_neg_threshold': cfg.segment_expansion.resplit_neg_threshold,
            'resplit_speech_pad_ms': cfg.segment_expansion.resplit_speech_pad_ms,
            'segment_upload': cfg.segment_upload,
            'input_storage': cfg.data.input_storage,
            'output_storage': cfg.data.output_storage
        }
        pipeline.add_stage(
            stage_class=SegmentExpansionProcessor,
            stage_config=stage4_config,
            stage_name='segment_expansion',
            num_workers=stage_workers.get('segment_expansion', 4),
            resources={'num_cpus': 1}
        )
        
        # ==================== 4. 打印配置 ====================
        logger.info("\n" + "=" * 70)
        logger.info("Pipeline配置:")
        logger.info("-" * 70)
        logger.info(f"{'Stage':<25} {'Workers':<10} {'Type':<10}")
        logger.info("-" * 70)
        for stage_cfg in pipeline.stage_configs:
            logger.info(f"{stage_cfg['name']:<25} {stage_cfg['num_workers']:<10} CPU")
        logger.info("-" * 70)
        logger.info(f"批次大小: {cfg.pipeline.batch_size}")
        if max_batches:
            logger.info(f"最大批次数: {max_batches} (测试模式)")
        logger.info("=" * 70)
        
        # ==================== 5. 进度回调 ====================
        last_log_time = [time.time()]
        
        def progress_callback(completed: int, total: int, stage_name: str):
            """进度回调 - 每10秒或每完成10个批次打印一次"""
            current_time = time.time()
            if completed % 10 == 0 or current_time - last_log_time[0] > 10:
                progress_pct = completed / total * 100 if total > 0 else 0
                logger.info(f"[{stage_name}] 进度: {completed}/{total} ({progress_pct:.1f}%)")
                last_log_time[0] = current_time
        
        # ==================== 6. 运行Pipeline ====================
        logger.info("\n开始处理...\n")
        start_time = time.time()
        
        stats = pipeline.run(
            max_batches=max_batches,
            max_retries=3,
            progress_callback=progress_callback
        )
        
        duration = time.time() - start_time
        
        # ==================== 7. 打印结果 ====================
        logger.info("\n" + "=" * 70)
        logger.info(" Pipeline执行完成 ")
        logger.info("=" * 70)
        logger.info(f"总耗时: {duration:.2f}秒 ({duration/60:.1f}分钟)")
        logger.info(f"总批次数: {stats['total_batches']}")
        logger.info(f"成功批次: {stats['successful_batches']}")
        logger.info(f"失败批次: {stats['failed_batches']}")
        
        success_rate = stats['successful_batches'] / stats['total_batches'] * 100 if stats['total_batches'] > 0 else 0
        logger.info(f"成功率: {success_rate:.1f}%")
        
        throughput = stats['successful_batches'] / duration if duration > 0 else 0
        logger.info(f"吞吐量: {throughput:.2f} 批次/秒")
        
        # 各Stage统计
        logger.info("\n" + "-" * 70)
        logger.info("各阶段统计:")
        logger.info("-" * 70)
        
        for stage_name, stage_stat in stats.get('stage_stats', {}).items():
            logger.info(f"\n{stage_name}:")
            logger.info(f"  Workers: {stage_stat['workers']}")
            logger.info(f"  处理数: {stage_stat['processed']}")
            logger.info(f"  错误数: {stage_stat['errors']}")
            logger.info(f"  成功率: {stage_stat['success_rate']:.1%}")
        
        logger.info("\n" + "=" * 70)
        
        # ==================== 8. 清理 ====================
        pipeline.shutdown()
        logger.info("✓ Pipeline已关闭")
        
    except KeyboardInterrupt:
        logger.warning("\n⚠ 用户中断")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"\n❌ Pipeline失败: {e}", exc_info=True)
        sys.exit(1)


@app.command()
def status(
    checkpoint_dir: str = typer.Option("./checkpoints", "--checkpoint-dir", help="检查点目录")
):
    """查看处理状态"""
    checkpoint_path = Path(checkpoint_dir)
    
    print("=" * 60)
    print(" 处理状态 ")
    print("=" * 60)
    
    if not checkpoint_path.exists():
        print(f"检查点目录不存在: {checkpoint_dir}")
        return
    
    # 读取checkpoint
    checkpoint_file = checkpoint_path / "producer_checkpoint.pkl"
    if checkpoint_file.exists():
        import pickle
        with open(checkpoint_file, 'rb') as f:
            processed_files = pickle.load(f)
        
        print(f"已处理文件数: {len(processed_files)}")
        print(f"检查点文件: {checkpoint_file}")
        print(f"最后修改时间: {time.ctime(checkpoint_file.stat().st_mtime)}")
    else:
        print("未找到检查点文件")
    
    print("=" * 60)


@app.command()
def clear_checkpoint(
    checkpoint_dir: str = typer.Option("./checkpoints", "--checkpoint-dir", help="检查点目录"),
    confirm: bool = typer.Option(False, "--yes", "-y", help="跳过确认")
):
    """清除检查点（重新开始处理）"""
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_file = checkpoint_path / "producer_checkpoint.pkl"
    
    if not checkpoint_file.exists():
        print("未找到检查点文件")
        return
    
    if not confirm:
        response = input("确认要清除检查点吗？这将重新开始处理所有文件。(y/N): ")
        if response.lower() != 'y':
            print("已取消")
            return
    
    checkpoint_file.unlink()
    print(f"✓ 已清除检查点: {checkpoint_file}")


if __name__ == "__main__":
    app()