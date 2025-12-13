"""简化的ASR蒸馏主程序 - 无Ray，串行处理"""

import sys
from pathlib import Path
from typing import Optional

import typer

from src.config.manager import ConfigManager, create_sample_config
import threading
import asyncio
import time
from typing import Dict, Any, List

from src.simple_pipeline import SimplePipeline
from src.simple_producer import SimpleDataProducer
from src.storage.result_writer import AsyncResultWriter, WriteConfig
from src.data.storage import AudioStorageManager
from src.common import BatchData, FileResultItem

# 导入所有stage
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

class SyncResultWriter:
    """Synchronous wrapper for AsyncResultWriter"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Parse configurations
        writer_config = WriteConfig(**config.get('writer', {}))
        
        # Setup storage manager if configured
        self.storage_manager = None
        if 'input_storage' in config and 'output_storage' in config:
            # 使用分离的输入和输出存储配置
            self.storage_manager = MediaStorageManager(
                input_config=config['input_storage'],
                output_config=config['output_storage']
            )
            
        # Initialize async writer
        self.async_writer = AsyncResultWriter(writer_config, self.storage_manager)
        
        # Setup background loop
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        
        # Start writer in background
        future = asyncio.run_coroutine_threadsafe(self.async_writer.start(), self.loop)
        future.result()  # Wait for start
        
    def _run_loop(self):
        """Run asyncio loop in background thread"""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
        
    def write(self, batch: BatchData) -> None:
        """Write batch results"""
        items_to_write = []
        for item in batch.items:
            if isinstance(item, FileResultItem):
                # Convert to dict for writer
                result_dict = {
                    'file_id': item.file_id,
                    'transcription': item.transcription,
                    'segments': item.segments,
                    'metadata': item.metadata,
                    'stats': item.stats
                }
                items_to_write.append(result_dict)
                
        if items_to_write:
            asyncio.run_coroutine_threadsafe(
                self.async_writer.write_batch(items_to_write), 
                self.loop
            )
            
    def close(self):
        """Stop writer and clean up"""
        if self.loop.is_running():
            # Stop writer
            future = asyncio.run_coroutine_threadsafe(self.async_writer.stop(), self.loop)
            try:
                future.result(timeout=5.0)
            except Exception as e:
                print(f"Error stopping writer: {e}")
            
            # Stop loop
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=2.0)
            
    def get_stats(self) -> Dict[str, Any]:
        """Get writer stats"""
        # This is not thread-safe strictly but stats are simple types
        stats = self.async_writer.get_stats()
        # Add a few sync stats if needed
        return {
            'total_written': stats.get('items_written', 0),
            'files_created': stats.get('files_written', 0),
            'output_dir': self.async_writer.config.output_format, # Just for info
            'errors': stats.get('errors', 0)
        }


def setup_pipeline(config) -> SimplePipeline:
    """设置Pipeline的所有阶段
    
    Args:
        config: 配置对象
        
    Returns:
        配置好的SimplePipeline实例
    """
    pipeline_config = {
        'batch_size': config.pipeline.batch_size,
        'checkpoint_dir': config.pipeline.checkpoint_dir
    }
    
    pipeline = SimplePipeline(pipeline_config)
    
    # 检查是否启用多媒体处理
    enable_multimedia = config.media is not None
    
    # 1. 音频下载阶段
    pipeline.add_stage(
            'audio_download',
            AudioDownloadStage({
                'data': config.data.__dict__,
                'audio': config.audio.__dict__,
                'input_storage': config.data.input_storage,
                'output_storage': config.data.output_storage,
                'enable_multimedia': enable_multimedia,
                'media': config.media.__dict__ if enable_multimedia else {}
            })
        )    
    # 2. 音频预处理阶段
    pipeline.add_stage(
            'audio_preprocessing',
            AudioPreprocessingStage({
                'data': config.data.__dict__,
                'audio': config.audio.__dict__,
                'input_storage': config.data.input_storage,
                'output_storage': config.data.output_storage
            })
        )    
    # 3. VAD处理阶段
    pipeline.add_stage(
        'vad_processing',
        VADProcessingStage({
            'vad': config.vad.__dict__,
            'batch_size': config.pipeline.batch_size
        })
    )
    
    # 4. 片段展开阶段
    pipeline.add_stage(
        'segment_expansion',
        SegmentExpansionStage({
            'min_segment_duration': 0.1,
            'preserve_order': True
        })
    )
    
    # 5. 特征提取阶段
    pipeline.add_stage(
            'feature_extraction',
            AudioFeatureStage({
                'data': config.data.__dict__,
                'audio': config.audio.__dict__
            })
        )    
    # 6. 批量推理阶段（GPU）
    pipeline.add_stage(
            'batch_inference',
            BatchInferenceStage({
                'inference': config.inference.__dict__,
                'writer': config.writer.__dict__
            })
        )    
    # 7. 片段聚合阶段
    pipeline.add_stage(
        'segment_aggregation',
        SegmentAggregationStage({
            'sort_by_timestamp': True,
            'include_segment_details': True,
            'calculate_file_stats': True
        })
    )
    
    # 8. 后处理阶段
    pipeline.add_stage(
        'post_processing',
            PostProcessingStage({
                'inference': config.inference.__dict__,
                'writer': config.writer.__dict__,
                'input_storage': config.data.input_storage,
                'output_storage': config.data.output_storage
            })
        )    
    return pipeline


def run_pipeline(config_path: str,
                max_batches: Optional[int] = None,
                log_interval: int = 10) -> None:
    """运行ASR蒸馏Pipeline
    
    Args:
        config_path: 配置文件路径
        max_batches: 最大处理批次数（None表示全部）
        log_interval: 日志打印间隔（每N个batch）
    """
    # 1. 加载配置
    print(f"加载配置: {config_path}")
    config_manager = ConfigManager(config_path)
    config = config_manager.load_config()
    
    if not config_manager.validate_config(config):
        print("配置验证失败")
        sys.exit(1)
    
    # 2. 创建必要目录
    Path("logs").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    Path(config.pipeline.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # 3. 创建数据生产者
    producer_config = {
        'data': config.data.__dict__,
        'batch_size': config.pipeline.batch_size,
        'checkpoint_dir': config.pipeline.checkpoint_dir
    }
    producer = SimpleDataProducer(producer_config)
    
    # 4. 创建结果写入器
    writer_config = {
        'writer': config.writer.__dict__,
        'input_storage': config.data.input_storage,
        'output_storage': config.data.output_storage,
        'output_dir': './results'  # Configured in writer config usually, but keeping for compat
    }
    writer = SyncResultWriter(writer_config)
    
    # 5. 设置Pipeline
    print("设置Pipeline...")
    pipeline = setup_pipeline(config)
    
    # 6. 运行Pipeline
    print(f"开始处理 (max_batches={max_batches or '全部'})")
    try:
        stats = pipeline.run(
            producer=producer,
            writer=writer,
            max_batches=max_batches,
            log_interval=log_interval
        )
        
        # 7. 打印生产者和写入器统计
        print("\n数据生产统计:")
        producer_stats = producer.get_stats()
        print(f"  生成批次: {producer_stats['total_produced']}")
        print(f"  已处理文件: {producer_stats['processed_files']}")
        
        print("\n写入器统计:")
        writer_stats = writer.get_stats()
        print(f"  写入数据: {writer_stats['total_written']}")
        print(f"  创建文件: {writer_stats['files_created']}")
        print(f"  输出目录: {writer_stats['output_dir']}")
        
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"\nPipeline失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# CLI应用
app = typer.Typer(help="简化的ASR蒸馏框架")


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
    log_interval: int = typer.Option(
        10,
        "--log-interval",
        help="日志打印间隔（每N个batch）"
    )
):
    """运行简化的ASR蒸馏Pipeline"""
    run_pipeline(config, max_batches, log_interval)


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
    
    # 检查生产者checkpoint
    producer_ckpt = checkpoint_path / "producer_checkpoint.pkl"
    if producer_ckpt.exists():
        import pickle
        with open(producer_ckpt, 'rb') as f:
            ckpt = pickle.load(f)
        
        print("="*60)
        print("Pipeline状态")
        print("="*60)
        print(f"检查点目录: {checkpoint_dir}")
        print(f"已生成批次: {ckpt.get('total_produced', 0)}")
        print(f"已处理文件: {len(ckpt.get('processed_file_ids', set()))}")
        print("="*60)
    else:
        print("未找到checkpoint文件")


if __name__ == "__main__":
    app()