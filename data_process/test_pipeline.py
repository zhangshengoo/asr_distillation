"""快速测试脚本 - 验证新Pipeline架构是否正常工作

功能:
1. 创建模拟数据
2. 运行4个stage
3. 验证输出结果
"""

import sys
import logging
from pathlib import Path

# 添加路径
sys.path.insert(0, '/home/claude')

from simple_ray_pipeline import SimplifiedPipeline, ProcessBatch, StageProcessor
import ray


# ==================== 模拟Stage ====================

class MockDownloadStage(StageProcessor):
    """模拟下载stage"""
    def process(self, batch: ProcessBatch) -> ProcessBatch:
        print(f"[下载] 处理批次 {batch.batch_id}: {len(batch.data)} 个文件")
        
        # 模拟下载：添加audio_bytes
        processed = []
        for item in batch.data:
            item['audio_bytes'] = b'mock_audio_data'
            processed.append(item)
        
        batch.data = processed
        return batch


class MockPreprocessStage(StageProcessor):
    """模拟预处理stage"""
    def process(self, batch: ProcessBatch) -> ProcessBatch:
        print(f"[预处理] 处理批次 {batch.batch_id}: {len(batch.data)} 个文件")
        
        import numpy as np
        
        # 模拟预处理：添加waveform
        processed = []
        for item in batch.data:
            item['waveform'] = np.random.randn(16000)  # 1秒音频
            item['sample_rate'] = 16000
            processed.append(item)
        
        batch.data = processed
        return batch


class MockVADStage(StageProcessor):
    """模拟VAD stage"""
    def process(self, batch: ProcessBatch) -> ProcessBatch:
        print(f"[VAD] 处理批次 {batch.batch_id}: {len(batch.data)} 个文件")
        
        # 模拟VAD：添加segments
        processed = []
        for item in batch.data:
            # 模拟2个segment
            item['segments'] = [
                {'start': 0.0, 'end': 0.5, 'audio': item['waveform'][:8000]},
                {'start': 0.5, 'end': 1.0, 'audio': item['waveform'][8000:]}
            ]
            processed.append(item)
        
        batch.data = processed
        return batch


class MockExpansionStage(StageProcessor):
    """模拟片段展开stage"""
    def process(self, batch: ProcessBatch) -> ProcessBatch:
        print(f"[展开] 处理批次 {batch.batch_id}: {len(batch.data)} 个文件")
        
        # 展开segments
        expanded = []
        for item in batch.data:
            file_id = item['file_id']
            for idx, seg in enumerate(item['segments']):
                seg_item = {
                    'file_id': f"{file_id}_seg_{idx}",
                    'parent_file_id': file_id,
                    'start_time': seg['start'],
                    'end_time': seg['end'],
                    'waveform': seg['audio']
                }
                expanded.append(seg_item)
        
        batch.data = expanded
        return batch


# ==================== 模拟Producer ====================

@ray.remote
class MockProducer:
    """模拟数据生产者"""
    
    def __init__(self, num_files: int = 100):
        self.num_files = num_files
    
    def load_batches(self, max_batches=None):
        """生成模拟批次"""
        batch_size = 10
        batches = []
        
        for i in range(0, self.num_files, batch_size):
            if max_batches and len(batches) >= max_batches:
                break
            
            # 生成模拟数据
            data = []
            for j in range(batch_size):
                file_idx = i + j
                if file_idx >= self.num_files:
                    break
                
                data.append({
                    'file_id': f'file_{file_idx:04d}',
                    'oss_path': f'oss://bucket/audio/file_{file_idx:04d}.wav'
                })
            
            batch = ProcessBatch(
                batch_id=f'batch_{len(batches)}',
                data=data,
                metadata={'num_files': len(data)}
            )
            batches.append(batch)
        
        return batches
    
    def mark_completed(self, file_ids):
        """标记完成"""
        print(f"✓ 标记完成: {len(file_ids)} 个文件")


# ==================== 测试函数 ====================

def test_pipeline():
    """测试Pipeline"""
    
    print("=" * 70)
    print(" 测试新Pipeline架构 ")
    print("=" * 70)
    
    # 创建Pipeline
    config = {
        'data': {},
        'pipeline': {'object_store_memory': 100 * 1024 * 1024},  # 100MB
        'batch_size': 10
    }
    
    pipeline = SimplifiedPipeline(config)
    
    # 设置模拟Producer
    print("\n1. 设置Producer...")
    pipeline.producer = MockProducer.remote(num_files=50)
    
    # 添加4个stage
    print("\n2. 添加Stage...")
    
    pipeline.add_stage(
        stage_class=MockDownloadStage,
        stage_config={},
        stage_name='download',
        num_workers=2
    )
    
    pipeline.add_stage(
        stage_class=MockPreprocessStage,
        stage_config={},
        stage_name='preprocess',
        num_workers=2
    )
    
    pipeline.add_stage(
        stage_class=MockVADStage,
        stage_config={},
        stage_name='vad',
        num_workers=2
    )
    
    pipeline.add_stage(
        stage_class=MockExpansionStage,
        stage_config={},
        stage_name='expansion',
        num_workers=2
    )
    
    print("\n3. 运行Pipeline...")
    print("-" * 70)
    
    # 运行
    stats = pipeline.run(max_batches=3, max_retries=2)
    
    # 打印结果
    print("\n" + "=" * 70)
    print(" 测试结果 ")
    print("=" * 70)
    print(f"总批次数: {stats['total_batches']}")
    print(f"成功批次: {stats['successful_batches']}")
    print(f"失败批次: {stats['failed_batches']}")
    print(f"耗时: {stats['duration']:.2f}秒")
    
    print("\n各Stage统计:")
    for stage_name, stage_stats in stats['stage_stats'].items():
        print(f"  {stage_name}:")
        print(f"    Workers: {stage_stats['workers']}")
        print(f"    处理数: {stage_stats['processed']}")
        print(f"    错误数: {stage_stats['errors']}")
    
    # 验证
    print("\n" + "-" * 70)
    if stats['successful_batches'] == stats['total_batches']:
        print("✅ 测试通过！所有批次处理成功")
    else:
        print(f"⚠️  警告: {stats['failed_batches']} 个批次失败")
    
    print("=" * 70)
    
    # 清理
    pipeline.shutdown()
    
    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    try:
        stats = test_pipeline()
        
        if stats['failed_batches'] > 0:
            sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)