"""简单串行Pipeline - 无并发，易调试"""

import time
from typing import Dict, List, Any, Optional, Tuple

from src.common import BatchData


class SimplePipeline:
    """简单的串行Pipeline
    
    特点：
    - 串行处理：batch依次通过所有stages
    - 无并发：简单稳定，易调试
    - 错误处理：记录失败batch，继续处理
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Pipeline配置
        """
        self.config = config
        self.stages: List[Tuple[str, Any]] = []  # [(stage_name, stage_instance), ...]
        
        # 统计信息
        self.stats = {
            'total_batches': 0,
            'successful_batches': 0,
            'failed_batches': 0,
            'total_items': 0,
            'successful_items': 0,
            'stage_times': {},
            'start_time': None,
            'end_time': None
        }
        
        # 失败记录
        self.failed_batches: List[Dict[str, Any]] = []
    
    def add_stage(self, stage_name: str, stage_instance: Any) -> None:
        """添加一个处理阶段
        
        Args:
            stage_name: 阶段名称（用于日志和统计）
            stage_instance: 阶段实例（必须有process方法）
        """
        if not hasattr(stage_instance, 'process'):
            raise ValueError(f"Stage '{stage_name}' must have a 'process' method")
        
        self.stages.append((stage_name, stage_instance))
        self.stats['stage_times'][stage_name] = 0.0
    
    def run(self, 
            producer: Any,
            writer: Any,
            max_batches: Optional[int] = None,
            log_interval: int = 10) -> Dict[str, Any]:
        """运行Pipeline
        
        Args:
            producer: 数据生产者（SimpleDataProducer）
            writer: 结果写入器（SimpleResultWriter）
            max_batches: 最大处理批次数（None表示全部）
            log_interval: 日志打印间隔（每N个batch）
            
        Returns:
            Pipeline执行统计信息
        """
        self.stats['start_time'] = time.time()
        
        print(f"Pipeline启动，共{len(self.stages)}个阶段")
        
        # 逐个处理batch
        for batch in producer.generate_batches(max_batches):
            self.stats['total_batches'] += 1
            batch_start_time = time.time()
            
            try:
                # 依次通过所有stages
                current_batch = batch
                for stage_name, stage in self.stages:
                    stage_start = time.time()
                    
                    current_batch = stage.process(current_batch)
                    
                    stage_time = time.time() - stage_start
                    self.stats['stage_times'][stage_name] += stage_time
                
                # 写入结果
                writer.write(current_batch)
                
                # 标记为已处理
                file_ids = [item.file_id for item in batch.items]
                producer.mark_processed(file_ids)
                
                # 更新统计
                self.stats['successful_batches'] += 1
                self.stats['total_items'] += len(batch.items)
                self.stats['successful_items'] += len(current_batch.items)
                
                # 定期打印进度
                if self.stats['total_batches'] % log_interval == 0:
                    self._print_progress()
                
            except Exception as e:
                # 记录失败
                self.stats['failed_batches'] += 1
                self.failed_batches.append({
                    'batch_id': batch.batch_id,
                    'error': str(e),
                    'file_ids': [item.file_id for item in batch.items]
                })
                
                # 继续处理下一个batch
                continue
        
        self.stats['end_time'] = time.time()
        
        # 关闭writer
        writer.close()
        
        # 打印最终统计
        self._print_final_stats()
        
        return self.get_stats()
    
    def _print_progress(self) -> None:
        """打印进度信息"""
        total = self.stats['total_batches']
        success = self.stats['successful_batches']
        failed = self.stats['failed_batches']
        items = self.stats['successful_items']
        
        elapsed = time.time() - self.stats['start_time']
        throughput = items / elapsed if elapsed > 0 else 0
        
        print(f"进度: {total}批次 (成功:{success}, 失败:{failed}) | "
              f"{items}条数据 | 吞吐:{throughput:.1f}条/秒")
    
    def _print_final_stats(self) -> None:
        """打印最终统计信息"""
        duration = self.stats['end_time'] - self.stats['start_time']
        
        print("\n" + "="*60)
        print("Pipeline执行完成")
        print("="*60)
        print(f"总耗时: {duration:.2f}秒")
        print(f"总批次: {self.stats['total_batches']}")
        print(f"成功批次: {self.stats['successful_batches']}")
        print(f"失败批次: {self.stats['failed_batches']}")
        print(f"成功数据: {self.stats['successful_items']}")
        print(f"成功率: {self._get_success_rate():.1%}")
        print(f"吞吐量: {self._get_throughput():.2f}条/秒")
        
        print("\n各阶段耗时:")
        for stage_name, stage_time in self.stats['stage_times'].items():
            avg_time = stage_time / self.stats['successful_batches'] if self.stats['successful_batches'] > 0 else 0
            print(f"  {stage_name}: {stage_time:.2f}秒 (平均:{avg_time:.2f}秒/批次)")
        
        if self.failed_batches:
            print(f"\n失败批次: {len(self.failed_batches)}个")
        
        print("="*60)
    
    def _get_success_rate(self) -> float:
        """计算成功率"""
        total = self.stats['total_batches']
        return self.stats['successful_batches'] / total if total > 0 else 0
    
    def _get_throughput(self) -> float:
        """计算吞吐量（条/秒）"""
        duration = self.stats['end_time'] - self.stats['start_time']
        return self.stats['successful_items'] / duration if duration > 0 else 0
    
    def get_stats(self) -> Dict[str, Any]:
        """获取详细统计信息"""
        return {
            **self.stats,
            'success_rate': self._get_success_rate(),
            'throughput': self._get_throughput(),
            'failed_batches': self.failed_batches
        }