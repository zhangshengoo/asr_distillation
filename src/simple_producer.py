"""简单的数据生产者 - 生成SourceItem批次"""

import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Generator, Set

from src.common import BatchData, SourceItem
from src.data.media_indexer import MediaDataLoader
from src.data.storage import MediaStorageManager


class SimpleDataProducer:
    """简单的批次数据生产者
    
    功能：
    - 从索引加载音频文件列表
    - 分批生成SourceItem
    - 支持断点续传（checkpoint）
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 包含 'data' 配置的字典
        """
        data_config = config['data']
        
        self.batch_size = config.get('batch_size', 32)
        self.checkpoint_dir = Path(config.get('checkpoint_dir', './checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据加载器和存储管理器
        self.data_loader = MediaDataLoader(data_config)
        
        # 使用分离的输入和输出存储配置
        self.storage_manager = MediaStorageManager(
            input_config=data_config['input_storage'],
            output_config=data_config['output_storage']
        )
        
        # 处理状态
        self.processed_file_ids: Set[str] = set()
        self.total_produced = 0
        
        # 加载checkpoint
        self._load_checkpoint()
    
    def _load_checkpoint(self) -> None:
        """加载生产者checkpoint"""
        checkpoint_file = self.checkpoint_dir / "producer_checkpoint.pkl"
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'rb') as f:
                    checkpoint = pickle.load(f)
                self.processed_file_ids = checkpoint.get('processed_file_ids', set())
                self.total_produced = checkpoint.get('total_produced', 0)
            except Exception:
                pass
    
    def _save_checkpoint(self) -> None:
        """保存生产者checkpoint"""
        checkpoint_file = self.checkpoint_dir / "producer_checkpoint.pkl"
        try:
            checkpoint = {
                'processed_file_ids': self.processed_file_ids,
                'total_produced': self.total_produced
            }
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint, f)
        except Exception:
            pass
    
    def _load_index(self) -> List[Dict[str, Any]]:
        """加载音频索引"""
        df = self.data_loader.load_index()
        
        if df.empty:
            # 从存储构建索引
            media_files = self.storage_manager.list_media_files()
            if media_files:
                df = self.data_loader.create_index(media_files)
        
        return df.to_dict('records') if not df.empty else []
    
    def generate_batches(self, 
                        max_batches: Optional[int] = None) -> Generator[BatchData[SourceItem], None, None]:
        """生成器：逐个yield批次数据
        
        Args:
            max_batches: 最大批次数（用于测试），None表示处理全部
            
        Yields:
            BatchData[SourceItem]: 包含SourceItem的批次
        """
        # 加载索引
        all_records = self._load_index()
        
        # 过滤已处理的文件
        remaining_records = [
            r for r in all_records 
            if r['file_id'] not in self.processed_file_ids
        ]
        
        batch_count = 0
        checkpoint_interval = 100  # 每100个batch保存一次checkpoint
        
        # 分批生成
        for i in range(0, len(remaining_records), self.batch_size):
            if max_batches and batch_count >= max_batches:
                break
            
            batch_records = remaining_records[i:i + self.batch_size]
            
            # 转换为SourceItem对象
            items = []
            for record in batch_records:
                item = SourceItem(
                    file_id=record['file_id'],
                    oss_path=record['oss_path'],
                    format=record.get('format', 'wav'),
                    duration=record.get('duration', 0.0),
                    metadata={
                        k: v for k, v in record.items()
                        if k not in ['file_id', 'oss_path', 'format', 'duration']
                    }
                )
                items.append(item)
            
            # 创建批次
            batch = BatchData(
                batch_id=f"batch_{self.total_produced}",
                items=items,
                metadata={'stage': 'producer', 'batch_index': self.total_produced}
            )
            
            yield batch
            
            self.total_produced += 1
            batch_count += 1
            
            # 定期保存checkpoint
            if batch_count % checkpoint_interval == 0:
                self._save_checkpoint()
        
        # 最终保存checkpoint
        self._save_checkpoint()
    
    def mark_processed(self, file_ids: List[str]) -> None:
        """标记文件为已处理
        
        Args:
            file_ids: 已处理的文件ID列表
        """
        self.processed_file_ids.update(file_ids)
        self._save_checkpoint()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'total_produced': self.total_produced,
            'processed_files': len(self.processed_file_ids)
        }