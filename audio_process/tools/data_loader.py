"""DataLoader实现 - 支持多进程并行"""
from torch.utils.data import Dataset, DataLoader
from typing import Set, Dict, List
import logging


class AudioFileDataset(Dataset):
    """音频文件数据集 - worker中完成CPU预处理"""
    
    def __init__(self, 
                 index_file: str, 
                 config: dict,
                 processed_ids: Set[str] = None):
        """
        Args:
            index_file: 索引文件路径
            config: 完整配置
            processed_ids: 已处理的file_id集合
        """
        self.config = config
        self.processed_ids = processed_ids or set()
        
        # 读取索引
        self.items = []
        with open(index_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('\t')
                if len(parts) != 2:
                    continue
                
                file_id, oss_path = parts
                if file_id not in self.processed_ids:
                    self.items.append({
                        'file_id': file_id,
                        'oss_path': oss_path
                    })
        
        # 延迟初始化（在worker进程中初始化）
        self.pipeline = None
        self.worker_id = None
    
    def _init_worker(self):
        """在worker进程中初始化Pipeline"""
        if self.pipeline is None:
            import torch
            from ..data.storage import MediaStorageManager
            from .pipeline import Pipeline
            from .stages import (
                DownloadStage,
                AudioFormatStage,
                CoarseVADStage,
                SegmentSplitStage,
                SegmentExpandAndUploadStage
            )
            
            # 获取worker信息
            worker_info = torch.utils.data.get_worker_info()
            self.worker_id = worker_info.id if worker_info else 0
            
            # 初始化存储管理器（每个worker独立）
            storage_manager = MediaStorageManager(
                input_config=self.config['input_storage'],
                output_config=self.config['output_storage']
            )
            
            # 创建Pipeline（每个worker独立加载VAD模型）
            stages = [
                DownloadStage(storage_manager),
                AudioFormatStage(
                    target_sr=self.config['media']['target_sample_rate'],
                    target_channels=self.config['media']['target_channels']
                ),
                CoarseVADStage(self.config['vad']),
                SegmentSplitStage(
                    max_duration=self.config['segment_expansion']['max_segment_duration'],
                    target_duration=self.config['segment_expansion']['segment_threshold']
                ),
                SegmentExpandAndUploadStage(
                    storage_manager,
                    self.config['segment_upload'],
                    max_upload_workers=self.config['segment_upload'].get('max_concurrent_parts', 4)
                )
            ]
            
            self.pipeline = Pipeline(stages)
            
            logger = logging.getLogger(__name__)
            logger.info(f"Worker-{self.worker_id} initialized pipeline")
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx) -> Dict:
        """处理单个文件，返回segments"""
        self._init_worker()
        
        from .data_structures import ProcessingItem
        
        file_dict = self.items[idx]
        item = ProcessingItem(**file_dict)
        
        # 在worker中运行Pipeline（CPU预处理+上传）
        results = []
        for seg_item in self.pipeline.process_one(item):
            # ✅ seg_item.audio_data已经在SegmentExpandAndUploadStage中清空
            # 只返回轻量级的metadata
            results.append({
                'segment_item': seg_item,  # audio_data已清空，pickle时很小
                'file_id': file_dict['file_id']
            })
        
        return results


def collate_fn(batch):
    """展平batch中的segments"""
    all_segments = []
    file_ids = set()
    
    for item_results in batch:
        for result in item_results:
            all_segments.append(result['segment_item'])
            file_ids.add(result['file_id'])
    
    return {
        'segments': all_segments,
        'processed_file_ids': file_ids
    }


def create_data_loader(index_file: str,
                       config: dict,
                       processed_ids: Set[str] = None,
                       num_workers: int = 8,
                       batch_size: int = 32) -> DataLoader:
    """
    创建多进程DataLoader
    
    Args:
        index_file: 索引文件
        config: 完整配置
        processed_ids: 已处理的ID
        num_workers: worker进程数（推荐4-8）
        batch_size: 批次大小（推荐32）
    """
    dataset = AudioFileDataset(index_file, config, processed_ids)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        persistent_workers=True if num_workers > 0 else False  # ✅ 保持worker存活，避免重复初始化
    )