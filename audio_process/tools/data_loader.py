"""DataLoader实现 - 支持多进程并行"""
from torch.utils.data import Dataset, DataLoader
from typing import Set, Dict, List, Tuple
import logging


class AudioFileDataset(Dataset):
    """音频文件数据集 - worker中完成CPU预处理"""
    
    def __init__(self, 
                 file_list: List[Tuple[str, str]], 
                 config: dict):
        """
        Args:
            file_list: [(file_id, oss_path), ...] 文件列表
            config: 完整配置
        """
        self.config = config
        self.items = [{'file_id': fid, 'oss_path': path} for fid, path in file_list]
        self.pipeline = None
        self.worker_id = None
    
    def _init_worker(self):
        """在worker进程中初始化Pipeline"""
        if self.pipeline is None:
            import torch
            from ..storage import MediaStorageManager
            from .pipeline import Pipeline
            from ..stages.download import DownloadStage
            from ..stages.audio_format import AudioFormatStage
            from ..stages.vad import CoarseVADStage
            from ..stages.segment_split import SegmentSplitStage
            from ..stages.segment_upload import SegmentExpandAndUploadStage
            
            worker_info = torch.utils.data.get_worker_info()
            self.worker_id = worker_info.id if worker_info else 0
            
            storage_manager = MediaStorageManager(
                input_config=self.config['data']['input_storage'],
                output_config=self.config['data']['output_storage']
            )
            
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
    
    def __getitem__(self, idx) -> List[Dict]:
        """处理单个文件，返回segments"""
        self._init_worker()
        
        from ..data_structures import ProcessingItem
        
        file_dict = self.items[idx]
        item = ProcessingItem(**file_dict)
        
        results = []
        for seg_item in self.pipeline.process_one(item):
            results.append({
                'segment_item': seg_item,
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


def create_data_loader(file_list: List[Tuple[str, str]],
                       config: dict,
                       num_workers: int = 8,
                       batch_size: int = 32) -> DataLoader:
    """
    创建多进程DataLoader
    
    Args:
        file_list: [(file_id, oss_path), ...] 文件列表
        config: 完整配置
        num_workers: worker进程数（推荐4-8）
        batch_size: 批次大小（推荐32）
    """
    dataset = AudioFileDataset(file_list, config)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        persistent_workers=True if num_workers > 0 else False
    )