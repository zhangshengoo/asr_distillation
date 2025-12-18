"""主程序入口 - 简化版"""
import yaml
import logging
from tqdm import tqdm

from .data_loader import create_data_loader
from .metadata_writer import MetadataWriter
from .checkpoint import CheckpointManager
from ..data.storage import MediaStorageManager


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [%(processName)s] - %(levelname)s - %(message)s'
    )


def load_config(config_file: str) -> dict:
    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main(config_file: str):
    """主函数 - 简化版"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # 加载配置
    config = load_config(config_file)
    logger.info(f"Loaded config from {config_file}")
    
    # Storage manager
    storage_manager = MediaStorageManager(
        input_config=config['input_storage'],
        output_config=config['output_storage']
    )
    
    # Checkpoint管理
    checkpoint_mgr = CheckpointManager(
        config['monitoring']['checkpoint_dir'] + '/preprocessing_checkpoint.json'
    )
    processed_ids = checkpoint_mgr.load_processed_ids()
    logger.info(f"Loaded {len(processed_ids)} processed file IDs")
    
    # 创建多进程DataLoader
    num_workers = config.get('pipeline', {}).get('num_cpu_workers', 8)
    data_loader = create_data_loader(
        config['data']['index_path'],
        config=config,
        processed_ids=processed_ids,
        num_workers=num_workers,
        batch_size=1
    )
    logger.info(f"Created data loader with {num_workers} workers")
    
    # Metadata writer
    meta_writer = MetadataWriter(
        storage_manager,
        metadata_prefix=config['segment_upload']['metadata_prefix'],
        local_buffer_size=config['segment_upload'].get('segment_metadata_batch_size', 1000)
    )
    
    # 处理
    stats = {'success': 0, 'failed': 0, 'processed_files': 0}
    
    try:
        for batch_dict in tqdm(data_loader, desc="Processing files"):
            segments = batch_dict['segments']
            file_ids = batch_dict['processed_file_ids']
            
            # 写metadata
            for seg_item in segments:
                meta_writer.write(seg_item.to_meta_dict())
                
                if seg_item.status == "success":
                    stats['success'] += 1
                else:
                    stats['failed'] += 1
            
            # ✅ 同步更新checkpoint
            processed_ids.update(file_ids)
            stats['processed_files'] += len(file_ids)
            
            # 每100个文件保存一次
            if stats['processed_files'] % 100 == 0:
                checkpoint_mgr.save_processed_ids(processed_ids)
                logger.info(f"Checkpoint saved: {stats['processed_files']} files processed")
    
    finally:
        # 最终保存
        checkpoint_mgr.save_processed_ids(processed_ids)
        meta_writer.close()
        
        logger.info(f"Processing completed!")
        logger.info(f"Files: {stats['processed_files']}, "
                   f"Segments - Success: {stats['success']}, Failed: {stats['failed']}")


if __name__ == '__main__':
    import sys
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'config.yaml'
    main(config_file)