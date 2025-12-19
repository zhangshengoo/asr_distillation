"""主程序 - 批处理版本"""
import yaml
import logging
import time
from typing import List, Tuple, Dict, Set
from pathlib import Path

from tools.storage import MediaStorageManager
from tools.checkpoint import CheckpointManager
from tools.metadata_writer import MetadataWriter
from tools.data_structures import ProcessingItem, SegmentItem
from stages.download import DownloadStage
from stages.audio_format import AudioFormatStage
from stages.vad import CoarseVADStage
from stages.segment_split import SegmentSplitStage
from stages.segment_upload import SegmentExpandAndUploadStage


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def load_config(config_file: str) -> dict:
    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def init_stages(config: dict, storage_manager) -> dict:
    """初始化所有stage"""
    return {
        'download': DownloadStage(
            storage_manager=storage_manager,
            max_workers=config['pipeline'].get('stage_workers', {}).get('audio_download', 8)
        ),
        'audio_format': AudioFormatStage(
            target_sr=config['media']['target_sample_rate'],
            target_channels=config['media']['target_channels'],
            max_workers=config['media'].get('ffmpeg_num_workers', 8)
        ),
        'coarse_vad': CoarseVADStage(
            vad_config=config['vad'],
            num_workers=config['vad'].get('parallel_workers', 4)
        ),
        'segment_split': SegmentSplitStage(
            max_duration=config['segment_expansion']['max_segment_duration'],
            target_duration=config['segment_expansion']['segment_threshold'],
            num_workers=config['pipeline'].get('stage_workers', {}).get('segment_expansion', 4)
        ),
        'segment_expand': SegmentExpandAndUploadStage(
            storage_manager=storage_manager,
            config=config['segment_upload'],
            max_upload_workers=config['segment_upload'].get('max_concurrent_parts', 4)
        )
    }


def process_one_batch(file_batch: List[Tuple[str, str]], 
                     stages: dict) -> List[SegmentItem]:
    """处理一批文件，返回所有segments"""
    
    items = [ProcessingItem(file_id=fid, oss_path=path) 
             for fid, path in file_batch]
    
    items = stages['download'].process_batch(items)
    
    items = stages['audio_format'].process_batch(items)
    for item in items:
        item.audio_bytes = None
    
    items = stages['coarse_vad'].process_batch(items)
    
    items = stages['segment_split'].process_batch(items)
    
    segment_batches = stages['segment_expand'].process_batch(items)
    
    all_segments = []
    for seg_list in segment_batches:
        all_segments.extend(seg_list)
    
    for item in items:
        item.clear_audio_data()
    
    return all_segments


def group_segments_by_file(segments: List[SegmentItem]) -> Dict[str, List[SegmentItem]]:
    """按parent_file_id分组segments"""
    file_to_segs = {}
    for seg in segments:
        parent_id = seg.parent_file_id
        if parent_id not in file_to_segs:
            file_to_segs[parent_id] = []
        file_to_segs[parent_id].append(seg)
    return file_to_segs


def get_successful_file_ids(segments: List[SegmentItem]) -> Set[str]:
    """获取所有segments都成功的file_id集合"""
    file_to_segs = group_segments_by_file(segments)
    
    successful_files = set()
    for file_id, segs in file_to_segs.items():
        if all(seg.status == "success" for seg in segs):
            successful_files.add(file_id)
    
    return successful_files


def main(config_file: str):
    """主函数"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    config = load_config(config_file)
    
    storage_manager = MediaStorageManager(
        input_config=config['data']['input_storage'],
        output_config=config['data']['output_storage']
    )
    
    logger.info("Fetching file list...")
    file_list = storage_manager.list_audio_files()
    logger.info(f"Found {len(file_list)} files")
    
    checkpoint_mgr = CheckpointManager(
        config.get('monitoring', {}).get('checkpoint_dir', './checkpoints') + 
        '/preprocessing_checkpoint.json'
    )
    processed_ids = checkpoint_mgr.load_processed_ids()
    
    unprocessed_files = [(fid, path) for fid, path in file_list 
                        if fid not in processed_ids]
    logger.info(f"{len(unprocessed_files)} files to process")
    
    if not unprocessed_files:
        logger.info("No files to process")
        return
    
    logger.info("Initializing stages...")
    stages = init_stages(config, storage_manager)
    
    meta_writer = MetadataWriter(
        storage_manager,
        metadata_prefix=config['segment_upload'].get('metadata_prefix', 'metadata/'),
        local_buffer_size=config['segment_upload'].get('segment_metadata_batch_size', 1000)
    )
    
    batch_size = config['pipeline'].get('batch_size', 64)
    checkpoint_interval = config['pipeline'].get('checkpoint_interval', 100)
    
    stats = {
        'success_segments': 0,
        'failed_segments': 0,
        'processed_files': 0,
        'failed_files': 0,
        'start_time': time.time()
    }
    
    total_batches = (len(unprocessed_files) + batch_size - 1) // batch_size
    logger.info(f"Starting: {len(unprocessed_files)} files, {total_batches} batches, batch_size={batch_size}")
    
    try:
        for batch_idx in range(0, len(unprocessed_files), batch_size):
            batch_num = batch_idx // batch_size + 1
            file_batch = unprocessed_files[batch_idx:batch_idx + batch_size]
            
            segments = process_one_batch(file_batch, stages)
            
            for seg in segments:
                meta_writer.write(seg.to_meta_dict())
                if seg.status == "success":
                    stats['success_segments'] += 1
                else:
                    stats['failed_segments'] += 1
            
            successful_file_ids = get_successful_file_ids(segments)
            failed_file_ids = set(fid for fid, _ in file_batch) - successful_file_ids
            
            processed_ids.update(successful_file_ids)
            stats['processed_files'] += len(successful_file_ids)
            stats['failed_files'] += len(failed_file_ids)
            
            logger.info(f"Batch {batch_num}/{total_batches}: "
                       f"{len(file_batch)} files -> "
                       f"{len(segments)} segments "
                       f"(success: {sum(1 for s in segments if s.status == 'success')}, "
                       f"failed: {sum(1 for s in segments if s.status == 'failed')}), "
                       f"complete files: {len(successful_file_ids)}, "
                       f"failed files: {len(failed_file_ids)}")
            
            if stats['processed_files'] % checkpoint_interval < batch_size:
                checkpoint_mgr.save_processed_ids(processed_ids)
                logger.info(f"Checkpoint: {stats['processed_files']} files")
            
            if batch_num % 10 == 0:
                elapsed = time.time() - stats['start_time']
                rate = stats['processed_files'] / elapsed if elapsed > 0 else 0
                remaining = len(unprocessed_files) - (batch_idx + len(file_batch))
                eta = remaining / rate / 60 if rate > 0 else 0
                logger.info(f"Progress: {stats['processed_files']}/{len(unprocessed_files)} "
                           f"({rate:.1f} files/s, ETA: {eta:.1f}min)")
    
    except KeyboardInterrupt:
        logger.warning("Interrupted, saving checkpoint...")
        checkpoint_mgr.save_processed_ids(processed_ids)
        meta_writer.close()
        return
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        checkpoint_mgr.save_processed_ids(processed_ids)
        meta_writer.close()
        raise
    
    finally:
        checkpoint_mgr.save_processed_ids(processed_ids)
        meta_writer.close()
        
        total_time = time.time() - stats['start_time']
        logger.info(f"\nCompleted in {total_time/60:.1f} min")
        logger.info(f"Files: {stats['processed_files']} success, {stats['failed_files']} failed")
        logger.info(f"Segments: {stats['success_segments']} success, {stats['failed_segments']} failed")
        if stats['processed_files'] > 0:
            logger.info(f"Rate: {stats['processed_files']/(total_time/60):.1f} files/min")


if __name__ == '__main__':
    import sys
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'config.yaml'
    main(config_file)