"""Media data indexing and caching management"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import webdataset as wds
from loguru import logger


@dataclass
class MediaMetadata:
    """多媒体文件元数据"""
    file_id: str
    oss_path: str
    media_type: str  # 'audio' or 'video'
    size_bytes: int
    # 可选字段
    checksum: Optional[str] = None


class MediaIndexer(ABC):
    """Abstract base class for media indexing"""
    
    @abstractmethod
    def create_index(self, media_files: List[str]) -> pd.DataFrame:
        """Create index from media files"""
        pass
    
    @abstractmethod
    def load_index(self) -> pd.DataFrame:
        """Load existing index"""
        pass
    
    @abstractmethod
    def save_index(self, df: pd.DataFrame) -> None:
        """Save index to storage"""
        pass


class ParquetIndexer(MediaIndexer):
    """Parquet-based media indexer for large-scale data"""
    
    def __init__(self, index_path: str):
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        
    def create_index(self, media_files: List[Dict[str, Any]]) -> pd.DataFrame:
        """Create index from media file metadata"""
        metadata_list = []
        
        for file_info in media_files:
            # 只保留可以从文件系统直接获取的信息
            metadata = MediaMetadata(
                file_id=file_info['file_id'],
                oss_path=file_info['oss_path'], 
                media_type=file_info.get('media_type', 'audio'),
                size_bytes=file_info.get('size_bytes', 0),
                checksum=file_info.get('checksum')
            )
            metadata_list.append(metadata.__dict__)
            
        df = pd.DataFrame(metadata_list)
        self.save_index(df)
        return df
    
    def load_index(self) -> pd.DataFrame:
        """Load index from parquet file"""
        index_file = self.index_path / "media_index.parquet"
        if index_file.exists():
            return pd.read_parquet(index_file)
        return pd.DataFrame()
    
    def save_index(self, df: pd.DataFrame) -> None:
        """Save index to parquet file"""
        index_file = self.index_path / "media_index.parquet"
        df.to_parquet(index_file, index=False)
        logger.info(f"Saved {len(df)} media records to {index_file}")


class WebDatasetBuilder:
    """Build WebDataset shards from media files"""
    
    def __init__(self, shard_size: int = 100 * 1024 * 1024):  # 100MB
        self.shard_size = shard_size
        
    def create_shards(self, 
                     media_df: pd.DataFrame, 
                     output_dir: str,
                     pattern: str = "data-%06d.tar") -> None:
        """Create WebDataset shards from indexed media files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        with wds.ShardWriter(str(output_path / pattern), maxcount=1000) as shard_writer:
            for _, row in media_df.iterrows():
                # Create sample dictionary with only the fields that exist in the simplified MediaMetadata
                sample = {
                    "__key__": row['file_id'],
                    "oss_path": row['oss_path'],
                    "media_type": row['media_type'],
                    "size_bytes": row['size_bytes']
                }
                
                # Add optional fields if available
                if 'checksum' in row and pd.notna(row['checksum']):
                    sample['checksum'] = row['checksum']
                
                # Additional fields will be added during processing stages
                shard_writer.write(sample)
                
        logger.info(f"Created WebDataset shards in {output_path}")


class MediaCache:
    """本地多媒体文件缓存"""
    
    def __init__(self, cache_dir: str, max_size_gb: float = 100.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self.cache_index_file = self.cache_dir / "cache_index.json"
        
    def _load_cache_index(self) -> Dict[str, Dict[str, Any]]:
        """加载缓存索引"""
        if self.cache_index_file.exists():
            try:
                with open(self.cache_index_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"加载缓存索引失败: {e}")
        return {}
    
    def _save_cache_index(self, index: Dict[str, Dict[str, Any]]) -> None:
        """保存缓存索引"""
        try:
            with open(self.cache_index_file, 'w') as f:
                json.dump(index, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"保存缓存索引失败: {e}")
    
    def _get_cache_size(self) -> int:
        """获取缓存总大小"""
        total_size = 0
        for file_path in self.cache_dir.glob("*"):
            if file_path.is_file() and file_path != self.cache_index_file:
                total_size += file_path.stat().st_size
        return total_size
    
    def _evict_if_needed(self) -> None:
        """Evict old files if cache is full"""
        current_size = self._get_cache_size()
        if current_size > self.max_size_bytes:
            # Simple LRU: sort by modification time
            files = []
            for file_path in self.cache_dir.glob("*"):
                if file_path.is_file() and file_path != self.cache_index_file:
                    files.append((file_path.stat().st_mtime, file_path))
            
            files.sort()  # Oldest first
            
            # Remove oldest files until under limit
            for mtime, file_path in files:
                file_path.unlink()
                current_size -= file_path.stat().st_size
                if current_size <= self.max_size_bytes * 0.8:  # Leave 20% headroom
                    break
                    
            logger.info(f"Evicted cache files, new size: {current_size / 1024 / 1024:.1f} MB")
    
    def get(self, file_id: str, media_type: str = 'audio') -> Optional[Path]:
        """Get cached file path"""
        # Determine file extension based on media type
        extension = 'wav' if media_type == 'audio' else 'mp4'
        cache_file = self.cache_dir / f"{file_id}.{extension}"
        
        if cache_file.exists():
            # Update access time
            cache_file.touch()
            return cache_file
        return None
    
    def put(self, file_id: str, data: bytes, media_type: str = 'audio') -> Path:
        """Cache file data"""
        self._evict_if_needed()
        
        # Determine file extension based on media type
        extension = 'wav' if media_type == 'audio' else 'mp4'
        cache_file = self.cache_dir / f"{file_id}.{extension}"
        
        with open(cache_file, 'wb') as f:
            f.write(data)
            
        # Update index
        index = self._load_cache_index()
        index[file_id] = {
            'path': str(cache_file),
            'size': len(data),
            'media_type': media_type,
            'timestamp': pd.Timestamp.now()
        }
        self._save_cache_index(index)
        
        return cache_file
    
    def clear(self) -> None:
        """Clear all cache"""
        for file_path in self.cache_dir.glob("*"):
            if file_path.is_file():
                file_path.unlink()
        
        if self.cache_index_file.exists():
            self.cache_index_file.unlink()


class MediaDataLoader:
    """多媒体数据加载器 - 统一索引和缓存接口"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.index_path = config['index_path']
        self.cache_dir = config['cache_dir']
        self.cache_size_gb = config.get('cache_size_gb', 100.0)
        
        # Initialize components
        self.indexer = ParquetIndexer(self.index_path)
        self.cache = MediaCache(self.cache_dir, self.cache_size_gb)
        
    def load_index(self) -> pd.DataFrame:
        """加载多媒体索引"""
        return self.indexer.load_index()
    
    def create_index(self, media_files: List[Dict[str, Any]]) -> pd.DataFrame:
        """创建多媒体索引"""
        return self.indexer.create_index(media_files)
    
    def get_cached_media(self, file_id: str, media_type: str = 'audio') -> Optional[Path]:
        """获取缓存的多媒体文件"""
        return self.cache.get(file_id, media_type)
    
    def cache_media(self, file_id: str, media_data: bytes, media_type: str = 'audio') -> Path:
        """缓存多媒体数据"""
        return self.cache.put(file_id, media_data, media_type)
    
    def clear_cache(self) -> None:
        """清空缓存"""
        self.cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        index = self.cache._load_cache_index()
        total_size = self.cache._get_cache_size()
        
        return {
            'num_files': len(index),
            'total_size_mb': total_size / 1024 / 1024,
            'total_size_gb': total_size / 1024 / 1024 / 1024,
            'max_size_gb': self.cache_size_gb,
            'utilization': total_size / self.cache.max_size_bytes
        }
    
    def filter_by_media_type(self, media_type: str) -> pd.DataFrame:
        """根据媒体类型过滤索引"""
        df = self.load_index()
        if df.empty:
            return df
        
        return df[df['media_type'] == media_type]
    
    def get_audio_files(self) -> pd.DataFrame:
        """获取所有音频文件索引"""
        return self.filter_by_media_type('audio')
    
    def get_video_files(self) -> pd.DataFrame:
        """获取所有视频文件索引"""
        return self.filter_by_media_type('video')


# 向后兼容的别名
DataLoader = MediaDataLoader