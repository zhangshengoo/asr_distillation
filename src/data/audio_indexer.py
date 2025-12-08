"""Audio data indexing and caching management"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import webdataset as wds
from loguru import logger


@dataclass
class AudioMetadata:
    """音频文件元数据"""
    file_id: str
    oss_path: str
    duration: float
    sample_rate: int
    size_bytes: int
    format: str
    checksum: Optional[str] = None


class AudioIndexer(ABC):
    """Abstract base class for audio indexing"""
    
    @abstractmethod
    def create_index(self, audio_files: List[str]) -> pd.DataFrame:
        """Create index from audio files"""
        pass
    
    @abstractmethod
    def load_index(self) -> pd.DataFrame:
        """Load existing index"""
        pass
    
    @abstractmethod
    def save_index(self, df: pd.DataFrame) -> None:
        """Save index to storage"""
        pass


class ParquetIndexer(AudioIndexer):
    """Parquet-based audio indexer for large-scale data"""
    
    def __init__(self, index_path: str):
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        
    def create_index(self, audio_files: List[Dict[str, Any]]) -> pd.DataFrame:
        """Create index from audio file metadata"""
        metadata_list = []
        
        for file_info in audio_files:
            metadata = AudioMetadata(
                file_id=file_info['file_id'],
                s3_path=file_info['s3_path'], 
                duration=file_info['duration'],
                sample_rate=file_info.get('sample_rate', 16000),
                size_bytes=file_info.get('size_bytes', 0),
                format=file_info.get('format', 'wav')
            )
            metadata_list.append(metadata.__dict__)
            
        df = pd.DataFrame(metadata_list)
        self.save_index(df)
        return df
    
    def load_index(self) -> pd.DataFrame:
        """Load index from parquet file"""
        index_file = self.index_path / "audio_index.parquet"
        if index_file.exists():
            return pd.read_parquet(index_file)
        return pd.DataFrame()
    
    def save_index(self, df: pd.DataFrame) -> None:
        """Save index to parquet file"""
        index_file = self.index_path / "audio_index.parquet"
        df.to_parquet(index_file, index=False)
        logger.info(f"Saved {len(df)} audio records to {index_file}")


class WebDatasetBuilder:
    """Build WebDataset shards from audio files"""
    
    def __init__(self, shard_size: int = 100 * 1024 * 1024):  # 100MB
        self.shard_size = shard_size
        
    def create_shards(self, 
                     audio_df: pd.DataFrame, 
                     output_dir: str,
                     pattern: str = "data-%06d.tar") -> None:
        """Create WebDataset shards from indexed audio files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        with wds.ShardWriter(str(output_path / pattern), maxcount=1000) as shard_writer:
            for _, row in audio_df.iterrows():
                # Create sample dictionary
                sample = {
                    "__key__": row['file_id'],
                    "s3_path": row['s3_path'],
                    "duration": row['duration'],
                    "sample_rate": row['sample_rate'],
                    "format": row['format']
                }
                
                shard_writer.write(sample)
                
        logger.info(f"Created WebDataset shards in {output_path}")


class AudioCache:
    """Audio data caching management"""
    
    def __init__(self, 
                 cache_dir: str,
                 max_size_gb: float = 100.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self.cache_index_file = self.cache_dir / "cache_index.pkl"
        
    def _load_cache_index(self) -> Dict[str, Dict]:
        """Load cache index"""
        if self.cache_index_file.exists():
            with open(self.cache_index_file, 'rb') as f:
                return pickle.load(f)
        return {}
    
    def _save_cache_index(self, index: Dict[str, Dict]) -> None:
        """Save cache index"""
        with open(self.cache_index_file, 'wb') as f:
            pickle.dump(index, f)
    
    def _get_cache_size(self) -> int:
        """Get current cache size in bytes"""
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
    
    def get(self, file_id: str) -> Optional[Path]:
        """Get cached file path"""
        cache_file = self.cache_dir / f"{file_id}.wav"
        if cache_file.exists():
            # Update access time
            cache_file.touch()
            return cache_file
        return None
    
    def put(self, file_id: str, data: bytes) -> Path:
        """Cache file data"""
        self._evict_if_needed()
        
        cache_file = self.cache_dir / f"{file_id}.wav"
        with open(cache_file, 'wb') as f:
            f.write(data)
            
        # Update index
        index = self._load_cache_index()
        index[file_id] = {
            'path': str(cache_file),
            'size': len(data),
            'timestamp': pd.Timestamp.now()
        }
        self._save_cache_index(index)
        
        return cache_file
    
    def clear(self) -> None:
        """Clear all cache"""
        for file_path in self.cache_dir.glob("*"):
            if file_path.is_file() and file_path != self.cache_index_file:
                file_path.unlink()
        
        if self.cache_index_file.exists():
            self.cache_index_file.unlink()
            
        logger.info("Cache cleared")


class DataLoader:
    """Main data loading interface"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.indexer = ParquetIndexer(config['index_path'])
        self.cache = AudioCache(
            config['cache_dir'], 
            config.get('cache_size_gb', 100)
        )
        self.wds_builder = WebDatasetBuilder(
            config.get('shard_size_mb', 100) * 1024 * 1024
        )
        
    def build_index(self, audio_files: List[Dict[str, Any]]) -> pd.DataFrame:
        """Build audio index from file list"""
        logger.info(f"Building index for {len(audio_files)} audio files")
        return self.indexer.create_index(audio_files)
    
    def load_index(self) -> pd.DataFrame:
        """Load existing audio index"""
        return self.indexer.load_index()
    
    def create_webdataset_shards(self, audio_df: pd.DataFrame) -> None:
        """Create WebDataset shards from indexed data"""
        self.wds_builder.create_shards(
            audio_df, 
            self.config['webdataset_output_dir']
        )
    
    def get_cached_audio(self, file_id: str) -> Optional[Path]:
        """Get audio from cache"""
        return self.cache.get(file_id)
    
    def cache_audio(self, file_id: str, data: bytes) -> Path:
        """Cache audio data"""
        return self.cache.put(file_id, data)