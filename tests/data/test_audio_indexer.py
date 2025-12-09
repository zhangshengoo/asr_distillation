"""
AudioIndexer 简单单元测试

测试音频索引功能
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch
from pathlib import Path

from src.data.audio_indexer import ParquetIndexer, AudioMetadata


class TestParquetIndexer:
    """ParquetIndexer简单测试"""

    def test_init(self):
        """测试初始化"""
        with patch('pathlib.Path.mkdir'):
            indexer = ParquetIndexer("./test_index")
            assert indexer.index_path.name == "test_index"

    def test_create_index(self):
        """测试创建索引"""
        # 模拟音频文件信息
        audio_files = [
            {
                'file_id': 'test_1',
                's3_path': 's3://bucket/test_1.wav',
                'duration': 5.0,
                'sample_rate': 16000,
                'size_bytes': 80000,
                'format': 'wav'
            },
            {
                'file_id': 'test_2',
                's3_path': 's3://bucket/test_2.wav',
                'duration': 3.0,
                'sample_rate': 16000,
                'size_bytes': 48000,
                'format': 'wav'
            }
        ]
        
        with patch('pathlib.Path.mkdir'), \
             patch('pandas.DataFrame.to_parquet') as mock_save:
            
            indexer = ParquetIndexer("./test_index")
            df = indexer.create_index(audio_files)
            
            # 验证结果
            assert len(df) == 2
            assert 'file_id' in df.columns
            assert 'duration' in df.columns
            assert df.iloc[0]['file_id'] == 'test_1'
            assert df.iloc[1]['duration'] == 3.0

    def test_load_empty_index(self):
        """测试加载空索引"""
        with patch('pathlib.Path.mkdir'), \
             patch('pathlib.Path.exists', return_value=False):
            
            indexer = ParquetIndexer("./test_index")
            df = indexer.load_index()
            
            # 验证空索引
            assert df.empty

    def test_audio_metadata(self):
        """测试音频元数据"""
        metadata = AudioMetadata(
            file_id='test_1',
            oss_path='oss://bucket/test_1.wav',
            duration=5.0,
            sample_rate=16000,
            size_bytes=80000,
            format='wav'
        )
        
        assert metadata.file_id == 'test_1'
        assert metadata.duration == 5.0
        assert metadata.sample_rate == 16000