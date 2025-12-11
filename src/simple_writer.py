"""简单的结果写入器 - 同步写入结果"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any

from src.common import BatchData, FileResultItem


class SimpleResultWriter:
    """简单的同步结果写入器
    
    功能：
    - 同步写入结果到JSONL文件
    - 支持文件大小限制（自动切分）
    - 统计写入数量
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 包含 'writer' 配置的字典
        """
        writer_config = config.get('writer', {})
        
        self.output_dir = Path(config.get('output_dir', './results'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_file_size_mb = writer_config.get('max_file_size_mb', 100)
        self.output_format = writer_config.get('output_format', 'jsonl')
        
        # 当前文件状态
        self.current_file = None
        self.current_file_size = 0
        self.file_index = 0
        
        # 统计信息
        self.total_written = 0
        self.files_created = 0
        
        # 创建第一个文件
        self._create_new_file()
    
    def _create_new_file(self) -> None:
        """创建新的输出文件"""
        timestamp = int(time.time())
        filename = f"results_{timestamp}_{self.file_index}.{self.output_format}"
        
        if self.current_file:
            self.current_file.close()
        
        self.current_file = open(self.output_dir / filename, 'w', encoding='utf-8')
        self.current_file_size = 0
        self.file_index += 1
        self.files_created += 1
    
    def _check_file_size(self) -> None:
        """检查文件大小，超过限制则创建新文件"""
        max_size_bytes = self.max_file_size_mb * 1024 * 1024
        if self.current_file_size >= max_size_bytes:
            self._create_new_file()
    
    def write(self, batch: BatchData[FileResultItem]) -> None:
        """写入一个批次的结果
        
        Args:
            batch: 包含FileResultItem的批次
        """
        for item in batch.items:
            if not isinstance(item, FileResultItem):
                continue
            
            # 构建输出字典
            result = {
                'file_id': item.file_id,
                'transcription': item.transcription,
                'segments': item.segments,
                'stats': item.stats,
                'metadata': item.metadata
            }
            
            # 写入JSONL
            line = json.dumps(result, ensure_ascii=False) + '\n'
            self.current_file.write(line)
            self.current_file.flush()
            
            # 更新文件大小
            self.current_file_size += len(line.encode('utf-8'))
            self.total_written += 1
            
            # 检查是否需要切分文件
            self._check_file_size()
    
    def close(self) -> None:
        """关闭写入器"""
        if self.current_file:
            self.current_file.close()
            self.current_file = None
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'total_written': self.total_written,
            'files_created': self.files_created,
            'output_dir': str(self.output_dir)
        }