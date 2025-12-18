"""Checkpoint管理器"""
import json
from pathlib import Path
from typing import Set


class CheckpointManager:
    """管理处理进度的checkpoint"""
    
    def __init__(self, checkpoint_file: str):
        self.checkpoint_file = Path(checkpoint_file)
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    
    def load_processed_ids(self) -> Set[str]:
        """加载已处理的file_id"""
        if not self.checkpoint_file.exists():
            return set()
        
        try:
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return set(data.get('processed_ids', []))
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            return set()
    
    def save_processed_ids(self, processed_ids: Set[str]):
        """保存已处理的file_id"""
        try:
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'processed_ids': list(processed_ids),
                    'total': len(processed_ids)
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")