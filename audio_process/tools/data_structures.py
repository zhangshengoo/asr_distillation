"""数据结构定义"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import numpy as np
import time


@dataclass
class ProcessingItem:
    """流水线数据单元"""
    # 身份
    file_id: str
    oss_path: str
    
    # 音频数据
    audio_bytes: Optional[bytes] = None
    audio_data: Optional[np.ndarray] = None
    sample_rate: Optional[int] = None
    duration: Optional[float] = None
    
    # VAD结果
    segments: List[Dict] = field(default_factory=list)
    
    # 状态
    status: str = "pending"  # pending/success/failed
    error: Optional[str] = None
    error_stage: Optional[str] = None
    
    # 元数据
    metadata: Dict = field(default_factory=dict)
    
    def mark_failed(self, stage_name: str, error: Exception):
        """标记失败"""
        self.status = "failed"
        self.error = str(error)[:200]  # 精炼错误信息
        self.error_stage = stage_name
    
    def mark_success(self):
        """标记成功"""
        self.status = "success"
    
    def clear_audio_data(self):
        """清理音频数据（节省内存）"""
        self.audio_bytes = None
        self.audio_data = None


@dataclass
class SegmentItem(ProcessingItem):
    """Segment级别的数据单元"""
    parent_file_id: str = None
    segment_index: int = None
    start_time: float = None
    end_time: float = None
    segment_duration: float = None
    segment_oss_path: Optional[str] = None
    
    def to_meta_dict(self) -> Dict:
        """转为输出格式"""
        if self.status == "failed":
            return {
                "status": "failed",
                "segment_id": self.file_id,
                "original_oss_path": self.oss_path,
                "error": self.error,
                "error_stage": self.error_stage
            }
        
        return {
            "status": "success",
            "segment_id": self.file_id,
            "original_oss_path": self.oss_path,
            "segment_oss_path": self.segment_oss_path,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.segment_duration,
            "original_file_id": self.parent_file_id,
            "timestamp": time.time()
        }