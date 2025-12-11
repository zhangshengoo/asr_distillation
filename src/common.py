"""Common classes and utilities for ASR Distillation Framework"""

from dataclasses import dataclass, field
from typing import Dict, List, Any


@dataclass
class DataBatch:
    """Data batch for pipeline processing
    
    统一的DataBatch类，用于在不同流水线组件间传递数据
    设计原则：
    1. 每个item包含自己的metadata，表示原始文件的属性
    2. batch.metadata用于批次级别的状态信息
    3. 清晰分离批次状态信息和文件元数据信息
    """
    batch_id: str
    items: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    
    def __post_init__(self):
        """初始化默认metadata字典"""
        if not self.metadata:
            self.metadata = {}