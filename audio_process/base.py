"""Stage基础接口"""
from abc import ABC, abstractmethod
from typing import List
from ..data_structures import ProcessingItem


class Stage(ABC):
    """处理阶段基类"""
    
    @abstractmethod
    def name(self) -> str:
        """阶段名称"""
        pass
    
    @abstractmethod
    def process(self, item: ProcessingItem) -> ProcessingItem:
        """
        处理单个item
        注意：应该返回修改后的item，不要in-place修改
        """
        pass
    
    def __call__(self, item: ProcessingItem) -> ProcessingItem:
        """统一的调用入口"""
        if item.status == "failed":
            return item  # 已失败，跳过
        
        try:
            return self.process(item)
        except Exception as e:
            item.mark_failed(self.name(), e)
            return item


class ExpandStage(Stage):
    """支持1→N展开的特殊Stage"""
    
    @abstractmethod
    def expand(self, item: ProcessingItem) -> List[ProcessingItem]:
        """将一个item展开为多个"""
        pass
    
    def process(self, item: ProcessingItem):
        """重写process，返回List"""
        return self.expand(item)