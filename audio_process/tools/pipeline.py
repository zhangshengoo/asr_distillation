"""Pipeline实现"""
from typing import List, Iterator
from .stages.base import Stage
from .data_structures import ProcessingItem


class Pipeline:
    """处理流水线"""
    
    def __init__(self, stages: List[Stage]):
        self.stages = stages
    
    def process_one(self, item: ProcessingItem) -> Iterator[ProcessingItem]:
        """
        处理单个item，支持1→N展开
        
        Args:
            item: 输入item
            
        Yields:
            处理后的items（可能多个）
        """
        current_items = [item]
        
        for stage in self.stages:
            next_items = []
            
            for it in current_items:
                result = stage(it)
                
                # 处理展开情况
                if isinstance(result, list):
                    next_items.extend(result)
                else:
                    next_items.append(result)
            
            current_items = next_items
            
            # 优化：如果所有items都失败了，提前终止
            if all(it.status == "failed" for it in current_items):
                break
        
        # 返回所有最终结果
        yield from current_items
    
    def process_stream(self, items: Iterator[ProcessingItem]) -> Iterator[ProcessingItem]:
        """
        流式处理多个items
        
        Args:
            items: 输入item流
            
        Yields:
            处理后的items
        """
        for item in items:
            yield from self.process_one(item)