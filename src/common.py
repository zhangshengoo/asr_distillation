"""Common classes and utilities for ASR Distillation Framework"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Any, Dict, Union, Generic, TypeVar, Callable
from abc import ABC
import numpy as np
import time

# --- 1. Item States (The "What") ---

@dataclass
class PipelineItem(ABC):
    """Base class for all items in the pipeline."""
    file_id: str
    # Note: metadata has no default here to allow subclasses to add fields without defaults
    # Subclasses should add: metadata: Dict[str, Any] = field(default_factory=dict) as their last field
    
    @property
    def stage_name(self) -> str:
        return self.__class__.__name__

@dataclass
class SourceItem(PipelineItem):
    """Initial state: Reference to a file in storage."""
    oss_path: str
    format: str = "wav"
    duration: float = 0.0

@dataclass
class RawAudioItem(SourceItem):
    """State after download: Contains raw bytes."""
    audio_bytes: bytes = b''  # Default empty bytes to satisfy dataclass inheritance rules
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TensorItem(SourceItem):
    """State after decoding: Contains waveform tensor."""
    # Using np.ndarray for Ray Zero-Copy efficiency
    waveform: np.ndarray 
    sample_rate: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Note: audio_bytes is DROPPED here to save memory

@dataclass
class SegmentItem(PipelineItem):
    """State after VAD/Segmentation: Represents a slice of the original file."""
    # Link back to parent usually via ID convention or explicit field
    parent_file_id: str 
    segment_id: str
    segment_index: int
    start_time: float
    end_time: float
    # Slice of the original tensor (NumPy)
    waveform: np.ndarray 
    original_duration: float
    # Inherited source metadata
    oss_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class InferenceItem(SegmentItem):
    """State after Inference: Contains text."""
    transcription: str
    confidence: float
    inference_timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FileResultItem(PipelineItem):
    """Final aggregated state: Ready for writing."""
    transcription: str  # Aggregated text
    segments: List[Dict[str, Any]] # Detailed breakdown
    stats: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    output: Optional[Dict[str, Any]] = None # Formatted output for writer

# --- 2. The Container (The "Batch") ---

T = TypeVar("T", bound=PipelineItem)
U = TypeVar("U", bound=PipelineItem)

@dataclass
class BatchData(Generic[T]):
    """
    A strongly-typed container for a batch of items.
    
    Features:
    - Generic typing: BatchData[SourceItem], BatchData[TensorItem], etc.
    - transformations: functional methods to map processed items to new batch.
    """
    batch_id: str
    items: List[T]
    metadata: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    
    def map(self, func: Callable[[T], U], new_batch_id: str = None) -> 'BatchData[U]':
        """Apply a function to all items and return a new batch."""
        new_items = []
        for item in self.items:
            try:
                result = func(item)
                if result is not None:
                    new_items.append(result)
            except Exception as e:
                # Log error but don't crash pipeline? 
                # Ideally, we put an ErrorItem in the list, but for now we skip.
                # In production, we'd handle errors more robustly.
                print(f"Error processing item {item.file_id}: {e}")
                
        return BatchData(
            batch_id=new_batch_id or self.batch_id, 
            items=new_items, 
            metadata=self.metadata.copy()
        )
        
    def flat_map(self, func: Callable[[T], List[U]], new_batch_id: str = None) -> 'BatchData[U]':
        """One-to-Many transformation (e.g., Segmentation)."""
        new_items = []
        for item in self.items:
            try:
                results = func(item)
                new_items.extend(results)
            except Exception as e:
                print(f"Error processing item {item.file_id}: {e}")
                
        return BatchData(
            batch_id=new_batch_id or self.batch_id, 
            items=new_items, 
            metadata={**self.metadata, 'expanded': True}
        )
    
    def transform(self, func: Callable[[List[T]], List[U]], new_batch_id: str = None) -> 'BatchData[U]':
        """Batch-to-Batch transformation (e.g. for batch inference optimization)."""
        new_items = func(self.items)
        return BatchData(
            batch_id=new_batch_id or self.batch_id,
            items=new_items,
            metadata=self.metadata.copy()
        )

    def group_by(self, key_func: Callable[[T], str]) -> Dict[str, List[T]]:
        """Helper for aggregation stage."""
        groups = {}
        for item in self.items:
            k = key_func(item)
            if k not in groups: groups[k] = []
            groups[k].append(item)
        return groups
    
    # Backwards compatibility stub for old code accessing .items as dict list
    # Remove after full conversion
    def __post_init__(self):
        if not self.metadata:
            self.metadata = {}