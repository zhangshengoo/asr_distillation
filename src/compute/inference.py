"""GPU inference workers with vLLM integration for Qwen3-Omni"""

import os
import time
import asyncio
from typing import Dict, List, Any
from dataclasses import dataclass

import torch
import numpy as np
from vllm import SamplingParams, AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from transformers import Qwen3OmniMoeProcessor

# Set environment variables for Qwen3-Omni
os.environ['VLLM_USE_V1'] = '0'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

from ..common import BatchData, SegmentItem, InferenceItem, FileResultItem
from ..scheduling.pipeline import PipelineStage

# Import Qwen3-Omni utilities
from qwen_omni_utils import process_mm_info


@dataclass
class InferenceConfig:
    """Inference configuration"""
    model_name: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    tensor_parallel_size: int = 1
    max_num_batched_tokens: int = 8192
    max_model_len: int = 32768
    gpu_memory_utilization: float = 0.95
    trust_remote_code: bool = True
    dtype: str = "auto"
    temperature: float = 1e-2
    max_tokens: int = 8192
    top_p: float = 0.1
    repetition_penalty: float = 1.1
    max_num_seqs: int = 1
    limit_mm_per_prompt: Dict[str, int] = None
    seed: int = 1234
    
    def __post_init__(self):
        if self.limit_mm_per_prompt is None:
            self.limit_mm_per_prompt = {'image': 1, 'video': 3, 'audio': 3}


class AudioModelProcessor:
    """Audio model processor for multimodal models"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code
        )
    
    def prepare_inputs(self, 
                      audio_features: np.ndarray,
                      sample_rate: int,
                      prompt: str = "请将这段语音转换为纯文本。") -> Dict[str, Any]:
        """Prepare inputs for Qwen3-Omni model"""
        # audio_features is already numpy array from SegmentItem.waveform
        
        # Create messages in Qwen3-Omni format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_features},
                    {"type": "text", "text": prompt},
                ]
            }
        ]
        
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Process multimodal information
        audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
        
        # Prepare inputs for vLLM
        inputs = {
            'prompt': text, 
            'multi_modal_data': {}, 
            "mm_processor_kwargs": {"use_audio_in_video": False}
        }
        
        if images is not None:
            inputs['multi_modal_data']['image'] = images
        if videos is not None:
            inputs['multi_modal_data']['video'] = videos
        if audios is not None:
            inputs['multi_modal_data']['audio'] = audios
        
        return inputs


class VLLMInferenceEngine:
    """vLLM-based async inference engine"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.engine = self._setup_engine()
        
    def _setup_engine(self) -> AsyncLLMEngine:
        """Setup vLLM async engine"""
        engine_args = AsyncEngineArgs(
            model=self.config.model_name,
            tensor_parallel_size=self.config.tensor_parallel_size,
            max_num_batched_tokens=self.config.max_num_batched_tokens,
            max_model_len=self.config.max_model_len,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            trust_remote_code=self.config.trust_remote_code,
            dtype=self.config.dtype
        )
        
        return AsyncLLMEngine.from_engine_args(engine_args)
    
    async def generate_batch_async(self, 
                                  batch_inputs: List[Dict[str, Any]],
                                  sampling_params: SamplingParams = None) -> List[str]:
        """Batch async inference - 唯一的公开接口
        
        Args:
            batch_inputs: List of prepared inputs from AudioModelProcessor
            sampling_params: Optional sampling parameters
            
        Returns:
            List of transcription strings
            
        Raises:
            Exception: Any error during inference
        """
        if sampling_params is None:
            sampling_params = SamplingParams(
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                repetition_penalty=self.config.repetition_penalty
            )
        
        # Submit all requests to vLLM (continuous batching will handle them)
        tasks = []
        for i, inputs in enumerate(batch_inputs):
            request_id = f"req_{int(time.time() * 1000000)}_{i}"
            task = self._generate_single(inputs, sampling_params, request_id)
            tasks.append(task)
        
        # Wait for all results (vLLM will batch internally)
        results = await asyncio.gather(*tasks)
        return results
    
    async def _generate_single(self,
                               inputs: Dict[str, Any],
                               sampling_params: SamplingParams,
                               request_id: str) -> str:
        """Internal method for single inference"""
        final_output = None
        
        async for request_output in self.engine.generate(
            inputs['prompt'], 
            sampling_params, 
            request_id
        ):
            final_output = request_output.outputs[0].text
        
        return final_output if final_output else ""


class BatchInferenceStage(PipelineStage):
    """Batch inference stage with async support"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        inference_config = InferenceConfig(**config.get('inference', {}))
        
        # Initialize components
        self.inference_engine = VLLMInferenceEngine(inference_config)
        self.model_processor = AudioModelProcessor(inference_config)
        self.prompt_template = config.get('prompt_template', '请将这段语音转换为纯文本。')
        
    async def process_async(self, batch: BatchData[SegmentItem]) -> BatchData[InferenceItem]:
        """Async processing method"""
        if not batch.items:
            return BatchData(
                batch_id=batch.batch_id,
                items=[],
                metadata={**batch.metadata, 'stage': 'batch_inference'}
            )
        
        # Filter out items with empty waveforms to prevent vLLM errors
        valid_items = []
        for item in batch.items:
            if item.waveform is None:
                print(f"⚠️  Warning: Skipping None waveform for {item.segment_id}")
                continue
            if item.waveform.size == 0:
                print(f"⚠️  Warning: Skipping empty waveform for {item.segment_id}, shape={item.waveform.shape}")
                continue
            valid_items.append(item)
        
        if not valid_items:
            print(f"⚠️  Warning: Batch {batch.batch_id} has no valid waveforms, skipping inference")
            return BatchData(
                batch_id=batch.batch_id,
                items=[],
                metadata={**batch.metadata, 'stage': 'batch_inference', 'skipped_reason': 'all_empty_waveforms'}
            )
        
        # Log if some items were filtered
        if len(valid_items) < len(batch.items):
            print(f"⚠️  Filtered {len(batch.items) - len(valid_items)} items with empty waveforms from batch {batch.batch_id}")
        
        # Prepare batch inputs
        batch_inputs = []
        for item in valid_items:
            # item.waveform is already numpy array from SegmentItem
            sample_rate = item.metadata.get('sample_rate', 16000)
            
            inputs = self.model_processor.prepare_inputs(
                item.waveform,  # Direct numpy array, no conversion needed
                sample_rate,
                self.prompt_template
            )
            batch_inputs.append(inputs)
        
        # Batch inference (errors will raise)
        transcriptions = await self.inference_engine.generate_batch_async(batch_inputs)
        
        # Create InferenceItem objects
        inference_items = []
        for item, transcription in zip(valid_items, transcriptions):
            inference_item = InferenceItem(
                file_id=item.file_id,
                parent_file_id=item.parent_file_id,
                segment_id=item.segment_id,
                segment_index=item.segment_index,
                start_time=item.start_time,
                end_time=item.end_time,
                waveform=item.waveform,
                original_duration=item.original_duration,
                oss_path=item.oss_path,
                metadata=item.metadata,
                transcription=transcription,
                confidence=0.95,
                inference_timestamp=time.time()
            )
            inference_items.append(inference_item)
        
        return BatchData(
            batch_id=batch.batch_id,
            items=inference_items,
            metadata={**batch.metadata, 'stage': 'batch_inference'}
        )
    
    def process(self, batch: BatchData[SegmentItem]) -> BatchData[InferenceItem]:
        """Sync wrapper for compatibility"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                raise RuntimeError("Cannot call sync process() in running event loop")
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.process_async(batch))


class PostProcessingStage(PipelineStage):
    """Post-processing stage for inference results"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.output_format = config.get('output_format', 'json')
        
    def process(self, batch: BatchData[FileResultItem]) -> BatchData[FileResultItem]:
        """Post-process inference results"""
        processed_items = []
        
        for item in batch.items:
            # Clean up transcription
            cleaned_transcription = item.transcription.strip()
            
            # Format output
            if self.output_format == 'json':
                output = {
                    'file_id': item.file_id,
                    'transcription': cleaned_transcription,
                    'timestamp': time.time(),
                    'metadata': {
                        'stats': item.stats
                    }
                }
            else:
                output = {
                    'file_id': item.file_id,
                    'text': cleaned_transcription
                }
            
            # Create updated FileResultItem with output
            updated_item = FileResultItem(
                file_id=item.file_id,
                transcription=cleaned_transcription,
                segments=item.segments,
                stats=item.stats,
                metadata=item.metadata,
                output=output
            )
            processed_items.append(updated_item)
        
        return BatchData(
            batch_id=batch.batch_id,
            items=processed_items,
            metadata={**batch.metadata, 'stage': 'post_processing'}
        )