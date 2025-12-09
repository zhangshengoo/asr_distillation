"""GPU inference workers with vLLM integration for Qwen3-Omni"""

import os
import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import json

import torch
import numpy as np
from vllm import SamplingParams, AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from transformers import Qwen3OmniMoeProcessor
from loguru import logger

# Set environment variables for Qwen3-Omni
os.environ['VLLM_USE_V1'] = '0'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

from ..scheduling.pipeline import PipelineStage, DataBatch

# Import Qwen3-Omni utilities
try:
    from qwen_omni_utils import process_mm_info
except ImportError:
    logger.warning("qwen_omni_utils not found, using fallback implementation")
    
    def process_mm_info(messages, use_audio_in_video=False):
        """Fallback implementation for processing multimodal info"""
        audios = None
        images = None
        videos = None
        
        for message in messages:
            if isinstance(message.get('content'), list):
                for content in message['content']:
                    if content.get('type') == 'audio':
                        if audios is None:
                            audios = []
                        audios.append(content['audio'])
                    elif content.get('type') == 'image':
                        if images is None:
                            images = []
                        images.append(content['image'])
                    elif content.get('type') == 'video':
                        if videos is None:
                            videos = []
                        videos.append(content['video'])
        
        return audios, images, videos


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
        self.processor = None
        self.tokenizer = None
        self._load_processor()
        
    def _load_processor(self) -> None:
        """Load model processor and tokenizer"""
        try:
            self.processor = Qwen3OmniMoeProcessor.from_pretrained(
                self.config.model_name,
                trust_remote_code=self.config.trust_remote_code
            )
            
            logger.info(f"Loaded Qwen3-Omni processor for {self.config.model_name}")
            
        except Exception as e:
            logger.error(f"Error loading Qwen3-Omni processor: {e}")
            raise
    
    def prepare_inputs(self, 
                      audio_features: torch.Tensor,
                      sample_rate: int,
                      prompt: str = "请将这段语音转换为纯文本。") -> Dict[str, Any]:
        """Prepare inputs for Qwen3-Omni model"""
        try:
            # Convert audio tensor to the format expected by Qwen3-Omni
            audio_data = {
                'array': audio_features.numpy(),
                'sampling_rate': sample_rate
            }
            
            # Create messages in Qwen3-Omni format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": audio_data},
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
            
        except Exception as e:
            logger.error(f"Error preparing Qwen3-Omni inputs: {e}")
            raise
    
    def decode_output(self, output_text: str) -> str:
        """Decode model output to text (Qwen3-Omni returns text directly)"""
        try:
            # Qwen3-Omni returns text directly, no need to decode tokens
            cleaned_text = output_text.strip()
            
            return cleaned_text
            
        except Exception as e:
            logger.error(f"Error decoding Qwen3-Omni output: {e}")
            return ""


class VLLMInferenceEngine:
    """vLLM-based inference engine"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.engine = None
        self.model_processor = AudioModelProcessor(config)
        self._setup_engine()
        
    def _setup_engine(self) -> None:
        """Setup vLLM async engine"""
        try:
            engine_args = AsyncEngineArgs(
                model=self.config.model_name,
                tensor_parallel_size=self.config.tensor_parallel_size,
                max_num_batched_tokens=self.config.max_num_batched_tokens,
                max_model_len=self.config.max_model_len,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                trust_remote_code=self.config.trust_remote_code,
                dtype=self.config.dtype
            )
            
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            logger.info("vLLM engine setup complete")
            
        except Exception as e:
            logger.error(f"Error setting up vLLM engine: {e}")
            raise
    
    async def generate_async(self, 
                           inputs: Dict[str, Any],
                           sampling_params: Optional[SamplingParams] = None) -> str:
        """Generate text asynchronously"""
        if sampling_params is None:
            sampling_params = SamplingParams(
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                repetition_penalty=self.config.repetition_penalty
            )
        
        try:
            # Generate using vLLM
            request_id = f"req_{int(time.time() * 1000000)}"
            
            # For simplicity, we'll use a text-based approach
            # In practice, you'd need to handle the multimodal inputs properly
            text_input = inputs.get('text', 'Transcribe the audio:')
            
            results = []
            async for request_output in self.engine.generate(text_input, sampling_params, request_id):
                results.append(request_output.outputs[0].text)
            
            return results[-1] if results else ""
            
        except Exception as e:
            logger.error(f"Error in async generation: {e}")
            return ""


class AudioInferenceStage(PipelineStage):
    """Stage for audio inference using vLLM"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        inference_config = InferenceConfig(**config.get('inference', {}))
        self.inference_engine = VLLMInferenceEngine(inference_config)
        self.prompt_template = config.get('prompt_template', '请将这段语音转换为纯文本。')
        
    def process(self, batch: DataBatch) -> DataBatch:
        """Process batch through inference"""
        processed_items = []
        
        for item in batch.items:
            try:
                if 'error' in item:
                    processed_items.append(item)
                    continue
                    
                audio_features = item['audio_features']
                sample_rate = item['sample_rate']
                
                # Prepare inputs for Qwen3-Omni model
                inputs = self.inference_engine.model_processor.prepare_inputs(
                    audio_features,
                    sample_rate,
                    self.prompt_template
                )
                
                # Run inference (simplified for this example)
                # In practice, you'd use the async engine properly
                transcription = self._run_inference_sync(inputs)
                
                # Update item with results
                item['transcription'] = transcription
                item['inference_timestamp'] = time.time()
                
                processed_items.append(item)
                
            except Exception as e:
                logger.error(f"Error in inference for {item['file_id']}: {e}")
                item['error'] = str(e)
                processed_items.append(item)
        
        # Create new batch with inference results
        new_batch = DataBatch(
            batch_id=batch.batch_id,
            items=processed_items,
            metadata={**batch.metadata, 'stage': 'inference'}
        )
        
        return new_batch
    
    def _run_inference_sync(self, inputs: Dict[str, Any]) -> str:
        """Run inference synchronously (simplified)"""
        # This is a placeholder for actual vLLM integration
        # In practice, you'd implement proper async handling
        
        # For now, return a dummy transcription
        # Replace this with actual vLLM inference
        return "This is a dummy transcription. Replace with actual model output."


class BatchInferenceStage(PipelineStage):
    """Batch inference stage for better GPU utilization"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        inference_config = InferenceConfig(**config.get('inference', {}))
        self.inference_engine = VLLMInferenceEngine(inference_config)
        self.batch_size = config.get('batch_size', 8)
        self.prompt_template = config.get('prompt_template', 'Transcribe the audio to text:')
        
    def process(self, batch: DataBatch) -> DataBatch:
        """Process batch with batched inference"""
        processed_items = []
        
        # Filter out items with errors
        valid_items = [item for item in batch.items if 'error' not in item]
        error_items = [item for item in batch.items if 'error' in item]
        
        # Process valid items in batches
        for i in range(0, len(valid_items), self.batch_size):
            batch_items = valid_items[i:i + self.batch_size]
            
            try:
                # Prepare batch inputs
                batch_inputs = []
                for item in batch_items:
                    audio_features = item['audio_features']
                    inputs = self.inference_engine.model_processor.prepare_inputs(
                        audio_features,
                        self.prompt_template
                    )
                    batch_inputs.append(inputs)
                
                # Run batch inference
                batch_results = self._run_batch_inference(batch_inputs)
                
                # Update items with results
                for item, result in zip(batch_items, batch_results):
                    item['transcription'] = result
                    item['inference_timestamp'] = time.time()
                    # 为segment级别的处理添加置信度（如果模型提供）
                    if isinstance(result, dict) and 'confidence' in result:
                        item['confidence'] = result['confidence']
                    else:
                        item['confidence'] = 0.0  # 默认值
                    processed_items.append(item)
                    
            except Exception as e:
                logger.error(f"Error in batch inference: {e}")
                # Mark all items in this batch as failed
                for item in batch_items:
                    item['error'] = str(e)
                    processed_items.append(item)
        
        # Add items that already had errors
        processed_items.extend(error_items)
        
        # Create new batch with inference results
        new_batch = DataBatch(
            batch_id=batch.batch_id,
            items=processed_items,
            metadata={**batch.metadata, 'stage': 'batch_inference'}
        )
        
        return new_batch
    
    def _run_batch_inference(self, batch_inputs: List[Dict[str, Any]]) -> List[str]:
        """Run batched inference"""
        # This is a placeholder for actual batched vLLM integration
        # In practice, you'd implement proper batched inference
        
        results = []
        for inputs in batch_inputs:
            # Replace with actual inference
            result = f"Batch transcription for input {len(results)}"
            results.append(result)
            
        return results


class PostProcessingStage(PipelineStage):
    """Post-processing stage for inference results"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.output_format = config.get('output_format', 'json')
        
    def process(self, batch: DataBatch) -> DataBatch:
        """Post-process inference results"""
        processed_items = []
        
        for item in batch.items:
            try:
                if 'error' in item:
                    processed_items.append(item)
                    continue
                    
                transcription = item.get('transcription', '')
                
                # Clean up transcription
                cleaned_transcription = self._clean_transcription(transcription)
                
                # Format output
                if self.output_format == 'json':
                    output = {
                        'file_id': item['file_id'],
                        'transcription': cleaned_transcription,
                        'timestamp': item.get('inference_timestamp', time.time()),
                        'metadata': {
                            'duration': item.get('audio_tensor', {}).get('duration', 0),
                            'sample_rate': item.get('audio_tensor', {}).get('sample_rate', 16000)
                        }
                    }
                else:
                    output = {
                        'file_id': item['file_id'],
                        'text': cleaned_transcription
                    }
                
                item['output'] = output
                item['transcription'] = cleaned_transcription  # Keep for compatibility
                
                processed_items.append(item)
                
            except Exception as e:
                logger.error(f"Error in post-processing for {item['file_id']}: {e}")
                item['error'] = str(e)
                processed_items.append(item)
        
        # Create new batch with post-processed results
        new_batch = DataBatch(
            batch_id=batch.batch_id,
            items=processed_items,
            metadata={**batch.metadata, 'stage': 'post_processing'}
        )
        
        return new_batch
    
    def _clean_transcription(self, transcription: str) -> str:
        """Clean up transcription text"""
        if not transcription:
            return ""
            
        # Basic cleaning
        cleaned = transcription.strip()
        
        return cleaned.strip()