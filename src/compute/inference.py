"""GPU inference workers with vLLM integration for Qwen3-Omni"""

import os
import time
import asyncio
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import json

import torch
import numpy as np
from vllm import SamplingParams, AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from transformers import Qwen3OmniMoeProcessor

# Set environment variables for Qwen3-Omni
os.environ['VLLM_USE_V1'] = '0'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

from ..scheduling.pipeline import PipelineStage, DataBatch

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
        self.processor = None
        self.tokenizer = None
        self._load_processor()
        
    def _load_processor(self) -> None:
        """Load model processor and tokenizer"""
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code
        )
    
    def prepare_inputs(self, 
                      audio_features: torch.Tensor,
                      sample_rate: int,
                      prompt: str = "请将这段语音转换为纯文本。") -> Dict[str, Any]:
        """Prepare inputs for Qwen3-Omni model"""
        # Convert audio tensor directly to numpy array (not dict) as expected by qwen_omni_utils
        audio_array = audio_features.numpy()
        
        # Create messages in Qwen3-Omni format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_array},
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
    
    def decode_output(self, output_text: str) -> str:
        """Decode model output to text (Qwen3-Omni returns text directly)"""
        # Qwen3-Omni returns text directly, no need to decode tokens
        cleaned_text = output_text.strip()
        return cleaned_text


class VLLMInferenceEngine:
    """vLLM-based inference engine"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.engine = None
        self.model_processor = AudioModelProcessor(config)
        self._setup_engine()
        
    def _setup_engine(self) -> None:
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
        
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    async def _generate_single_async(self, 
                                   inputs: Dict[str, Any],
                                   sampling_params: Optional[SamplingParams],
                                   request_id: str) -> str:
        """Internal method to handle single audio async inference"""
        try:
            results = []
            async for request_output in self.engine.generate(
                inputs['prompt'], 
                sampling_params, 
                request_id,
                multi_modal_data=inputs.get('multi_modal_data', {}),
                mm_processor_kwargs=inputs.get('mm_processor_kwargs', {})
            ):
                results.append(request_output.outputs[0].text)
            
            return results[-1] if results else ""
        except Exception as e:
            # Return error message instead of raising exception
            return f"Error during inference: {str(e)}"
    
    async def generate_async(self, 
                           inputs: Dict[str, Any],
                           sampling_params: Optional[SamplingParams] = None) -> str:
        """Generate text asynchronously for single input"""
        if sampling_params is None:
            sampling_params = SamplingParams(
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                repetition_penalty=self.config.repetition_penalty
            )
        
        # Generate using vLLM
        request_id = f"req_{int(time.time() * 1000000)}"
        
        return await self._generate_single_async(inputs, sampling_params, request_id)
    
    async def generate_batch_async(self, 
                                 batch_inputs: List[Dict[str, Any]], 
                                 sampling_params: Optional[SamplingParams] = None) -> List[str]:
        """Generate text asynchronously for batch inputs"""
        if sampling_params is None:
            sampling_params = SamplingParams(
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                repetition_penalty=self.config.repetition_penalty
            )
        
        # Create tasks for concurrent processing
        tasks = []
        for i, inputs in enumerate(batch_inputs):
            request_id = f"batch_req_{int(time.time() * 1000000)}_{i}"
            task = self._generate_single_async(inputs, sampling_params, request_id)
            tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle any exceptions
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append(f"Error during batch inference: {str(result)}")
            else:
                processed_results.append(result)
        
        return processed_results

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
        
        # Filter out items with errors - only process valid items
        valid_items = [item for item in batch.items if 'error' not in item]
        
        # Process valid items in batches
        for i in range(0, len(valid_items), self.batch_size):
            batch_items = valid_items[i:i + self.batch_size]
            # Prepare batch inputs
            batch_inputs = []
            for item in batch_items:
                audio_features = item['audio_features']
                # 准备输入，使用正确的sample_rate
                inputs = self.inference_engine.model_processor.prepare_inputs(
                    audio_features,
                    item.get('sample_rate', 16000),
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
                    # 其他情况也使用默认置信度
                    item['confidence'] = 0.0
                
                processed_items.append(item)
                    
        # Create new batch with inference results - exclude error items
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
        
        # 使用线程池来处理异步事件循环问题
        import asyncio
        import concurrent.futures
        from threading import Thread
        
        # 使用异步方法进行批量推理
        async def run_async_batch():
            return await self.inference_engine.generate_batch_async(batch_inputs)
        
        # 在新线程中运行异步函数
        def run_in_thread():
            # 创建新的事件循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(run_async_batch())
            finally:
                loop.close()
        
        # 使用线程池运行异步函数
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_thread)
            results = future.result()
        
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