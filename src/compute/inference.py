"""GPU inference workers with vLLM integration"""

import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import json

import torch
import numpy as np
from vllm import SamplingParams, AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from transformers import AutoProcessor, AutoTokenizer
from loguru import logger

from ..scheduling.pipeline import PipelineStage, DataBatch


@dataclass
class InferenceConfig:
    """Inference configuration"""
    model_name: str = "Qwen/Qwen2-Audio-7B-Instruct"
    tensor_parallel_size: int = 1
    max_num_batched_tokens: int = 8192
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.9
    trust_remote_code: bool = True
    dtype: str = "auto"
    temperature: float = 0.0
    max_tokens: int = 512
    top_p: float = 0.9
    repetition_penalty: float = 1.1


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
            self.processor = AutoProcessor.from_pretrained(
                self.config.model_name,
                trust_remote_code=self.config.trust_remote_code
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=self.config.trust_remote_code
            )
            
            # Set padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info(f"Loaded processor for {self.config.model_name}")
            
        except Exception as e:
            logger.error(f"Error loading processor: {e}")
            raise
    
    def prepare_inputs(self, 
                      audio_features: torch.Tensor,
                      prompt: str = "Transcribe the audio to text:") -> Dict[str, Any]:
        """Prepare inputs for the model"""
        try:
            # For Qwen2-Audio, we need to format the input properly
            # This is a simplified example - actual implementation may vary
            inputs = self.processor(
                text=prompt,
                audio=audio_features,
                return_tensors="pt",
                padding=True
            )
            
            return inputs
            
        except Exception as e:
            logger.error(f"Error preparing inputs: {e}")
            raise
    
    def decode_output(self, output_ids: torch.Tensor) -> str:
        """Decode model output to text"""
        try:
            # Decode generated tokens
            decoded_text = self.tokenizer.decode(
                output_ids[0], 
                skip_special_tokens=True
            )
            
            # Clean up the output
            decoded_text = decoded_text.strip()
            
            return decoded_text
            
        except Exception as e:
            logger.error(f"Error decoding output: {e}")
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
        self.prompt_template = config.get('prompt_template', 'Transcribe the audio to text:')
        
    def process(self, batch: DataBatch) -> DataBatch:
        """Process batch through inference"""
        processed_items = []
        
        for item in batch.items:
            try:
                if 'error' in item:
                    processed_items.append(item)
                    continue
                    
                audio_features = item['audio_features']
                
                # Prepare inputs for model
                inputs = self.inference_engine.model_processor.prepare_inputs(
                    audio_features,
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
        
        # Remove common artifacts (customize as needed)
        artifacts = [
            "Transcribe the audio to text:",
            "Here is the transcription:",
            "The audio says:",
        ]
        
        for artifact in artifacts:
            cleaned = cleaned.replace(artifact, "")
        
        return cleaned.strip()