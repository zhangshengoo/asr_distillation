"""GPU inference workers with vLLM integration for Qwen3-Omni"""

import os
import time
import logging
from typing import Dict, List, Any
from dataclasses import dataclass

import torch
import numpy as np
from vllm import LLM, SamplingParams
from transformers import Qwen3OmniMoeProcessor

# 抑制vLLM日志
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
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_features},
                    {"type": "text", "text": prompt},
                ]
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
        
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


class BatchInferenceStage(PipelineStage):
    """Batch inference stage with synchronous LLM"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger = logging.getLogger("BatchInferenceStage")
        
        # 运行时检测GPU
        available_gpus = torch.cuda.device_count()
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'not_set')
        if cuda_visible != 'not_set' or available_gpus > 0:
            self.logger.info(f"GPU initialized: CUDA_VISIBLE_DEVICES={cuda_visible}, detected={available_gpus}")
        
        # 初始化配置
        self.inference_config = InferenceConfig(**config.get('inference', {}))
        
        # 初始化同步LLM
        self.llm = LLM(
            model=self.inference_config.model_name,
            tensor_parallel_size=self.inference_config.tensor_parallel_size,
            max_num_batched_tokens=self.inference_config.max_num_batched_tokens,
            max_model_len=self.inference_config.max_model_len,
            gpu_memory_utilization=self.inference_config.gpu_memory_utilization,
            trust_remote_code=self.inference_config.trust_remote_code,
            dtype=self.inference_config.dtype
        )
        
        # 初始化采样参数
        self.sampling_params = SamplingParams(
            temperature=self.inference_config.temperature,
            max_tokens=self.inference_config.max_tokens,
            top_p=self.inference_config.top_p,
            repetition_penalty=self.inference_config.repetition_penalty
        )
        
        # 初始化音频处理器
        self.model_processor = AudioModelProcessor(self.inference_config)
        self.prompt_template = config.get('prompt_template', '请将这段语音转换为纯文本。')
    
    def process(self, batch: BatchData[SegmentItem]) -> BatchData[InferenceItem]:
        """同步推理处理"""
        if not batch.items:
            return BatchData(
                batch_id=batch.batch_id,
                items=[],
                metadata={**batch.metadata, 'stage': 'batch_inference'}
            )
        
        # 过滤空waveform
        valid_items = []
        for item in batch.items:
            if item.waveform is None:
                continue
            if item.waveform.size == 0:
                continue
            valid_items.append(item)
        
        if not valid_items:
            return BatchData(
                batch_id=batch.batch_id,
                items=[],
                metadata={**batch.metadata, 'stage': 'batch_inference', 'skipped_reason': 'all_empty_waveforms'}
            )
        
        # 准备batch inputs
        batch_inputs = []
        for item in valid_items:
            sample_rate = item.metadata.get('sample_rate', 16000)
            inputs = self.model_processor.prepare_inputs(
                item.waveform,
                sample_rate,
                self.prompt_template
            )
            batch_inputs.append(inputs)
        
        # 同步batch推理
        outputs = self.llm.generate(
            [inp['prompt'] for inp in batch_inputs],
            self.sampling_params,
            use_tqdm=False
        )
        
        # 构造结果
        inference_items = []
        for item, output in zip(valid_items, outputs):
            transcription = output.outputs[0].text if output.outputs else ""
            
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


class PostProcessingStage(PipelineStage):
    """Post-processing stage for inference results"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger = logging.getLogger("PostProcessingStage")
        self.output_format = config.get('output_format', 'json')
        
    def process(self, batch: BatchData[FileResultItem]) -> BatchData[FileResultItem]:
        """Post-process inference results"""
        processed_items = []
        
        for item in batch.items:
            try:
                cleaned_transcription = item.transcription.strip()
                
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
                
                updated_item = FileResultItem(
                    file_id=item.file_id,
                    transcription=cleaned_transcription,
                    segments=item.segments,
                    stats=item.stats,
                    metadata=item.metadata,
                    output=output
                )
                processed_items.append(updated_item)
            except Exception as e:
                self.logger.error(f"Post-processing failed for item {item.file_id}: {e}")
        
        return BatchData(
            batch_id=batch.batch_id,
            items=processed_items,
            metadata={**batch.metadata, 'stage': 'post_processing'}
        )