import os
# os.environ['VLLM_USE_V1'] = '0'
# os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
# os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
import warnings
import numpy as np

# warnings.filterwarnings('ignore')
# warnings.filterwarnings('ignore', category=DeprecationWarning)
# warnings.filterwarnings('ignore', category=FutureWarning)
# warnings.filterwarnings('ignore', category=UserWarning)

from vllm import LLM
from vllm import SamplingParams
from qwen_omni_utils import process_mm_info
from transformers import Qwen3OmniMoeProcessor


MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

def _load_model_processor():
    model = LLM(
        model=MODEL_PATH, trust_remote_code=True, gpu_memory_utilization=0.95,
        tensor_parallel_size=torch.cuda.device_count(),
        limit_mm_per_prompt={'image': 1, 'video': 3, 'audio': 3},
        max_num_seqs=1,
        max_model_len=32768,
        seed=1234,
        )

    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)
    return model, processor

def run_model(model, processor, messages):
    sampling_params = SamplingParams(temperature=1e-2, top_p=0.1, top_k=1, max_tokens=8192)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
    inputs = {'prompt': text, 'multi_modal_data': {}, "mm_processor_kwargs": {"use_audio_in_video": False}}
    if images is not None: inputs['multi_modal_data']['image'] = images
    if videos is not None: inputs['multi_modal_data']['video'] = videos
    if audios is not None: inputs['multi_modal_data']['audio'] = audios
    outputs = model.generate(inputs, sampling_params=sampling_params)
    response = outputs[0].outputs[0].text
    return response, None


if __name__ == "__main__":
    model, processor = _load_model_processor()

    audio_path = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/asr_zh.wav"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": "请将这段中文语音转换为纯文本。"},
            ]
        }
    ]

    response, _ = run_model(model=model, messages=messages, processor=processor)

    print(response)