from silero_vad import load_silero_vad, get_speech_timestamps

worker_vad_model = load_silero_vad(onnx=True)

vad_params = {
    'sampling_rate': 16000,             # 强烈建议将音频重采样到 16000 Hz
    'return_seconds': False,
    'min_speech_duration_ms': 1500,     # 调高， 忽略短促杂音
    'min_silence_duration_ms': 1000,    # 调高，停顿不切分，保证句子完整性
    "threshold": 0.4,                   # 设置阈值，过滤BGM和高噪
    "neg_threshold": 0.15,              # 默认
    "speech_pad_ms": 100                # 调高，前后各多留pad，防止吞字
}

speech_timestamps = get_speech_timestamps(
    wav_data,
    worker_vad_model,
    **vad_params
)