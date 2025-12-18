"""音频格式转换Stage"""
import tempfile
import os
import subprocess
import torchaudio
import torch
from ..stages.base import Stage
from ..tools.data_structures import ProcessingItem


class AudioFormatStage(Stage):
    """音频格式转换：统一为16k采样率，单声道，WAV格式"""
    
    def __init__(self, target_sr: int = 16000, target_channels: int = 1):
        self.target_sr = target_sr
        self.target_channels = target_channels
    
    def name(self) -> str:
        return "audio_format"
    
    def process(self, item: ProcessingItem) -> ProcessingItem:
        """转换音频格式"""
        # 写入临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as input_file:
            input_path = input_file.name
            input_file.write(item.audio_bytes)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as output_file:
            output_path = output_file.name
        
        try:
            # 使用ffmpeg转换
            cmd = [
                'ffmpeg', '-y',
                '-i', input_path,
                '-ar', str(self.target_sr),
                '-ac', str(self.target_channels),
                '-f', 'wav',
                '-loglevel', 'error',
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=300)
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg error: {result.stderr.decode()}")
            
            # 加载为numpy array
            waveform, sample_rate = torchaudio.load(output_path)
            
            # 转为mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # 保存到item
            item.audio_data = waveform.squeeze(0).numpy()
            item.sample_rate = sample_rate
            item.duration = len(item.audio_data) / sample_rate
            
            # 释放原始bytes（节省内存）
            item.audio_bytes = None
            
            return item
            
        finally:
            # 清理临时文件
            for path in [input_path, output_path]:
                if os.path.exists(path):
                    os.unlink(path)