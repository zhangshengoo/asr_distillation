"""音频格式转换Stage"""
import tempfile
import os
import subprocess
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
import torchaudio
import torch
from base import Stage
from tools.data_structures import ProcessingItem


class AudioFormatStage(Stage):
    """音频格式转换：统一为16k采样率，单声道，WAV格式"""
    
    def __init__(self, target_sr: int = 16000, target_channels: int = 1, 
                 max_workers: int = 8):
        self.target_sr = target_sr
        self.target_channels = target_channels
        self.max_workers = max_workers
    
    def name(self) -> str:
        return "audio_format"
    
    def process(self, item: ProcessingItem) -> ProcessingItem:
        """转换音频格式"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as input_file:
            input_path = input_file.name
            input_file.write(item.audio_bytes)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as output_file:
            output_path = output_file.name
        
        try:
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
            
            waveform, sample_rate = torchaudio.load(output_path)
            
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            item.audio_data = waveform.squeeze(0).numpy()
            item.sample_rate = sample_rate
            item.duration = len(item.audio_data) / sample_rate
            
            item.audio_bytes = None
            
            return item
            
        finally:
            for path in [input_path, output_path]:
                if os.path.exists(path):
                    os.unlink(path)
    
    def process_batch(self, items: List[ProcessingItem]) -> List[ProcessingItem]:
        """并行转换批次样本"""
        if not items:
            return []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._convert_single, item): i
                for i, item in enumerate(items)
            }
            
            results = [None] * len(items)
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
            
            return results
    
    def _convert_single(self, item: ProcessingItem) -> ProcessingItem:
        """转换单个音频（线程安全）"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as input_file:
            input_path = input_file.name
            input_file.write(item.audio_bytes)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as output_file:
            output_path = output_file.name
        
        try:
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
            
            waveform, sample_rate = torchaudio.load(output_path)
            
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            item.audio_data = waveform.squeeze(0).numpy()
            item.sample_rate = sample_rate
            item.duration = len(item.audio_data) / sample_rate
            
            item.audio_bytes = None
            
            return item
            
        finally:
            for path in [input_path, output_path]:
                if os.path.exists(path):
                    os.unlink(path)