"""VAD处理Stage"""
from ..stages.base import Stage
from ..tools.data_structures import ProcessingItem
from src.compute.vad import VADProcessor


class CoarseVADStage(Stage):
    """粗VAD处理：产生较长的语音片段"""
    
    def __init__(self, vad_config: dict):
        self.vad_processor = VADProcessor(vad_config)
    
    def name(self) -> str:
        return "coarse_vad"
    
    def process(self, item: ProcessingItem) -> ProcessingItem:
        """执行VAD检测"""
        # 执行VAD
        vad_result = self.vad_processor.process_audio(
            item.file_id,
            item.audio_data,
            item.sample_rate
        )
        
        # 保存VAD结果到metadata
        item.metadata['vad_result'] = {
            'total_speech_duration': vad_result.total_speech_duration,
            'speech_ratio': vad_result.speech_ratio,
            'num_segments': len(vad_result.speech_segments)
        }
        
        # 转换为segment info格式（包含采样点索引）
        segments = []
        for seg_info in vad_result.speech_segments:
            start_idx = int(seg_info['start'] * item.sample_rate)
            end_idx = int(seg_info['end'] * item.sample_rate)
            
            segments.append({
                'start_time': seg_info['start'],
                'end_time': seg_info['end'],
                'duration': seg_info['end'] - seg_info['start'],
                'start_idx': start_idx,
                'end_idx': end_idx
            })
        
        item.segments = segments
        
        return item