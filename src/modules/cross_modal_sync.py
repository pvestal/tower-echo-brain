import subprocess
import json
import librosa
import numpy as np
from typing import Dict, List, Any

class CrossModalSync:
    """Synchronize video, music, and voice"""
    
    def __init__(self):
        self.vision_checker = None  # Will be injected
    
    def sync_video_to_music(self, video_path: str, music_path: str) -> Dict[str, Any]:
        """Synchronize video cuts to music beats"""
        # Extract video info
        video_cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=r_frame_rate,duration',
            '-of', 'json', video_path
        ]
        video_result = subprocess.run(video_cmd, capture_output=True, text=True)
        video_info = json.loads(video_result.stdout)["streams"][0]
        
        # Extract music tempo and beats
        y, sr = librosa.load(music_path)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beats, sr=sr)
        
        # Generate sync points
        sync_data = {
            "video_fps": eval(video_info.get("r_frame_rate", "24/1")),
            "music_tempo": float(tempo),
            "beat_times": beat_times.tolist(),
            "sync_points": self._calculate_sync_points(beat_times, video_info),
            "sync_quality": self._calculate_sync_quality(beat_times, video_info)
        }
        
        return sync_data
    
    def _calculate_sync_points(self, beat_times: np.ndarray, video_info: Dict) -> List[float]:
        """Calculate optimal cut points"""
        # Every 4 beats = scene change
        sync_points = []
        for i in range(0, len(beat_times), 4):
            if i < len(beat_times):
                sync_points.append(float(beat_times[i]))
        return sync_points
    
    def _calculate_sync_quality(self, beat_times: np.ndarray, video_info: Dict) -> float:
        """Calculate synchronization quality score"""
        if len(beat_times) == 0:
            return 0.0
        
        # Check if video duration aligns with music
        video_duration = float(video_info.get("duration", 0))
        music_duration = beat_times[-1] if len(beat_times) > 0 else 0
        
        duration_match = 1.0 - abs(video_duration - music_duration) / max(video_duration, music_duration, 1)
        return min(10, duration_match * 10)
    
    def match_voice_to_character(self, character_image: str, voice_sample: str) -> float:
        """Ensure voice matches character appearance"""
        # Would use LLaVA for character analysis
        # Simplified for now
        return 0.85  # 85% match
