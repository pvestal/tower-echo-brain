import librosa
import numpy as np
import json
from typing import Dict, Any

class MusicQualityChecker:
    """Verify music quality using audio analysis"""
    
    def __init__(self):
        self.min_quality = 7.0
    
    def check_audio_quality(self, audio_path: str) -> Dict[str, Any]:
        """Analyze audio quality metrics"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=None)
            
            # Calculate metrics
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            rms_energy = librosa.feature.rms(y=y)
            
            # Tempo consistency check
            beat_times = librosa.frames_to_time(beats, sr=sr)
            tempo_consistency = self._calculate_tempo_consistency(beat_times)
            
            # Dynamic range
            dynamic_range = 10 * np.log10(np.max(rms_energy) / (np.min(rms_energy) + 1e-10))
            
            # Harmonic quality (simplified)
            harmonic_quality = min(10, spectral_centroid.std() / 100)
            
            quality_metrics = {
                "tempo": float(tempo),
                "tempo_consistency": min(10, tempo_consistency * 10),
                "harmonic_quality": harmonic_quality,
                "dynamic_range": min(10, dynamic_range / 3),
                "overall_quality": 0
            }
            
            # Calculate overall
            quality_metrics["overall_quality"] = np.mean([
                quality_metrics["tempo_consistency"],
                quality_metrics["harmonic_quality"],
                quality_metrics["dynamic_range"]
            ])
            
            return quality_metrics
            
        except Exception as e:
            return {
                "error": str(e),
                "overall_quality": 0
            }
    
    def _calculate_tempo_consistency(self, beat_times):
        """Check if tempo is consistent"""
        if len(beat_times) < 2:
            return 1.0
        intervals = np.diff(beat_times)
        return 1.0 - (intervals.std() / (intervals.mean() + 1e-10))
