import json
import sqlite3
import time
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from src.modules.vision_quality_checker import VisionQualityChecker
from src.modules.music_quality_checker import MusicQualityChecker
from src.modules.voice_quality_checker import VoiceQualityChecker
from src.modules.cross_modal_sync import CrossModalSync
from src.modules.kb_video_verifier import KBVideoVerifier

class UnifiedMultimediaPipeline:
    """Unified pipeline for all multimedia generation"""
    
    def __init__(self):
        self.vision_checker = VisionQualityChecker()
        self.music_checker = MusicQualityChecker()
        self.voice_checker = VoiceQualityChecker()
        self.sync_checker = CrossModalSync()
        self.kb_verifier = KBVideoVerifier()
        
        # Learning system
        self.db = sqlite3.connect('/opt/tower-echo-brain/quality_metrics.db')
        self.workflows_path = '/opt/tower-echo-brain/successful_workflows.json'
        
        # Quality thresholds
        self.min_quality = 7.0
        self.max_retries = 3
        
    def generate_multimedia(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Generate complete multimedia with quality checks"""
        media_type = request.get("type", "video")
        
        for attempt in range(self.max_retries):
            # Generate media
            result = self._generate_media(media_type, request, attempt)
            
            if not result["success"]:
                continue
            
            # Quality check
            quality = self._check_quality(media_type, result["output_path"])
            
            if quality["overall_quality"] >= self.min_quality:
                # Cross-modal sync if multiple media
                if "music_path" in result and "video_path" in result:
                    sync_score = self.sync_checker.sync_video_to_music(
                        result["video_path"], result["music_path"]
                    )
                    quality["sync_quality"] = sync_score["sync_quality"]
                
                # Save successful pattern
                self._save_success(media_type, request, quality)
                
                return {
                    "success": True,
                    "output": result,
                    "quality": quality,
                    "attempts": attempt + 1
                }
            
            # Adjust parameters for retry
            request = self._adjust_parameters(media_type, quality, request)
            
            # Exponential backoff
            time.sleep(5 * (attempt + 1))
        
        return {
            "success": False,
            "error": "Failed to meet quality standards after retries",
            "attempts": self.max_retries
        }
    
    def _generate_media(self, media_type: str, request: Dict, attempt: int) -> Dict:
        """Generate media based on type"""
        # Placeholder - would call actual generators
        return {
            "success": True,
            "output_path": f"/tmp/{media_type}_{attempt}.mp4"
        }
    
    def _check_quality(self, media_type: str, path: str) -> Dict:
        """Check quality based on media type"""
        if media_type == "video":
            return self.vision_checker.check_frame_quality(path)
        elif media_type == "music":
            return self.music_checker.check_audio_quality(path)
        elif media_type == "voice":
            return self.voice_checker.check_voice_quality(path)
        return {"overall_quality": 0}
    
    def _adjust_parameters(self, media_type: str, quality: Dict, params: Dict) -> Dict:
        """Intelligent parameter adjustment"""
        new_params = params.copy()
        
        if media_type == "video" and quality["overall_quality"] < 7:
            new_params["steps"] = params.get("steps", 20) + 10
            new_params["cfg_scale"] = params.get("cfg_scale", 7) + 1
            
        elif media_type == "music" and quality.get("tempo_consistency", 10) < 7:
            new_params["quantization"] = True
            new_params["tempo_lock"] = True
            
        elif media_type == "voice" and quality.get("clarity", 10) < 7:
            new_params["noise_reduction"] = True
            new_params["enhance_clarity"] = True
        
        return new_params
    
    def _save_success(self, media_type: str, params: Dict, quality: Dict):
        """Save successful generation pattern"""
        # Load existing workflows
        workflows = []
        if Path(self.workflows_path).exists():
            with open(self.workflows_path, 'r') as f:
                workflows = json.load(f)
        
        # Add new success
        workflows.append({
            "media_type": media_type,
            "parameters": params,
            "quality": quality,
            "timestamp": datetime.now().isoformat()
        })
        
        # Save
        with open(self.workflows_path, 'w') as f:
            json.dump(workflows, f, indent=2)
        
        # Update database
        cursor = self.db.cursor()
        cursor.execute("""
            INSERT INTO quality_scores
            (media_type, parameters, quality_score, success)
            VALUES (?, ?, ?, ?)
        """, (media_type, json.dumps(params), quality["overall_quality"], True))
        self.db.commit()

    def _set_proper_permissions(self, file_path: str):
        """Set proper permissions for Jellyfin access - learned from Patrick's feedback"""
        import os
        import stat
        
        try:
            # Set 644 permissions (owner read/write, group/others read)
            os.chmod(file_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
            
            # Log this success pattern
            self._save_permission_success(file_path)
            
            return True
        except Exception as e:
            self._log_permission_failure(file_path, str(e))
            return False
    
    def _save_permission_success(self, file_path: str):
        """Save successful permission setting as training data"""
        success_data = {
            "timestamp": datetime.now().isoformat(),
            "action": "set_video_permissions", 
            "file": file_path,
            "permissions": "644",
            "result": "jellyfin_accessible",
            "lesson": "Always set 644 on generated videos for media server access"
        }
        
        # Append to training data
        with open('/opt/tower-echo-brain/permission_training.json', 'a') as f:
            f.write(json.dumps(success_data) + '\n')

