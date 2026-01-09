"""Echo Video Generation with Error Handling, Retries, and Persistence"""
import json
import requests
import time
import uuid
import subprocess
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoGenerator:
    def __init__(self):
        self.comfyui_api = "http://127.0.0.1:8188"
        self.output_dir = Path("***REMOVED***/ComfyUI-Working/output")
        self.video_dir = Path("***REMOVED***")
        self.video_dir.mkdir(exist_ok=True)
        self.db_path = Path("/opt/tower-echo-brain/video_projects.db")
        self.init_database()
        
    def init_database(self):
        """Initialize persistence database"""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS video_projects (
                id TEXT PRIMARY KEY,
                prompt TEXT,
                status TEXT,
                frames TEXT,
                video_path TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                error_log TEXT
            )
        ''')
        conn.commit()
        conn.close()
        
    def save_project(self, project_id: str, data: Dict):
        """Save project state to database"""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            INSERT OR REPLACE INTO video_projects 
            (id, prompt, status, frames, video_path, updated_at, error_log)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            project_id,
            data.get('prompt', ''),
            data.get('status', 'pending'),
            json.dumps(data.get('frames', [])),
            data.get('video_path', ''),
            datetime.now().isoformat(),
            data.get('error_log', '')
        ))
        conn.commit()
        conn.close()
        
    def check_comfyui_health(self) -> bool:
        """Check if ComfyUI is running and responsive"""
        try:
            resp = requests.get(f"{self.comfyui_api}/queue", timeout=2)
            return resp.status_code == 200
        except:
            return False
            
    def generate_frame_with_retry(self, prompt: str, seed: int, frame_id: str, max_retries: int = 3) -> Optional[str]:
        """Generate a single frame with retry logic"""
        workflow = {
            "1": {"class_type": "CheckpointLoaderSimple", 
                  "inputs": {"ckpt_name": "animagine_xl_3.1.safetensors"}},
            "2": {"class_type": "CLIPTextEncode", 
                  "inputs": {"text": prompt, "clip": ["1", 1]}},
            "3": {"class_type": "CLIPTextEncode", 
                  "inputs": {"text": "worst quality, low quality", "clip": ["1", 1]}},
            "4": {"class_type": "EmptyLatentImage", 
                  "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
            "5": {"class_type": "KSampler", 
                  "inputs": {
                      "seed": seed, "steps": 20, "cfg": 7.0,
                      "sampler_name": "euler", "scheduler": "normal",
                      "denoise": 1.0, "model": ["1", 0],
                      "positive": ["2", 0], "negative": ["3", 0],
                      "latent_image": ["4", 0]
                  }},
            "6": {"class_type": "VAEDecode", 
                  "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
            "7": {"class_type": "SaveImage", 
                  "inputs": {"filename_prefix": frame_id, "images": ["6", 0]}}
        }
        
        for attempt in range(max_retries):
            try:
                # Check ComfyUI health first
                if not self.check_comfyui_health():
                    logger.error(f"ComfyUI not responding, attempt {attempt+1}/{max_retries}")
                    time.sleep(5)
                    continue
                    
                resp = requests.post(f"{self.comfyui_api}/prompt", json={"prompt": workflow}, timeout=10)
                if resp.status_code == 200:
                    result = resp.json()
                    prompt_id = result.get('prompt_id')
                    logger.info(f"Frame {frame_id} queued: {prompt_id}")
                    return prompt_id
                else:
                    logger.error(f"API error {resp.status_code}: {resp.text[:100]}")
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed (attempt {attempt+1}): {e}")
                
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))  # Exponential backoff
                
        return None
        
    def wait_for_frame(self, prompt_id: str, frame_id: str, timeout: int = 60) -> Optional[Path]:
        """Wait for frame to be generated and return path"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check if frame exists
            pattern = self.output_dir / f"{frame_id}_*.png"
            found = list(self.output_dir.glob(f"{frame_id}_*.png"))
            if found:
                logger.info(f"Frame ready: {found[0].name}")
                return found[0]
                
            # Check queue status
            try:
                resp = requests.get(f"{self.comfyui_api}/history/{prompt_id}", timeout=2)
                if resp.status_code == 200:
                    history = resp.json()
                    if prompt_id in history:
                        status = history[prompt_id].get('status', {})
                        if status.get('completed'):
                            time.sleep(2)  # Give file system time
                            found = list(self.output_dir.glob(f"{frame_id}_*.png"))
                            if found:
                                return found[0]
            except:
                pass
                
            time.sleep(2)
            
        logger.error(f"Timeout waiting for frame {frame_id}")
        return None
        
    def generate_frames(self, prompt: str, num_frames: int = 10) -> List[Path]:
        """Generate frames with error handling and persistence"""
        project_id = f"video_{uuid.uuid4().hex[:8]}"
        self.save_project(project_id, {'prompt': prompt, 'status': 'generating'})
        
        generated_frames = []
        errors = []
        
        for i in range(num_frames):
            frame_id = f"echo_{project_id}_{i:03d}"
            seed = 42 + i * 100
            
            logger.info(f"Generating frame {i+1}/{num_frames}")
            prompt_id = self.generate_frame_with_retry(prompt, seed, frame_id)
            
            if prompt_id:
                frame_path = self.wait_for_frame(prompt_id, frame_id)
                if frame_path:
                    generated_frames.append(frame_path)
                else:
                    errors.append(f"Frame {i+1} generation failed")
            else:
                errors.append(f"Frame {i+1} queue failed")
                
            # Save progress
            self.save_project(project_id, {
                'prompt': prompt,
                'status': 'generating',
                'frames': [str(f) for f in generated_frames],
                'error_log': '\n'.join(errors)
            })
            
        return generated_frames
        
    def create_video_with_retry(self, frames: List[Path], output_name: str, 
                               duration: int = 30, max_retries: int = 3) -> Optional[Path]:
        """Create video with retry logic"""
        if not frames:
            return None
            
        output_path = self.video_dir / f"{output_name}.mp4"
        
        # Create input file list
        list_file = Path(f"/tmp/{output_name}_frames.txt")
        with open(list_file, "w") as f:
            frame_duration = duration / len(frames)
            for frame in frames:
                if frame.exists():
                    f.write(f"file '{frame}'\n")
                    f.write(f"duration {frame_duration}\n")
            # Last frame
            if frames[-1].exists():
                f.write(f"file '{frames[-1]}'\n")
                
        for attempt in range(max_retries):
            try:
                cmd = [
                    "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                    "-i", str(list_file),
                    "-c:v", "libx264", "-preset", "medium", "-crf", "23",
                    "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,fps=24",
                    "-pix_fmt", "yuv420p",
                    str(output_path)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                
                if result.returncode == 0 and output_path.exists():
                    logger.info(f"Video created successfully: {output_path}")
                    return output_path
                else:
                    logger.error(f"FFmpeg error (attempt {attempt+1}): {result.stderr[:200]}")
                    
            except subprocess.TimeoutExpired:
                logger.error(f"FFmpeg timeout (attempt {attempt+1})")
            except Exception as e:
                logger.error(f"Video creation error: {e}")
                
            if attempt < max_retries - 1:
                time.sleep(5)
                
        return None
        
    def generate_video(self, prompt: str, duration: int = 30) -> Dict:
        """Main pipeline with full error handling"""
        logger.info(f"Starting video generation: {prompt[:50]}...")
        
        # Check ComfyUI first
        if not self.check_comfyui_health():
            return {"status": "error", "message": "ComfyUI not responding"}
            
        # Generate frames
        num_frames = max(5, min(30, duration // 2))
        frames = self.generate_frames(prompt, num_frames)
        
        if not frames:
            return {"status": "error", "message": "No frames could be generated"}
            
        logger.info(f"Generated {len(frames)} frames successfully")
        
        # Create video
        video_name = f"echo_{uuid.uuid4().hex[:8]}"
        video_path = self.create_video_with_retry(frames, video_name, duration)
        
        if video_path and video_path.exists():
            # Save successful project
            project_data = {
                'prompt': prompt,
                'status': 'completed',
                'frames': [str(f) for f in frames],
                'video_path': str(video_path)
            }
            self.save_project(video_name, project_data)
            
            return {
                "status": "success",
                "video_path": str(video_path),
                "frames_generated": len(frames),
                "duration": duration
            }
        else:
            return {"status": "error", "message": "Video creation failed after retries"}

# Test with proper error handling
if __name__ == "__main__":
    gen = VideoGenerator()
    if gen.check_comfyui_health():
        print("ComfyUI is healthy, generating video...")
        result = gen.generate_video("goblin slayer cyberpunk armor, neon city", duration=15)
        print(f"Result: {result}")
    else:
        print("ERROR: ComfyUI is not responding!")
