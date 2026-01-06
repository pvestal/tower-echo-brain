"""Echo Brain Video Generation Module"""
import json
import requests
import time
import uuid
import subprocess
import asyncio
from pathlib import Path
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoGenerator:
    def __init__(self):
        self.comfyui_api = "http://127.0.0.1:8188"
        self.output_dir = Path("/home/patrick/Projects/ComfyUI-Working/output")
        self.video_dir = Path("/home/patrick/Videos")
        
        self.style_presets = {
            "cyberpunk": {
                "suffix": ", neon lights, rain, dystopian city, cyber enhancements",
                "model": "animagine_xl_3.1.safetensors"
            },
            "anime": {
                "suffix": ", anime style, masterpiece, best quality",
                "model": "animagine_xl_3.1.safetensors"
            }
        }
    
    async def generate_video(self, prompt: str, style: str = "anime", duration: int = 30) -> Dict:
        """Generate video from prompt"""
        preset = self.style_presets.get(style, self.style_presets["anime"])
        full_prompt = prompt + preset["suffix"]
        
        # Generate frames
        num_frames = max(10, duration // 3)
        frames = []
        
        for i in range(num_frames):
            workflow = {
                "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": preset["model"]}},
                "2": {"class_type": "CLIPTextEncode", "inputs": {"text": full_prompt, "clip": ["1", 1]}},
                "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "worst quality", "clip": ["1", 1]}},
                "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
                "5": {"class_type": "KSampler", "inputs": {
                    "seed": 42 + i * 100, "steps": 20, "cfg": 7.0, "sampler_name": "euler",
                    "scheduler": "normal", "denoise": 1.0, "model": ["1", 0],
                    "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0]
                }},
                "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
                "7": {"class_type": "SaveImage", "inputs": {
                    "filename_prefix": f"echo_vid_{uuid.uuid4().hex[:8]}", "images": ["6", 0]
                }}
            }
            
            try:
                response = requests.post(f"{self.comfyui_api}/prompt", json={"prompt": workflow})
                if response.status_code == 200:
                    frames.append(workflow["7"]["inputs"]["filename_prefix"])
                    logger.info(f"Frame {i+1}/{num_frames} queued")
            except Exception as e:
                logger.error(f"Frame generation error: {e}")
            
            await asyncio.sleep(2)
        
        # Wait for generation
        await asyncio.sleep(num_frames * 3 + 10)
        
        # Create video
        video_name = f"echo_{style}_{uuid.uuid4().hex[:8]}"
        video_path = self.video_dir / f"{video_name}.mp4"
        
        # Create frame list
        list_file = Path("/tmp/echo_frames.txt")
        with open(list_file, "w") as f:
            for frame in frames:
                frame_path = self.output_dir / f"{frame}_00001_.png"
                if frame_path.exists():
                    for _ in range(3):  # Repeat each frame
                        f.write(f"file '{frame_path}'\n")
        
        # FFmpeg command
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
            "-c:v", "libx264", "-preset", "slow", "-crf", "18",
            "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080",
            "-pix_fmt", "yuv420p", "-r", "24", str(video_path)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return {"status": "success", "video_path": str(video_path), "frames": len(frames)}
        except Exception as e:
            return {"status": "error", "message": str(e)}
