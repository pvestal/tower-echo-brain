"""Echo Brain Video Generation with SVD Support"""
import json
import requests
import time
import uuid
import subprocess
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
        self.video_dir.mkdir(exist_ok=True)
        
    def generate_frames(self, prompt: str, num_frames: int = 10) -> List[Path]:
        """Generate frames using ComfyUI"""
        generated = []
        
        for i in range(num_frames):
            frame_id = f"echo_{uuid.uuid4().hex[:8]}"
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
                          "seed": 42 + i * 100,
                          "steps": 20,
                          "cfg": 7.0,
                          "sampler_name": "euler",
                          "scheduler": "normal",
                          "denoise": 1.0,
                          "model": ["1", 0],
                          "positive": ["2", 0],
                          "negative": ["3", 0],
                          "latent_image": ["4", 0]
                      }},
                "6": {"class_type": "VAEDecode", 
                      "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
                "7": {"class_type": "SaveImage", 
                      "inputs": {"filename_prefix": frame_id, "images": ["6", 0]}}
            }
            
            try:
                resp = requests.post(f"{self.comfyui_api}/prompt", json={"prompt": workflow})
                if resp.status_code == 200:
                    generated.append(frame_id)
                    logger.info(f"Frame {i+1}/{num_frames}: {frame_id}")
                time.sleep(3)
            except Exception as e:
                logger.error(f"Frame error: {e}")
        
        # Wait for generation
        logger.info(f"Waiting for {len(generated)} frames to complete...")
        time.sleep(len(generated) * 5 + 10)
        
        # Find actual generated files
        actual_frames = []
        for frame_id in generated:
            pattern = self.output_dir / f"{frame_id}_*.png"
            found = list(self.output_dir.glob(f"{frame_id}_*.png"))
            if found:
                actual_frames.append(found[0])
                logger.info(f"Found: {found[0].name}")
        
        return actual_frames
    
    def apply_svd_interpolation(self, frames: List[Path]) -> List[Path]:
        """Apply SVD interpolation between frames for smoothness"""
        # Check if SVD model is available
        svd_workflow = {
            "1": {"class_type": "CheckpointLoaderSimple",
                  "inputs": {"ckpt_name": "svd.safetensors"}},
            "2": {"class_type": "LoadImage",
                  "inputs": {"image": str(frames[0])}},
            "3": {"class_type": "SVDimg2vid",
                  "inputs": {
                      "init_image": ["2", 0],
                      "frames": 14,
                      "motion_bucket_id": 127,
                      "fps": 7,
                      "augmentation_level": 0.0,
                      "model": ["1", 0]
                  }},
            "4": {"class_type": "SaveAnimatedWEBP",
                  "inputs": {"images": ["3", 0], "filename_prefix": "svd_interpolated"}}
        }
        
        # For now, return original frames
        # TODO: Implement actual SVD interpolation when model is available
        logger.info("SVD interpolation not yet implemented, using original frames")
        return frames
    
    def create_video(self, frames: List[Path], output_name: str, duration: int = 30) -> Path:
        """Create video from frames"""
        if not frames:
            raise ValueError("No frames to create video")
        
        output_path = self.video_dir / f"{output_name}.mp4"
        
        # Calculate how many times to repeat each frame
        target_fps = 24
        total_frames_needed = duration * target_fps
        repeat_count = max(1, total_frames_needed // len(frames))
        
        # Create concat file
        concat_file = Path("/tmp/echo_concat.txt")
        with open(concat_file, "w") as f:
            for frame in frames:
                if frame.exists():
                    # Each frame duration
                    frame_duration = duration / len(frames)
                    f.write(f"file '{frame}'\n")
                    f.write(f"duration {frame_duration}\n")
            # Last frame needs explicit duration
            if frames and frames[-1].exists():
                f.write(f"file '{frames[-1]}'\n")
        
        # FFmpeg command
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,fps=24",
            "-pix_fmt", "yuv420p",
            str(output_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr}")
                return None
            logger.info(f"Video created: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Video creation failed: {e}")
            return None
    
    def generate_video(self, prompt: str, duration: int = 30, use_svd: bool = False) -> Dict:
        """Complete pipeline"""
        logger.info(f"Generating video: {prompt[:50]}...")
        
        # Generate frames
        num_frames = max(5, min(30, duration // 2))
        frames = self.generate_frames(prompt, num_frames)
        
        if not frames:
            return {"status": "error", "message": "No frames generated"}
        
        # Apply SVD if requested
        if use_svd:
            frames = self.apply_svd_interpolation(frames)
        
        # Create video
        video_name = f"echo_{uuid.uuid4().hex[:8]}"
        video_path = self.create_video(frames, video_name, duration)
        
        if video_path and video_path.exists():
            return {
                "status": "success",
                "video_path": str(video_path),
                "frames_generated": len(frames),
                "duration": duration
            }
        else:
            return {"status": "error", "message": "Video creation failed"}

# Test it
if __name__ == "__main__":
    gen = VideoGenerator()
    result = gen.generate_video("goblin slayer cyberpunk armor, neon city", duration=15)
    print(f"Result: {result}")
