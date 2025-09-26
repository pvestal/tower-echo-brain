#!/usr/bin/env python3
"""
Echo Brain Video Generation Module
Supports multiple video generation methods with Article 71 compliance
"""

import json
import requests
import time
import uuid
import os
import subprocess
from typing import Optional, Dict, List
import asyncio

class EchoVideoGenerator:
    """Echo's intelligent video generation system"""
    
    def __init__(self, comfyui_url="http://localhost:8188"):
        self.comfyui_url = comfyui_url
        self.output_dir = "/home/patrick/Videos"
        self.comfyui_output = "/home/patrick/ComfyUI/output"
        
    def generate_video_from_prompt(self, prompt: str, character: str = None, duration: int = 30) -> Dict:
        """
        Main entry point for video generation
        Intelligently selects best method based on requirements
        """
        
        print(f"🎬 Echo Brain Video Generation")
        print(f"   Prompt: {prompt[:100]}...")
        print(f"   Duration: {duration}s")
        print(f"   Character: {character or 'auto-detect'}")
        
        # Step 1: Generate high-quality base image
        image_path = self._generate_base_image(prompt, character)
        
        if not image_path:
            return {"error": "Failed to generate base image"}
        
        # Step 2: Choose video generation method
        if duration <= 5:
            # Short video - use AnimateDiff
            video_path = self._generate_animatediff_video(prompt, duration)
        elif duration <= 30:
            # Medium video - use FFmpeg with effects
            video_path = self._generate_ffmpeg_video(image_path, duration)
        else:
            # Long video - combine multiple methods
            video_path = self._generate_long_video(prompt, image_path, duration)
        
        # Step 3: Verify Article 71 compliance
        if video_path and os.path.exists(video_path):
            quality_check = self._verify_quality(video_path)
            
            return {
                "success": True,
                "video_path": video_path,
                "duration": duration,
                "quality_score": quality_check['score'],
                "method_used": quality_check['method'],
                "character": character,
                "prompt": prompt
            }
        else:
            return {"error": "Video generation failed"}
    
    def _generate_base_image(self, prompt: str, character: str = None) -> Optional[str]:
        """Generate high-quality base image using ComfyUI"""
        
        # Build quality prompt
        full_prompt = prompt
        if character:
            full_prompt = f"{character}, {prompt}"
        
        full_prompt += ", ((solo)), ((single character only)), high quality, detailed, sharp focus, 8k, HD, production quality"
        
        workflow = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "animagine_xl_3.1.safetensors"}
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": full_prompt,
                    "clip": ["1", 1]
                }
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": "((multiple people)), ((crowd)), blur, low quality, distorted",
                    "clip": ["1", 1]
                }
            },
            "4": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": 1920,
                    "height": 1080,
                    "batch_size": 1
                }
            },
            "5": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": int(time.time()),
                    "steps": 35,
                    "cfg": 9.5,
                    "sampler_name": "dpmpp_2m_sde",
                    "scheduler": "karras",
                    "denoise": 1.0,
                    "model": ["1", 0],
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "latent_image": ["4", 0]
                }
            },
            "6": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["5", 0],
                    "vae": ["1", 2]
                }
            },
            "7": {
                "class_type": "SaveImage",
                "inputs": {
                    "images": ["6", 0],
                    "filename_prefix": "echo_video_base"
                }
            }
        }
        
        # Send to ComfyUI
        response = requests.post(
            f"{self.comfyui_url}/api/prompt",
            json={"prompt": workflow, "client_id": str(uuid.uuid4())}
        )
        
        if response.status_code == 200:
            # Wait for generation
            time.sleep(15)
            
            # Find generated image
            latest = max(
                [f for f in os.listdir(self.comfyui_output) if f.startswith("echo_video_base")],
                key=lambda x: os.path.getctime(os.path.join(self.comfyui_output, x)),
                default=None
            )
            
            if latest:
                return os.path.join(self.comfyui_output, latest)
        
        return None
    
    def _generate_ffmpeg_video(self, image_path: str, duration: int) -> Optional[str]:
        """Generate video using FFmpeg effects"""
        
        output_path = os.path.join(self.output_dir, f"echo_video_{int(time.time())}.mp4")
        
        # Advanced FFmpeg with multiple effects
        cmd = f'''ffmpeg -loop 1 -i {image_path} \
        -c:v libx264 -t {duration} -pix_fmt yuv420p \
        -vf "scale=1920:1080,zoompan=z='min(zoom+0.0015,1.5)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={duration*24}:s=1920x1080:fps=24,\
        fade=in:0:24,fade=out:{duration*24-24}:24" \
        -preset medium -crf 23 -b:v 10M {output_path} -y'''
        
        result = subprocess.run(cmd, shell=True, capture_output=True)
        
        if result.returncode == 0:
            return output_path
        
        return None
    
    def _generate_animatediff_video(self, prompt: str, duration: int) -> Optional[str]:
        """Generate video using AnimateDiff"""
        
        # AnimateDiff workflow (simplified)
        num_frames = min(duration * 24, 96)  # Cap at 96 frames
        
        # Similar workflow to earlier
        # ... (AnimateDiff implementation)
        
        return None  # Placeholder
    
    def _generate_long_video(self, prompt: str, image_path: str, duration: int) -> Optional[str]:
        """Generate long video by combining multiple segments"""
        
        segments = []
        segment_duration = 10
        
        for i in range(0, duration, segment_duration):
            seg_path = self._generate_ffmpeg_video(image_path, segment_duration)
            if seg_path:
                segments.append(seg_path)
        
        # Combine segments
        if segments:
            output_path = os.path.join(self.output_dir, f"echo_long_video_{int(time.time())}.mp4")
            concat_file = "/tmp/concat.txt"
            
            with open(concat_file, 'w') as f:
                for seg in segments:
                    f.write(f"file '{seg}'\n")
            
            cmd = f"ffmpeg -f concat -safe 0 -i {concat_file} -c copy {output_path} -y"
            subprocess.run(cmd, shell=True)
            
            return output_path
        
        return None
    
    def _verify_quality(self, video_path: str) -> Dict:
        """Verify video meets Article 71 standards"""
        
        cmd = f"ffprobe -v error -show_entries format=duration -show_entries stream=width,height,r_frame_rate -of json {video_path}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            info = json.loads(result.stdout)
            duration = float(info['format']['duration'])
            width = info['streams'][0]['width']
            height = info['streams'][0]['height']
            
            # Calculate quality score
            score = 100
            if duration < 30:
                score -= 20
            if width < 1920 or height < 1080:
                score -= 15
            
            return {
                "score": score,
                "duration": duration,
                "resolution": f"{width}x{height}",
                "method": "ffmpeg" if duration == 30 else "combined"
            }
        
        return {"score": 0, "method": "unknown"}

# Integration with Echo Brain
async def handle_video_request(message: str) -> Dict:
    """Echo Brain handler for video generation requests"""
    
    generator = EchoVideoGenerator()
    
    # Parse request
    if "goblin slayer" in message.lower():
        character = "goblin slayer"
        prompt = "cyberpunk warrior, armored, neon city, rain, dark atmosphere"
    else:
        character = None
        prompt = message
    
    # Generate video
    result = generator.generate_video_from_prompt(prompt, character, duration=30)
    
    return result

if __name__ == "__main__":
    # Test video generation
    print("=== ECHO BRAIN VIDEO GENERATION TEST ===")
    generator = EchoVideoGenerator()
    
    result = generator.generate_video_from_prompt(
        "cyberpunk goblin slayer in neon city",
        character="goblin slayer",
        duration=30
    )
    
    print(f"\nResult: {json.dumps(result, indent=2)}")
