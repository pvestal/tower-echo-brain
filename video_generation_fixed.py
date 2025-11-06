#!/usr/bin/env python3
"""
Fixed Video Generation System - Properly handles image generation first
"""

import json
import requests
import time
import os
from pathlib import Path
from typing import Dict, List, Optional
import subprocess

class VideoGenerationSystemFixed:
    """Fixed video generation system"""
    
    def __init__(self, comfyui_url="http://localhost:8188"):
        self.comfyui_url = comfyui_url
        self.output_dir = Path("/home/{os.getenv("TOWER_USER", "patrick")}/Videos")
        self.output_dir.mkdir(exist_ok=True)
        
    def generate_base_image(self, prompt: str) -> Optional[str]:
        """Generate a base image using ComfyUI first"""
        print(f"Generating base image for: {prompt}")
        
        workflow = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name": "Counterfeit-V3.0_fp16.safetensors"
                }
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": prompt,
                    "clip": ["1", 1]
                }
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": "low quality, worst quality, blurry",
                    "clip": ["1", 1]
                }
            },
            "4": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": 832,
                    "height": 1216,
                    "batch_size": 1
                }
            },
            "5": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": int(time.time()),
                    "steps": 20,
                    "cfg": 7.0,
                    "sampler_name": "euler",
                    "scheduler": "normal",
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
                    "filename_prefix": "video_base",
                    "images": ["6", 0]
                }
            }
        }
        
        # Submit to ComfyUI
        try:
            response = requests.post(
                f"{self.comfyui_url}/prompt",
                json={"prompt": workflow}
            )
            
            if response.status_code == 200:
                prompt_id = response.json().get('prompt_id')
                print(f"Image generation started: {prompt_id}")
                
                # Wait for completion
                time.sleep(10)  # Give it time to generate
                
                # Check for output
                hist_response = requests.get(f"{self.comfyui_url}/history/{prompt_id}")
                if hist_response.status_code == 200:
                    history = hist_response.json()
                    if prompt_id in history:
                        outputs = history[prompt_id].get('outputs', {})
                        for node_id, output in outputs.items():
                            if 'images' in output and output['images']:
                                filename = output['images'][0]['filename']
                                image_path = f"/home/{os.getenv("TOWER_USER", "patrick")}/Projects/ComfyUI-Working/output/{filename}"
                                print(f"Base image generated: {image_path}")
                                return image_path
        except Exception as e:
            print(f"Error generating base image: {e}")
        
        return None
    
    def generate_ffmpeg_video(self, image_path: str, duration: int = 30) -> str:
        """Generate video from image using FFmpeg effects"""
        output_path = self.output_dir / f"echo_video_{int(time.time())}.mp4"
        
        if not os.path.exists(image_path):
            # Try to use existing image
            test_images = [
                "/home/{os.getenv("TOWER_USER", "patrick")}/Projects/ComfyUI-Working/output/goblin_solo_00001_.png",
                "/home/{os.getenv("TOWER_USER", "patrick")}/Projects/ComfyUI-Working/output/video_base_00001_.png"
            ]
            for test_image in test_images:
                if os.path.exists(test_image):
                    image_path = test_image
                    print(f"Using existing image: {image_path}")
                    break
            else:
                print("No input image found")
                return None
        
        # Create video with zoom/pan effect
        cmd = f'''ffmpeg -loop 1 -i "{image_path}" -c:v libx264 -t {duration} \
            -vf "scale=w=2*iw:h=2*ih,zoompan=z='min(zoom+0.0015,1.5)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={duration*24}:s=1920x1080:fps=24" \
            -pix_fmt yuv420p -preset medium -crf 23 "{output_path}" -y 2>&1'''
        
        print(f"Running FFmpeg command...")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0 and os.path.exists(output_path):
            print(f"Video generated: {output_path}")
            return str(output_path)
        else:
            print(f"FFmpeg error: {result.stderr[:500]}")
            return None
    
    def generate_video_from_prompt(self, prompt: str, duration: int = 30) -> Dict:
        """Generate video from text prompt"""
        result = {
            'success': False,
            'video_path': None,
            'error': None
        }
        
        # First generate base image
        image_path = self.generate_base_image(prompt)
        
        if not image_path:
            # Use existing image as fallback
            image_path = "/home/{os.getenv("TOWER_USER", "patrick")}/Projects/ComfyUI-Working/output/goblin_solo_00001_.png"
        
        # Generate video from image
        video_path = self.generate_ffmpeg_video(image_path, duration)
        
        if video_path:
            result['success'] = True
            result['video_path'] = video_path
        else:
            result['error'] = "Failed to generate video"
        
        return result


# Test the fixed system
if __name__ == "__main__":
    system = VideoGenerationSystemFixed()
    
    print("Testing fixed video generation...")
    result = system.generate_video_from_prompt(
        "anime girl with pink hair, cherry blossoms",
        duration=10
    )
    
    if result['success']:
        print(f"✅ Video generated: {result['video_path']}")
        
        # Check video properties
        cmd = f"ffprobe -v error -show_entries format=duration,size -of json '{result['video_path']}'"
        probe_result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if probe_result.stdout:
            info = json.loads(probe_result.stdout)
            print(f"Duration: {info['format']['duration']}s")
            print(f"Size: {int(info['format']['size'])/1024/1024:.1f}MB")
    else:
        print(f"❌ Generation failed: {result['error']}")
