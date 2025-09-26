#!/usr/bin/env python3
"""
Multi-method Video Generator for Echo Brain
Supports AnimateDiff (text-to-video) and frame interpolation
"""

import json
import requests
import time
import uuid
import os
import subprocess

def create_animatediff_video(prompt, num_frames=96, fps=24):
    """Create video using AnimateDiff (text-to-video)"""
    
    print(f"🎬 Generating AnimateDiff video: {prompt[:50]}...")
    
    workflow = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": "animagine_xl_3.1.safetensors"
            }
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt + ", high quality, detailed, sharp focus, 8k",
                "clip": ["1", 1]
            }
        },
        "3": {
            "class_type": "CLIPTextEncode", 
            "inputs": {
                "text": "low quality, blurry, distorted, disfigured",
                "clip": ["1", 1]
            }
        },
        "4": {
            "class_type": "ADE_AnimateDiffLoaderGen1",
            "inputs": {
                "model_name": "mm_sd_v15_v2.ckpt",
                "beta_schedule": "sqrt_linear (AnimateDiff)",
                "model": ["1", 0]
            }
        },
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "seed": int(time.time()),
                "steps": 20,
                "cfg": 8.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["4", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["6", 0]
            }
        },
        "6": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": 512,
                "height": 512,
                "batch_size": num_frames
            }
        },
        "7": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["5", 0],
                "vae": ["1", 2]
            }
        },
        "8": {
            "class_type": "ADE_AnimateDiffCombine",
            "inputs": {
                "images": ["7", 0],
                "frame_rate": fps,
                "loop_count": 0,
                "filename_prefix": "animatediff_video",
                "format": "image/gif",
                "pingpong": False,
                "save_image": True
            }
        }
    }
    
    prompt_data = {
        "prompt": workflow,
        "client_id": str(uuid.uuid4())
    }
    
    response = requests.post("http://localhost:8188/api/prompt", json=prompt_data)
    
    if response.status_code == 200:
        result = response.json()
        prompt_id = result.get('prompt_id')
        print(f"✅ AnimateDiff started: {prompt_id}")
        return prompt_id
    else:
        print(f"❌ Failed: {response.text[:200]}")
        return None

def generate_30s_video_ffmpeg(base_image, output_path):
    """Generate 30+ second video using frame repetition and effects"""
    
    print(f"🎬 Creating 30s video from: {base_image}")
    
    cmd = f'''
    ffmpeg -loop 1 -i {base_image} -c:v libx264 -t 30 -pix_fmt yuv420p \
    -vf "scale=1920:1080,zoompan=z='min(zoom+0.0015,1.5)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=720:s=1920x1080:fps=24" \
    -preset medium -crf 23 {output_path} -y
    '''
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Created 30s video: {output_path}")
        return True
    else:
        print(f"❌ FFmpeg failed: {result.stderr[:200]}")
        return False

if __name__ == "__main__":
    # Option 1: AnimateDiff for Goblin Slayer
    prompt = "cyberpunk goblin slayer, armored warrior, neon city, rain, dark atmosphere, cinematic"
    create_animatediff_video(prompt, num_frames=96, fps=24)
    
    # Option 2: FFmpeg zoom effect from existing image
    goblin_image = "/home/patrick/ComfyUI/output/echo_goblin_slayer_cyberpunk_00001_.png"
    if os.path.exists(goblin_image):
        generate_30s_video_ffmpeg(
            goblin_image,
            "/home/patrick/Videos/goblin_slayer_30s.mp4"
        )
