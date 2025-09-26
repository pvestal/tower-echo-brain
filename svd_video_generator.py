#!/usr/bin/env python3
"""
SVD (Stable Video Diffusion) Video Generator for Echo Brain
Turns static images into dynamic videos with motion
"""

import json
import requests
import time
import uuid
import os

def create_svd_workflow(image_path, num_frames=25, motion_bucket_id=127, fps=8, augmentation_level=0.0):
    """
    Create SVD workflow to turn image into video
    
    Args:
        image_path: Path to input image
        num_frames: Number of frames to generate (14 or 25 for SVD)
        motion_bucket_id: Controls amount of motion (1-255, higher = more motion)
        fps: Frames per second for output
        augmentation_level: Noise augmentation (0.0-1.0)
    """
    
    workflow = {
        "1": {
            "class_type": "LoadImage",
            "inputs": {
                "image": os.path.basename(image_path),
                "upload": "image"
            }
        },
        "2": {
            "class_type": "ImageOnlyCheckpointLoader", 
            "inputs": {
                "ckpt_name": "svd.safetensors"
            }
        },
        "3": {
            "class_type": "SVD_img2vid_Conditioning",
            "inputs": {
                "clip_vision": ["2", 1],
                "init_image": ["1", 0],
                "vae": ["2", 2],
                "width": 1024,
                "height": 576,
                "num_frames": num_frames,
                "motion_bucket_id": motion_bucket_id,
                "fps": fps,
                "augmentation_level": augmentation_level
            }
        },
        "4": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["2", 0],
                "positive": ["3", 0],
                "negative": ["3", 1],
                "seed": int(time.time()),
                "steps": 25,
                "cfg": 2.5,
                "sampler_name": "euler",
                "scheduler": "sgm_uniform",
                "denoise": 1.0,
                "latent_image": ["3", 2]
            }
        },
        "5": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["4", 0],
                "vae": ["2", 2]
            }
        },
        "6": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["5", 0],
                "frame_rate": fps,
                "loop_count": 0,
                "filename_prefix": "svd_goblin_slayer",
                "format": "video/h264-mp4",
                "pingpong": False,
                "save_output": True
            }
        }
    }
    
    return workflow

def generate_svd_video(image_path, output_prefix="svd_video", **kwargs):
    """Generate video from image using SVD"""
    
    print(f"🎬 Generating video from: {image_path}")
    
    # Create workflow
    workflow = create_svd_workflow(image_path, **kwargs)
    
    # Send to ComfyUI
    prompt = {
        "prompt": workflow,
        "client_id": str(uuid.uuid4())
    }
    
    response = requests.post(
        "http://localhost:8188/api/prompt",
        json=prompt
    )
    
    if response.status_code == 200:
        result = response.json()
        prompt_id = result.get('prompt_id')
        print(f"✅ SVD generation started: {prompt_id}")
        return prompt_id
    else:
        print(f"❌ Failed to start SVD: {response.text}")
        return None

def create_extended_video(image_path, duration_seconds=30):
    """
    Create extended video by generating multiple SVD clips
    and combining them for longer duration
    """
    
    print(f"🎬 Creating {duration_seconds}s video using SVD")
    
    # SVD generates ~3 seconds per run at 8fps
    # We'll generate multiple variations and combine
    
    clips_needed = duration_seconds // 3
    prompt_ids = []
    
    # Generate multiple clips with different motion settings
    for i in range(clips_needed):
        motion = 100 + (i * 20)  # Vary motion for variety
        prompt_id = generate_svd_video(
            image_path,
            output_prefix=f"svd_clip_{i}",
            num_frames=25,
            motion_bucket_id=min(motion, 200),
            fps=8
        )
        if prompt_id:
            prompt_ids.append(prompt_id)
        time.sleep(2)  # Space out requests
    
    return prompt_ids

if __name__ == "__main__":
    # Test with Goblin Slayer image
    goblin_image = "/home/patrick/ComfyUI/output/echo_goblin_slayer_cyberpunk_00001_.png"
    
    if os.path.exists(goblin_image):
        # Generate single SVD clip
        print("=== GENERATING SINGLE SVD CLIP ===")
        generate_svd_video(goblin_image, motion_bucket_id=150)
        
        # For 30+ second video, we'd combine multiple clips
        print("\n=== FOR 30+ SECOND VIDEO ===")
        print("Would generate multiple clips and combine with FFmpeg")
        print("Each SVD run = ~3 seconds, so need 10+ clips for 30s")
    else:
        print(f"❌ Image not found: {goblin_image}")
