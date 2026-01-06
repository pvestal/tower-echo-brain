#!/usr/bin/env python3
"""Fixed SVD Video Generator using available nodes"""

import json
import requests
import time
import uuid
import os

def create_svd_workflow(image_path, num_frames=25, motion_bucket_id=127, fps=8):
    """Create SVD workflow with available nodes"""
    
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
                "augmentation_level": 0.0
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
            "class_type": "SaveAnimatedWEBP",  # Use this instead of VHS
            "inputs": {
                "images": ["5", 0],
                "filename_prefix": "svd_goblin",
                "fps": fps,
                "lossless": False,
                "quality": 80,
                "method": "default"
            }
        }
    }
    
    return workflow

def generate_svd_frames(image_path):
    """Generate SVD frames and save as images"""
    print(f"🎬 Generating SVD frames from: {image_path}")
    
    # First, upload the image if needed
    with open(image_path, 'rb') as f:
        files = {'image': (os.path.basename(image_path), f, 'image/png')}
        upload_resp = requests.post('http://localhost:8188/upload/image', files=files)
        print(f"Upload response: {upload_resp.status_code}")
    
    workflow = create_svd_workflow(image_path, num_frames=25, motion_bucket_id=150)
    
    prompt = {
        "prompt": workflow,
        "client_id": str(uuid.uuid4())
    }
    
    response = requests.post("http://localhost:8188/api/prompt", json=prompt)
    
    if response.status_code == 200:
        result = response.json()
        prompt_id = result.get('prompt_id')
        print(f"✅ SVD generation started: {prompt_id}")
        return prompt_id
    else:
        print(f"❌ Failed: {response.text}")
        return None

if __name__ == "__main__":
    goblin_image = "/home/patrick/ComfyUI/output/echo_goblin_slayer_cyberpunk_00001_.png"
    if os.path.exists(goblin_image):
        generate_svd_frames(goblin_image)
    else:
        print(f"❌ Image not found: {goblin_image}")
