#!/usr/bin/env python3
import requests
import json
import time
import sys

def create_working_video_workflow(prompt, frames=16):
    """Create a WORKING video workflow that actually produces output"""
    
    workflow = {
        # Load checkpoint
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": "animagine-xl-3.1.safetensors"  # This model EXISTS
            }
        },
        
        # Text encode positive  
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": f"anime masterpiece, {prompt}, studio quality, 4k, detailed",
                "clip": ["1", 1]
            }
        },
        
        # Text encode negative
        "3": {
            "class_type": "CLIPTextEncode", 
            "inputs": {
                "text": "low quality, blurry, static, slideshow, bad anatomy",
                "clip": ["1", 1]
            }
        },
        
        # Empty latent for video frames
        "4": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": 512,  # Start smaller to ensure GPU can handle
                "height": 512,
                "batch_size": frames
            }
        },
        
        # KSampler to generate frames
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 42,
                "steps": 20,
                "cfg": 7.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
                "model": ["1", 0],
                "denoise": 1.0
            }
        },
        
        # VAE Decode
        "6": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["5", 0],
                "vae": ["1", 2]
            }
        },
        
        # Save as video
        "7": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["6", 0],
                "frame_rate": 8,
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 19,
                "save_metadata": True,
                "filename_prefix": "anime_video"
            }
        }
    }
    
    return workflow

# Test it RIGHT NOW
print("Creating video workflow...")
workflow = create_working_video_workflow("magical girl transformation sequence", frames=16)

print("Sending to ComfyUI...")
response = requests.post(
    "http://127.0.0.1:8188/prompt",
    json={"prompt": workflow}
)

if response.status_code == 200:
    result = response.json()
    prompt_id = result.get('prompt_id')
    print(f"Queued: {prompt_id}")
    
    # Wait and check
    for i in range(60):
        time.sleep(2)
        history = requests.get(f"http://127.0.0.1:8188/history/{prompt_id}")
        if history.status_code == 200:
            data = history.json()
            if prompt_id in data:
                status = data[prompt_id].get('status', {})
                if status.get('status_str') == 'success' and status.get('completed'):
                    outputs = data[prompt_id].get('outputs', {})
                    if outputs:
                        print(f"SUCCESS! Video generated: {outputs}")
                        sys.exit(0)
                elif status.get('status_str') == 'error':
                    print(f"ERROR: {status.get('messages', 'Unknown error')}")
                    sys.exit(1)
        print(f"Processing... ({i*2}s)")
    
    print("Timeout after 120 seconds")
else:
    print(f"Failed to queue: {response.text}")
EOF'
