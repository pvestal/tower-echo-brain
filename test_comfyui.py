#!/usr/bin/env python3
import requests
import json
import random

# Test ComfyUI connection
comfyui_url = "http://localhost:8188"

# Simple test workflow
workflow = {
    "prompt": {
        "1": {
            "inputs": {
                "ckpt_name": "realisticVision_v51.safetensors"
            },
            "class_type": "CheckpointLoaderSimple"
        },
        "2": {
            "inputs": {
                "text": "young Japanese man with two beautiful women, photorealistic",
                "clip": ["1", 1]
            },
            "class_type": "CLIPTextEncode"
        },
        "3": {
            "inputs": {
                "text": "cartoon, anime, bad quality",
                "clip": ["1", 1]
            },
            "class_type": "CLIPTextEncode"
        },
        "4": {
            "inputs": {
                "seed": random.randint(1, 1000000000),
                "steps": 20,
                "cfg": 7,
                "sampler_name": "dpmpp_2m",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["5", 0]
            },
            "class_type": "KSampler"
        },
        "5": {
            "inputs": {
                "width": 512,
                "height": 512,
                "batch_size": 1
            },
            "class_type": "EmptyLatentImage"
        },
        "6": {
            "inputs": {
                "samples": ["4", 0],
                "vae": ["1", 2]
            },
            "class_type": "VAEDecode"
        },
        "7": {
            "inputs": {
                "filename_prefix": "test_multi_character",
                "images": ["6", 0]
            },
            "class_type": "SaveImage"
        }
    }
}

try:
    print("Testing ComfyUI...")

    # Check status
    status = requests.get(f"{comfyui_url}/system_stats")
    print(f"ComfyUI Status: {status.status_code}")

    # Send workflow
    response = requests.post(
        f"{comfyui_url}/prompt",
        json={"prompt": workflow["prompt"]}
    )

    print(f"Response Status: {response.status_code}")
    print(f"Response: {response.text[:500]}")

    if response.status_code == 200:
        result = response.json()
        if "prompt_id" in result:
            print(f"✅ Success! Prompt ID: {result['prompt_id']}")
        else:
            print(f"❌ No prompt_id in response: {result}")
    else:
        print(f"❌ Failed with status {response.status_code}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()