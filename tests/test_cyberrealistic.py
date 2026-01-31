#!/usr/bin/env python3
"""Test CyberRealistic v9 for sharp photorealistic quality"""
import requests
import json
import random

comfyui_url = "http://localhost:8188"

# Test with CyberRealistic settings
workflow = {
    "prompt": {
        "1": {
            "inputs": {"ckpt_name": "cyberrealistic_v9.safetensors"},
            "class_type": "CheckpointLoaderSimple"
        },
        "2": {
            "inputs": {
                "text": "RAW photo, Japanese businessman, 30 years old, masculine features, sharp focus, detailed skin texture, natural lighting, photographic, DSLR quality, 85mm lens",
                "clip": ["1", 1]
            },
            "class_type": "CLIPTextEncode"
        },
        "3": {
            "inputs": {
                "text": "blurry, soft focus, anime, cartoon, painting, airbrushed, smooth skin",
                "clip": ["1", 1]
            },
            "class_type": "CLIPTextEncode"
        },
        "4": {
            "inputs": {
                "seed": random.randint(1, 1000000000),
                "steps": 30,
                "cfg": 5.0,  # CyberRealistic recommended CFG
                "sampler_name": "dpmpp_sde",  # Recommended sampler
                "scheduler": "karras",
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
                "height": 768,
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
                "filename_prefix": "CYBER_TEST",
                "images": ["6", 0]
            },
            "class_type": "SaveImage"
        }
    }
}

response = requests.post(f"{comfyui_url}/prompt", json={"prompt": workflow["prompt"]})
if response.status_code == 200:
    print("✅ CyberRealistic v9 test submitted")
else:
    print(f"❌ Failed: {response.status_code}")