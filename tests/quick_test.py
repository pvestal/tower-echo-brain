#!/usr/bin/env python3
"""Quick test of ChilloutMix and Realistic Vision"""
import requests
import json
import random

comfyui_url = "http://localhost:8188"

# Test ChilloutMix for Asian faces
workflow = {
    "prompt": {
        "1": {
            "inputs": {"ckpt_name": "chilloutmix.safetensors"},
            "class_type": "CheckpointLoaderSimple"
        },
        "2": {
            "inputs": {
                "text": "beautiful Japanese woman, long black hair, detailed face, photorealistic portrait, high quality",
                "clip": ["1", 1]
            },
            "class_type": "CLIPTextEncode"
        },
        "3": {
            "inputs": {
                "text": "bad anatomy, blurry, anime",
                "clip": ["1", 1]
            },
            "class_type": "CLIPTextEncode"
        },
        "4": {
            "inputs": {
                "seed": random.randint(1, 1000000000),
                "steps": 25,
                "cfg": 7.0,
                "sampler_name": "euler",
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
                "filename_prefix": "CHILLOUT_TEST",
                "images": ["6", 0]
            },
            "class_type": "SaveImage"
        }
    }
}

response = requests.post(f"{comfyui_url}/prompt", json={"prompt": workflow["prompt"]})
if response.status_code == 200:
    print("✅ ChilloutMix test submitted")
else:
    print(f"❌ Failed: {response.status_code}")