#!/usr/bin/env python3
"""Test Realistic Vision v5.1 with various characters"""
import requests
import json
import random
import time

comfyui_url = "http://localhost:8188"

test_prompts = [
    {
        "name": "RV_Yuki_nervous_male",
        "prompt": "RAW photo, young nervous Japanese man, short black hair, worried expression, counting money, masculine features, detailed skin, photorealistic, 85mm lens",
        "negative": "woman, female, breasts, long hair, anime, cartoon, smooth skin"
    },
    {
        "name": "RV_Mei_gentle_female",
        "prompt": "RAW photo, beautiful Japanese woman, long black hair, gentle smile, cooking in kitchen, apron, photorealistic, detailed face, natural lighting",
        "negative": "man, male, masculine, anime, cartoon, airbrushed"
    },
    {
        "name": "RV_Rina_confident_female",
        "prompt": "RAW photo, confident Japanese businesswoman, short brown hair, assertive pose, mini skirt, office, photorealistic, sharp focus, DSLR quality",
        "negative": "man, male, masculine, anime, soft focus"
    },
    {
        "name": "RV_Takeshi_yakuza_male",
        "prompt": "RAW photo, intimidating middle-aged Japanese man, dark suit, cold expression, yakuza boss, masculine features, photorealistic, dramatic lighting",
        "negative": "woman, female, young, anime, soft"
    }
]

for test in test_prompts:
    workflow = {
        "prompt": {
            "1": {
                "inputs": {"ckpt_name": "realistic_vision_v51.safetensors"},
                "class_type": "CheckpointLoaderSimple"
            },
            "2": {
                "inputs": {
                    "text": test["prompt"],
                    "clip": ["1", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "3": {
                "inputs": {
                    "text": test["negative"],
                    "clip": ["1", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "4": {
                "inputs": {
                    "seed": random.randint(1, 1000000000),
                    "steps": 25,
                    "cfg": 6.0,
                    "sampler_name": "dpmpp_2m",
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
                    "filename_prefix": test["name"],
                    "images": ["6", 0]
                },
                "class_type": "SaveImage"
            }
        }
    }

    response = requests.post(f"{comfyui_url}/prompt", json={"prompt": workflow["prompt"]})
    if response.status_code == 200:
        print(f"✅ {test['name']}")
    else:
        print(f"❌ {test['name']}: {response.status_code}")

    time.sleep(15)  # Wait between generations

print("\nCheck results: ls -la /mnt/1TB-storage/ComfyUI/output/RV_*")