#!/usr/bin/env python3
"""
Test the newly downloaded MajicMix v7 - supposed to be best for Asian faces
"""
import requests
import json
import random

comfyui_url = "http://localhost:8188"

tests = [
    {
        "name": "majicmix_trio",
        "prompt": "one nervous young Japanese man in center, two beautiful Japanese women on sides, man has short black hair, left woman has long hair, right woman has short hair, exactly 3 people, living room, photorealistic, detailed faces",
        "negative": "4 people, 5 people, more than 3 people, less than 3 people, anime, cartoon"
    },
    {
        "name": "majicmix_duo",
        "prompt": "young Japanese couple, man and woman talking, man has masculine features short hair, woman has long black hair feminine features, two people only, photorealistic",
        "negative": "solo, single person, more than 2 people, anime"
    },
    {
        "name": "majicmix_male_solo",
        "prompt": "handsome Japanese man, masculine features, short black hair, business suit, confident expression, solo portrait, photorealistic",
        "negative": "woman, female, breasts, long hair, multiple people"
    }
]

for test in tests:
    workflow = {
        "prompt": {
            "1": {
                "inputs": {"ckpt_name": "majicmixRealistic_v7.safetensors"},
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
                    "steps": 30,
                    "cfg": 7.5,
                    "sampler_name": "euler_a",  # MajicMix recommends euler_a
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
                    "width": 768 if "duo" in test["name"] or "trio" in test["name"] else 512,
                    "height": 512 if "duo" in test["name"] or "trio" in test["name"] else 768,
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
                    "filename_prefix": f"MAJICMIX_{test['name']}",
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

print("\nCheck: ls -la /mnt/1TB-storage/ComfyUI/output/MAJICMIX_*")