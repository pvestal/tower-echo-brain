#!/usr/bin/env python3
"""Test different models for multi-character and gender accuracy"""
import requests
import json
import random
import time

comfyui_url = "http://localhost:8188"

# Models to test
models = [
    "counterfeit_v3.safetensors",  # Anime style
    "deliberate_v2.safetensors",   # Semi-realistic
    "dreamshaper_8.safetensors",   # Fantasy/artistic
    "chilloutmix_NiPrunedFp32Fix.safetensors",  # Asian faces
    "AOM3A1B.safetensors"  # Anime
]

# Test prompt - clear gender and count
test_prompt = "1 young Japanese man with masculine features and 2 beautiful Japanese women, the man in center, women on sides, all facing camera, group photo, detailed faces"

def test_model(model_name):
    """Test a single model"""

    workflow = {
        "prompt": {
            "1": {
                "inputs": {
                    "ckpt_name": model_name
                },
                "class_type": "CheckpointLoaderSimple"
            },
            "2": {
                "inputs": {
                    "text": test_prompt + ", high quality, detailed",
                    "clip": ["1", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "3": {
                "inputs": {
                    "text": "bad anatomy, wrong gender, fused bodies, bad faces, low quality",
                    "clip": ["1", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "4": {
                "inputs": {
                    "seed": random.randint(1, 1000000000),
                    "steps": 20,
                    "cfg": 7,
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
                    "width": 768,
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
                    "filename_prefix": f"model_test_{model_name.split('.')[0]}",
                    "images": ["6", 0]
                },
                "class_type": "SaveImage"
            }
        }
    }

    try:
        response = requests.post(
            f"{comfyui_url}/prompt",
            json={"prompt": workflow["prompt"]},
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            if "prompt_id" in result:
                print(f"✅ {model_name}: Queued {result['prompt_id']}")
                return True
        else:
            print(f"❌ {model_name}: Failed - {response.status_code}")

    except Exception as e:
        print(f"❌ {model_name}: Error - {e}")

    return False

print("🧪 Testing models for multi-character generation...")
print(f"Prompt: {test_prompt}\n")

for model in models:
    print(f"\nTesting: {model}")
    if test_model(model):
        time.sleep(15)  # Wait for generation

print("\n✅ Tests complete! Check images:")
print("ls -la /mnt/1TB-storage/ComfyUI/output/model_test_*")