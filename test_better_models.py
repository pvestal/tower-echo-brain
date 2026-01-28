#!/usr/bin/env python3
"""
Test DIFFERENT models because chilloutmix sucks for multi-character scenes
"""
import requests
import json
import random
import time

comfyui_url = "http://localhost:8188"

# Test the SAME trio prompt with DIFFERENT models
test_models = [
    "dreamshaper_8.safetensors",
    "deliberate_v2.safetensors",
    "AOM3A1B.safetensors",
    "Counterfeit-V2.5.safetensors"
]

trio_prompt = "one young Japanese man in center nervous expression, two beautiful Japanese women on sides flirting with him, exactly 3 people, living room scene, photorealistic"
negative = "4 people, 5 people, more than 3 people, less than 3 people, crowd"

for model in test_models:
    print(f"\nTesting: {model}")

    workflow = {
        "prompt": {
            "1": {
                "inputs": {"ckpt_name": model},
                "class_type": "CheckpointLoaderSimple"
            },
            "2": {
                "inputs": {
                    "text": trio_prompt,
                    "clip": ["1", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "3": {
                "inputs": {
                    "text": negative,
                    "clip": ["1", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "4": {
                "inputs": {
                    "seed": 999999,  # Fixed seed for comparison
                    "steps": 25,
                    "cfg": 7.5,
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
                    "filename_prefix": f"better_model_test_{model.split('.')[0]}",
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
                print(f"✅ Queued: {result['prompt_id']}")
        else:
            print(f"❌ Failed: {response.status_code}")

        time.sleep(20)

    except Exception as e:
        print(f"❌ Error: {e}")

print("\nDone! Check which model got the trio RIGHT:")
print("ls -la /mnt/1TB-storage/ComfyUI/output/better_model_test_*")