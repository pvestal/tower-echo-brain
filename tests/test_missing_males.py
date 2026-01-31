#!/usr/bin/env python3
"""Test the missing male characters specifically"""
import requests
import json

comfyui_url = "http://localhost:8188"

# Test male characters that didn't generate
test_males = {
    "Yuki_Tanaka": {
        "checkpoint": "deliberate_v2.safetensors",
        "prompt": "portrait of young Japanese man, nervous expression, masculine features, short black hair, male face, casual t-shirt, detailed face, solo, photorealistic",
        "negative": "woman, female, breasts, feminine, multiple people, girl, long hair",
        "seed": 99999
    },
    "Takeshi_Sato": {
        "checkpoint": "deliberate_v2.safetensors",
        "prompt": "intimidating Japanese businessman, middle-aged man, masculine face, dark suit, cold expression, masculine features, male portrait, solo, photorealistic",
        "negative": "woman, female, young, feminine, multiple people, breasts, long hair",
        "seed": 88888
    }
}

for char_name, data in test_males.items():
    workflow = {
        "prompt": {
            "1": {
                "inputs": {"ckpt_name": data["checkpoint"]},
                "class_type": "CheckpointLoaderSimple"
            },
            "2": {
                "inputs": {
                    "text": data["prompt"],
                    "clip": ["1", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "3": {
                "inputs": {
                    "text": data["negative"],
                    "clip": ["1", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "4": {
                "inputs": {
                    "seed": data["seed"],
                    "steps": 20,
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
                    "filename_prefix": f"TEST_{char_name}_male",
                    "images": ["6", 0]
                },
                "class_type": "SaveImage"
            }
        }
    }

    response = requests.post(
        f"{comfyui_url}/prompt",
        json={"prompt": workflow["prompt"]},
        timeout=10
    )

    if response.status_code == 200:
        result = response.json()
        if "prompt_id" in result:
            print(f"✅ {char_name}: Queued {result['prompt_id']}")
        else:
            print(f"❌ {char_name}: No prompt_id")
    else:
        print(f"❌ {char_name}: Failed {response.status_code}")
        print(response.text[:200])

print("\nWait 30 seconds then check:")
print("ls -la /mnt/1TB-storage/ComfyUI/output/TEST_*male*")