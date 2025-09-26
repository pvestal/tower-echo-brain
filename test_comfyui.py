import requests
import json
import time

# ComfyUI simple text-to-image workflow
workflow = {
    "3": {
        "class_type": "KSampler",
        "inputs": {
            "cfg": 7.5,
            "denoise": 1,
            "latent_image": ["5", 0],
            "model": ["4", 0],
            "negative": ["7", 0],
            "positive": ["6", 0],
            "sampler_name": "euler",
            "scheduler": "normal",
            "seed": 12345,
            "steps": 20
        }
    },
    "4": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {
            "ckpt_name": "deliberate_v2.safetensors"
        }
    },
    "5": {
        "class_type": "EmptyLatentImage",
        "inputs": {
            "batch_size": 1,
            "height": 512,
            "width": 512
        }
    },
    "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "clip": ["4", 1],
            "text": "anime cyberpunk girl named Luna, purple hair, neon city background, detailed, high quality"
        }
    },
    "7": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "clip": ["4", 1],
            "text": "blurry, bad quality, deformed"
        }
    },
    "8": {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["3", 0],
            "vae": ["4", 2]
        }
    },
    "9": {
        "class_type": "SaveImage",
        "inputs": {
            "filename_prefix": "luna_test",
            "images": ["8", 0]
        }
    }
}

# Send to ComfyUI
response = requests.post("http://localhost:8188/prompt", json={"prompt": workflow})
print(f"Response: {response.status_code}")
print(json.dumps(response.json(), indent=2))

# Check if image was generated
time.sleep(10)
import os
output_dir = "/home/patrick/Projects/ComfyUI-Working/output"
files = sorted([f for f in os.listdir(output_dir) if "luna_test" in f])
if files:
    print(f"\n✅ SUCCESS! Generated image: {files[-1]}")
    print(f"Full path: {output_dir}/{files[-1]}")
else:
    print("\n❌ FAILED - No image generated")
