#!/usr/bin/env python3
"""
High Quality LTX Generation Test
Using REAL trained LoRAs and optimized settings
"""

import json
import requests
import time

# Use a working workflow from history
workflow = {
    "1": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "ltxv-2b-fp8.safetensors"}
    },
    "2": {
        "class_type": "CLIPLoader",
        "inputs": {"clip_name": "t5xxl_fp16.safetensors", "type": "ltxv"}
    },
    "3": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            # Much more detailed prompt for quality
            "text": "beautiful woman, intimate scene, high quality, detailed face, realistic skin, professional cinematography, 4k quality, sharp focus, dramatic lighting, adult content",
            "clip": ["2", 0]
        }
    },
    "4": {
        "class_type": "LoraLoader",
        "inputs": {
            # Use REAL trained LoRA, not our test garbage
            "lora_name": "SexGod_Nudity_LTX2_v1_5.safetensors",
            "strength_model": 0.8,  # Not too strong
            "strength_clip": 0.6,
            "model": ["1", 0],
            "clip": ["2", 0]
        }
    },
    "5": {
        "class_type": "LoraLoader",
        "inputs": {
            # Stack another REAL LoRA for better quality
            "lora_name": "LTX-2_-_Better_Female_Nudity.safetensors",
            "strength_model": 0.5,
            "strength_clip": 0.4,
            "model": ["4", 0],
            "clip": ["4", 1]
        }
    },
    # Empty negative prompt
    "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "cartoon, anime, low quality, blurry, child, underage, bad anatomy, deformed",
            "clip": ["5", 1]
        }
    },
    # Scheduler
    "7": {
        "class_type": "LTXVScheduler",
        "inputs": {
            "scheduler": "euler",
            "beta_schedule": "scaled_linear",
            "model": ["5", 0],
            "steps": 50,  # MORE STEPS for quality
            "alpha": 0.5,
            "beta_min": 0.3,
            "beta_max": 20.0
        }
    },
    # Guider
    "8": {
        "class_type": "BasicGuider",
        "inputs": {
            "conditioning": ["3", 0],
            "model": ["5", 0]
        }
    },
    # Sampler
    "9": {
        "class_type": "SamplerCustom",
        "inputs": {
            "cfg": 5.0,  # Higher CFG for more adherence
            "sampler": ["10", 0],
            "sigmas": ["7", 0],
            "latent_image": ["11", 0],
            "guider": ["8", 0],
            "noise": ["12", 0]
        }
    },
    "10": {
        "class_type": "KSamplerSelect",
        "inputs": {"sampler_name": "euler"}
    },
    "11": {
        "class_type": "EmptyLTXVLatentVideo",
        "inputs": {
            "height": 512,  # Higher resolution
            "width": 768,
            "length": 97,  # Longer video
            "batch_size": 1
        }
    },
    "12": {
        "class_type": "RandomNoise",
        "inputs": {"noise_seed": 42}
    },
    "13": {
        "class_type": "VAEDecodeLTXV",
        "inputs": {
            "samples": ["9", 0],
            "vae": ["1", 2]
        }
    },
    "14": {
        "class_type": "VHS_VideoCombine",
        "inputs": {
            "images": ["13", 0],
            "frame_rate": 12,  # Higher framerate
            "loop_count": 0,
            "filename_prefix": "quality_test_real_loras",
            "format": "video/h264-mp4",
            "pingpong": False,
            "save_output": True,
            "videopreview": {"format": "webp"}
        }
    }
}

print("🎬 Testing with REAL trained LoRAs and optimized settings...")
print("Using:")
print("  - SexGod_Nudity_LTX2_v1_5.safetensors (769MB trained)")
print("  - LTX-2_-_Better_Female_Nudity.safetensors (769MB trained)")
print("  - 50 steps, CFG 5.0, 512x768 resolution")

response = requests.post("http://localhost:8188/prompt", json={"prompt": workflow})

if response.status_code == 200:
    result = response.json()
    if "prompt_id" in result:
        print(f"✅ Submitted: {result['prompt_id']}")
        print("Generating high quality video...")

        # Wait and check
        for i in range(60):
            time.sleep(3)
            history = requests.get(f"http://localhost:8188/history/{result['prompt_id']}").json()
            if result['prompt_id'] in history:
                if history[result['prompt_id']].get("outputs"):
                    print("✅ Generation complete!")
                    break
    else:
        print(f"Error: {result}")
else:
    print(f"Failed: {response.text}")

print("\nCheck /mnt/1TB-storage/ComfyUI/output/quality_test_real_loras*.mp4")