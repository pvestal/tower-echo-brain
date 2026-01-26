#!/usr/bin/env python3
"""
Test the created LoRA with LTX pipeline
"""

import json
import requests
import time
from pathlib import Path

# ComfyUI API endpoint
COMFYUI_URL = "http://localhost:8188"

def test_ltx_with_lora():
    """Test our minimal LoRA with LTX"""

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
                "text": "cowgirl_position, a woman in cowgirl position, intimate scene, adult content",
                "clip": ["2", 0]
            }
        },
        "4": {
            "class_type": "LoraLoader",
            "inputs": {
                "lora_name": "test_cowgirl_ltx_minimal.safetensors",
                "strength_model": 1.0,
                "strength_clip": 1.0,
                "model": ["1", 0],
                "clip": ["2", 0]
            }
        },
        "5": {
            "class_type": "LTXVSampler",
            "inputs": {
                "model": ["4", 0],
                "cond": ["3", 0],
                "uncond": ["6", 0],
                "fps": 8,
                "width": 768,
                "height": 512,
                "length": 65,
                "seed": 42,
                "steps": 30,
                "cfg": 3.5,
                "denoise": 1.0,
                "sampler_name": "euler"
            }
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "",
                "clip": ["4", 1]
            }
        },
        "7": {
            "class_type": "VAEDecodeLTXV",
            "inputs": {
                "samples": ["5", 0],
                "vae": ["1", 1]
            }
        },
        "8": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["7", 0],
                "frame_rate": 8,
                "loop_count": 0,
                "filename_prefix": "ltx_lora_test",
                "format": "video/h264-mp4",
                "pingpong": False,
                "save_output": True,
                "videopreview": {"format": "webp"}
            }
        }
    }

    # Submit workflow
    response = requests.post(f"{COMFYUI_URL}/prompt", json={"prompt": workflow})
    if response.status_code != 200:
        print(f"Error submitting workflow: {response.text}")
        return False

    result = response.json()
    if "error" in result:
        print(f"Workflow error: {result['error']}")
        return False

    prompt_id = result.get("prompt_id")
    if not prompt_id:
        print(f"No prompt_id in response: {result}")
        return False

    print(f"Submitted prompt: {prompt_id}")

    # Wait for completion
    print("Generating video with test LoRA...")
    for i in range(60):
        time.sleep(2)
        history = requests.get(f"{COMFYUI_URL}/history/{prompt_id}").json()
        if prompt_id in history:
            if history[prompt_id].get("outputs"):
                print("Generation complete!")
                outputs = history[prompt_id]["outputs"]["8"]["videos"]
                print(f"Output video: {outputs[0]['filename']}")
                return True
            elif "exception" in history[prompt_id]:
                print(f"Error: {history[prompt_id]['exception']}")
                return False

    print("Timeout waiting for generation")
    return False

if __name__ == "__main__":
    success = test_ltx_with_lora()
    if success:
        print("\n✅ LoRA integration test successful!")
        print("Check /mnt/1TB-storage/ComfyUI/output/ for the generated video")
    else:
        print("\n❌ LoRA integration test failed")