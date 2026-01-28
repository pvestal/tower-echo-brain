#!/usr/bin/env python3
"""
Test OpenPose ControlNet for multi-character generation
This ensures proper gender and positioning
"""
import requests
import json
import random
import base64
from PIL import Image, ImageDraw
import io
import numpy as np

comfyui_url = "http://localhost:8188"

def create_simple_pose_image():
    """
    Create a simple 3-person pose reference image
    Just colored rectangles to indicate positions
    """
    width, height = 768, 512
    img = Image.new('RGB', (width, height), 'black')
    draw = ImageDraw.Draw(img)

    # Draw 3 stick figures as colored regions
    # Left person (woman) - pink
    draw.rectangle([100, 100, 250, 450], fill=(255, 192, 203))

    # Center person (man) - blue, slightly taller
    draw.rectangle([300, 80, 450, 450], fill=(173, 216, 230))

    # Right person (woman) - pink
    draw.rectangle([500, 100, 650, 450], fill=(255, 192, 203))

    # Save as base64 for embedding
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def test_with_controlnet():
    """Test using ControlNet for composition control"""

    # Create pose reference
    pose_base64 = create_simple_pose_image()

    workflow = {
        "prompt": {
            # Load checkpoint
            "1": {
                "inputs": {
                    "ckpt_name": "deliberate_v2.safetensors"
                },
                "class_type": "CheckpointLoaderSimple"
            },
            # Load ControlNet
            "2": {
                "inputs": {
                    "control_net_name": "control_v11p_sd15_openpose.pth"
                },
                "class_type": "ControlNetLoader"
            },
            # Load pose image
            "3": {
                "inputs": {
                    "image": f"data:image/png;base64,{pose_base64}",
                    "upload": "image"
                },
                "class_type": "LoadImage"
            },
            # Apply ControlNet
            "4": {
                "inputs": {
                    "net": ["2", 0],
                    "image": ["3", 0],
                    "strength": 1.0
                },
                "class_type": "ControlNetApply"
            },
            # Positive prompt
            "5": {
                "inputs": {
                    "text": "group photo of 3 people, left beautiful Japanese woman long hair, center handsome Japanese man masculine features, right beautiful Japanese woman short hair, all facing camera, professional photo",
                    "clip": ["1", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            # Negative prompt
            "6": {
                "inputs": {
                    "text": "bad anatomy, wrong gender, merged bodies, extra limbs",
                    "clip": ["1", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            # KSampler with ControlNet
            "7": {
                "inputs": {
                    "seed": 42,  # Fixed seed for testing
                    "steps": 30,
                    "cfg": 8,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0,
                    "model": ["1", 0],
                    "positive": ["4", 0],  # ControlNet output
                    "negative": ["6", 0],
                    "latent_image": ["8", 0]
                },
                "class_type": "KSampler"
            },
            # Empty latent
            "8": {
                "inputs": {
                    "width": 768,
                    "height": 512,
                    "batch_size": 1
                },
                "class_type": "EmptyLatentImage"
            },
            # VAE Decode
            "9": {
                "inputs": {
                    "samples": ["7", 0],
                    "vae": ["1", 2]
                },
                "class_type": "VAEDecode"
            },
            # Save
            "10": {
                "inputs": {
                    "filename_prefix": "controlnet_test_3person",
                    "images": ["9", 0]
                },
                "class_type": "SaveImage"
            }
        }
    }

    try:
        print("🎯 Testing ControlNet with 3-person composition...")

        response = requests.post(
            f"{comfyui_url}/prompt",
            json={"prompt": workflow["prompt"]},
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            if "prompt_id" in result:
                print(f"✅ Queued: {result['prompt_id']}")
                return True
        else:
            print(f"❌ Failed: {response.status_code}")
            print(response.text[:500])

    except Exception as e:
        print(f"❌ Error: {e}")

    return False

def test_without_controlnet():
    """Test without ControlNet for comparison"""

    workflow = {
        "prompt": {
            "1": {
                "inputs": {
                    "ckpt_name": "deliberate_v2.safetensors"
                },
                "class_type": "CheckpointLoaderSimple"
            },
            "2": {
                "inputs": {
                    "text": "group photo of exactly 3 people: beautiful Japanese woman with long hair on left, handsome Japanese man with masculine features in center, beautiful Japanese woman with short hair on right, all facing camera",
                    "clip": ["1", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "3": {
                "inputs": {
                    "text": "bad anatomy, wrong gender, wrong number of people",
                    "clip": ["1", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "4": {
                "inputs": {
                    "seed": 42,  # Same seed for comparison
                    "steps": 30,
                    "cfg": 8,
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
                    "filename_prefix": "no_controlnet_test_3person",
                    "images": ["6", 0]
                },
                "class_type": "SaveImage"
            }
        }
    }

    try:
        print("\n📊 Testing WITHOUT ControlNet for comparison...")

        response = requests.post(
            f"{comfyui_url}/prompt",
            json={"prompt": workflow["prompt"]},
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            if "prompt_id" in result:
                print(f"✅ Queued: {result['prompt_id']}")
                return True
        else:
            print(f"❌ Failed: {response.status_code}")

    except Exception as e:
        print(f"❌ Error: {e}")

    return False

# Run tests
print("=" * 60)
print("CONTROLNET MULTI-CHARACTER TEST")
print("=" * 60)

# Test without ControlNet first
test_without_controlnet()

# Test with ControlNet
# Note: This may fail if ControlNet nodes aren't installed
# test_with_controlnet()

print("\n⏳ Wait 30 seconds then check:")
print("ls -la /mnt/1TB-storage/ComfyUI/output/*controlnet*3person*")
print("ls -la /mnt/1TB-storage/ComfyUI/output/no_controlnet*3person*")