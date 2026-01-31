#!/usr/bin/env python3
"""
Test high-quality models for Tokyo Debt Desire
ChilloutMix for Asian faces, Realistic Vision for general
"""
import requests
import json
import random
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

comfyui_url = "http://localhost:8188"

# Test scenarios focusing on quality
test_scenes = [
    {
        "name": "TDD_Yuki_male_chillout",
        "model": "chilloutmix.safetensors",
        "prompt": "portrait of young Japanese man, nervous expression, short black hair, masculine features, detailed face, realistic photo, high quality",
        "negative": "woman, female, breasts, long hair, anime, cartoon",
        "width": 512, "height": 768
    },
    {
        "name": "TDD_Mei_female_chillout",
        "model": "chilloutmix.safetensors",
        "prompt": "beautiful Japanese woman, long black hair, gentle smile, feminine features, apron, kitchen, detailed face, realistic photo",
        "negative": "man, male, masculine, anime, cartoon",
        "width": 512, "height": 768
    },
    {
        "name": "TDD_Rina_female_realistic",
        "model": "realistic_vision_v51.safetensors",
        "prompt": "confident Japanese businesswoman, short brown hair, assertive expression, mini skirt, blouse, office, photorealistic",
        "negative": "man, male, masculine, anime, cartoon",
        "width": 512, "height": 768
    },
    {
        "name": "TDD_Takeshi_male_realistic",
        "model": "realistic_vision_v51.safetensors",
        "prompt": "intimidating middle-aged Japanese man, dark suit, cold expression, yakuza boss, masculine features, photorealistic",
        "negative": "woman, female, young, anime",
        "width": 512, "height": 768
    }
]

def test_model(scene):
    """Test a single model configuration"""

    workflow = {
        "prompt": {
            "1": {
                "inputs": {"ckpt_name": scene["model"]},
                "class_type": "CheckpointLoaderSimple"
            },
            "2": {
                "inputs": {
                    "text": scene["prompt"],
                    "clip": ["1", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "3": {
                "inputs": {
                    "text": scene["negative"],
                    "clip": ["1", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "4": {
                "inputs": {
                    "seed": random.randint(1, 1000000000),
                    "steps": 30,
                    "cfg": 7.0,
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
                    "width": scene["width"],
                    "height": scene["height"],
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
                    "filename_prefix": f"QUALITY_{scene['name']}",
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
            logger.info(f"✅ {scene['name']} - {scene['model']}")
            return True
        else:
            logger.error(f"❌ {scene['name']}: {response.status_code}")
            return False

    except Exception as e:
        logger.error(f"❌ {scene['name']}: {e}")
        return False

def main():
    logger.info("🎯 Testing High-Quality Models for Tokyo Debt Desire")
    logger.info("=" * 60)

    # Wait for models to download if needed
    logger.info("Checking model availability...")
    time.sleep(2)

    success_count = 0

    for scene in test_scenes:
        if test_model(scene):
            success_count += 1
            time.sleep(20)  # Wait for generation
        else:
            time.sleep(5)

    logger.info("=" * 60)
    logger.info(f"✅ Success: {success_count}/{len(test_scenes)}")
    logger.info("Check results: ls -la /mnt/1TB-storage/ComfyUI/output/QUALITY_*")

if __name__ == "__main__":
    main()