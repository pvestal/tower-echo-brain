#!/usr/bin/env python3
"""
REAL testing with models that actually work
Testing 5 critical scenarios
"""
import requests
import json
import random
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

comfyui_url = "http://localhost:8188"

# Use ONLY models we know work
test_scenarios = [
    # Test 1: Solo male with dreamshaper
    {
        "name": "Yuki_solo",
        "model": "dreamshaper_8.safetensors",
        "prompt": "young nervous Japanese man, masculine features, short black hair, worried expression, counting money bills, Tokyo apartment, photorealistic",
        "negative": "woman, female, breasts, long hair",
        "width": 512, "height": 768
    },
    # Test 2: Solo female with chilloutmix
    {
        "name": "Mei_solo",
        "model": "chilloutmix_NiPrunedFp32Fix.safetensors",
        "prompt": "beautiful Japanese woman, long black hair, gentle smile, cooking in kitchen with apron, medium breasts, photorealistic",
        "negative": "man, male, masculine, beard",
        "width": 512, "height": 768
    },
    # Test 3: Duo with dreamshaper (can it handle both genders?)
    {
        "name": "Yuki_Mei_duo_dreamshaper",
        "model": "dreamshaper_8.safetensors",
        "prompt": "young Japanese man talking to beautiful Japanese woman, man has short hair, woman has long black hair, two people only, living room, photorealistic",
        "negative": "solo, single person, more than 2 people",
        "width": 768, "height": 512
    },
    # Test 4: Anime solo with counterfeit
    {
        "name": "GoblinSlayer_solo",
        "model": "counterfeit_v3.safetensors",
        "prompt": "armored warrior, cyberpunk armor with glowing neon accents, helmet, standing ready for battle, solo character, anime style",
        "negative": "realistic, photograph, multiple people",
        "width": 512, "height": 768
    },
    # Test 5: Check download status
    {
        "name": "check_downloads",
        "skip_generation": True
    }
]

def generate_test(scenario):
    """Generate a test image"""

    if scenario.get("skip_generation"):
        # Just check downloads
        logger.info("\n📥 Checking model downloads...")
        import subprocess
        result = subprocess.run("ls -lh /mnt/1TB-storage/ComfyUI/models/checkpoints/*.safetensors | tail -5",
                              shell=True, capture_output=True, text=True)
        logger.info(result.stdout)
        return

    workflow = {
        "prompt": {
            "1": {
                "inputs": {"ckpt_name": scenario["model"]},
                "class_type": "CheckpointLoaderSimple"
            },
            "2": {
                "inputs": {
                    "text": scenario["prompt"],
                    "clip": ["1", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "3": {
                "inputs": {
                    "text": scenario["negative"],
                    "clip": ["1", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "4": {
                "inputs": {
                    "seed": random.randint(1, 1000000000),
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
                    "width": scenario["width"],
                    "height": scenario["height"],
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
                    "filename_prefix": f"REAL_TEST_{scenario['name']}",
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
                logger.info(f"✅ Queued: {scenario['name']}")
                return True
        else:
            logger.error(f"❌ Failed: {response.status_code}")

    except Exception as e:
        logger.error(f"❌ Error: {e}")

    return False

# Run tests
logger.info("=" * 60)
logger.info("REAL TESTING WITH WORKING MODELS")
logger.info("=" * 60)

for i, scenario in enumerate(test_scenarios, 1):
    logger.info(f"\n[{i}/5] Testing: {scenario.get('name', 'Status check')}")

    if scenario.get("skip_generation"):
        generate_test(scenario)
    else:
        logger.info(f"     Model: {scenario['model'].split('.')[0]}")
        if generate_test(scenario):
            time.sleep(20)

logger.info("\n" + "=" * 60)
logger.info("TEST COMPLETE")
logger.info("Check: ls -la /mnt/1TB-storage/ComfyUI/output/REAL_TEST_*")
logger.info("=" * 60)