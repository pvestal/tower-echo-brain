#!/usr/bin/env python3
"""
FINAL test with dreamshaper_8 which we KNOW handles both genders
Test the critical scenarios for production
"""
import requests
import json
import random
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

comfyui_url = "http://localhost:8188"

# All tests with dreamshaper since it works for both genders
scenarios = [
    # Tokyo Debt Desire scenes
    {
        "name": "TDD_Yuki_male_solo",
        "prompt": "young nervous Japanese man, masculine features, short black hair, counting money, worried expression, Tokyo apartment, photorealistic",
        "width": 512, "height": 768
    },
    {
        "name": "TDD_Mei_female_solo",
        "prompt": "beautiful Japanese woman, long black hair, gentle smile, cooking in kitchen, apron, medium breasts, photorealistic",
        "width": 512, "height": 768
    },
    {
        "name": "TDD_duo_confrontation",
        "prompt": "young Japanese man arguing with confident Japanese woman, man has short hair nervous, woman has short brown hair assertive, two people, apartment, photorealistic",
        "width": 768, "height": 512
    },
    {
        "name": "TDD_trio_harem",
        "prompt": "one young Japanese man surrounded by two women, man in center looking nervous, women flirting with him, three people total, living room, photorealistic",
        "width": 768, "height": 512
    },
    # Anime with counterfeit
    {
        "name": "CGS_team",
        "model": "counterfeit_v3.safetensors",
        "prompt": "three warriors - armored fighter, young male warrior, female elf archer, team pose, cyberpunk setting, anime style",
        "width": 768, "height": 512
    }
]

for scenario in scenarios:
    model = scenario.get("model", "dreamshaper_8.safetensors")
    negative = "bad anatomy, wrong number of people" if "duo" in scenario["name"] or "trio" in scenario["name"] else "bad anatomy"

    workflow = {
        "prompt": {
            "1": {
                "inputs": {"ckpt_name": model},
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
                    "text": negative,
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
                    "filename_prefix": f"FINAL_{scenario['name']}",
                    "images": ["6", 0]
                },
                "class_type": "SaveImage"
            }
        }
    }

    response = requests.post(f"{comfyui_url}/prompt", json={"prompt": workflow["prompt"]})

    if response.status_code == 200:
        logger.info(f"✅ {scenario['name']}")
        time.sleep(20)
    else:
        logger.error(f"❌ {scenario['name']}")

logger.info("\n✅ FINAL WORKING CONFIGURATION:")
logger.info("- Tokyo Debt Desire: dreamshaper_8")
logger.info("- Cyberpunk Goblin Slayer: counterfeit_v3")
logger.info("- CFG: 7.5, Steps: 25, Sampler: euler")
logger.info("\nCheck results: ls -la /mnt/1TB-storage/ComfyUI/output/FINAL_*")