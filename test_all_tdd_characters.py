#!/usr/bin/env python3
"""Test all Tokyo Debt Desire characters with SSOT configuration"""
import requests
import json
import random
import time
import logging
from project_config_ssot import config_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

comfyui_url = "http://localhost:8188"
seed = random.randint(1, 1000000000)

# Character-specific prompts
character_prompts = {
    "Yuki_Tanaka": "RAW photo, young Japanese man, nervous expression, short black hair, masculine features, counting money, apartment, photorealistic, 85mm lens",
    "Mei_Kobayashi": "RAW photo, beautiful Japanese woman, long black hair, gentle smile, cooking in kitchen, apron, medium breasts, photorealistic, 85mm lens",
    "Rina_Suzuki": "RAW photo, confident Japanese woman, long dark hair, assertive expression, business attire, office setting, photorealistic, 85mm lens",
    "Takeshi_Sato": "RAW photo, intimidating Japanese man, expensive suit, cold eyes, masculine jawline, yakuza boss, photorealistic, 85mm lens"
}

for char_key, char_config in config_manager.characters["tokyo_debt_desire"].items():
    logger.info(f"Generating {char_config.name} ({char_config.gender})...")

    # Get model and settings
    model = config_manager.get_model_for_character("tokyo_debt_desire", char_key)
    project = config_manager.projects["tokyo_debt_desire"]

    # Build prompt
    base_prompt = character_prompts.get(char_key, f"portrait of {char_config.name}")
    full_prompt = f"{project['style_prompt']}, {base_prompt}"

    # Add gender-specific negative prompts
    if char_config.gender == "male":
        negative = f"{project['negative_prompt']}, woman, female, breasts, feminine, long hair"
    else:
        negative = f"{project['negative_prompt']}, man, male, masculine, beard, muscular"

    workflow = {
        "prompt": {
            "1": {
                "inputs": {"ckpt_name": model.file},
                "class_type": "CheckpointLoaderSimple"
            },
            "2": {
                "inputs": {
                    "text": full_prompt,
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
                    "seed": seed + hash(char_key) % 1000000,  # Unique seed per character
                    "steps": model.settings["steps"],
                    "cfg": model.settings["cfg"],
                    "sampler_name": model.settings["sampler"],
                    "scheduler": model.settings["scheduler"],
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
                    "width": project["resolution"]["width"],
                    "height": project["resolution"]["height"],
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
                    "filename_prefix": f"TDD_SSOT_{char_key}_{char_config.gender}",
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
            logger.info(f"✅ {char_config.name} submitted")
            time.sleep(15)  # Wait between generations
        else:
            logger.error(f"❌ {char_config.name}: {response.status_code}")

    except Exception as e:
        logger.error(f"❌ {char_config.name}: {e}")

logger.info("\n" + "="*60)
logger.info("All TDD characters generated with SSOT configuration")
logger.info("Files: /mnt/1TB-storage/ComfyUI/output/TDD_SSOT_*")