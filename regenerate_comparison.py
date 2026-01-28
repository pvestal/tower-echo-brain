#!/usr/bin/env python3
"""Regenerate comparison images for all models using SSOT configuration"""
import requests
import json
import random
import time
import logging
from project_config_ssot import config_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

comfyui_url = "http://localhost:8188"

# Get models from SSOT
models_to_test = [
    (config_manager.models["primary"].file, "Primary"),
    (config_manager.models["backup"].file, "Backup"),
    (config_manager.models["fallback"].file, "Fallback"),
    (config_manager.models["asian_specialized"].file, "Asian")
]

# Use same seed for consistency
seed = random.randint(1, 1000000000)

for model_key, (model_file, prefix) in zip(["primary", "backup", "fallback", "asian_specialized"], models_to_test):
    logger.info(f"Generating with {model_file} ({prefix})...")

    # Get settings from SSOT
    model_config = config_manager.models[model_key]
    project_config = config_manager.projects["tokyo_debt_desire"]

    workflow = {
        "prompt": {
            "1": {
                "inputs": {"ckpt_name": model_file},
                "class_type": "CheckpointLoaderSimple"
            },
            "2": {
                "inputs": {
                    "text": f"{project_config['style_prompt']}, young Japanese man, nervous expression, short black hair, masculine features, counting money, apartment",
                    "clip": ["1", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "3": {
                "inputs": {
                    "text": project_config["negative_prompt"],
                    "clip": ["1", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "4": {
                "inputs": {
                    "seed": seed,  # Same seed for all
                    "steps": model_config.settings["steps"],
                    "cfg": model_config.settings["cfg"],
                    "sampler_name": model_config.settings["sampler"],
                    "scheduler": model_config.settings["scheduler"],
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
                    "filename_prefix": f"SSOT_{prefix}_Yuki_{model_config.sharpness_score}sharp",
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
            logger.info(f"✅ {prefix} submitted")
            time.sleep(20)  # Wait for generation
        else:
            logger.error(f"❌ {prefix}: {response.status_code}")

    except Exception as e:
        logger.error(f"❌ {prefix}: {e}")

logger.info("\n" + "=" * 60)
logger.info("SSOT Model Comparison Complete!")
logger.info(f"Used seed: {seed}")
logger.info("Files: /mnt/1TB-storage/ComfyUI/output/SSOT_*")
logger.info("\nModel Rankings by Sharpness:")
for model_key in ["primary", "backup", "fallback", "asian_specialized"]:
    model = config_manager.models[model_key]
    logger.info(f"  {model.name}: {model.sharpness_score}/10")