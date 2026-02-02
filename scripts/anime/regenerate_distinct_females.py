#!/usr/bin/env python3
"""Generate Mei and Rina with distinct visual features"""
import requests
import json
import random
import time
import logging
from project_config_ssot import config_manager
from character_distinctions import get_distinguishing_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

comfyui_url = "http://localhost:8188"
seed = random.randint(1, 1000000000)

characters = {
    "Mei_Kobayashi": {
        "prompt": "RAW photo, 24 year old Japanese woman, shoulder-length black hair with side-swept bangs, soft round face, gentle smile, beauty mark under left eye, dimples, petite build, wearing apron over simple dress, cooking in kitchen, warm lighting, photorealistic, 85mm lens",
        "personality": "gentle housewife"
    },
    "Rina_Suzuki": {
        "prompt": "RAW photo, 28 year old Japanese businesswoman, straight black hair in professional bob cut, sharp angular features, high cheekbones, wearing glasses, red lipstick, tall athletic build, business suit with blazer and pencil skirt, office setting, confident posture, photorealistic, 85mm lens",
        "personality": "confident professional"
    }
}

for char_key, char_data in characters.items():
    logger.info(f"Generating distinct {char_key}...")

    model = config_manager.get_model_for_character("tokyo_debt_desire", char_key)
    project = config_manager.projects["tokyo_debt_desire"]

    # Build prompts with distinct features
    full_prompt = f"{project['style_prompt']}, {char_data['prompt']}, {get_distinguishing_features(char_key)}"
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
                    "seed": seed + hash(char_key) % 1000000,
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
                    "filename_prefix": f"TDD_DISTINCT_{char_key}_{char_data['personality'].replace(' ', '_')}",
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
            logger.info(f"✅ {char_key} submitted with distinct features")
            time.sleep(15)
        else:
            logger.error(f"❌ {char_key}: {response.status_code}")

    except Exception as e:
        logger.error(f"❌ {char_key}: {e}")

logger.info("\n" + "="*60)
logger.info("Distinct character comparison:")
logger.info("Mei: Soft features, shoulder-length hair, beauty mark, dimples")
logger.info("Rina: Sharp features, bob cut, glasses, professional look")
logger.info("Files: /mnt/1TB-storage/ComfyUI/output/TDD_DISTINCT_*")