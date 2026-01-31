#!/usr/bin/env python3
"""
Generate exactly 15 test images to tweak settings and get the proper look
Mix of solo and group scenes using ONE model per project
"""
import asyncio
import requests
import json
import random
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TweakingTestGenerator:
    """Generate 15 test images with different settings to find optimal config"""

    def __init__(self):
        self.comfyui_url = "http://localhost:8188"
        self.output_dir = Path("/mnt/1TB-storage/ComfyUI/output")
        self.image_count = 0
        self.max_images = 15

        # Test variations - 15 total images
        self.test_plan = [
            # Tokyo Debt Desire - using chilloutmix for ALL (7 images)
            {
                "project": "TDD",
                "model": "chilloutmix_NiPrunedFp32Fix.safetensors",
                "type": "solo",
                "prompt": "portrait of young nervous Japanese man, masculine features, short black hair, casual t-shirt, worried expression, counting money, solo, photorealistic",
                "negative": "woman, female, breasts, multiple people",
                "cfg": 7, "steps": 20, "name": "TDD_Yuki_solo_cfg7"
            },
            {
                "project": "TDD",
                "model": "chilloutmix_NiPrunedFp32Fix.safetensors",
                "type": "solo",
                "prompt": "beautiful Japanese woman, long black hair, gentle smile, cooking in kitchen, apron, medium breasts, solo portrait, photorealistic",
                "negative": "man, male, masculine, multiple people",
                "cfg": 8, "steps": 25, "name": "TDD_Mei_solo_cfg8"
            },
            {
                "project": "TDD",
                "model": "chilloutmix_NiPrunedFp32Fix.safetensors",
                "type": "solo",
                "prompt": "confident Japanese woman, short brown hair, assertive expression, mini skirt, hands on hips, solo, photorealistic",
                "negative": "man, male, long hair, multiple people",
                "cfg": 7.5, "steps": 30, "name": "TDD_Rina_solo_cfg7.5"
            },
            {
                "project": "TDD",
                "model": "chilloutmix_NiPrunedFp32Fix.safetensors",
                "type": "duo",
                "prompt": "young nervous Japanese man with short hair talking to beautiful Japanese woman with long black hair, two people interacting, indoor scene, photorealistic",
                "negative": "solo, single person, more than 2 people",
                "cfg": 8, "steps": 25, "name": "TDD_Yuki_Mei_duo"
            },
            {
                "project": "TDD",
                "model": "chilloutmix_NiPrunedFp32Fix.safetensors",
                "type": "trio",
                "prompt": "one young Japanese man in center surrounded by two beautiful Japanese women, man looks nervous, women are flirting, 3 people total, living room scene, photorealistic",
                "negative": "solo, single person, more than 3 people",
                "cfg": 9, "steps": 30, "name": "TDD_harem_trio_cfg9"
            },
            {
                "project": "TDD",
                "model": "chilloutmix_NiPrunedFp32Fix.safetensors",
                "type": "duo",
                "prompt": "intimidating middle-aged Japanese businessman in dark suit confronting nervous young man, two men facing each other, tense scene, photorealistic",
                "negative": "woman, female, solo, more than 2 people",
                "cfg": 7, "steps": 20, "name": "TDD_Takeshi_Yuki_confrontation"
            },
            {
                "project": "TDD",
                "model": "chilloutmix_NiPrunedFp32Fix.safetensors",
                "type": "solo",
                "prompt": "intimidating Japanese businessman, middle-aged, expensive dark suit, cold expression, masculine features, standing in office, solo portrait, photorealistic",
                "negative": "woman, female, young, multiple people",
                "cfg": 6.5, "steps": 25, "name": "TDD_Takeshi_solo_cfg6.5"
            },

            # Cyberpunk Goblin Slayer - using counterfeit_v3 for ALL (8 images)
            {
                "project": "CGS",
                "model": "counterfeit_v3.safetensors",
                "type": "solo",
                "prompt": "armored warrior, cyberpunk armor with neon accents, glowing helmet, standing ready, solo character, anime style",
                "negative": "realistic, photograph, multiple people",
                "cfg": 7, "steps": 20, "name": "CGS_GoblinSlayer_solo"
            },
            {
                "project": "CGS",
                "model": "counterfeit_v3.safetensors",
                "type": "solo",
                "prompt": "young male fighter, cyberpunk clothing, energy sword, confident pose, solo character, anime style",
                "negative": "realistic, female, old, multiple people",
                "cfg": 8, "steps": 25, "name": "CGS_Kai_solo"
            },
            {
                "project": "CGS",
                "model": "counterfeit_v3.safetensors",
                "type": "solo",
                "prompt": "beautiful elf archer girl, long silver hair, energy bow, cyberpunk outfit, pointy ears, solo portrait, anime style",
                "negative": "realistic, male, short hair, multiple people",
                "cfg": 7.5, "steps": 20, "name": "CGS_Ryuu_solo"
            },
            {
                "project": "CGS",
                "model": "counterfeit_v3.safetensors",
                "type": "trio",
                "prompt": "three warriors standing together - armored warrior in center, young male fighter on left, elf archer girl with silver hair on right, team pose, cyberpunk setting, anime style",
                "negative": "realistic, solo, less than 3 people, more than 3 people",
                "cfg": 9, "steps": 30, "name": "CGS_team_trio_cfg9"
            },
            {
                "project": "CGS",
                "model": "counterfeit_v3.safetensors",
                "type": "duo",
                "prompt": "armored warrior and young male fighter fighting side by side, dynamic action pose, two warriors in combat, neon Tokyo street, anime style",
                "negative": "realistic, female, solo, more than 2 people",
                "cfg": 8.5, "steps": 25, "name": "CGS_combat_duo"
            },

            # Test different samplers (3 images)
            {
                "project": "TEST",
                "model": "chilloutmix_NiPrunedFp32Fix.safetensors",
                "type": "solo",
                "prompt": "beautiful Japanese woman, detailed face, photorealistic portrait",
                "negative": "low quality, bad anatomy",
                "cfg": 7, "steps": 20, "sampler": "euler_a", "name": "TEST_euler_a"
            },
            {
                "project": "TEST",
                "model": "chilloutmix_NiPrunedFp32Fix.safetensors",
                "type": "solo",
                "prompt": "beautiful Japanese woman, detailed face, photorealistic portrait",
                "negative": "low quality, bad anatomy",
                "cfg": 7, "steps": 20, "sampler": "dpmpp_2m", "name": "TEST_dpmpp_2m"
            },
            {
                "project": "TEST",
                "model": "chilloutmix_NiPrunedFp32Fix.safetensors",
                "type": "solo",
                "prompt": "beautiful Japanese woman, detailed face, photorealistic portrait",
                "negative": "low quality, bad anatomy",
                "cfg": 7, "steps": 20, "sampler": "dpmpp_sde", "name": "TEST_dpmpp_sde"
            }
        ]

    async def generate_test_image(self, config: dict) -> bool:
        """Generate a single test image with specific settings"""

        # Get sampler from config or default
        sampler = config.get("sampler", "euler")

        workflow = {
            "prompt": {
                "1": {
                    "inputs": {"ckpt_name": config["model"]},
                    "class_type": "CheckpointLoaderSimple"
                },
                "2": {
                    "inputs": {
                        "text": config["prompt"],
                        "clip": ["1", 1]
                    },
                    "class_type": "CLIPTextEncode"
                },
                "3": {
                    "inputs": {
                        "text": config["negative"],
                        "clip": ["1", 1]
                    },
                    "class_type": "CLIPTextEncode"
                },
                "4": {
                    "inputs": {
                        "seed": random.randint(1, 1000000000),
                        "steps": config["steps"],
                        "cfg": config["cfg"],
                        "sampler_name": sampler,
                        "scheduler": "karras" if "dpmpp" in sampler else "normal",
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
                        "width": 768 if config["type"] in ["duo", "trio"] else 512,
                        "height": 512 if config["type"] in ["duo", "trio"] else 768,
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
                        "filename_prefix": f"tweak_{config['name']}",
                        "images": ["6", 0]
                    },
                    "class_type": "SaveImage"
                }
            }
        }

        try:
            response = requests.post(
                f"{self.comfyui_url}/prompt",
                json={"prompt": workflow["prompt"]},
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                if "prompt_id" in result:
                    self.image_count += 1
                    logger.info(f"  [{self.image_count}/15] {config['name']} - CFG:{config['cfg']} Steps:{config['steps']}")
                    return True
            else:
                logger.error(f"  ❌ Failed {config['name']}: {response.status_code}")

        except Exception as e:
            logger.error(f"  ❌ Error: {e}")

        return False

    async def run_test(self):
        """Run the 15-image test"""

        logger.info("=" * 60)
        logger.info("15-IMAGE TWEAKING TEST")
        logger.info("=" * 60)
        logger.info("Testing different settings to find optimal configuration\n")

        for config in self.test_plan:
            if self.image_count >= self.max_images:
                break

            logger.info(f"\n{config['project']} - {config['type'].upper()} - {config['model'].split('.')[0]}")

            success = await self.generate_test_image(config)
            if success:
                await asyncio.sleep(15)  # Wait for generation

        logger.info("\n" + "=" * 60)
        logger.info(f"GENERATED {self.image_count}/15 TEST IMAGES")
        logger.info("=" * 60)
        logger.info("\nCheck results:")
        logger.info("ls -la /mnt/1TB-storage/ComfyUI/output/tweak_*")
        logger.info("\nLook for:")
        logger.info("  1. Correct gender in solos")
        logger.info("  2. Correct count in duos/trios")
        logger.info("  3. Best CFG value (7-9)")
        logger.info("  4. Best sampler (euler/euler_a/dpmpp_2m/dpmpp_sde)")
        logger.info("\n✅ Review these before mass generation!")

async def main():
    generator = TweakingTestGenerator()
    await generator.run_test()

if __name__ == "__main__":
    asyncio.run(main())