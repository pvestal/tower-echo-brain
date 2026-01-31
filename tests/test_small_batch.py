#!/usr/bin/env python3
"""
PROPER TESTING - Generate 5 images per character to verify quality FIRST
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

class SmallBatchTester:
    """Generate just 5 images per character to verify everything works"""

    def __init__(self):
        self.comfyui_url = "http://localhost:8188"
        self.output_dir = Path("/mnt/1TB-storage/ComfyUI/output")
        self.test_limit = 5  # ONLY 5 images per character for testing
        self.generation_count = {}

        # Test with just 2 characters first - one male, one female
        self.test_characters = {
            "Yuki_Tanaka": {
                "gender": "male",
                "checkpoint": "dreamshaper_8.safetensors",
                "prompts": [
                    "portrait of young Japanese man, nervous expression, masculine features, casual clothing, detailed face, solo",
                    "young Japanese man sitting alone, worried expression, masculine body, t-shirt and jeans, solo portrait",
                    "Japanese male in his 20s, anxious face, masculine build, counting money, solo shot",
                    "profile view of young Japanese man, stressed expression, masculine jawline, casual wear, solo",
                    "full body shot of nervous Japanese man, masculine physique, standing alone, modern clothing"
                ],
                "negative": "woman, female, breasts, feminine, multiple people, girl"
            },
            "Mei_Kobayashi": {
                "gender": "female",
                "checkpoint": "chilloutmix_NiPrunedFp32Fix.safetensors",
                "prompts": [
                    "beautiful Japanese woman, long black hair, gentle smile, medium breasts, casual dress, solo portrait",
                    "attractive Asian woman cooking, long dark hair, apron, kitchen scene, solo, feminine curves",
                    "Japanese woman in her 20s, long hair, caring expression, modern clothing, sitting alone",
                    "full body portrait of beautiful Japanese woman, long black hair, sundress, standing pose, solo",
                    "profile view of pretty Japanese woman, long dark hair, gentle expression, indoor lighting, solo"
                ],
                "negative": "man, male, masculine, multiple people, beard"
            }
        }

    async def generate_test_image(self, char_name: str, char_data: dict, image_num: int) -> bool:
        """Generate a single test image"""

        prompt = char_data["prompts"][image_num % len(char_data["prompts"])]
        seed = random.randint(1, 1000000000)

        workflow = {
            "prompt": {
                "1": {
                    "inputs": {"ckpt_name": char_data["checkpoint"]},
                    "class_type": "CheckpointLoaderSimple"
                },
                "2": {
                    "inputs": {
                        "text": prompt,
                        "clip": ["1", 1]
                    },
                    "class_type": "CLIPTextEncode"
                },
                "3": {
                    "inputs": {
                        "text": char_data["negative"],
                        "clip": ["1", 1]
                    },
                    "class_type": "CLIPTextEncode"
                },
                "4": {
                    "inputs": {
                        "seed": seed,
                        "steps": 20,
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
                        "filename_prefix": f"small_test_{char_name}_{image_num:02d}",
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
                    logger.info(f"  Image {image_num+1}/5: Queued (seed: {seed})")
                    return True
            else:
                logger.error(f"  ❌ Failed: {response.status_code}")

        except Exception as e:
            logger.error(f"  ❌ Error: {e}")

        return False

    async def run_test(self):
        """Run small batch test"""

        logger.info("=" * 60)
        logger.info("SMALL BATCH TEST - 5 IMAGES PER CHARACTER")
        logger.info("=" * 60)
        logger.info("Testing 2 characters with 5 images each\n")

        for char_name, char_data in self.test_characters.items():
            logger.info(f"\n📸 {char_name} ({char_data['gender']})")
            logger.info(f"   Model: {char_data['checkpoint'].split('.')[0]}")

            for i in range(self.test_limit):
                success = await self.generate_test_image(char_name, char_data, i)
                if success:
                    await asyncio.sleep(15)  # Wait between generations
                else:
                    logger.error(f"   Failed at image {i+1}")
                    break

        logger.info("\n" + "=" * 60)
        logger.info("TEST COMPLETE - CHECK RESULTS:")
        logger.info("=" * 60)
        logger.info("\n1. Check if all 10 images generated:")
        logger.info("   ls -la /mnt/1TB-storage/ComfyUI/output/small_test_* | wc -l")
        logger.info("\n2. Verify Yuki images are MALE:")
        logger.info("   Check /mnt/1TB-storage/ComfyUI/output/small_test_Yuki_*")
        logger.info("\n3. Verify Mei images are FEMALE:")
        logger.info("   Check /mnt/1TB-storage/ComfyUI/output/small_test_Mei_*")
        logger.info("\n⚠️  ONLY proceed to full generation if ALL images are correct!")

async def main():
    tester = SmallBatchTester()
    await tester.run_test()

if __name__ == "__main__":
    asyncio.run(main())