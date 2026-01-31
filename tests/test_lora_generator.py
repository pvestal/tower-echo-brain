#!/usr/bin/env python3
"""
Test LoRA generator with small batch to verify correct gender/character generation
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

class TestLoRAGenerator:
    """Test version - generates 1 image per character for verification"""

    def __init__(self):
        self.comfyui_url = "http://localhost:8188"
        self.output_dir = Path("/mnt/1TB-storage/ComfyUI/output")

        # Test ONE character from each gender/style combination
        self.test_characters = {
            "Yuki_Tanaka": {
                "gender": "male",
                "checkpoint": "deliberate_v2.safetensors",
                "prompt": "portrait of young Japanese man, nervous expression, masculine features, casual clothing, detailed face, solo, photorealistic",
                "negative": "woman, female, breasts, feminine, multiple people, girl",
                "seed": 12345  # Fixed seed for testing
            },
            "Mei_Kobayashi": {
                "gender": "female",
                "checkpoint": "chilloutmix_NiPrunedFp32Fix.safetensors",
                "prompt": "beautiful Japanese woman, long black hair, gentle smile, medium breasts, casual dress, solo portrait, photorealistic",
                "negative": "man, male, masculine, multiple people, beard",
                "seed": 23456
            },
            "Takeshi_Sato": {
                "gender": "male",
                "checkpoint": "deliberate_v2.safetensors",
                "prompt": "intimidating Japanese businessman, middle-aged, dark suit, cold expression, masculine features, solo portrait, photorealistic",
                "negative": "woman, female, young, feminine, multiple people, breasts",
                "seed": 34567
            },
            "Goblin_Slayer": {
                "gender": "male",
                "checkpoint": "counterfeit_v3.safetensors",
                "prompt": "armored warrior, cyberpunk armor with neon accents, helmet, standing alone, anime style, solo character",
                "negative": "realistic, photograph, multiple people, unarmored, female",
                "seed": 45678
            },
            "Ryuu": {
                "gender": "female",
                "checkpoint": "counterfeit_v3.safetensors",
                "prompt": "beautiful elf archer, long silver hair, energy bow, cyberpunk outfit, solo portrait, anime style",
                "negative": "realistic, photograph, male, short hair, multiple people",
                "seed": 56789
            }
        }

    async def generate_test_image(self, char_name: str, char_data: dict) -> bool:
        """Generate a single test image"""

        workflow = {
            "prompt": {
                "1": {
                    "inputs": {"ckpt_name": char_data["checkpoint"]},
                    "class_type": "CheckpointLoaderSimple"
                },
                "2": {
                    "inputs": {
                        "text": char_data["prompt"],
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
                        "seed": char_data["seed"],
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
                        "filename_prefix": f"TEST_{char_name}_{char_data['gender']}",
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
                    logger.info(f"✅ Queued {char_name} ({char_data['gender']}) - Checkpoint: {char_data['checkpoint'].split('.')[0]}")
                    return True
            else:
                logger.error(f"❌ Failed {char_name}: {response.status_code}")

        except Exception as e:
            logger.error(f"❌ Error generating {char_name}: {e}")

        return False

    async def run_test(self):
        """Run test generation"""

        logger.info("=" * 60)
        logger.info("🧪 TESTING LoRA GENERATOR - GENDER VERIFICATION")
        logger.info("=" * 60)
        logger.info("Generating 1 test image per character...")
        logger.info("Using FIXED seeds for reproducibility\n")

        # Generate test images
        for char_name, char_data in self.test_characters.items():
            logger.info(f"\n📸 Testing: {char_name}")
            logger.info(f"   Expected: {char_data['gender']}")
            logger.info(f"   Model: {char_data['checkpoint'].split('.')[0]}")

            success = await self.generate_test_image(char_name, char_data)

            if success:
                await asyncio.sleep(15)  # Wait for generation
            else:
                logger.error(f"   FAILED TO QUEUE!")

        logger.info("\n" + "=" * 60)
        logger.info("✅ TEST COMPLETE - CHECK IMAGES:")
        logger.info("=" * 60)

        # List expected files
        for char_name, char_data in self.test_characters.items():
            expected = f"TEST_{char_name}_{char_data['gender']}_*.png"
            logger.info(f"   {expected}")

        logger.info("\n📁 Command to view:")
        logger.info("ls -la /mnt/1TB-storage/ComfyUI/output/TEST_*")
        logger.info("\n⚠️  VERIFY each image shows correct gender before mass generation!")

async def main():
    tester = TestLoRAGenerator()
    await tester.run_test()

if __name__ == "__main__":
    asyncio.run(main())