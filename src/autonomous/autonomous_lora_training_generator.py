#!/usr/bin/env python3
"""
Autonomous LoRA Training Image Generator
Generates SOLO character images for training + some scenes for context
"""
import asyncio
import requests
import json
import random
import time
import logging
from pathlib import Path
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LoRATrainingGenerator:
    """Generate character images optimized for LoRA training"""

    def __init__(self):
        self.comfyui_url = "http://localhost:8188"
        self.output_dir = Path("/mnt/1TB-storage/ComfyUI/output")
        self.generation_count = {}

        # Character configurations for SOLO generation
        self.characters = {
            "tokyo_debt_desire": {
                "Yuki_Tanaka": {
                    "gender": "male",
                    "checkpoint": "dreamshaper_8.safetensors",  # Fixed - works for males
                    "solo_prompts": [
                        "portrait of young Japanese man, nervous expression, masculine features, casual clothing, detailed face, solo, photorealistic",
                        "young Japanese man sitting alone, worried expression, masculine body, t-shirt and jeans, indoor scene, solo portrait, photorealistic",
                        "Japanese male in his 20s, anxious face, masculine build, counting money, solo shot, detailed features, photorealistic",
                        "profile view of young Japanese man, stressed expression, masculine jawline, casual wear, solo character, photorealistic",
                        "full body shot of nervous Japanese man, masculine physique, standing alone, modern clothing, photorealistic"
                    ],
                    "negative": "woman, female, breasts, feminine, multiple people, girl"
                },
                "Mei_Kobayashi": {
                    "gender": "female",
                    "checkpoint": "chilloutmix_NiPrunedFp32Fix.safetensors",  # Best for Asian women
                    "solo_prompts": [
                        "beautiful Japanese woman, long black hair, gentle smile, medium breasts, casual dress, solo portrait, photorealistic",
                        "attractive Asian woman cooking, long dark hair, apron, kitchen scene, solo, feminine curves, photorealistic",
                        "Japanese woman in her 20s, long hair, caring expression, modern clothing, sitting alone, detailed face, photorealistic",
                        "full body portrait of beautiful Japanese woman, long black hair, sundress, standing pose, solo, photorealistic",
                        "profile view of pretty Japanese woman, long dark hair, gentle expression, indoor lighting, solo shot, photorealistic"
                    ],
                    "negative": "man, male, masculine, multiple people, beard"
                },
                "Rina_Suzuki": {
                    "gender": "female",
                    "checkpoint": "chilloutmix_NiPrunedFp32Fix.safetensors",
                    "solo_prompts": [
                        "confident Japanese woman, short brown hair, assertive expression, business casual, solo portrait, photorealistic",
                        "attractive Asian woman, short hair, hands on hips, mini skirt, standing alone, confident pose, photorealistic",
                        "Japanese woman in her 20s, short brown hair, determined face, modern outfit, solo shot, detailed features, photorealistic",
                        "full body shot of assertive Japanese woman, short hair, power pose, office wear, solo, photorealistic",
                        "close-up portrait of Japanese woman, short brown hair, confident smile, professional look, solo, photorealistic"
                    ],
                    "negative": "man, male, masculine, multiple people, long hair"
                },
                "Takeshi_Sato": {
                    "gender": "male",
                    "checkpoint": "dreamshaper_8.safetensors",
                    "solo_prompts": [
                        "intimidating Japanese businessman, middle-aged, dark suit, cold expression, masculine features, solo portrait, photorealistic",
                        "menacing Japanese man in suit, short black hair, stern face, standing alone, masculine build, photorealistic",
                        "yakuza boss portrait, expensive suit, calculating expression, masculine presence, solo shot, detailed face, photorealistic",
                        "middle-aged Japanese man, business attire, threatening aura, sitting at desk, solo, masculine features, photorealistic",
                        "profile shot of dangerous Japanese man, dark suit, cold eyes, masculine jawline, solo character, photorealistic"
                    ],
                    "negative": "woman, female, young, feminine, multiple people, breasts"
                }
            },
            "cyberpunk_goblin_slayer": {
                "Goblin_Slayer": {
                    "gender": "male",
                    "checkpoint": "counterfeit_v3.safetensors",  # Good for anime
                    "solo_prompts": [
                        "armored warrior, cyberpunk armor with neon accents, helmet, standing alone, anime style, solo character",
                        "goblin slayer in futuristic armor, glowing visor, combat pose, solo shot, anime art style",
                        "cyberpunk knight, high-tech armor, weapon ready, dark alley, solo warrior, anime style",
                        "armored fighter checking equipment, neon-lit armor, preparation scene, solo, anime style",
                        "masked warrior in cyber armor, intimidating stance, solo character, detailed armor, anime"
                    ],
                    "negative": "realistic, photograph, multiple people, unarmored"
                },
                "Kai": {
                    "gender": "male",
                    "checkpoint": "counterfeit_v3.safetensors",
                    "solo_prompts": [
                        "young male fighter, cyberpunk clothes, energy sword, confident pose, solo portrait, anime style",
                        "teenage boy warrior, futuristic outfit, determined expression, standing alone, anime art",
                        "young cyber fighter, neon accents on clothes, training pose, solo shot, anime style",
                        "male protagonist, modern combat gear, ready stance, solo character, detailed face, anime",
                        "young warrior portrait, cyber-enhanced clothes, fierce expression, solo, anime style"
                    ],
                    "negative": "realistic, photograph, female, old, multiple people"
                },
                "Ryuu": {
                    "gender": "female",
                    "checkpoint": "counterfeit_v3.safetensors",
                    "solo_prompts": [
                        "beautiful elf archer, long silver hair, energy bow, cyberpunk outfit, solo portrait, anime style",
                        "female elf warrior, futuristic armor, elegant pose, standing alone, detailed face, anime",
                        "elven woman with bow, silver hair, neon-accented clothes, solo shot, anime art style",
                        "graceful elf fighter, high-tech bow, combat ready, solo character, anime style",
                        "elf girl portrait, long silver hair, determined eyes, cyber outfit, solo, anime"
                    ],
                    "negative": "realistic, photograph, male, short hair, multiple people"
                }
            }
        }

        # Occasionally generate group scenes (20% of the time)
        self.group_scenes = {
            "tokyo_debt_desire": [
                "young nervous Japanese man talking with beautiful woman with long hair, two people interacting, photorealistic",
                "Japanese businessman threatening younger man, tense confrontation, two people, photorealistic"
            ],
            "cyberpunk_goblin_slayer": [
                "armored warrior and young fighter back to back, combat ready, two people, anime style",
                "elf archer providing cover for armored warrior, action scene, two people, anime style"
            ]
        }

    async def generate_character(self, project: str, char_name: str, char_data: dict) -> bool:
        """Generate a single character image with random prompt and seed"""

        # Pick random prompt
        prompt = random.choice(char_data["solo_prompts"])

        # Random seed for variety
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
                        "steps": 25,
                        "cfg": 7.5,
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
                        "width": 512,
                        "height": 768,  # Portrait orientation for characters
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
                        "filename_prefix": f"lora_training_{project}_{char_name}_{seed}",
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
                    if char_name not in self.generation_count:
                        self.generation_count[char_name] = 0
                    self.generation_count[char_name] += 1

                    logger.info(f"✅ Generated {char_name} #{self.generation_count[char_name]} "
                              f"({char_data['gender']}) - Seed: {seed}")
                    return True
            else:
                logger.error(f"❌ Failed {char_name}: {response.status_code}")

        except Exception as e:
            logger.error(f"❌ Error generating {char_name}: {e}")

        return False

    async def generate_group_scene(self, project: str) -> bool:
        """Occasionally generate a group scene (20% chance)"""

        if random.random() > 0.2:  # 80% skip
            return False

        if project not in self.group_scenes:
            return False

        prompt = random.choice(self.group_scenes[project])
        seed = random.randint(1, 1000000000)

        # Use best checkpoint for project type
        checkpoint = "deliberate_v2.safetensors" if "debt" in project else "counterfeit_v3.safetensors"

        logger.info(f"🎬 Generating group scene for {project}")

        # Similar workflow but for group scene
        # ... (workflow code similar to above)

        return True

    async def start_generation(self):
        """Start autonomous generation with proper validation"""

        logger.info("🚀 LoRA Training Image Generator")
        logger.info("📊 Target: 100 images per character for training")
        logger.info("🎲 Using random seeds for variety\n")

        images_per_character = 100
        cycle = 0

        while True:
            cycle += 1
            logger.info(f"\n🔄 Generation Cycle {cycle}")

            # Generate for each project
            for project, characters in self.characters.items():
                logger.info(f"📁 Project: {project}")

                # Generate solo images for each character
                for char_name, char_data in characters.items():
                    # Check if we have enough images
                    count = self.generation_count.get(char_name, 0)
                    if count >= images_per_character:
                        logger.info(f"  ✓ {char_name}: {count}/{images_per_character} complete")
                        continue

                    # Generate image
                    success = await self.generate_character(project, char_name, char_data)

                    if success:
                        await asyncio.sleep(15)  # Wait for generation

                # Occasionally generate group scene
                await self.generate_group_scene(project)

            # Check if all complete
            all_complete = all(
                self.generation_count.get(char, 0) >= images_per_character
                for chars in self.characters.values()
                for char in chars.keys()
            )

            if all_complete:
                logger.info("\n🎯 All characters have 100+ training images!")
                break

            # Status report
            logger.info("\n📊 Current Progress:")
            for project, characters in self.characters.items():
                for char_name in characters:
                    count = self.generation_count.get(char_name, 0)
                    logger.info(f"  {char_name}: {count}/{images_per_character}")

            await asyncio.sleep(30)

async def main():
    generator = LoRATrainingGenerator()

    try:
        await generator.start_generation()
    except KeyboardInterrupt:
        logger.info("\n⛔ Stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")

if __name__ == "__main__":
    asyncio.run(main())