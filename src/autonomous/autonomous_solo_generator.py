#!/usr/bin/env python3
"""
Autonomous SOLO Image Generator for LoRA Training
Generates ONLY solo images which actually work properly
Using SSOT configuration for model selection
"""
import asyncio
import requests
import json
import random
import time
import logging
from pathlib import Path
from typing import Dict, List
from project_config_ssot import config_manager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AutonomousSoloGenerator:
    """Generate solo character images for LoRA training"""

    def __init__(self):
        self.comfyui_url = "http://localhost:8188"
        self.output_dir = Path("/mnt/1TB-storage/ComfyUI/output")
        self.generation_count = {}
        self.images_per_character = 100  # Target for LoRA training

        # Character configurations - SOLO ONLY
        self.characters = {
            "tokyo_debt_desire": {
                "model": config_manager.models["primary"].file,  # Use SSOT primary model
                "style": config_manager.projects["tokyo_debt_desire"]["style_prompt"],
                "characters": {
                    "Yuki_Tanaka": {
                        "gender": "male",
                        "prompts": [
                            "young nervous Japanese man, masculine features, short black hair, worried expression, casual t-shirt, counting money, solo portrait",
                            "Japanese man in his 20s, anxious face, masculine build, sitting alone at table with bills, indoor lighting, solo",
                            "profile view of young Japanese man, stressed expression, masculine jawline, casual wear, holding calculator, solo",
                            "nervous young man, Japanese, short dark hair, masculine features, looking at phone worried, apartment setting, solo",
                            "full body shot of Japanese man, masculine physique, casual clothes, standing alone looking anxious, modern apartment, solo"
                        ],
                        "negative": "woman, female, breasts, feminine, multiple people, long hair"
                    },
                    "Mei_Kobayashi": {
                        "gender": "female",
                        "prompts": [
                            "beautiful Japanese woman, long black hair, gentle smile, cooking in kitchen, apron over dress, medium breasts, solo portrait",
                            "attractive Asian woman, long dark hair, caring expression, sitting alone reading, casual dress, feminine curves, solo",
                            "Japanese woman in her 20s, long black hair, sweet smile, cleaning apartment, casual clothes, solo shot",
                            "pretty Japanese woman, long hair tied back, concentrating while cooking, kitchen scene, apron, solo",
                            "full body portrait of Japanese woman, long black hair, sundress, standing by window, gentle expression, solo"
                        ],
                        "negative": "man, male, masculine, multiple people, beard, short hair"
                    },
                    "Rina_Suzuki": {
                        "gender": "female",
                        "prompts": [
                            "confident Japanese woman, short brown hair, assertive expression, business casual, hands on hips, solo portrait",
                            "attractive Asian woman, short hair, determined face, mini skirt and blouse, standing pose, solo",
                            "Japanese businesswoman, short brown hair, serious expression, checking phone, modern outfit, solo shot",
                            "assertive Japanese woman, short hair, confident smile, casual but stylish clothes, apartment, solo",
                            "full body shot of confident woman, short brown hair, power pose, fitted dress, office setting, solo"
                        ],
                        "negative": "man, male, masculine, multiple people, long hair"
                    },
                    "Takeshi_Sato": {
                        "gender": "male",
                        "prompts": [
                            "intimidating Japanese businessman, middle-aged, dark suit, cold expression, masculine features, solo portrait",
                            "menacing Japanese man, short black hair, expensive suit, stern face, sitting at desk, solo",
                            "middle-aged yakuza boss, dark suit, calculating expression, office with city view, masculine presence, solo",
                            "dangerous looking Japanese man, business attire, threatening aura, standing by window, solo shot",
                            "profile shot of intimidating man, expensive dark suit, cold eyes, masculine jawline, solo"
                        ],
                        "negative": "woman, female, young, feminine, multiple people, breasts"
                    }
                }
            },
            "cyberpunk_goblin_slayer": {
                "model": config_manager.models["fallback"].file,  # Use fallback for anime style
                "style": config_manager.projects["cyberpunk_goblin_slayer"]["style_prompt"],
                "characters": {
                    "Goblin_Slayer": {
                        "gender": "male",
                        "prompts": [
                            "armored warrior, cyberpunk armor with neon accents, glowing helmet, standing ready, solo character, anime",
                            "goblin slayer in futuristic armor, energy weapon, combat pose, neon lighting, solo shot, anime",
                            "cyberpunk knight, high-tech armor, checking equipment, dark alley background, solo, anime style",
                            "armored fighter, cyber helmet with visor, intimidating stance, neon Tokyo street, solo character, anime",
                            "full body armor warrior, glowing armor details, weapon at ready, night scene, solo, anime"
                        ],
                        "negative": "realistic, photograph, multiple people, unarmored, female"
                    },
                    "Kai": {
                        "gender": "male",
                        "prompts": [
                            "young male fighter, cyberpunk clothes, energy sword, confident pose, solo portrait, anime style",
                            "teenage boy warrior, futuristic outfit, determined expression, training stance, solo, anime",
                            "young cyber fighter, neon accents on clothes, ready for battle, urban background, solo shot, anime",
                            "male protagonist, modern combat gear, fierce expression, holding weapon, solo character, anime",
                            "young warrior, cyber-enhanced clothes, action pose, Tokyo street, solo, anime style"
                        ],
                        "negative": "realistic, photograph, female, old, multiple people"
                    },
                    "Ryuu": {
                        "gender": "female",
                        "prompts": [
                            "beautiful elf archer, long silver hair, energy bow, cyberpunk outfit, pointy ears, solo portrait, anime",
                            "female elf warrior, silver hair flowing, futuristic armor, graceful pose, solo, anime style",
                            "elven woman with bow, long silver hair, neon-accented clothes, determined expression, solo shot, anime",
                            "graceful elf fighter, high-tech bow, silver hair, combat ready stance, solo character, anime",
                            "elf girl, long silver hair, cyber outfit, aiming bow, rooftop scene, solo, anime style"
                        ],
                        "negative": "realistic, photograph, male, short hair, multiple people"
                    }
                }
            }
        }

    async def generate_character(self, project: str, char_name: str, char_data: dict, project_config: dict) -> bool:
        """Generate a single solo character image"""

        # Random prompt from character's list
        prompt = random.choice(char_data["prompts"])
        full_prompt = f"{prompt}, {project_config['style']}"

        # Random seed for variety
        seed = random.randint(1, 1000000000)

        # Get optimal settings from SSOT
        model_config = config_manager.get_model_for_character(project, char_name)

        workflow = {
            "prompt": {
                "1": {
                    "inputs": {"ckpt_name": model_config.file},
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
                        "text": char_data["negative"],
                        "clip": ["1", 1]
                    },
                    "class_type": "CLIPTextEncode"
                },
                "4": {
                    "inputs": {
                        "seed": seed,
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
                        "height": 768,  # Portrait orientation
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
                        "filename_prefix": f"lora_training_{project}_{char_name}_solo",
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

                    count = self.generation_count[char_name]
                    logger.info(f"  ✅ {char_name} #{count}/{self.images_per_character} ({char_data['gender']})")
                    return True

        except Exception as e:
            logger.error(f"  ❌ Error generating {char_name}: {e}")

        return False

    async def start_generation(self):
        """Start autonomous solo generation"""

        logger.info("🚀 Autonomous SOLO Image Generator")
        logger.info(f"🎯 Target: {self.images_per_character} solo images per character")
        logger.info("📊 Projects: Tokyo Debt Desire (4 chars) + Cyberpunk Goblin Slayer (3 chars)")
        logger.info("=" * 60)

        cycle = 0

        while True:
            cycle += 1
            logger.info(f"\n🔄 Cycle {cycle}")

            all_complete = True

            for project_name, project_config in self.characters.items():
                logger.info(f"\n📁 {project_name} ({project_config['model'].split('.')[0]})")

                for char_name, char_data in project_config["characters"].items():
                    count = self.generation_count.get(char_name, 0)

                    if count >= self.images_per_character:
                        logger.info(f"  ✓ {char_name}: Complete ({count}/{self.images_per_character})")
                        continue

                    all_complete = False

                    # Generate image
                    success = await self.generate_character(
                        project_name,
                        char_name,
                        char_data,
                        project_config
                    )

                    if success:
                        await asyncio.sleep(15)  # Wait for generation

            if all_complete:
                logger.info("\n" + "=" * 60)
                logger.info("🎯 ALL COMPLETE!")
                logger.info(f"Generated {self.images_per_character} solo images for each character")
                logger.info("Ready for LoRA training!")
                break

            # Progress summary
            logger.info("\n📊 Progress:")
            total = 0
            for name, count in self.generation_count.items():
                logger.info(f"  {name}: {count}/{self.images_per_character}")
                total += count
            logger.info(f"  Total: {total}/{self.images_per_character * 7} images")

            # Wait before next cycle
            await asyncio.sleep(30)

async def main():
    generator = AutonomousSoloGenerator()

    try:
        await generator.start_generation()
    except KeyboardInterrupt:
        logger.info("\n⛔ Stopped by user")
        logger.info(f"Generated images so far:")
        for name, count in generator.generation_count.items():
            logger.info(f"  {name}: {count}")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")

if __name__ == "__main__":
    asyncio.run(main())