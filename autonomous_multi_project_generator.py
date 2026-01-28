#!/usr/bin/env python3
"""
Autonomous Multi-Project Generator for Echo Brain
Handles autonomous generation for all active anime projects
"""

import asyncio
import requests
import json
import time
import logging
import psycopg2
from pathlib import Path
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AutonomousMultiProjectGenerator:
    """Multi-project autonomous image generator"""

    def __init__(self):
        self.comfyui_url = "http://localhost:8188"
        self.output_dir = Path("/mnt/1TB-storage/ComfyUI/output")
        self.is_running = False
        self.generation_count = 0
        self.db_config = {
            'host': 'localhost',
            'database': 'anime_production',
            'user': 'patrick',
            'password': 'RP78eIrW7cI2jYvL5akt1yurE'
        }

        # Project-specific configurations
        self.project_configs = {
            "Tokyo Debt Desire": {
                "style": "photorealistic",
                "content_rating": "nsfw",
                "model": "realisticVision_v51.safetensors",
                "limit": 1000,
                "characters": {
                    "Mei_Kobayashi": {
                        "prompts": [
                            "beautiful Japanese woman, female anatomy, long dark hair, seductive gentle expression, wearing skimpy apron with visible cleavage, medium natural breasts, vagina, feminine curves, photorealistic, NSFW, aroused expression",
                            "beautiful curvy Japanese woman, female anatomy, long dark hair, sultry smile, wearing tight low-cut top, visible cleavage, medium natural breasts, feminine body, photorealistic, slightly aroused",
                            "beautiful Japanese woman, female anatomy, long dark hair, caring but sexy expression, wearing revealing crop top, bent over pose, feminine curves, photorealistic, arousal visible"
                        ]
                    },
                    "Rina_Suzuki": {
                        "prompts": [
                            "attractive confident Japanese woman, female anatomy, short brown hair, assertive seductive expression, wearing tight mini dress with deep neckline, medium natural breasts, vagina, feminine curves, photorealistic, aroused",
                            "attractive Japanese woman, female anatomy, short brown hair, seductive confident expression, wearing lingerie, bedroom setting, medium natural breasts, feminine body, photorealistic, arousal state",
                            "attractive Japanese woman, female anatomy, short brown hair, arguing but sexy pose, wearing revealing tight clothing, feminine curves, photorealistic, sexually excited"
                        ]
                    },
                    "Yuki_Tanaka": {
                        "prompts": [
                            "young nervous Japanese man, worried anxious expression, surrounded by two sexy women in revealing clothing, photorealistic, visible male arousal, masculine body, NSFW situation",
                            "young Japanese man, nervous worried expression, sitting shirtless between two curvy women touching him, masculine chest, photorealistic, sexually excited, NSFW scene",
                            "young Japanese man, stressed but sexually aroused expression, attractive woman in lingerie distracts him, masculine features, photorealistic, visible male arousal, NSFW content"
                        ]
                    },
                    "Takeshi_Sato": {
                        "prompts": [
                            "intimidating middle-aged Japanese man, cold menacing expression, expensive dark business suit, masculine features, photorealistic, dangerous aura",
                            "intimidating Japanese man in business suit, menacing expression, while sexy women in background, masculine build, photorealistic, power dynamic",
                            "intimidating yakuza man, dark suit, cold calculating expression, office with attractive secretary, masculine presence, photorealistic, authority figure"
                        ]
                    }
                }
            },

            "Cyberpunk Goblin Slayer: Neon Shadows": {
                "style": "cyberpunk_anime",
                "content_rating": "mature",
                "model": "counterfeit_v3.safetensors",
                "limit": 500,
                "characters": {
                    "Kai": {
                        "prompts": [
                            "young female, dark hair with neon blue streaks, glowing cybernetic eye, angular face, mechanical joints at jawline, black tactical vest with glowing circuits, cyberpunk aesthetic",
                            "young female protagonist, neon blue hair streaks, cybernetic enhancements, tactical gear, cyberpunk street setting, anime style",
                            "female cyberpunk warrior, dark hair with blue highlights, glowing right eye, mechanical jaw, tactical vest, futuristic city background, anime style"
                        ]
                    },
                    "Goblin Slayer": {
                        "prompts": [
                            "cybernetic warrior with glowing red visor, tactical armor with neon accents, plasma sword, cyberpunk aesthetic, anime style",
                            "armored cyberpunk warrior, red glowing visor, neon-lit tactical gear, dark urban setting, anime style",
                            "futuristic armored figure, red cybernetic visor, high-tech armor with neon details, cyberpunk city, anime style"
                        ]
                    },
                    "Ryuu": {
                        "prompts": [
                            "sniper with silver hair, cyberpunk aesthetic, long-range weapons specialist, tactical gear, anime style",
                            "silver-haired marksman, cyberpunk setting, sniper rifle, tactical equipment, neon city background, anime style",
                            "cyberpunk sniper character, silver hair, precision weapons, tactical gear, urban setting, anime style"
                        ]
                    }
                }
            }
        }

    async def start_autonomous_generation(self):
        """Start autonomous generation for all active projects"""

        logger.info("🧠 Starting Multi-Project Autonomous Generator")
        self.is_running = True

        # Get active projects from database
        active_projects = await self._get_active_projects()
        logger.info(f"📋 Found {len(active_projects)} active projects")

        generation_cycle = 0

        while self.is_running:
            try:
                generation_cycle += 1
                logger.info(f"🎯 Starting multi-project generation cycle {generation_cycle}")

                for project in active_projects:
                    project_name = project['name']

                    if project_name not in self.project_configs:
                        logger.warning(f"⚠️  No configuration for project: {project_name}")
                        continue

                    config = self.project_configs[project_name]

                    # Check if project has reached its limit
                    project_count = await self._get_project_generation_count(project_name)
                    if project_count >= config['limit']:
                        logger.info(f"✅ {project_name} reached limit ({config['limit']}), skipping")
                        continue

                    logger.info(f"🎨 Generating for {project_name} ({project_count}/{config['limit']})")

                    # Generate images for each character in the project
                    for char_name, char_config in config['characters'].items():
                        # Rotate through prompts
                        prompt_index = (generation_cycle - 1) % len(char_config['prompts'])
                        prompt = char_config['prompts'][prompt_index]

                        success = await self._generate_project_image(
                            project_name, char_name, prompt, config, generation_cycle
                        )

                        if success:
                            self.generation_count += 1
                            logger.info(f"✅ Generated {project_name}/{char_name} #{self.generation_count}")
                        else:
                            logger.warning(f"❌ Failed {project_name}/{char_name}")

                        # Small delay between characters
                        await asyncio.sleep(2)

                    # Delay between projects
                    await asyncio.sleep(5)

                logger.info(f"🔄 Multi-project cycle {generation_cycle} complete. Total generated: {self.generation_count}")

                # Wait before next cycle
                await asyncio.sleep(60)

            except KeyboardInterrupt:
                logger.info("🛑 Stopping multi-project generation...")
                self.is_running = False
                break
            except Exception as e:
                logger.error(f"❌ Error in multi-project cycle: {e}")
                await asyncio.sleep(30)

    async def _get_active_projects(self) -> List[Dict]:
        """Get active projects from database, with fallback to configured projects"""

        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            cur.execute("SELECT name, description, status FROM projects WHERE status IN ('active', 'generating')")
            projects = cur.fetchall()
            conn.close()

            return [{'name': p[0], 'description': p[1], 'status': p[2]} for p in projects]
        except Exception as e:
            logger.error(f"❌ Database error: {e}")
            logger.info("🔄 Using fallback: all configured projects as active")

            # Fallback: return all configured projects as active
            return [
                {'name': name, 'description': f'Autonomous generation for {name}', 'status': 'active'}
                for name in self.project_configs.keys()
            ]

    async def _get_project_generation_count(self, project_name: str) -> int:
        """Count existing generated images for a project"""

        pattern = f"autonomous_{project_name.replace(' ', '_').replace(':', '').lower()}_*"
        files = list(self.output_dir.glob(pattern))
        return len(files)

    async def _generate_project_image(self, project_name: str, character: str, prompt: str, config: Dict, cycle: int) -> bool:
        """Generate image for specific project character"""

        try:
            # Create filename prefix
            safe_project = project_name.replace(' ', '_').replace(':', '').lower()
            filename_prefix = f"autonomous_{safe_project}_{character}_cycle{cycle}"

            # Build full prompt based on project style
            if config['content_rating'] == 'nsfw':
                full_prompt = f"photograph, {prompt}, ultra realistic, high detail, professional photography, 4k, masterpiece, NSFW, explicit, adult content, sexually suggestive"
                negative_prompt = "cartoon, anime, illustration, drawing, painting, low quality, blurry, deformed, ugly, distorted"
            else:
                full_prompt = f"{prompt}, high detail, masterpiece, best quality"
                negative_prompt = "low quality, blurry, deformed, ugly, bad anatomy"

            workflow = {
                "prompt": {
                    "1": {
                        "inputs": {"ckpt_name": config['model']},
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
                            "text": negative_prompt,
                            "clip": ["1", 1]
                        },
                        "class_type": "CLIPTextEncode"
                    },
                    "4": {
                        "inputs": {
                            "width": 768,
                            "height": 1024,
                            "batch_size": 1
                        },
                        "class_type": "EmptyLatentImage"
                    },
                    "5": {
                        "inputs": {
                            "seed": int(time.time()) % 1000000 + cycle,
                            "steps": 25,
                            "cfg": 7.0 if config['content_rating'] == 'nsfw' else 7.5,
                            "sampler_name": "dpmpp_2m_sde",
                            "scheduler": "karras",
                            "denoise": 1,
                            "model": ["1", 0],
                            "positive": ["2", 0],
                            "negative": ["3", 0],
                            "latent_image": ["4", 0]
                        },
                        "class_type": "KSampler"
                    },
                    "6": {
                        "inputs": {
                            "samples": ["5", 0],
                            "vae": ["1", 2]
                        },
                        "class_type": "VAEDecode"
                    },
                    "7": {
                        "inputs": {
                            "filename_prefix": filename_prefix,
                            "images": ["6", 0]
                        },
                        "class_type": "SaveImage"
                    }
                }
            }

            response = requests.post(f"{self.comfyui_url}/prompt", json={"prompt": workflow["prompt"]}, timeout=15)
            result = response.json()

            if "prompt_id" in result:
                logger.info(f"📷 Queued {project_name}/{character}: {result['prompt_id']}")

                # Wait for generation
                await asyncio.sleep(15)

                # Check if file was created
                expected_file = self.output_dir / f"{filename_prefix}_00001_.png"
                if expected_file.exists():
                    file_size = expected_file.stat().st_size
                    if file_size > 100000:
                        return True

                return False
            else:
                return False

        except Exception as e:
            logger.error(f"❌ Generation failed for {project_name}/{character}: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get current generation status"""

        return {
            "is_running": self.is_running,
            "total_generated": self.generation_count,
            "output_directory": str(self.output_dir),
            "projects_configured": list(self.project_configs.keys())
        }

async def main():
    """Main function"""

    generator = AutonomousMultiProjectGenerator()

    logger.info("🚀 Echo Brain Multi-Project Autonomous Generator")
    logger.info("📷 Will generate for all active anime projects")
    logger.info("🔄 Press Ctrl+C to stop")

    try:
        await generator.start_autonomous_generation()
    except KeyboardInterrupt:
        logger.info("🛑 Multi-project generation stopped by user")

    # Show final status
    status = generator.get_status()
    logger.info(f"📊 Final Status: Generated {status['total_generated']} total images across all projects")

if __name__ == "__main__":
    asyncio.run(main())