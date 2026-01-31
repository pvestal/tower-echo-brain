#!/usr/bin/env python3
"""
Autonomous Multi-Character Scene Generator with Echo Brain Integration
Generates complete scenes with multiple characters interacting, not individual portraits
"""
import asyncio
import requests
import json
import time
import random
import logging
from pathlib import Path
from typing import List, Dict, Any
from image_quality_reviewer import EchoBrainImageReviewer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AutonomousSceneGenerator:
    """Generate multi-character scenes for anime projects with Echo Brain feedback"""

    def __init__(self):
        self.comfyui_url = "http://localhost:8188"
        self.output_dir = Path("/mnt/1TB-storage/ComfyUI/output")
        self.is_running = False
        self.generation_count = 0
        self.reviewer = EchoBrainImageReviewer()

        # Project configurations with SCENE-BASED generation
        self.projects = {
            "tokyo_debt_desire": {
                "style": "photorealistic, natural lighting, 8k, detailed skin texture",
                "negative": "cartoon, anime, drawn, painted, sketch, 3d render, cgi, illustration",
                "max_images": 1000,
                "scenes": {
                    "domestic_tension": {
                        "prompt": "young nervous Japanese man (Yuki) with masculine features and visible male arousal, being seduced by beautiful Japanese woman with long dark hair (Mei) wearing skimpy apron showing cleavage and medium breasts, she's touching his chest while he counts money bills, Tokyo apartment kitchen, sexual tension, NSFW, photorealistic",
                        "characters": ["Yuki", "Mei"]
                    },
                    "confrontation": {
                        "prompt": "young anxious Japanese man (Yuki) with masculine body sitting on couch, argued with by attractive confident Japanese woman with short brown hair (Rina) wearing tight mini dress with visible cleavage, she's standing with hands on hips while he looks worried, Tokyo apartment living room, sexual and financial tension, NSFW, photorealistic",
                        "characters": ["Yuki", "Rina"]
                    },
                    "harem_scene": {
                        "prompt": "young overwhelmed Japanese man (Yuki) with masculine features surrounded by two sexy women - one with long dark hair (Mei) and one with short brown hair (Rina) both in revealing clothing touching him from both sides, he's holding money looking stressed but aroused with visible male excitement, Tokyo apartment, harem situation, NSFW, photorealistic",
                        "characters": ["Yuki", "Mei", "Rina"]
                    },
                    "yakuza_threat": {
                        "prompt": "intimidating middle-aged Japanese man (Takeshi) in expensive dark suit with masculine presence threatening younger nervous Japanese man (Yuki) who looks scared, Takeshi grabbing Yuki's collar aggressively, dark Tokyo alley at night, dangerous confrontation, photorealistic",
                        "characters": ["Takeshi", "Yuki"]
                    },
                    "women_plotting": {
                        "prompt": "two beautiful Japanese women conspiring together - one with long dark hair (Mei) wearing revealing outfit and one with short brown hair (Rina) in tight dress, both with seductive expressions whispering to each other, Tokyo apartment bedroom, plotting something, visible cleavage and curves, NSFW, photorealistic",
                        "characters": ["Mei", "Rina"]
                    },
                    "debt_collection": {
                        "prompt": "intimidating Japanese yakuza man (Takeshi) in business suit standing at apartment door while young nervous man (Yuki) hands over money, two women (Mei and Rina) watching anxiously from behind in revealing clothing, tense atmosphere, Tokyo apartment entrance, photorealistic",
                        "characters": ["Takeshi", "Yuki", "Mei", "Rina"]
                    }
                }
            },
            "cyberpunk_goblin_slayer": {
                "style": "anime style, cyberpunk aesthetic, neon lighting, detailed armor",
                "negative": "photorealistic, real person, photograph, low quality",
                "max_images": 500,
                "scenes": {
                    "team_preparation": {
                        "prompt": "armored warrior (Goblin Slayer) checking weapons with young male fighter (Kai) and female elf archer (Ryuu), all in cyberpunk armor with neon accents, preparing for battle in neon-lit Tokyo alley, anime style",
                        "characters": ["Goblin Slayer", "Kai", "Ryuu"]
                    },
                    "goblin_battle": {
                        "prompt": "armored warrior (Goblin Slayer) fighting multiple cyber-goblins with glowing red eyes, young fighter (Kai) backing him up with energy sword, dynamic action scene, neon Tokyo street, anime style combat",
                        "characters": ["Goblin Slayer", "Kai", "goblins"]
                    },
                    "ryuu_support": {
                        "prompt": "beautiful elf archer (Ryuu) with long silver hair providing cover fire with energy bow while armored warrior (Goblin Slayer) advances, cyberpunk Tokyo rooftop, neon signs in background, anime style",
                        "characters": ["Ryuu", "Goblin Slayer"]
                    },
                    "victory_scene": {
                        "prompt": "three warriors standing victorious - armored Goblin Slayer in center, young male fighter Kai on left, elf archer Ryuu on right, defeated cyber-goblins at their feet, neon Tokyo skyline behind them, anime style epic scene",
                        "characters": ["Goblin Slayer", "Kai", "Ryuu"]
                    }
                }
            }
        }

        # Track quality metrics per scene
        self.scene_metrics = {}

    async def _generate_scene(self, project_name: str, scene_name: str, scene_data: dict, cycle: int) -> bool:
        """Generate a multi-character scene"""

        project = self.projects[project_name]

        # Build the complete prompt with style
        full_prompt = f"{scene_data['prompt']}, {project['style']}"

        # Prepare proper ComfyUI workflow
        workflow = {
            "prompt": {
                "1": {
                    "inputs": {
                        "ckpt_name": "realisticVision_v51.safetensors"
                    },
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
                        "text": project['negative'],
                        "clip": ["1", 1]
                    },
                    "class_type": "CLIPTextEncode"
                },
                "4": {
                    "inputs": {
                        "seed": random.randint(1, 1000000000),
                        "steps": 30,
                        "cfg": 7,
                        "sampler_name": "dpmpp_2m",
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
                        "width": 1024,
                        "height": 1024,
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
                        "filename_prefix": f"scene_{project_name}_{scene_name}_cycle{cycle}",
                        "images": ["6", 0]
                    },
                    "class_type": "SaveImage"
                }
            }
        }

        try:
            # Queue generation
            response = requests.post(
                f"{self.comfyui_url}/prompt",
                json={"prompt": workflow["prompt"]}
            )

            if response.status_code == 200:
                result = response.json()
                if "prompt_id" in result:
                    logger.info(f"🎬 Queued {project_name} scene '{scene_name}' with {len(scene_data['characters'])} characters")

                    # Wait for generation
                    await asyncio.sleep(25)

                    # Check if file was created
                    pattern = f"scene_{project_name}_{scene_name}_cycle{cycle}_*.png"
                    files = list(self.output_dir.glob(pattern))

                    if files:
                        self.generation_count += 1
                        file_size = files[0].stat().st_size / 1024
                        logger.info(f"✅ Generated scene '{scene_name}' ({len(scene_data['characters'])} characters, {file_size:.0f}KB)")

                        # Analyze with Echo Brain every 5 generations
                        if self.generation_count % 5 == 0:
                            await self._analyze_scene_quality(project_name, scene_name, files[0])

                        return True
                    else:
                        logger.warning(f"⚠️ Scene file not found: {pattern}")

        except Exception as e:
            logger.error(f"❌ Scene generation error for {scene_name}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

        return False

    async def _analyze_scene_quality(self, project: str, scene: str, image_path: Path):
        """Use Echo Brain to analyze scene quality"""

        analysis = await self.reviewer.analyze_image(
            image_path=str(image_path),
            expected_keywords=self.projects[project]["scenes"][scene]["characters"],
            project_name=project
        )

        if analysis:
            score = analysis.get("quality_score", 0)
            suggestions = analysis.get("suggestions", [])

            # Track metrics
            scene_key = f"{project}_{scene}"
            if scene_key not in self.scene_metrics:
                self.scene_metrics[scene_key] = []
            self.scene_metrics[scene_key].append(score)

            logger.info(f"🧠 Echo Brain Analysis - {scene}: Quality {score}/10")
            if suggestions:
                logger.info(f"💡 Suggestions: {', '.join(suggestions)}")

            # Store in Echo Brain memory
            await self.reviewer.store_analysis_memory(
                project_name=project,
                character_name=scene,
                analysis=analysis
            )

    async def start_generation(self):
        """Start autonomous multi-project scene generation"""

        logger.info("🚀 Autonomous Multi-Character Scene Generator")
        logger.info("🎬 Generating complete scenes with character interactions")
        logger.info("🧠 Echo Brain quality feedback enabled")

        self.is_running = True
        cycle = 0

        while self.is_running:
            cycle += 1
            logger.info(f"\n🔄 Generation Cycle {cycle}")

            for project_name, project_config in self.projects.items():
                # Check if project reached its limit
                project_count = len(list(self.output_dir.glob(f"scene_{project_name}_*")))
                if project_count >= project_config["max_images"]:
                    logger.info(f"✓ {project_name} reached {project_config['max_images']} image limit")
                    continue

                logger.info(f"📽️ Generating scenes for {project_name}")

                # Generate each scene type
                for scene_name, scene_data in project_config["scenes"].items():
                    success = await self._generate_scene(
                        project_name,
                        scene_name,
                        scene_data,
                        cycle
                    )

                    if not success:
                        logger.warning(f"Failed to generate {scene_name}")

                    await asyncio.sleep(2)

            # Check if all projects completed
            all_complete = all(
                len(list(self.output_dir.glob(f"scene_{name}_*"))) >= config["max_images"]
                for name, config in self.projects.items()
            )

            if all_complete:
                logger.info("🎯 All projects reached their limits!")
                await self._print_final_report()
                break

            # Pause between cycles
            logger.info(f"⏸️ Completed cycle {cycle}, waiting 30 seconds...")
            await asyncio.sleep(30)

    async def _print_final_report(self):
        """Print quality metrics report"""

        logger.info("\n📊 === FINAL ECHO BRAIN QUALITY REPORT ===")

        for scene_key, scores in self.scene_metrics.items():
            if scores:
                avg_score = sum(scores) / len(scores)
                logger.info(f"🎬 {scene_key}: Avg Quality {avg_score:.1f}/10 ({len(scores)} samples)")

        logger.info(f"📈 Total scenes generated: {self.generation_count}")

async def main():
    generator = AutonomousSceneGenerator()

    try:
        await generator.start_generation()
    except KeyboardInterrupt:
        logger.info("\n⛔ Generation stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")

if __name__ == "__main__":
    asyncio.run(main())