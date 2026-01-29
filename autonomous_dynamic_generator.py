#!/usr/bin/env python3
"""
Autonomous Dynamic Generator
Generates images for ALL projects loaded from database/config
No hardcoding - works with any new projects automatically
"""

import asyncio
import requests
import json
import random
import time
import logging
from pathlib import Path
from typing import Dict, List, Any
from dynamic_project_loader import DynamicProjectLoader
from character_distinctions import get_distinguishing_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutonomousDynamicGenerator:
    """Generate images for all projects dynamically"""

    def __init__(self):
        self.comfyui_url = "http://localhost:8188"
        self.output_dir = Path("/mnt/1TB-storage/ComfyUI/output")
        self.loader = DynamicProjectLoader()
        self.generation_count = {}
        self.images_per_character = 10  # Start with 10 for testing

    async def initialize(self):
        """Load all projects and characters"""
        await self.loader.initialize()
        logger.info(f"Loaded {len(self.loader.projects)} projects")
        return self

    def get_model_settings(self, model_file: str) -> Dict:
        """Get optimal settings for a model from YAML config"""
        import yaml
        from pathlib import Path

        models_config_path = Path("/opt/tower-echo-brain/models_config.yaml")
        if models_config_path.exists():
            with open(models_config_path) as f:
                models = yaml.safe_load(f)['models']
                for model in models.values():
                    if model['file'] == model_file:
                        return model['settings']

        # Default settings
        return {
            'cfg': 7.0,
            'sampler': 'euler',
            'scheduler': 'normal',
            'steps': 25
        }

    async def generate_character_image(self, project: Dict, character: Dict, character_name: str) -> bool:
        """Generate a single image for a character"""

        # Get model settings
        model_file = project['model']
        settings = self.get_model_settings(model_file)

        # Build prompt based on project style
        style = project['style'].lower()
        if 'photorealistic' in style:
            base_prompt = f"RAW photo, photorealistic, detailed, sharp focus, 85mm lens"
            negative = "anime, cartoon, blurry, soft focus, smooth skin"
        elif 'anime' in style:
            base_prompt = f"anime style, vibrant colors, detailed, high quality"
            negative = "realistic, photograph, blurry"
        else:
            base_prompt = f"high quality, detailed"
            negative = "blurry, low quality"

        # Add ethnicity based on project/character context
        ethnicity = ""
        project_name_lower = project['name'].lower()
        if 'tokyo' in project_name_lower or 'japanese' in project_name_lower:
            ethnicity = "Japanese, Asian features, "
            negative += ", caucasian, european, western features"
        elif 'mario' in project_name_lower:
            if 'mario' in character_name.lower() or 'luigi' in character_name.lower():
                ethnicity = "Italian, "

        # Add character details
        char_prompt = f"{ethnicity}{character.get('description', character_name)}, {base_prompt}"

        # Add gender-specific negatives
        if character.get('gender') == 'male':
            negative += ", woman, female, breasts, feminine"
        elif character.get('gender') == 'female':
            negative += ", man, male, masculine, beard"

        # Add distinguishing features if available
        distinct = get_distinguishing_features(character_name)
        if distinct:
            char_prompt += f", {distinct}"

        workflow = {
            "prompt": {
                "1": {
                    "inputs": {"ckpt_name": model_file},
                    "class_type": "CheckpointLoaderSimple"
                },
                "2": {
                    "inputs": {
                        "text": char_prompt,
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
                        "seed": random.randint(1, 1000000000),
                        "steps": settings['steps'],
                        "cfg": settings['cfg'],
                        "sampler_name": settings['sampler'],
                        "scheduler": settings['scheduler'],
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
                        "width": project['resolution']['width'],
                        "height": project['resolution']['height'],
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
                        "filename_prefix": f"AUTO_{project['name'].replace(' ', '_')}_{character_name}",
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
                char_key = f"{project['id']}_{character_name}"
                self.generation_count[char_key] = self.generation_count.get(char_key, 0) + 1

                logger.info(f"✅ [{project['name']}] {character_name} - Image {self.generation_count[char_key]}/{self.images_per_character}")
                return True
            else:
                logger.error(f"❌ Failed: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"❌ Error: {e}")
            return False

    async def run_project(self, project: Dict):
        """Run generation for a single project"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting project: {project['name']}")
        logger.info(f"Style: {project['style']} | Model: {project['model']}")
        logger.info(f"Characters: {len(project['characters'])}")

        for char_name, char_data in project['characters'].items():
            char_key = f"{project['id']}_{char_name}"

            for i in range(self.images_per_character):
                success = await self.generate_character_image(project, char_data, char_name)
                if success:
                    await asyncio.sleep(10)  # Wait between generations
                else:
                    logger.warning(f"Skipping remaining images for {char_name}")
                    break

    async def run_all_projects(self):
        """Run generation for all loaded projects"""
        projects = self.loader.get_all_projects()

        if not projects:
            logger.error("No projects loaded!")
            return

        logger.info(f"\n{'='*60}")
        logger.info(f"AUTONOMOUS DYNAMIC GENERATION")
        logger.info(f"Projects: {len(projects)}")
        logger.info(f"Images per character: {self.images_per_character}")
        logger.info(f"{'='*60}")

        for project in projects:
            await self.run_project(project)
            logger.info(f"✅ Completed project: {project['name']}")

        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("GENERATION COMPLETE")
        logger.info(f"Total images generated: {sum(self.generation_count.values())}")
        for key, count in self.generation_count.items():
            logger.info(f"  {key}: {count} images")

    async def run_specific_project(self, project_name: str):
        """Run generation for a specific project"""
        projects = self.loader.get_all_projects()

        for project in projects:
            if project['name'].lower() == project_name.lower():
                await self.run_project(project)
                return

        logger.error(f"Project '{project_name}' not found")

# Main execution
async def main():
    generator = AutonomousDynamicGenerator()
    await generator.initialize()

    # Generate for all projects
    await generator.run_all_projects()

if __name__ == "__main__":
    asyncio.run(main())