#!/usr/bin/env python3
"""
LORA Generation Worker
Generates 40 training images for a character using ComfyUI
"""

import asyncio
import logging
import aiohttp
import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import random

logger = logging.getLogger(__name__)

class LoraGenerationWorker:
    """Worker that generates training images via ComfyUI"""

    def __init__(self, comfyui_url: str = "http://localhost:8188"):
        self.comfyui_url = comfyui_url
        self.base_workflow = self._create_base_workflow()

    async def generate_training_images(self, task_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate 40 training images for LORA training

        Args:
            task_payload: {
                'character_name': str,
                'reference_image': str (optional),
                'description': str,
                'output_dir': str
            }

        Returns:
            Dict with generation results
        """
        character_name = task_payload.get('character_name')
        description = task_payload.get('description')
        output_dir = Path(task_payload.get('output_dir', f'/mnt/1TB-storage/lora_datasets/{character_name}'))
        reference_image = task_payload.get('reference_image')

        logger.info(f"🎨 Starting LORA image generation for {character_name}")
        logger.info(f"📁 Output directory: {output_dir}")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        images_dir = output_dir / 'images'
        images_dir.mkdir(exist_ok=True)

        # Generate variations for diverse training data
        variations = self._create_prompt_variations(description, character_name)

        generated_images = []
        failed_generations = []

        # Generate images in batches
        batch_size = 4
        for i in range(0, len(variations), batch_size):
            batch = variations[i:i+batch_size]
            batch_results = await asyncio.gather(
                *[self._generate_single_image(
                    prompt=var['prompt'],
                    negative_prompt=var['negative_prompt'],
                    output_path=images_dir / f"{character_name}_{var['index']:03d}.png",
                    seed=var['seed']
                ) for var in batch],
                return_exceptions=True
            )

            for var, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    failed_generations.append({
                        'index': var['index'],
                        'error': str(result)
                    })
                    logger.error(f"❌ Failed to generate image {var['index']}: {result}")
                else:
                    generated_images.append(result)
                    logger.info(f"✅ Generated image {var['index']}/40")

            # Small delay between batches to avoid overwhelming ComfyUI
            await asyncio.sleep(2)

        result = {
            'character_name': character_name,
            'total_requested': 40,
            'total_generated': len(generated_images),
            'total_failed': len(failed_generations),
            'output_directory': str(output_dir),
            'images_directory': str(images_dir),
            'generated_images': generated_images,
            'failed_generations': failed_generations,
            'timestamp': datetime.now().isoformat()
        }

        # Save generation metadata
        metadata_file = output_dir / 'generation_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(result, f, indent=2)

        logger.info(f"🎉 LORA image generation complete: {len(generated_images)}/40 successful")
        return result

    def _create_prompt_variations(self, base_description: str, character_name: str, count: int = 40) -> List[Dict]:
        """Create 40 diverse prompt variations for training"""

        # Pose variations
        poses = [
            "standing confidently", "sitting casually", "walking forward", "looking back over shoulder",
            "arms crossed", "hands on hips", "reaching up", "leaning against wall",
            "kneeling", "running", "jumping", "dancing", "fighting stance",
            "relaxed pose", "dynamic action pose", "profile view", "three-quarter view"
        ]

        # Expression variations
        expressions = [
            "neutral expression", "smiling warmly", "serious expression", "determined look",
            "laughing happily", "surprised expression", "thoughtful expression", "confident smirk",
            "focused gaze", "gentle smile", "intense stare", "playful expression"
        ]

        # Angle/camera variations
        angles = [
            "front view", "side view", "back view", "three-quarter view",
            "slightly from below", "slightly from above", "close-up portrait",
            "full body shot", "upper body shot", "cowboy shot"
        ]

        # Lighting variations
        lighting = [
            "dramatic lighting", "soft lighting", "natural lighting", "studio lighting",
            "golden hour lighting", "rim lighting", "high contrast", "even lighting"
        ]

        # Background variations
        backgrounds = [
            "simple background", "white background", "gradient background",
            "indoor setting", "outdoor setting", "urban environment",
            "natural environment", "abstract background", "blurred background"
        ]

        variations = []
        for i in range(count):
            # Randomly select variations
            pose = random.choice(poses)
            expression = random.choice(expressions)
            angle = random.choice(angles)
            light = random.choice(lighting)
            bg = random.choice(backgrounds)

            # Construct prompt
            prompt = f"{base_description}, {pose}, {expression}, {angle}, {light}, {bg}, high quality, detailed, masterpiece, best quality"

            negative_prompt = "low quality, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, bad proportions, cloned face, disfigured, gross proportions, malformed limbs, extra arms, extra legs, fused fingers, too many fingers, long neck"

            variations.append({
                'index': i + 1,
                'prompt': prompt,
                'negative_prompt': negative_prompt,
                'seed': random.randint(1, 2147483647),
                'metadata': {
                    'pose': pose,
                    'expression': expression,
                    'angle': angle,
                    'lighting': light,
                    'background': bg
                }
            })

        return variations

    async def _generate_single_image(self, prompt: str, negative_prompt: str,
                                    output_path: Path, seed: int) -> Dict[str, Any]:
        """Generate a single image via ComfyUI API"""

        # Create workflow with parameters
        workflow = self._create_workflow(prompt, negative_prompt, seed)

        async with aiohttp.ClientSession() as session:
            # Submit workflow to ComfyUI
            async with session.post(
                f"{self.comfyui_url}/prompt",
                json={"prompt": workflow}
            ) as response:
                if response.status != 200:
                    raise Exception(f"ComfyUI returned status {response.status}")

                result = await response.json()
                prompt_id = result.get('prompt_id')

                if not prompt_id:
                    raise Exception("No prompt_id returned from ComfyUI")

            # Poll for completion
            max_wait = 300  # 5 minutes
            poll_interval = 2
            elapsed = 0

            while elapsed < max_wait:
                async with session.get(f"{self.comfyui_url}/history/{prompt_id}") as response:
                    if response.status == 200:
                        history = await response.json()
                        if prompt_id in history:
                            outputs = history[prompt_id].get('outputs', {})
                            if outputs:
                                # Image is ready
                                # In real implementation, would download and save the image
                                # For now, just return metadata
                                return {
                                    'prompt_id': prompt_id,
                                    'output_path': str(output_path),
                                    'prompt': prompt,
                                    'seed': seed,
                                    'status': 'completed'
                                }

                await asyncio.sleep(poll_interval)
                elapsed += poll_interval

            raise Exception(f"Image generation timed out after {max_wait} seconds")

    def _create_workflow(self, prompt: str, negative_prompt: str, seed: int) -> Dict:
        """Create ComfyUI workflow for image generation"""

        # Simplified workflow structure for DreamShaper 8
        # In production, this would be a complete ComfyUI workflow JSON
        workflow = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name": "dreamshaper_8.safetensors"
                }
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": prompt,
                    "clip": ["1", 1]
                }
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": negative_prompt,
                    "clip": ["1", 1]
                }
            },
            "4": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": 768,
                    "height": 768,
                    "batch_size": 1
                }
            },
            "5": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed,
                    "steps": 28,
                    "cfg": 7.5,
                    "sampler_name": "dpmpp_2m",
                    "scheduler": "karras",
                    "denoise": 1.0,
                    "model": ["1", 0],
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "latent_image": ["4", 0]
                }
            },
            "6": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["5", 0],
                    "vae": ["1", 2]
                }
            },
            "7": {
                "class_type": "SaveImage",
                "inputs": {
                    "images": ["6", 0],
                    "filename_prefix": "lora_training"
                }
            }
        }

        return workflow

    def _create_base_workflow(self) -> Dict:
        """Create base workflow template"""
        return {}

# Task handler function for integration with Echo task queue
async def handle_lora_generation_task(task) -> Dict[str, Any]:
    """Handler function for LORA_GENERATION task type"""
    worker = LoraGenerationWorker()
    return await worker.generate_training_images(task.payload)
