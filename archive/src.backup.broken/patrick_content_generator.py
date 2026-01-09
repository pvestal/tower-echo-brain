#!/usr/bin/env python3
"""
Patrick's Personal Content Generator
Direct ComfyUI integration for YOUR anime projects - no generic BS
"""

import asyncio
import aiohttp
import json
import uuid
import time
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class PatrickContentGenerator:
    """Generate content specifically for Patrick's anime projects"""

    def __init__(self):
        self.comfyui_url = "http://localhost:8188"
        self.output_dir = Path("/mnt/1TB-storage/ComfyUI/output")

        # Patrick's ACTUAL projects and characters
        self.projects = {
            "tokyo_debt_crisis": {
                "style": "modern anime, romantic comedy, vibrant colors",
                "characters": {
                    "riku": "male protagonist, 22 years old, brown hair, average build, casual clothes, friendly smile",
                    "yuki": "yakuza daughter, long black hair, intense eyes, traditional japanese beauty, confident",
                    "sakura": "childhood friend, pink hair, cheerful, school uniform, energetic personality"
                }
            },
            "goblin_slayer_neon": {
                "style": "cyberpunk, dark atmosphere, neon lights, tactical gear",
                "characters": {
                    "kai_nakamura": "tech specialist, spiky black hair, brown eyes, cyberpunk aesthetic, tactical gear",
                    "ryuu": "sniper, silver hair, sharp eyes, long coat, mysterious aura",
                    "hiroshi": "infiltrator, short dark hair, stealth suit, agile build"
                }
            }
        }

    async def generate_character_image(self, project: str, character: str, seed: Optional[int] = None) -> Dict:
        """Generate a SINGLE IMAGE for a character - NOT a video!"""

        if project not in self.projects:
            return {"error": f"Unknown project: {project}"}

        if character not in self.projects[project]["characters"]:
            return {"error": f"Unknown character: {character} in {project}"}

        # Get character details
        char_desc = self.projects[project]["characters"][character]
        project_style = self.projects[project]["style"]

        # Build PROPER image workflow (NOT video!)
        workflow = self._build_image_workflow(character, char_desc, project_style, seed)

        # Submit to ComfyUI
        prompt_id = await self._submit_workflow(workflow)

        if prompt_id:
            logger.info(f"✅ Generating IMAGE for {character} - ID: {prompt_id}")
            return {
                "status": "success",
                "prompt_id": prompt_id,
                "character": character,
                "project": project,
                "type": "IMAGE"  # NOT VIDEO!
            }
        else:
            return {"status": "error", "message": "Failed to submit workflow"}

    def _build_image_workflow(self, name: str, description: str, style: str, seed: Optional[int] = None) -> Dict:
        """Build PROPER IMAGE workflow - single frame, not 72 frames!"""

        if seed is None:
            seed = int(time.time() * 1000) % 1000000

        # Complete prompt with Patrick's style preferences
        full_prompt = f"{name}, {description}, {style}, masterpiece, best quality, detailed"

        return {
            "1": {  # Checkpoint
                "inputs": {"ckpt_name": "counterfeit_v3.safetensors"},  # Use actual available model
                "class_type": "CheckpointLoaderSimple"
            },
            "2": {  # Positive prompt
                "inputs": {
                    "text": full_prompt,
                    "clip": ["1", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "3": {  # Negative prompt
                "inputs": {
                    "text": "worst quality, low quality, blurry, ugly, distorted",
                    "clip": ["1", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "4": {  # Empty latent - SINGLE IMAGE!
                "inputs": {
                    "width": 1024,
                    "height": 1024,
                    "batch_size": 1  # NOT 72! JUST 1 IMAGE!
                },
                "class_type": "EmptyLatentImage"
            },
            "5": {  # KSampler
                "inputs": {
                    "seed": seed,
                    "steps": 20,
                    "cfg": 7.0,
                    "sampler_name": "euler_ancestral",
                    "scheduler": "normal",
                    "denoise": 1.0,
                    "model": ["1", 0],
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "latent_image": ["4", 0]
                },
                "class_type": "KSampler"
            },
            "6": {  # VAE Decode
                "inputs": {
                    "samples": ["5", 0],
                    "vae": ["1", 2]
                },
                "class_type": "VAEDecode"
            },
            "7": {  # Save Image
                "inputs": {
                    "images": ["6", 0],
                    "filename_prefix": f"patrick_{name}_{seed}"
                },
                "class_type": "SaveImage"
            }
        }

    async def _submit_workflow(self, workflow: Dict) -> Optional[str]:
        """Submit workflow to ComfyUI"""
        try:
            client_id = str(uuid.uuid4())
            payload = {
                "prompt": workflow,
                "client_id": client_id
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.comfyui_url}/prompt",
                    json=payload,
                    timeout=10
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("prompt_id")
                    else:
                        logger.error(f"ComfyUI submission failed: {response.status}")
                        return None
        except Exception as e:
            logger.error(f"Failed to submit to ComfyUI: {e}")
            return None

    async def generate_all_missing_characters(self) -> Dict:
        """Generate all characters for Patrick's projects"""
        results = {}

        for project, project_data in self.projects.items():
            for char_name in project_data["characters"]:
                # Skip Kai - already done
                if char_name == "kai_nakamura":
                    continue

                result = await self.generate_character_image(project, char_name)
                results[f"{project}_{char_name}"] = result

                # Small delay between submissions
                await asyncio.sleep(2)

        return results

# Singleton instance
patrick_generator = PatrickContentGenerator()