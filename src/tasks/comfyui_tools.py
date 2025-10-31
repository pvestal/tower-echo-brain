"""
ComfyUI tools for Echo Brain autonomous image generation
"""

import asyncio
import aiohttp
import json
import logging
import uuid
import time
from typing import Dict, Any, Optional, List
from pathlib import Path
from src.tasks.character_reference_system import CharacterReferenceSystem
import sys
sys.path.append('/opt/tower-anime-production')
from character_system import get_character_prompt

logger = logging.getLogger(__name__)

class ComfyUITools:
    """ComfyUI integration tools for Echo Brain"""

    def __init__(self, comfyui_url: str = "http://192.168.50.135:8188"):
        self.comfyui_url = comfyui_url
        self.output_dir = Path("/mnt/1TB-storage/ComfyUI/output")
        self.character_refs = CharacterReferenceSystem()

    async def generate_image(self, prompt: str, negative_prompt: str = "",
                           width: int = 1024, height: int = 1024,
                           steps: int = 25, cfg: float = 8.0,
                           model: str = "sd_xl_base_1.0.safetensors",
                           prefix: str = "echo_generated") -> Dict[str, Any]:
        """Generate image using ComfyUI"""

        workflow = self._build_image_workflow(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            cfg=cfg,
            model=model,
            prefix=prefix
        )

        return await self._execute_workflow(workflow)

    async def generate_anime_character(self, character_name: str,
                                     description: str = None,
                                     style: str = None) -> Dict[str, Any]:
        """Generate anime character image using project references"""

        # Get character data from anime production system
        try:
            character_data = get_character_prompt(character_name, description or 'standing portrait')

            if character_data.get('character_found'):
                prompt = character_data['prompt']
                negative_prompt = character_data['negative_prompt']
                logger.info(f"✅ Using anime production character system for {character_name}")
                logger.info(f"Project reference: {character_data.get('used_project_reference')}")
                used_project_reference = character_data.get('used_project_reference', False)
            else:
                # Fallback to provided description
                prompt = f"{character_name}, {description or 'anime character'}, photorealistic anime, hyperrealistic, high detail, professional lighting"
                negative_prompt = "blurry, low quality, bad anatomy, text, watermark, cartoon, chibi, simple anime, flat colors, 2D anime style"
                logger.warning(f"❌ No character found in anime production system for {character_name}, using fallback")
                used_project_reference = False
        except Exception as e:
            logger.error(f"Error accessing anime production character system: {e}")
            prompt = f"{character_name}, {description or 'anime character'}, photorealistic anime, hyperrealistic, high detail"
            negative_prompt = "blurry, low quality, bad anatomy, text, watermark"
            used_project_reference = False

        result = await self.generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            prefix=f"echo_{character_name.lower().replace(' ', '_')}"
        )

        # Add reference information to result
        result['reference_info'] = {
            'reference_image': None,
            'style_guide': {},
            'used_project_reference': used_project_reference
        }

        return result

    def _build_image_workflow(self, prompt: str, negative_prompt: str,
                            width: int, height: int, steps: int, cfg: float,
                            model: str, prefix: str) -> Dict[str, Any]:
        """Build ComfyUI workflow for image generation"""

        seed = int(time.time() * 1000) % 2147483647

        return {
            "1": {
                "inputs": {
                    "ckpt_name": model
                },
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
                    "text": negative_prompt,
                    "clip": ["1", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "4": {
                "inputs": {
                    "width": width,
                    "height": height,
                    "batch_size": 1
                },
                "class_type": "EmptyLatentImage"
            },
            "5": {
                "inputs": {
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": "dpmpp_2m",
                    "scheduler": "karras",
                    "denoise": 1.0,
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
                    "filename_prefix": prefix,
                    "images": ["6", 0]
                },
                "class_type": "SaveImage"
            }
        }

    async def _execute_workflow(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Execute ComfyUI workflow and wait for completion"""

        try:
            async with aiohttp.ClientSession() as session:
                # Submit workflow
                async with session.post(
                    f"{self.comfyui_url}/prompt",
                    json={"prompt": workflow}
                ) as resp:
                    if resp.status != 200:
                        return {
                            "success": False,
                            "error": f"ComfyUI returned status {resp.status}"
                        }

                    result = await resp.json()
                    prompt_id = result.get("prompt_id")

                    if not prompt_id:
                        return {
                            "success": False,
                            "error": "No prompt_id returned from ComfyUI"
                        }

                # Wait for completion
                return await self._wait_for_completion(prompt_id, session)

        except Exception as e:
            logger.error(f"ComfyUI workflow execution failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _wait_for_completion(self, prompt_id: str, session: aiohttp.ClientSession,
                                 timeout: int = 300) -> Dict[str, Any]:
        """Wait for ComfyUI generation to complete"""

        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                async with session.get(f"{self.comfyui_url}/history") as resp:
                    if resp.status == 200:
                        history = await resp.json()

                        if prompt_id in history:
                            # Generation completed
                            output_files = self._extract_output_files(history[prompt_id])

                            return {
                                "success": True,
                                "prompt_id": prompt_id,
                                "output_files": output_files,
                                "generation_time": time.time() - start_time
                            }

                await asyncio.sleep(2)  # Check every 2 seconds

            except Exception as e:
                logger.error(f"Error checking completion: {e}")
                await asyncio.sleep(5)

        return {
            "success": False,
            "error": f"Generation timed out after {timeout} seconds",
            "prompt_id": prompt_id
        }

    def _extract_output_files(self, history_entry: Dict) -> List[str]:
        """Extract output file paths from ComfyUI history"""

        output_files = []

        try:
            outputs = history_entry.get("outputs", {})

            for node_id, node_output in outputs.items():
                if "images" in node_output:
                    for image_info in node_output["images"]:
                        filename = image_info.get("filename")
                        if filename:
                            # Construct full path
                            full_path = self.output_dir / filename
                            if full_path.exists():
                                output_files.append(str(full_path))

        except Exception as e:
            logger.error(f"Error extracting output files: {e}")

        return output_files

    async def health_check(self) -> Dict[str, Any]:
        """Check ComfyUI service health"""

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.comfyui_url}/system_stats",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        stats = await resp.json()
                        return {
                            "success": True,
                            "status": "healthy",
                            "system_stats": stats
                        }
                    else:
                        return {
                            "success": False,
                            "status": "unhealthy",
                            "error": f"Status {resp.status}"
                        }

        except Exception as e:
            return {
                "success": False,
                "status": "unreachable",
                "error": str(e)
            }

    async def list_models(self) -> List[str]:
        """List available models in ComfyUI"""

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.comfyui_url}/object_info") as resp:
                    if resp.status == 200:
                        object_info = await resp.json()
                        checkpoint_loader = object_info.get("CheckpointLoaderSimple", {})
                        input_info = checkpoint_loader.get("input", {})
                        ckpt_name_info = input_info.get("ckpt_name", [])

                        if len(ckpt_name_info) > 1:
                            return ckpt_name_info[1]  # The list of available models

        except Exception as e:
            logger.error(f"Error listing models: {e}")

        return []