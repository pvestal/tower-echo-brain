#!/usr/bin/env python3
"""
ComfyUI Integration for Echo Brain
Provides image and video generation capabilities
"""

import aiohttp
import asyncio
import json
import logging
import uuid
from typing import Dict, Any, Optional, List
from pathlib import Path
import base64

logger = logging.getLogger(__name__)

class ComfyUIClient:
    """Client for interacting with ComfyUI API"""

    def __init__(self, base_url: str = "http://localhost:8188"):
        self.base_url = base_url
        self.client_id = str(uuid.uuid4())
        self.ws = None

    async def initialize(self):
        """Initialize and check ComfyUI connection"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/system_stats") as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"✅ Connected to ComfyUI - VRAM: {data.get('devices', [{}])[0].get('vram_used_gb', 0):.1f}GB / {data.get('devices', [{}])[0].get('vram_total_gb', 0):.1f}GB")
                        return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to ComfyUI: {e}")
            return False

    async def generate_image(self, prompt: str, negative_prompt: str = "",
                           width: int = 512, height: int = 512,
                           steps: int = 20, cfg: float = 7.0,
                           seed: int = -1) -> Optional[Dict[str, Any]]:
        """Generate an image using ComfyUI"""
        try:
            # Build workflow
            workflow = self._build_image_workflow(
                prompt, negative_prompt, width, height, steps, cfg, seed
            )

            # Queue the prompt
            prompt_id = await self._queue_prompt(workflow)

            if not prompt_id:
                return None

            # Wait for completion
            result = await self._wait_for_completion(prompt_id)

            if result and "images" in result:
                # Get the image path
                image_info = result["images"][0]
                image_path = f"/mnt/1TB-storage/ComfyUI/output/{image_info['filename']}"

                return {
                    "success": True,
                    "prompt_id": prompt_id,
                    "image_path": image_path,
                    "filename": image_info["filename"],
                    "metadata": {
                        "prompt": prompt,
                        "width": width,
                        "height": height,
                        "steps": steps,
                        "cfg": cfg,
                        "seed": seed
                    }
                }

        except Exception as e:
            logger.error(f"Image generation error: {e}")
            return None

    async def generate_video(self, prompt: str, negative_prompt: str = "",
                           width: int = 768, height: int = 768,
                           duration: int = 5, fps: int = 24) -> Optional[Dict[str, Any]]:
        """Generate a video using ComfyUI AnimateDiff"""
        try:
            # Build video workflow
            workflow = self._build_video_workflow(
                prompt, negative_prompt, width, height, duration, fps
            )

            # Queue the prompt
            prompt_id = await self._queue_prompt(workflow)

            if not prompt_id:
                return None

            # Wait for completion (longer timeout for video)
            result = await self._wait_for_completion(prompt_id, timeout=600)

            if result and "gifs" in result:
                # Get the video path
                video_info = result["gifs"][0]
                video_path = f"/mnt/1TB-storage/ComfyUI/output/{video_info['filename']}"

                return {
                    "success": True,
                    "prompt_id": prompt_id,
                    "video_path": video_path,
                    "filename": video_info["filename"],
                    "metadata": {
                        "prompt": prompt,
                        "width": width,
                        "height": height,
                        "duration": duration,
                        "fps": fps
                    }
                }

        except Exception as e:
            logger.error(f"Video generation error: {e}")
            return None

    def _build_image_workflow(self, prompt: str, negative_prompt: str,
                             width: int, height: int, steps: int,
                             cfg: float, seed: int) -> Dict[str, Any]:
        """Build a simple image generation workflow"""
        return {
            "3": {
                "inputs": {
                    "seed": seed if seed != -1 else random.randint(0, 2**32),
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": "dpmpp_2m",
                    "scheduler": "karras",
                    "denoise": 1,
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0]
                },
                "class_type": "KSampler"
            },
            "4": {
                "inputs": {
                    "ckpt_name": "sd_xl_base_1.0.safetensors"
                },
                "class_type": "CheckpointLoaderSimple"
            },
            "5": {
                "inputs": {
                    "width": width,
                    "height": height,
                    "batch_size": 1
                },
                "class_type": "EmptyLatentImage"
            },
            "6": {
                "inputs": {
                    "text": prompt,
                    "clip": ["4", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "7": {
                "inputs": {
                    "text": negative_prompt,
                    "clip": ["4", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "8": {
                "inputs": {
                    "samples": ["3", 0],
                    "vae": ["4", 2]
                },
                "class_type": "VAEDecode"
            },
            "9": {
                "inputs": {
                    "filename_prefix": "echo_image",
                    "images": ["8", 0]
                },
                "class_type": "SaveImage"
            }
        }

    def _build_video_workflow(self, prompt: str, negative_prompt: str,
                             width: int, height: int, duration: int, fps: int) -> Dict[str, Any]:
        """Build AnimateDiff video generation workflow"""
        frame_count = duration * fps

        # This would be a complex AnimateDiff workflow
        # Simplified for example
        return {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "frame_count": frame_count,
            "fps": fps
        }

    async def _queue_prompt(self, workflow: Dict[str, Any]) -> Optional[str]:
        """Queue a workflow prompt"""
        try:
            payload = {
                "prompt": workflow,
                "client_id": self.client_id
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/prompt",
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("prompt_id")
        except Exception as e:
            logger.error(f"Failed to queue prompt: {e}")
            return None

    async def _wait_for_completion(self, prompt_id: str, timeout: int = 120) -> Optional[Dict[str, Any]]:
        """Wait for prompt completion"""
        try:
            end_time = asyncio.get_event_loop().time() + timeout

            while asyncio.get_event_loop().time() < end_time:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.base_url}/history/{prompt_id}") as response:
                        if response.status == 200:
                            data = await response.json()
                            if prompt_id in data:
                                history = data[prompt_id]
                                if "outputs" in history:
                                    return history["outputs"]

                await asyncio.sleep(1)

            logger.warning(f"Prompt {prompt_id} timed out after {timeout} seconds")
            return None

        except Exception as e:
            logger.error(f"Error waiting for completion: {e}")
            return None

    async def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/queue") as response:
                    if response.status == 200:
                        return await response.json()
        except Exception as e:
            logger.error(f"Failed to get queue status: {e}")
            return {}

    async def interrupt_generation(self) -> bool:
        """Interrupt current generation"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.base_url}/interrupt") as response:
                    return response.status == 200
        except Exception as e:
            logger.error(f"Failed to interrupt: {e}")
            return False

# Singleton instance
import random
comfyui_client = ComfyUIClient()

async def get_comfyui_client() -> ComfyUIClient:
    """Get initialized ComfyUI client"""
    await comfyui_client.initialize()
    return comfyui_client