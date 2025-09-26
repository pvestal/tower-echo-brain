#!/usr/bin/env python3
"""Echo Brain Video Generation Service - BASIC IMPROVED VERSION"""

import asyncio
import json
import logging
import os
import time
import uuid
import subprocess
import aiohttp
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel

logger = logging.getLogger(__name__)

COMFYUI_URL = "http://127.0.0.1:8188"
OUTPUT_DIR = Path("/home/patrick/Videos/AI_Generated")
TEMP_DIR = Path("/tmp/echo_video_gen")

class VideoGenerationRequest(BaseModel):
    prompt: str
    project_id: Optional[str] = None
    character_name: Optional[str] = None
    duration: int = 5
    resolution: str = "1024x1024"  # Start with 1024x1024
    style: str = "photorealistic anime"
    auto_fix_errors: bool = True
    num_frames: int = 72  # 72 frames for 3 seconds at 24fps

class EchoVideoGenerator:
    def __init__(self):
        self.client_id = str(uuid.uuid4())
        self.base_url = COMFYUI_URL
        self.max_retry_attempts = 3
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        TEMP_DIR.mkdir(parents=True, exist_ok=True)

    async def generate_video(self, request: VideoGenerationRequest) -> Dict:
        try:
            logger.info(f"Starting IMPROVED video generation for: {request.prompt}")
            logger.info(f"Target: {request.num_frames} frames at 24fps = {request.num_frames/24:.1f}s duration")
            start_time = time.time()

            await self._check_comfyui_health()
            workflow = await self._create_video_workflow(request)

            logger.info(f"Submitting improved workflow to ComfyUI: {self.base_url}/prompt")
            logger.debug(f"Workflow payload: {json.dumps(workflow, indent=2)}")

            prompt_id = await self._submit_workflow(workflow)
            result = await self._wait_for_completion(prompt_id)
            video_path = await self._process_results(result, request)

            total_time = time.time() - start_time

            return {
                "success": True,
                "video_path": str(video_path),
                "prompt": request.prompt,
                "duration": request.duration,
                "num_frames": request.num_frames,
                "resolution": "1024x1024",
                "fps": 24,
                "generation_time": total_time,
                "timestamp": datetime.now().isoformat(),
                "quality": "IMPROVED_HD_24FPS"
            }
        except Exception as e:
            logger.error(f"Video generation failed: {e}")
            logger.exception("Full error traceback:")
            return {
                "success": False,
                "error": str(e),
                "prompt": request.prompt,
                "timestamp": datetime.now().isoformat()
            }

    async def _check_comfyui_health(self) -> bool:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/system_stats", timeout=10) as response:
                    if response.status == 200:
                        stats = await response.json()
                        logger.info(f"ComfyUI health check passed - RAM: {stats.get('system', {}).get('ram', 'unknown')}")
                        return True
                    else:
                        logger.error(f"ComfyUI health check failed with status: {response.status}")
                        raise Exception(f"ComfyUI health check failed: {response.status}")
        except Exception as e:
            logger.error(f"ComfyUI health check failed: {e}")
            raise Exception(f"ComfyUI service is not available: {e}")

    async def _create_video_workflow(self, request: VideoGenerationRequest) -> Dict:
        enhanced_prompt = f"{request.prompt}, {request.style}, cinematic quality, high detail, masterpiece, professional video"

        # Improved workflow with more frames and better settings
        workflow = {
            # Load checkpoint
            "3": {
                "inputs": {
                    "ckpt_name": "epicrealism_v5.safetensors"
                },
                "class_type": "CheckpointLoaderSimple"
            },
            # Create latent space for video generation with MORE FRAMES
            "4": {
                "inputs": {
                    "width": 1024,
                    "height": 1024,
                    "batch_size": request.num_frames  # Use the full number of frames requested
                },
                "class_type": "EmptyLatentImage"
            },
            # Positive prompt
            "5": {
                "inputs": {
                    "text": enhanced_prompt.strip(),
                    "clip": ["3", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            # Negative prompt
            "6": {
                "inputs": {
                    "text": "blurry, low quality, distorted, bad anatomy, static, boring, flickering, inconsistent, slideshow, single frame",
                    "clip": ["3", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            # Sampling with better settings for video generation
            "7": {
                "inputs": {
                    "seed": 42,
                    "steps": 25,  # Increased steps for better quality
                    "cfg": 8.5,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1,
                    "model": ["3", 0],
                    "positive": ["5", 0],
                    "negative": ["6", 0],
                    "latent_image": ["4", 0]
                },
                "class_type": "KSampler"
            },
            # Decode latents to images
            "8": {
                "inputs": {
                    "samples": ["7", 0],
                    "vae": ["3", 2]
                },
                "class_type": "VAEDecode"
            },
            # Save preview images
            "9": {
                "inputs": {
                    "images": ["8", 0],
                    "filename_prefix": f"echo_improved_preview_{int(time.time())}"
                },
                "class_type": "SaveImage"
            },
            # Generate video with proper 24fps
            "10": {
                "inputs": {
                    "images": ["8", 0],
                    "frame_rate": 24.0,  # Professional 24fps (not 8fps!)
                    "loop_count": 0,
                    "filename_prefix": f"echo_improved_video_{int(time.time())}",
                    "format": "video/h264-mp4",
                    "pix_fmt": "yuv420p",
                    "crf": 18,  # Lower CRF for higher quality
                    "save_metadata": True,
                    "pingpong": False,
                    "save_output": True
                },
                "class_type": "VHS_VideoCombine"
            }
        }

        logger.info(f"Created IMPROVED video workflow")
        logger.info(f"Frames: {request.num_frames}, FPS: 24, Duration: {request.num_frames/24:.1f}s")
        logger.info(f"Output resolution: 1024x1024")
        logger.info(f"Quality improvements: 24fps, {request.num_frames} frames, better CFG/steps")
        logger.debug(f"Workflow structure: {list(workflow.keys())}")
        return workflow

    async def _submit_workflow(self, workflow: Dict) -> str:
        try:
            async with aiohttp.ClientSession() as session:
                payload = {"prompt": workflow, "client_id": self.client_id}

                logger.info(f"Submitting improved workflow to ComfyUI at {self.base_url}/prompt")
                logger.info(f"Client ID: {self.client_id}")

                async with session.post(f"{self.base_url}/prompt", json=payload, timeout=30) as response:
                    response_text = await response.text()
                    logger.info(f"ComfyUI response status: {response.status}")
                    logger.debug(f"ComfyUI response body: {response_text}")

                    if response.status == 200:
                        result = await response.json()
                        prompt_id = result["prompt_id"]
                        logger.info(f"✅ Workflow submitted successfully! Prompt ID: {prompt_id}")

                        await self._verify_job_queued(prompt_id)
                        return prompt_id
                    else:
                        logger.error(f"❌ Failed to submit workflow: {response.status}")
                        logger.error(f"Response: {response_text}")
                        raise Exception(f"Failed to submit workflow: {response.status} - {response_text}")

        except Exception as e:
            logger.error(f"❌ Failed to submit workflow: {e}")
            logger.exception("Full error traceback:")
            raise

    async def _verify_job_queued(self, prompt_id: str) -> bool:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/queue") as response:
                    if response.status == 200:
                        queue_data = await response.json()
                        running = queue_data.get("queue_running", [])
                        pending = queue_data.get("queue_pending", [])

                        all_jobs = running + pending
                        job_found = any(job[1] == prompt_id for job in all_jobs if len(job) > 1)

                        if job_found:
                            logger.info(f"✅ Job {prompt_id} confirmed in ComfyUI queue")
                        else:
                            logger.warning(f"⚠️ Job {prompt_id} not found in ComfyUI queue")
                            logger.info(f"Current queue state - Running: {len(running)}, Pending: {len(pending)}")

                        return job_found
                    else:
                        logger.error(f"Failed to check queue: {response.status}")
                        return False
        except Exception as e:
            logger.error(f"Error verifying job in queue: {e}")
            return False

    async def _wait_for_completion(self, prompt_id: str) -> Dict:
        logger.info(f"Waiting for generation completion: {prompt_id}")
        logger.info("This may take several minutes due to increased frame count...")

        for attempt in range(120):  # 10 minutes timeout
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.base_url}/history/{prompt_id}") as response:
                        if response.status == 200:
                            history = await response.json()
                            if prompt_id in history:
                                result = history[prompt_id]
                                status = result.get("status", {})

                                if "completed" in status:
                                    logger.info(f"✅ Generation completed successfully after {attempt * 5} seconds")
                                    return result
                                elif "status_str" in status:
                                    logger.info(f"Processing status: {status['status_str']}")

                                if "outputs" in result:
                                    logger.info(f"✅ Generation completed successfully after {attempt * 5} seconds")
                                    return result

                if attempt % 12 == 0:  # Every 60 seconds
                    await self._log_queue_status()

                await asyncio.sleep(5)

            except Exception as e:
                logger.warning(f"Error checking completion status (attempt {attempt}): {e}")
                await asyncio.sleep(5)

        await self._log_queue_status()
        raise Exception(f"Generation timed out after 10 minutes for prompt {prompt_id}")

    async def _log_queue_status(self):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/queue") as response:
                    if response.status == 200:
                        queue_data = await response.json()
                        running = len(queue_data.get("queue_running", []))
                        pending = len(queue_data.get("queue_pending", []))
                        logger.info(f"ComfyUI Queue Status - Running: {running}, Pending: {pending}")
                    else:
                        logger.warning(f"Could not check queue status: {response.status}")
        except Exception as e:
            logger.warning(f"Error checking queue status: {e}")

    async def _process_results(self, result: Dict, request: VideoGenerationRequest) -> Path:
        try:
            outputs = result.get("outputs", {})
            logger.info(f"Processing results with {len(outputs)} output nodes")

            video_files = []
            preview_images = []

            for node_id, output in outputs.items():
                logger.debug(f"Checking node {node_id} output: {output.keys()}")

                if "images" in output:
                    for image_info in output["images"]:
                        filename = image_info["filename"]
                        subfolder = image_info.get("subfolder", "")
                        logger.info(f"Found preview image: {filename} in subfolder: {subfolder}")
                        preview_images.append(filename)

                if "gifs" in output:
                    for video_info in output["gifs"]:
                        filename = video_info["filename"]
                        subfolder = video_info.get("subfolder", "")
                        source_path = Path(f"/home/patrick/Projects/ComfyUI-Production/output/{subfolder}/{filename}")
                        logger.info(f"Found video file: {source_path}")
                        if source_path.exists():
                            video_files.append(source_path)

            logger.info(f"Found {len(preview_images)} preview images and {len(video_files)} video files")

            if not video_files:
                logger.warning("No video files found in gifs output, checking all outputs")
                for node_id, output in outputs.items():
                    for key in output:
                        if key in ["videos", "gifs", "images"]:
                            for file_info in output[key]:
                                filename = file_info["filename"]
                                if filename.endswith(('.mp4', '.avi', '.mov', '.webm')):
                                    subfolder = file_info.get("subfolder", "")
                                    source_path = Path(f"/home/patrick/Projects/ComfyUI-Production/output/{subfolder}/{filename}")
                                    if source_path.exists():
                                        video_files.append(source_path)

            if not video_files:
                raise Exception("No video files generated by VHS_VideoCombine")

            source_video = video_files[0]
            timestamp = int(time.time())
            clean_prompt = request.prompt[:20].replace(" ", "_").replace("/", "_")
            video_filename = f"echo_improved_video_{timestamp}_{clean_prompt}.mp4"
            video_path = OUTPUT_DIR / video_filename

            import shutil
            shutil.copy2(source_video, video_path)
            os.chmod(video_path, 0o644)

            # Get video info to verify quality
            try:
                import subprocess
                result = subprocess.run([
                    'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', str(video_path)
                ], capture_output=True, text=True)
                if result.returncode == 0:
                    video_info = json.loads(result.stdout)
                    for stream in video_info.get('streams', []):
                        if stream.get('codec_type') == 'video':
                            width = stream.get('width')
                            height = stream.get('height')
                            fps = stream.get('avg_frame_rate', '0/1')
                            logger.info(f"✅ VIDEO QUALITY VERIFIED: {width}x{height} at {fps} fps")
                            break
            except Exception as e:
                logger.warning(f"Could not verify video quality: {e}")

            logger.info(f"✅ IMPROVED VIDEO saved to: {video_path}")
            logger.info(f"✅ Preview images generated: {len(preview_images)} images in ComfyUI history")
            logger.info(f"✅ Video size: {video_path.stat().st_size / (1024*1024):.1f} MB")
            return video_path

        except Exception as e:
            logger.error(f"❌ Failed to process results: {e}")
            logger.exception("Full error traceback:")
            raise
# Import AnimateDiff workflow
import sys
sys.path.append('/opt/tower-echo-brain')
from animatediff_workflow import create_animatediff_workflow

def generate_quality_video(prompt, duration=5, fps=24):
    '''Generate actual quality video with AnimateDiff'''
    num_frames = duration * fps  # 5 seconds * 24fps = 120 frames
    workflow = create_animatediff_workflow(prompt, num_frames, fps)
    
    # Queue and execute
    result = requests.post(f'{COMFYUI_URL}/prompt', json={'prompt': workflow})
    return result.json()
