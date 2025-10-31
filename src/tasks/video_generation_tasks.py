#!/usr/bin/env python3
"""
Video Generation Task Types for Echo Brain Autonomous Operation
Extends the task queue system for anime video generation workflows
"""

import json
import logging
import os
import subprocess
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .task_queue import Task, TaskType, TaskPriority, TaskStatus

logger = logging.getLogger(__name__)

class VideoTaskType(Enum):
    """Video generation specific task types"""
    CHARACTER_TO_VIDEO = "character_to_video"
    BATCH_GENERATION = "batch_generation"
    QUALITY_CHECK = "quality_check"
    WORKFLOW_OPTIMIZATION = "workflow_optimization"
    VIDEO_ENHANCEMENT = "video_enhancement"
    STYLE_TRANSFER = "style_transfer"

@dataclass
class VideoGenerationConfig:
    """Configuration for video generation tasks"""
    input_image_path: str
    output_directory: str = "/mnt/1TB-storage/ComfyUI/output/autonomous"
    workflow_file: str = "/mnt/1TB-storage/ComfyUI/goblin_slayer_animation_working.json"
    comfyui_url: str = "http://127.0.0.1:8188"

    # Video parameters
    frame_count: int = 120  # 5 seconds at 24fps
    width: int = 1024
    height: int = 1024

    # Animation parameters
    prompt: str = "anime character, dynamic movement, action scene, high quality animation"
    negative_prompt: str = "static, blurry, low quality, distorted"

    # Generation settings
    steps: int = 25
    cfg: float = 8.0
    seed: int = -1  # -1 for random

    # Quality settings
    quality_threshold: float = 0.7  # Minimum quality score

class VideoGenerationExecutor:
    """Autonomous video generation executor for Echo Brain"""

    def __init__(self):
        self.comfyui_url = "http://127.0.0.1:8188"
        self.base_output_dir = "/mnt/1TB-storage/ComfyUI/output/autonomous"
        self.workflow_templates = {
            "character_animation": "/mnt/1TB-storage/ComfyUI/goblin_slayer_animation_working.json",
            "style_transfer": "/mnt/1TB-storage/ComfyUI/rife_video_scaling_10sec_workflow.json"
        }

        # Ensure output directory exists
        os.makedirs(self.base_output_dir, exist_ok=True)

    async def execute_character_to_video(self, task: Task) -> Dict[str, Any]:
        """Execute character to video generation task"""
        try:
            logger.info(f"🎬 Starting character to video generation: {task.name}")

            config = VideoGenerationConfig(**task.payload.get('config', {}))

            # Validate input image exists
            if not os.path.exists(config.input_image_path):
                raise FileNotFoundError(f"Input image not found: {config.input_image_path}")

            # Create unique output directory for this task
            task_output_dir = os.path.join(self.base_output_dir, f"task_{task.id}")
            os.makedirs(task_output_dir, exist_ok=True)

            # Load and customize workflow
            workflow = await self._load_workflow_template(config.workflow_file)
            workflow = await self._customize_workflow(workflow, config, task_output_dir)

            # Execute ComfyUI workflow
            result = await self._execute_comfyui_workflow(workflow, task.id)

            # Validate output
            output_files = await self._find_output_files(task_output_dir)
            if not output_files:
                raise Exception("No output files generated")

            # Quality check
            quality_score = await self._assess_video_quality(output_files[0])

            # Copy successful outputs to proper location
            final_outputs = []
            if quality_score >= config.quality_threshold:
                for output_file in output_files:
                    final_path = await self._archive_successful_output(output_file, task.name)
                    final_outputs.append(final_path)

            return {
                'status': 'completed',
                'output_files': final_outputs,
                'quality_score': quality_score,
                'generation_time': result.get('execution_time', 0),
                'task_output_dir': task_output_dir,
                'comfyui_result': result
            }

        except Exception as e:
            logger.error(f"❌ Character to video generation failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'output_files': []
            }

    async def execute_batch_generation(self, task: Task) -> Dict[str, Any]:
        """Execute batch video generation task"""
        try:
            logger.info(f"🎬 Starting batch generation: {task.name}")

            batch_config = task.payload.get('batch_config', {})
            input_images = batch_config.get('input_images', [])

            results = []
            for i, image_path in enumerate(input_images):
                # Create individual task for each image
                individual_task = Task(
                    id=f"{task.id}_batch_{i}",
                    name=f"{task.name}_item_{i}",
                    task_type=TaskType.USER_REQUEST,
                    priority=task.priority,
                    status=TaskStatus.PENDING,
                    payload={
                        'config': {
                            'input_image_path': image_path,
                            **batch_config.get('base_config', {})
                        }
                    },
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )

                result = await self.execute_character_to_video(individual_task)
                results.append({
                    'input_image': image_path,
                    'result': result
                })

            successful_results = [r for r in results if r['result'].get('status') == 'completed']

            return {
                'status': 'completed',
                'total_processed': len(results),
                'successful': len(successful_results),
                'failed': len(results) - len(successful_results),
                'results': results
            }

        except Exception as e:
            logger.error(f"❌ Batch generation failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }

    async def _load_workflow_template(self, workflow_path: str) -> Dict[str, Any]:
        """Load ComfyUI workflow template"""
        try:
            with open(workflow_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load workflow template: {e}")
            raise

    async def _customize_workflow(self, workflow: Dict[str, Any],
                                config: VideoGenerationConfig,
                                output_dir: str) -> Dict[str, Any]:
        """Customize workflow with task-specific parameters"""
        try:
            # Update prompts if they exist in the workflow
            for node_id, node in workflow.get('prompt', {}).items():
                if node.get('class_type') == 'CLIPTextEncode':
                    if 'positive' in node.get('_meta', {}).get('title', '').lower():
                        node['inputs']['text'] = config.prompt
                    elif 'negative' in node.get('_meta', {}).get('title', '').lower():
                        node['inputs']['text'] = config.negative_prompt

                # Update generation parameters
                elif node.get('class_type') == 'KSampler':
                    node['inputs']['steps'] = config.steps
                    node['inputs']['cfg'] = config.cfg
                    if config.seed != -1:
                        node['inputs']['seed'] = config.seed

                # Update image loading if present
                elif node.get('class_type') == 'LoadImage':
                    # Set input image path
                    image_filename = os.path.basename(config.input_image_path)
                    node['inputs']['image'] = image_filename

            return workflow

        except Exception as e:
            logger.error(f"Failed to customize workflow: {e}")
            raise

    async def _execute_comfyui_workflow(self, workflow: Dict[str, Any], task_id: str) -> Dict[str, Any]:
        """Execute workflow via ComfyUI API"""
        try:
            import aiohttp
            import asyncio

            start_time = datetime.now()

            async with aiohttp.ClientSession() as session:
                # Submit workflow to ComfyUI
                async with session.post(f"{self.comfyui_url}/prompt",
                                      json={"prompt": workflow["prompt"]}) as response:
                    if response.status != 200:
                        raise Exception(f"ComfyUI API error: {response.status}")

                    result = await response.json()
                    prompt_id = result.get('prompt_id')

                    if not prompt_id:
                        raise Exception("No prompt_id returned from ComfyUI")

                # Poll for completion
                max_wait = 600  # 10 minutes max
                wait_time = 0
                while wait_time < max_wait:
                    await asyncio.sleep(5)
                    wait_time += 5

                    async with session.get(f"{self.comfyui_url}/history/{prompt_id}") as hist_response:
                        if hist_response.status == 200:
                            history = await hist_response.json()
                            if prompt_id in history:
                                execution_time = (datetime.now() - start_time).total_seconds()
                                return {
                                    'prompt_id': prompt_id,
                                    'execution_time': execution_time,
                                    'history': history[prompt_id]
                                }

                raise Exception(f"ComfyUI execution timeout after {max_wait} seconds")

        except Exception as e:
            logger.error(f"ComfyUI execution failed: {e}")
            raise

    async def _find_output_files(self, output_dir: str) -> List[str]:
        """Find generated output files"""
        try:
            # Check both the specific output directory and ComfyUI's main output
            search_dirs = [
                output_dir,
                "/mnt/1TB-storage/ComfyUI/output"
            ]

            output_files = []
            for search_dir in search_dirs:
                if os.path.exists(search_dir):
                    for root, dirs, files in os.walk(search_dir):
                        for file in files:
                            if file.endswith(('.mp4', '.gif', '.avi', '.mov')):
                                file_path = os.path.join(root, file)
                                # Check if file was created recently (within last 10 minutes)
                                if datetime.now().timestamp() - os.path.getctime(file_path) < 600:
                                    output_files.append(file_path)

            return sorted(output_files, key=os.path.getctime, reverse=True)

        except Exception as e:
            logger.error(f"Failed to find output files: {e}")
            return []

    async def _assess_video_quality(self, video_path: str) -> float:
        """Assess video quality using basic metrics"""
        try:
            # Basic quality assessment - file size and duration
            if not os.path.exists(video_path):
                return 0.0

            file_size = os.path.getsize(video_path)

            # Use ffprobe to get video info
            try:
                result = subprocess.run([
                    'ffprobe', '-v', 'quiet', '-print_format', 'json',
                    '-show_format', '-show_streams', video_path
                ], capture_output=True, text=True, timeout=30)

                if result.returncode == 0:
                    info = json.loads(result.stdout)
                    duration = float(info.get('format', {}).get('duration', 0))

                    # Basic quality score based on file size, duration
                    if duration > 0 and file_size > 100000:  # At least 100KB and some duration
                        # Score based on bitrate (file_size / duration)
                        bitrate = file_size / duration
                        if bitrate > 500000:  # High quality
                            return 0.9
                        elif bitrate > 200000:  # Medium quality
                            return 0.7
                        else:  # Low but acceptable quality
                            return 0.5

            except Exception:
                pass

            # Fallback: basic file size check
            if file_size > 1000000:  # 1MB+
                return 0.8
            elif file_size > 100000:  # 100KB+
                return 0.6
            else:
                return 0.3

        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return 0.0

    async def _archive_successful_output(self, source_path: str, task_name: str) -> str:
        """Archive successful output to permanent location"""
        try:
            # Create archive directory
            archive_dir = "/mnt/1TB-storage/ComfyUI/output/successful/autonomous"
            os.makedirs(archive_dir, exist_ok=True)

            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_extension = os.path.splitext(source_path)[1]
            safe_task_name = "".join(c for c in task_name if c.isalnum() or c in "._- ")
            final_filename = f"{safe_task_name}_{timestamp}{file_extension}"
            final_path = os.path.join(archive_dir, final_filename)

            # Copy file
            import shutil
            shutil.copy2(source_path, final_path)

            logger.info(f"✅ Archived successful output: {final_path}")
            return final_path

        except Exception as e:
            logger.error(f"Failed to archive output: {e}")
            return source_path

# Utility functions for creating video generation tasks
def create_character_to_video_task(input_image_path: str,
                                 prompt: str = None,
                                 priority: TaskPriority = TaskPriority.HIGH) -> Task:
    """Create a character to video generation task"""
    config = {
        'input_image_path': input_image_path,
        'prompt': prompt or "anime character, dynamic movement, action scene, high quality animation",
        'output_directory': "/mnt/1TB-storage/ComfyUI/output/autonomous"
    }

    return Task(
        id=str(uuid.uuid4()),
        name=f"Character to Video: {os.path.basename(input_image_path)}",
        task_type=TaskType.USER_REQUEST,  # Use existing type, specify video in payload
        priority=priority,
        status=TaskStatus.PENDING,
        payload={
            'video_task_type': VideoTaskType.CHARACTER_TO_VIDEO.value,
            'config': config
        },
        created_at=datetime.now(),
        updated_at=datetime.now(),
        creator="echo_autonomous_video"
    )

def create_batch_generation_task(input_images: List[str],
                               batch_name: str = "Batch Generation",
                               priority: TaskPriority = TaskPriority.NORMAL) -> Task:
    """Create a batch video generation task"""

    return Task(
        id=str(uuid.uuid4()),
        name=batch_name,
        task_type=TaskType.USER_REQUEST,
        priority=priority,
        status=TaskStatus.PENDING,
        payload={
            'video_task_type': VideoTaskType.BATCH_GENERATION.value,
            'batch_config': {
                'input_images': input_images,
                'base_config': {
                    'output_directory': "/mnt/1TB-storage/ComfyUI/output/autonomous"
                }
            }
        },
        created_at=datetime.now(),
        updated_at=datetime.now(),
        creator="echo_autonomous_video"
    )

def create_quality_check_task(video_path: str,
                            min_quality: float = 0.7,
                            priority: TaskPriority = TaskPriority.LOW) -> Task:
    """Create a video quality check task"""

    return Task(
        id=str(uuid.uuid4()),
        name=f"Quality Check: {os.path.basename(video_path)}",
        task_type=TaskType.ANALYSIS,
        priority=priority,
        status=TaskStatus.PENDING,
        payload={
            'video_task_type': VideoTaskType.QUALITY_CHECK.value,
            'video_path': video_path,
            'min_quality': min_quality
        },
        created_at=datetime.now(),
        updated_at=datetime.now(),
        creator="echo_autonomous_video"
    )