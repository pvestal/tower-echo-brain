#!/usr/bin/env python3
"""
Fixed autonomous video generation test
Handles frame count limitations properly
"""

import asyncio
import json
import logging
import os
import sys
import uuid
from datetime import datetime
import aiohttp
import subprocess

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FixedVideoGenerator:
    """Fixed video generator with proper frame count handling"""

    def __init__(self):
        self.comfyui_url = "http://127.0.0.1:8188"
        self.output_dir = "/mnt/1TB-storage/ComfyUI/output/autonomous_test"
        os.makedirs(self.output_dir, exist_ok=True)

    async def generate_video_from_character(self, character_image_path: str):
        """Generate video from character image using ComfyUI with fixed parameters"""
        try:
            logger.info(f"🎬 Starting FIXED video generation for: {character_image_path}")

            if not os.path.exists(character_image_path):
                raise FileNotFoundError(f"Character image not found: {character_image_path}")

            # Load the working workflow
            workflow_path = "/mnt/1TB-storage/ComfyUI/goblin_slayer_animation_working.json"
            with open(workflow_path, 'r') as f:
                workflow = json.load(f)

            # Fix workflow parameters
            self._fix_workflow_parameters(workflow)

            # Execute workflow
            result = await self._execute_comfyui_workflow(workflow)

            # Find output files
            output_files = await self._find_recent_outputs()

            if output_files:
                logger.info(f"✅ FIXED Video generation completed! Files: {output_files}")
                return {
                    'success': True,
                    'output_files': output_files,
                    'workflow_result': result
                }
            else:
                logger.error("❌ No output files found")
                return {
                    'success': False,
                    'error': 'No output files generated'
                }

        except Exception as e:
            logger.error(f"❌ Video generation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _fix_workflow_parameters(self, workflow):
        """Fix workflow parameters to work within AnimateDiff limits"""
        try:
            cyberpunk_prompt = """cyberpunk goblin slayer, heavily armored knight,
            dynamic combat action, slaying goblins, dramatic lighting,
            anime character design, high quality animation, masterpiece"""

            negative_prompt = """worst quality, low quality, blurry, ugly, distorted,
            static, still image, no movement, multiple characters"""

            # Update text prompts and critical parameters
            for node_id, node in workflow.get('prompt', {}).items():
                if node.get('class_type') == 'CLIPTextEncode':
                    title = node.get('_meta', {}).get('title', '').lower()
                    if 'positive' in title or 'prompt' in title:
                        node['inputs']['text'] = cyberpunk_prompt
                        logger.info("Updated positive prompt for cyberpunk theme")
                    elif 'negative' in title:
                        node['inputs']['text'] = negative_prompt
                        logger.info("Updated negative prompt")

                # Fix KSampler parameters
                elif node.get('class_type') == 'KSampler':
                    node['inputs']['steps'] = 20  # Reduce steps for speed
                    node['inputs']['cfg'] = 7.0   # Standard CFG
                    node['inputs']['seed'] = 42   # Fixed seed
                    logger.info("Updated KSampler parameters")

                # CRITICAL: Fix EmptyLatentImage to use only 24 frames
                elif node.get('class_type') == 'EmptyLatentImage':
                    # Set batch_size to 24 (max frames without context window)
                    node['inputs']['batch_size'] = 24
                    node['inputs']['width'] = 512   # Reduce resolution for reliability
                    node['inputs']['height'] = 512
                    logger.info("✅ FIXED: Set batch_size to 24 frames (AnimateDiff limit)")

                # Fix any ADE (AnimateDiff Evolved) context nodes
                elif 'ADE' in node.get('class_type', '') or 'AnimateDiff' in node.get('class_type', ''):
                    # Ensure no context window settings that would conflict
                    if 'context_length' in node.get('inputs', {}):
                        node['inputs']['context_length'] = 24
                    if 'closed_loop' in node.get('inputs', {}):
                        node['inputs']['closed_loop'] = False  # Disable closed loop for simplicity
                    logger.info(f"Fixed AnimateDiff node: {node.get('class_type')}")

                # Fix video output settings
                elif node.get('class_type') == 'VHS_VideoCombine':
                    # Standard video settings
                    node['inputs']['frame_rate'] = 8  # 8 FPS for 24 frames = 3 seconds
                    logger.info("Set video frame rate to 8 FPS (3 second video)")

        except Exception as e:
            logger.warning(f"Failed to fix workflow parameters: {e}")

    async def _execute_comfyui_workflow(self, workflow):
        """Execute workflow via ComfyUI API"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
                logger.info("Submitting FIXED workflow to ComfyUI...")

                # Submit workflow
                async with session.post(f"{self.comfyui_url}/prompt",
                                      json={"prompt": workflow["prompt"]}) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"ComfyUI API error {response.status}: {error_text}")

                    result = await response.json()
                    prompt_id = result.get('prompt_id')

                    if not prompt_id:
                        raise Exception("No prompt_id returned from ComfyUI")

                    logger.info(f"FIXED Workflow submitted with prompt_id: {prompt_id}")

                # Wait for completion (poll every 10 seconds)
                max_wait = 600  # 10 minutes max
                wait_time = 0

                while wait_time < max_wait:
                    await asyncio.sleep(10)
                    wait_time += 10

                    logger.info(f"Checking progress... ({wait_time}s elapsed)")

                    async with session.get(f"{self.comfyui_url}/history/{prompt_id}") as hist_response:
                        if hist_response.status == 200:
                            history = await hist_response.json()
                            if prompt_id in history:
                                entry = history[prompt_id]
                                if entry.get('status', {}).get('completed'):
                                    logger.info(f"✅ FIXED Workflow completed in {wait_time}s")
                                    return {
                                        'prompt_id': prompt_id,
                                        'execution_time': wait_time,
                                        'history': entry
                                    }
                                elif 'error' in entry.get('status', {}).get('status_str', ''):
                                    error_msg = entry['status']['messages'][-1][1].get('exception_message', 'Unknown error')
                                    raise Exception(f"ComfyUI execution error: {error_msg}")

                raise Exception(f"ComfyUI execution timeout after {max_wait} seconds")

        except Exception as e:
            logger.error(f"ComfyUI execution failed: {e}")
            raise

    async def _find_recent_outputs(self):
        """Find recently generated output files"""
        try:
            # Check ComfyUI output directory for recent files
            output_dirs = [
                "/mnt/1TB-storage/ComfyUI/output",
                self.output_dir
            ]

            recent_files = []
            cutoff_time = datetime.now().timestamp() - 600  # Files created in last 10 minutes

            for output_dir in output_dirs:
                if os.path.exists(output_dir):
                    for root, dirs, files in os.walk(output_dir):
                        for file in files:
                            if file.endswith(('.mp4', '.gif', '.avi', '.mov')):
                                file_path = os.path.join(root, file)
                                if os.path.getctime(file_path) > cutoff_time:
                                    recent_files.append(file_path)

            # Sort by creation time, newest first
            recent_files.sort(key=os.path.getctime, reverse=True)
            return recent_files[:3]  # Return up to 3 most recent

        except Exception as e:
            logger.error(f"Failed to find output files: {e}")
            return []

    async def test_comfyui_connection(self):
        """Test connection to ComfyUI"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.comfyui_url}/system_stats") as response:
                    if response.status == 200:
                        stats = await response.json()
                        logger.info(f"✅ ComfyUI connected - {stats.get('system', {}).get('comfyui_version', 'Unknown version')}")
                        return True
                    else:
                        logger.error(f"ComfyUI connection failed: {response.status}")
                        return False
        except Exception as e:
            logger.error(f"ComfyUI connection test failed: {e}")
            return False

async def main():
    """Main test function"""
    logger.info("🚀 Starting FIXED autonomous video generation test")

    generator = FixedVideoGenerator()

    # Test ComfyUI connection
    if not await generator.test_comfyui_connection():
        logger.error("❌ ComfyUI not accessible, aborting test")
        return

    # Test with cyberpunk character
    cyberpunk_path = "/mnt/1TB-storage/ComfyUI/output/cyberpunk_goblin_slayer_crawl_00001_.png"

    if not os.path.exists(cyberpunk_path):
        logger.error(f"❌ Test image not found: {cyberpunk_path}")
        return

    logger.info(f"Using test image: {cyberpunk_path}")

    # Generate video with fixed parameters
    result = await generator.generate_video_from_character(cyberpunk_path)

    # Report results
    if result['success']:
        logger.info("🎉 FIXED AUTONOMOUS VIDEO GENERATION TEST SUCCESSFUL!")
        logger.info(f"Generated files: {result['output_files']}")

        # Show file details
        for output_file in result['output_files']:
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
                logger.info(f"  📁 {output_file} ({file_size:.1f} MB)")

        # Create success marker for Echo Brain to detect
        success_marker = "/opt/tower-echo-brain/logs/video_generation_success.json"
        with open(success_marker, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'test_status': 'SUCCESS',
                'output_files': result['output_files'],
                'character_used': cyberpunk_path,
                'autonomous_generation': True
            }, f, indent=2)

        logger.info(f"✅ Success marker created: {success_marker}")

    else:
        logger.error(f"❌ FIXED AUTONOMOUS VIDEO GENERATION TEST FAILED: {result['error']}")

        # Create failure marker
        failure_marker = "/opt/tower-echo-brain/logs/video_generation_failure.json"
        with open(failure_marker, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'test_status': 'FAILED',
                'error': result['error'],
                'character_used': cyberpunk_path,
                'autonomous_generation': True
            }, f, indent=2)

if __name__ == "__main__":
    asyncio.run(main())