#!/usr/bin/env python3
"""
Simple test script for autonomous video generation
Bypasses complex imports to test core functionality
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

class SimpleVideoGenerator:
    """Simplified video generator for testing"""

    def __init__(self):
        self.comfyui_url = "http://127.0.0.1:8188"
        self.output_dir = "/mnt/1TB-storage/ComfyUI/output/autonomous_test"
        os.makedirs(self.output_dir, exist_ok=True)

    async def generate_video_from_character(self, character_image_path: str):
        """Generate video from character image using ComfyUI"""
        try:
            logger.info(f"🎬 Starting video generation for: {character_image_path}")

            if not os.path.exists(character_image_path):
                raise FileNotFoundError(f"Character image not found: {character_image_path}")

            # Load the working workflow
            workflow_path = "/mnt/1TB-storage/ComfyUI/goblin_slayer_animation_working.json"
            with open(workflow_path, 'r') as f:
                workflow = json.load(f)

            # Customize workflow for cyberpunk character
            self._customize_workflow_for_cyberpunk(workflow)

            # Execute workflow
            result = await self._execute_comfyui_workflow(workflow)

            # Find output files
            output_files = await self._find_recent_outputs()

            if output_files:
                logger.info(f"✅ Video generation completed! Files: {output_files}")
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

    def _customize_workflow_for_cyberpunk(self, workflow):
        """Customize workflow for cyberpunk character"""
        try:
            cyberpunk_prompt = """cyberpunk goblin slayer, heavily armored knight in futuristic armor,
            full plate armor with cybernetic enhancements, helmet with glowing visor,
            dynamic combat action, slaying cybernetic goblins, neon-lit urban environment,
            dramatic lighting with neon glow, anime character design, high quality animation,
            cinematic movement, action sequence, masterpiece"""

            negative_prompt = """worst quality, low quality, blurry, ugly, distorted,
            static, still image, no movement, multiple characters, background focus only"""

            # Update text prompts in the workflow
            for node_id, node in workflow.get('prompt', {}).items():
                if node.get('class_type') == 'CLIPTextEncode':
                    title = node.get('_meta', {}).get('title', '').lower()
                    if 'positive' in title or 'prompt' in title:
                        node['inputs']['text'] = cyberpunk_prompt
                        logger.info("Updated positive prompt for cyberpunk theme")
                    elif 'negative' in title:
                        node['inputs']['text'] = negative_prompt
                        logger.info("Updated negative prompt")

                # Update generation parameters for better quality
                elif node.get('class_type') == 'KSampler':
                    node['inputs']['steps'] = 30  # More steps for better quality
                    node['inputs']['cfg'] = 8.5   # Slightly higher CFG
                    node['inputs']['seed'] = 42   # Fixed seed for reproducibility
                    logger.info("Updated sampling parameters")

        except Exception as e:
            logger.warning(f"Failed to customize workflow: {e}")

    async def _execute_comfyui_workflow(self, workflow):
        """Execute workflow via ComfyUI API"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
                logger.info("Submitting workflow to ComfyUI...")

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

                    logger.info(f"Workflow submitted with prompt_id: {prompt_id}")

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
                                logger.info(f"✅ Workflow completed in {wait_time}s")
                                return {
                                    'prompt_id': prompt_id,
                                    'execution_time': wait_time,
                                    'history': history[prompt_id]
                                }

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
    logger.info("🚀 Starting autonomous video generation test")

    generator = SimpleVideoGenerator()

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

    # Generate video
    result = await generator.generate_video_from_character(cyberpunk_path)

    # Report results
    if result['success']:
        logger.info("🎉 AUTONOMOUS VIDEO GENERATION TEST SUCCESSFUL!")
        logger.info(f"Generated files: {result['output_files']}")

        # Show file details
        for output_file in result['output_files']:
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
                logger.info(f"  📁 {output_file} ({file_size:.1f} MB)")
    else:
        logger.error(f"❌ AUTONOMOUS VIDEO GENERATION TEST FAILED: {result['error']}")

if __name__ == "__main__":
    asyncio.run(main())