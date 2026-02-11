#!/usr/bin/env python3
"""
Test Tokyo Debt Character Generation
Uses ComfyUI workflow with character data from story_bible
"""

import json
import requests
import time
import logging
from pathlib import Path

# Configuration
COMFYUI_URL = "http://localhost:8188"
WORKFLOW_PATH = "/opt/tower-anime-production/workflows/comfyui/anime_30sec_working_workflow.json"
OUTPUT_DIR = "/tmp/tokyo_debt_test_output"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_workflow():
    """Load the working anime workflow"""
    with open(WORKFLOW_PATH, 'r') as f:
        return json.load(f)

def customize_workflow_for_mei():
    """Customize workflow for Mei Kobayashi character"""
    workflow = load_workflow()

    # Update prompt node
    for node_id, node in workflow.items():
        if node.get("class_type") == "CLIPTextEncode":
            title = node.get("_meta", {}).get("title", "")
            if "Prompt" in title and "Negative" not in title:
                # Update with Mei's character description
                workflow[node_id]["inputs"]["text"] = (
                    "masterpiece, best quality, 1girl, Mei Kobayashi, "
                    "beautiful Japanese woman, gentle nurturing personality, "
                    "revealing clothing, curvy figure, large breasts, "
                    "competitive expression, Tokyo apartment setting, "
                    "anime style, detailed eyes, flowing hair, "
                    "high resolution, photorealistic"
                )
                logger.info(f"Updated prompt in node {node_id}")

        # Fix model name to match available checkpoint
        elif node.get("class_type") == "CheckpointLoaderSimple":
            if node["inputs"].get("ckpt_name") == "realisticVision_v51.safetensors":
                node["inputs"]["ckpt_name"] = "realistic_vision_v51.safetensors"
                logger.info(f"Fixed model name in node {node_id}")

    return workflow

def submit_to_comfyui(workflow):
    """Submit workflow to ComfyUI for generation"""
    try:
        # Generate unique prompt ID
        prompt_id = f"tokyo_debt_test_{int(time.time())}"

        # Submit workflow
        response = requests.post(
            f"{COMFYUI_URL}/prompt",
            json={
                "prompt": workflow,
                "client_id": prompt_id
            }
        )

        if response.status_code == 200:
            result = response.json()
            prompt_id = result.get("prompt_id")
            logger.info(f"Submitted workflow, prompt_id: {prompt_id}")
            return prompt_id
        else:
            logger.error(f"Failed to submit workflow: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        logger.error(f"Error submitting to ComfyUI: {e}")
        return None

def check_generation_status(prompt_id):
    """Check if generation is complete"""
    try:
        response = requests.get(f"{COMFYUI_URL}/history/{prompt_id}")
        if response.status_code == 200:
            history = response.json()
            if prompt_id in history:
                return history[prompt_id]
        return None
    except Exception as e:
        logger.error(f"Error checking status: {e}")
        return None

def wait_for_completion(prompt_id, timeout=300):
    """Wait for generation to complete"""
    start_time = time.time()

    while time.time() - start_time < timeout:
        status = check_generation_status(prompt_id)

        if status:
            logger.info(f"Generation completed: {prompt_id}")
            return status

        logger.info(f"Generation in progress... ({int(time.time() - start_time)}s)")
        time.sleep(5)

    logger.error(f"Generation timed out after {timeout}s")
    return None

def test_comfyui_connection():
    """Test if ComfyUI is responding"""
    try:
        response = requests.get(f"{COMFYUI_URL}/system_stats", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            logger.info("ComfyUI is responsive")
            logger.info(f"System: {stats.get('system', {}).get('os', 'Unknown')}")
            return True
        else:
            logger.error(f"ComfyUI returned {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"ComfyUI connection failed: {e}")
        return False

def main():
    """Main test function"""
    logger.info("Starting Tokyo Debt character generation test...")

    # Test ComfyUI connection
    if not test_comfyui_connection():
        logger.error("Cannot connect to ComfyUI. Is it running on port 8188?")
        return

    # Check if workflow file exists
    if not Path(WORKFLOW_PATH).exists():
        logger.error(f"Workflow file not found: {WORKFLOW_PATH}")
        return

    # Load and customize workflow
    try:
        workflow = customize_workflow_for_mei()
        logger.info("Loaded and customized workflow for Mei Kobayashi")
    except Exception as e:
        logger.error(f"Failed to load workflow: {e}")
        return

    # Submit to ComfyUI
    prompt_id = submit_to_comfyui(workflow)
    if not prompt_id:
        logger.error("Failed to submit workflow")
        return

    # Wait for completion
    logger.info("Waiting for generation to complete...")
    result = wait_for_completion(prompt_id)

    if result:
        logger.info("✅ Generation completed successfully!")
        logger.info(f"Outputs: {list(result.get('outputs', {}).keys())}")

        # Create output directory
        Path(OUTPUT_DIR).mkdir(exist_ok=True)

        # Save result info
        with open(f"{OUTPUT_DIR}/generation_result_{prompt_id}.json", 'w') as f:
            json.dump(result, f, indent=2)

        logger.info(f"Result saved to {OUTPUT_DIR}/generation_result_{prompt_id}.json")
    else:
        logger.error("❌ Generation failed")

if __name__ == "__main__":
    main()