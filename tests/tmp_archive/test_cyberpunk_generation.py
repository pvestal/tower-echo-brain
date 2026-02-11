#!/usr/bin/env python3
"""
Test Cyberpunk Goblins Character Generation
Uses action combat workflow with cyberpunk character data
"""

import json
import requests
import time
import logging

# Configuration
COMFYUI_URL = "http://localhost:8188"
WORKFLOW_PATH = "/opt/tower-anime-production/workflows/comfyui/ACTION_combat_workflow.json"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_cyberpunk_generation():
    """Test cyberpunk character generation with action workflow"""
    logger.info("Testing cyberpunk action workflow...")

    try:
        # Load workflow
        with open(WORKFLOW_PATH, 'r') as f:
            workflow = json.load(f)

        # Find and update the prompt for cyberpunk content
        for node_id, node in workflow.items():
            if node.get("class_type") == "CLIPTextEncode":
                title = node.get("_meta", {}).get("title", "")
                if "Prompt" in title and "Negative" not in title:
                    # Update with cyberpunk goblin scene
                    workflow[node_id]["inputs"]["text"] = (
                        "masterpiece, best quality, cyberpunk goblin slayer, "
                        "neon-lit Tokyo undercity, cybernetic enhancements, "
                        "action combat scene, mutant goblins, "
                        "futuristic weapons, dark atmosphere, "
                        "anime style, detailed cyberpunk aesthetic, "
                        "high resolution, dynamic action"
                    )
                    logger.info(f"Updated cyberpunk prompt in node {node_id}")
                    break

        # Submit quick test
        prompt_id = f"cyberpunk_test_{int(time.time())}"

        response = requests.post(
            f"{COMFYUI_URL}/prompt",
            json={
                "prompt": workflow,
                "client_id": prompt_id
            }
        )

        if response.status_code == 200:
            result = response.json()
            logger.info(f"✅ Cyberpunk workflow submitted successfully: {result.get('prompt_id')}")
            return True
        else:
            logger.error(f"❌ Cyberpunk workflow failed: {response.status_code} - {response.text}")
            return False

    except Exception as e:
        logger.error(f"❌ Cyberpunk workflow error: {e}")
        return False

def check_available_loras():
    """Check what LoRA files are available for cyberpunk"""
    try:
        import os
        lora_dir = "/mnt/1TB-storage/models/loras/"
        if os.path.exists(lora_dir):
            loras = [f for f in os.listdir(lora_dir) if f.endswith('.safetensors')]
            cyberpunk_loras = [f for f in loras if 'cyberpunk' in f.lower() or 'kai' in f.lower()]

            logger.info(f"Available cyberpunk LoRAs: {cyberpunk_loras}")
            return cyberpunk_loras
        else:
            logger.warning("LoRA directory not found")
            return []
    except Exception as e:
        logger.error(f"Failed to check LoRAs: {e}")
        return []

def main():
    logger.info("Starting cyberpunk workflow verification...")

    # Check available LoRAs
    cyberpunk_loras = check_available_loras()

    # Test action workflow
    success = test_cyberpunk_generation()

    if success:
        logger.info("🎯 Cyberpunk workflow verification PASSED")
        logger.info(f"Available cyberpunk LoRAs: {len(cyberpunk_loras)}")
    else:
        logger.error("❌ Cyberpunk workflow verification FAILED")

if __name__ == "__main__":
    main()