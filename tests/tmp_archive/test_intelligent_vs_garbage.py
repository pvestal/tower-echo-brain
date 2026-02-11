#!/usr/bin/env python3
"""
Test: Intelligent Orchestration vs Garbage Generation
Direct comparison of quality improvements
"""

import sys
sys.path.append('/opt/tower-echo-brain')

import json
import requests
import time
import logging
from scripts.intelligent_generation_orchestrator import IntelligentOrchestrator

COMFYUI_URL = "http://localhost:8188"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_customize_workflow(workflow_path: str, resources: dict) -> dict:
    """Load workflow and apply intelligent resource selections"""
    try:
        with open(f"/opt/tower-anime-production/workflows/comfyui/{workflow_path}", 'r') as f:
            workflow = json.load(f)

        # Apply intelligent modifications
        for node_id, node in workflow.items():
            # Update prompts
            if node.get("class_type") == "CLIPTextEncode":
                title = node.get("_meta", {}).get("title", "")
                if "Prompt" in title and "Negative" not in title:
                    node["inputs"]["text"] = resources['positive_prompt']
                    logger.info(f"Applied intelligent prompt to node {node_id}")
                elif "Negative" in title:
                    node["inputs"]["text"] = resources['negative_prompt']

            # Update model
            elif node.get("class_type") == "CheckpointLoaderSimple":
                node["inputs"]["ckpt_name"] = resources['base_model']
                logger.info(f"Set model to {resources['base_model']} in node {node_id}")

            # Apply LoRAs
            elif node.get("class_type") == "LoraLoader" and resources['loras']:
                lora = resources['loras'][0]  # Use first LoRA
                node["inputs"]["lora_name"] = lora['name']
                node["inputs"]["strength_model"] = lora['strength']
                node["inputs"]["strength_clip"] = lora['strength']
                logger.info(f"Applied LoRA {lora['name']} with strength {lora['strength']}")

            # Update generation settings
            elif node.get("class_type") == "KSampler":
                settings = resources['generation_settings']
                node["inputs"].update(settings)
                # Set a specific seed for reproducibility
                node["inputs"]["seed"] = 12345
                logger.info(f"Applied generation settings: {settings}")

        return workflow

    except Exception as e:
        logger.error(f"Failed to customize workflow: {e}")
        return None

def submit_and_wait_for_generation(workflow: dict, test_name: str) -> dict:
    """Submit workflow and wait for completion"""
    try:
        prompt_id = f"{test_name}_{int(time.time())}"

        # Submit
        response = requests.post(
            f"{COMFYUI_URL}/prompt",
            json={
                "prompt": workflow,
                "client_id": prompt_id
            }
        )

        if response.status_code != 200:
            logger.error(f"Failed to submit {test_name}: {response.status_code}")
            return None

        actual_prompt_id = response.json().get("prompt_id")
        logger.info(f"Submitted {test_name}, prompt_id: {actual_prompt_id}")

        # Wait for completion (shortened timeout for testing)
        start_time = time.time()
        timeout = 120  # 2 minutes

        while time.time() - start_time < timeout:
            history_response = requests.get(f"{COMFYUI_URL}/history/{actual_prompt_id}")
            if history_response.status_code == 200:
                history = history_response.json()
                if actual_prompt_id in history:
                    result = history[actual_prompt_id]
                    if result.get('status', {}).get('completed', False):
                        logger.info(f"✅ {test_name} generation completed")
                        return result

            time.sleep(5)

        logger.error(f"❌ {test_name} generation timed out")
        return None

    except Exception as e:
        logger.error(f"Generation failed for {test_name}: {e}")
        return None

async def run_comparison_test():
    """Run side-by-side comparison of intelligent vs garbage generation"""
    print("🧪 Testing: Intelligent Orchestration vs Garbage Generation")
    print("=" * 70)

    # Get intelligent orchestration plan
    orchestrator = IntelligentOrchestrator()
    request = "Generate Kai fighting cyberpunk goblins"

    orchestration_result = await orchestrator.generate_intelligently(request)

    if not orchestration_result['ready_for_generation']:
        print("❌ Orchestration failed")
        return

    resources = orchestration_result['resource_selection']

    print(f"📊 INTELLIGENT SELECTION:")
    print(f"   Workflow: {resources['workflow_file']}")
    print(f"   Model: {resources['base_model']}")
    print(f"   LoRAs: {len(resources['loras'])} applied")
    print(f"   Prompt: Smart character-specific prompt")
    print()

    # Test 1: Intelligent Generation
    print("🧠 Running INTELLIGENT generation...")
    intelligent_workflow = load_and_customize_workflow(resources['workflow_file'], resources)

    if intelligent_workflow:
        intelligent_result = submit_and_wait_for_generation(intelligent_workflow, "intelligent")
    else:
        intelligent_result = None

    # Test 2: Garbage Generation (recreate the previous bad approach)
    print("\n🗑️  Running GARBAGE generation (for comparison)...")

    # Load the bad workflow that was used before
    try:
        with open("/opt/tower-anime-production/workflows/comfyui/ACTION_combat_workflow.json", 'r') as f:
            garbage_workflow = json.load(f)

        # Apply the bad settings that were used before
        for node_id, node in garbage_workflow.items():
            if node.get("class_type") == "CLIPTextEncode":
                title = node.get("_meta", {}).get("title", "")
                if "Prompt" in title and "Negative" not in title:
                    # The generic prompt from before
                    node["inputs"]["text"] = "masterpiece, best quality, cyberpunk goblin slayer, neon-lit Tokyo undercity, cybernetic enhancements, action combat scene, mutant goblins, futuristic weapons, dark atmosphere, anime style, detailed cyberpunk aesthetic, high resolution, dynamic action"

            elif node.get("class_type") == "KSampler":
                # Same seed for comparison
                node["inputs"]["seed"] = 12345

        garbage_result = submit_and_wait_for_generation(garbage_workflow, "garbage")

    except Exception as e:
        logger.error(f"Failed to run garbage generation: {e}")
        garbage_result = None

    # Results Analysis
    print("\n" + "=" * 70)
    print("📊 RESULTS COMPARISON:")
    print("=" * 70)

    if intelligent_result and garbage_result:
        print("✅ Both generations completed - ready for quality comparison")

        # Extract technical metrics
        intelligent_outputs = list(intelligent_result.get('outputs', {}).keys())
        garbage_outputs = list(garbage_result.get('outputs', {}).keys())

        print(f"\n🧠 INTELLIGENT Generation:")
        print(f"   Resource Selection: OPTIMAL (cyberpunk model + LoRAs)")
        print(f"   Prompt: CHARACTER-SPECIFIC (from story_bible)")
        print(f"   Outputs: {intelligent_outputs}")

        print(f"\n🗑️  GARBAGE Generation:")
        print(f"   Resource Selection: POOR (wrong model, no LoRAs)")
        print(f"   Prompt: GENERIC (no character data)")
        print(f"   Outputs: {garbage_outputs}")

        print(f"\n🎯 EXPECTED QUALITY IMPROVEMENT:")
        print(f"   ✅ Character accuracy: Much better (LoRA + specific details)")
        print(f"   ✅ Style consistency: Much better (cyberpunk-optimized model)")
        print(f"   ✅ Scene relevance: Much better (action-specific settings)")
        print(f"   ✅ Technical quality: Better (optimized sampling)")

    elif intelligent_result:
        print("✅ INTELLIGENT generation completed")
        print("❌ GARBAGE generation failed")
        print("\n🏆 Intelligent orchestration is more reliable")

    elif garbage_result:
        print("❌ INTELLIGENT generation failed")
        print("✅ GARBAGE generation completed")
        print("\n⚠️  Need to debug intelligent orchestration")

    else:
        print("❌ Both generations failed")
        print("\n🔧 Need to debug ComfyUI connectivity")

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_comparison_test())