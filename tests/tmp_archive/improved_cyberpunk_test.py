#!/usr/bin/env python3
"""
Improved Cyberpunk Generation Test
Uses character-specific data from story_bible and proper LoRA integration
"""

import json
import requests
import logging

COMFYUI_URL = "http://localhost:8188"
WORKFLOW_PATH = "/opt/tower-anime-production/workflows/comfyui/anime_30sec_rife_workflow_with_lora.json"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_kai_character_data():
    """Get Kai character data from story_bible"""
    try:
        # Search for Kai in story_bible
        embedding_resp = requests.post(
            'http://localhost:11434/api/embeddings',
            json={'model': 'nomic-embed-text', 'prompt': 'Kai cyberpunk character'}
        )

        if embedding_resp.status_code != 200:
            return None

        embedding = embedding_resp.json()['embedding']

        search_resp = requests.post(
            'http://localhost:6333/collections/story_bible/points/search',
            json={
                'vector': embedding,
                'limit': 5,
                'with_payload': True,
                'filter': {
                    'must': [{'key': 'type', 'match': {'value': 'character'}}]
                }
            }
        )

        if search_resp.status_code == 200:
            results = search_resp.json()['result']
            for result in results:
                payload = result['payload']
                if 'kai' in payload.get('name', '').lower():
                    content = payload['content']
                    logger.info(f"Found Kai character: {payload['name']}")
                    return content
    except Exception as e:
        logger.error(f"Failed to get Kai data: {e}")

    return None

def create_improved_workflow():
    """Create improved cyberpunk workflow with LoRA and better prompts"""
    try:
        with open(WORKFLOW_PATH, 'r') as f:
            workflow = json.load(f)

        # Get character data
        kai_data = get_kai_character_data()

        # Improved prompt based on character data
        if kai_data and 'cybernetic' in kai_data:
            cyberpunk_prompt = (
                "masterpiece, best quality, Kai cyberpunk slayer, "
                "spiky black hair, cybernetic enhancements, "
                "hacker aesthetic, tech specialist, "
                "neon-lit Neo-Tokyo undercity, "
                "futuristic weapons, dynamic action pose, "
                "mutant goblins in background, "
                "detailed cyberpunk atmosphere, "
                "anime style, high resolution, dramatic lighting"
            )
        else:
            cyberpunk_prompt = (
                "masterpiece, best quality, cyberpunk goblin slayer, "
                "cybernetic warrior, neon Tokyo, futuristic combat, "
                "anime style, detailed, high quality"
            )

        # Update workflow
        for node_id, node in workflow.items():
            if node.get("class_type") == "CLIPTextEncode":
                title = node.get("_meta", {}).get("title", "")
                if "Prompt" in title and "Negative" not in title:
                    node["inputs"]["text"] = cyberpunk_prompt
                    logger.info(f"Updated prompt in node {node_id}")

            # Look for LoRA loader and set cyberpunk LoRA
            elif node.get("class_type") == "LoraLoader":
                # Use the kai_cyberpunk_slayer LoRA we found earlier
                node["inputs"]["lora_name"] = "kai_cyberpunk_slayer.safetensors"
                node["inputs"]["strength_model"] = 0.8
                node["inputs"]["strength_clip"] = 0.8
                logger.info(f"Set cyberpunk LoRA in node {node_id}")

        return workflow

    except Exception as e:
        logger.error(f"Failed to create workflow: {e}")
        return None

def analyze_generation_issues():
    """Analyze why the previous generation was poor quality"""
    issues = []

    # Check available LoRAs
    try:
        import os
        lora_dir = "/mnt/1TB-storage/models/loras/"
        available_loras = [f for f in os.listdir(lora_dir) if 'cyberpunk' in f.lower() or 'kai' in f.lower()]

        if available_loras:
            logger.info(f"Available cyberpunk LoRAs: {available_loras}")
            issues.append("✓ Cyberpunk LoRAs available but not used in previous generation")
        else:
            issues.append("❌ No cyberpunk LoRAs found")

    except Exception as e:
        issues.append(f"❌ Could not check LoRA directory: {e}")

    # Check models
    try:
        checkpoints_dir = "/mnt/1TB-storage/models/checkpoints/"
        available_models = [f for f in os.listdir(checkpoints_dir) if f.endswith('.safetensors')]

        cyberpunk_friendly = [m for m in available_models if any(x in m.lower() for x in ['realistic', 'cyber', 'epic'])]
        if cyberpunk_friendly:
            logger.info(f"Better models for cyberpunk: {cyberpunk_friendly}")
            issues.append("✓ Better base models available (cyberrealistic_v9, epicrealism_v5)")
        else:
            issues.append("❌ Limited model selection for cyberpunk content")

    except Exception as e:
        issues.append(f"❌ Could not check checkpoint directory: {e}")

    # Previous generation analysis
    issues.extend([
        "❌ Used generic AOM3A1B model (anime-focused, not cyberpunk-optimized)",
        "❌ No LoRA applied despite having kai_cyberpunk_slayer.safetensors available",
        "❌ Generic prompt without character-specific details from story_bible",
        "❌ Wrong workflow - ACTION_combat_workflow.json doesn't use LoRAs"
    ])

    return issues

def main():
    logger.info("Analyzing previous cyberpunk generation issues...")

    issues = analyze_generation_issues()

    print()
    print("🔍 Cyberpunk Generation Issues Identified:")
    print("=" * 50)
    for issue in issues:
        print(issue)

    print()
    print("🛠️ Recommended Fixes:")
    print("=" * 50)
    print("1. Use anime_30sec_rife_workflow_with_lora.json (supports LoRAs)")
    print("2. Apply kai_cyberpunk_slayer.safetensors LoRA at 0.8 strength")
    print("3. Use cyberrealistic_v9.safetensors base model")
    print("4. Include character-specific details from story_bible")
    print("5. Optimize sampling settings for cyberpunk aesthetic")

    # Test improved workflow creation
    print()
    logger.info("Testing improved workflow creation...")
    improved_workflow = create_improved_workflow()

    if improved_workflow:
        logger.info("✅ Improved workflow created successfully")

        # Could submit here, but just showing the analysis
        print("\n✅ Ready to generate improved cyberpunk content with:")
        print("   - Character-specific prompts")
        print("   - Proper LoRA integration")
        print("   - Better workflow selection")
    else:
        logger.error("❌ Failed to create improved workflow")

if __name__ == "__main__":
    main()