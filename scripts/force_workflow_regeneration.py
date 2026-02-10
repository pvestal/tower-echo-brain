#!/usr/bin/env python3
"""
Force Workflow Regeneration
============================
Modifies workflows to bypass ComfyUI's execution cache.
The cache issue causes workflows to return 0 outputs.

Usage: python3 force_workflow_regeneration.py <workflow.json>
"""

import json
import random
import sys
import time

def force_regenerate(workflow):
    """Modify workflow to bypass cache"""

    modified = False

    # Change seed in all KSampler nodes
    for node_id, node in workflow.items():
        if isinstance(node, dict):
            if node.get("class_type") == "KSampler":
                # Force new seed
                old_seed = node["inputs"].get("seed", 0)
                node["inputs"]["seed"] = random.randint(0, 2**31)
                modified = True
                print(f"  🎲 Changed KSampler seed: {old_seed} → {node['inputs']['seed']}")

            # Modify AnimateDiff seeds
            if node.get("class_type") == "AnimateDiffUniformContextOptions":
                if "seed_gen" in node.get("inputs", {}):
                    node["inputs"]["seed_gen"] = random.randint(0, 2**31)
                    modified = True

            # Add timestamp to positive prompts
            if node.get("class_type") == "CLIPTextEncode":
                title = node.get("_meta", {}).get("title", "").lower()
                if "positive" in title or ("negative" not in title and "prompt" in title):
                    text = node["inputs"].get("text", "")
                    # Add invisible timestamp comment
                    timestamp = int(time.time())
                    node["inputs"]["text"] = f"{text}\n<!-- regen_{timestamp} -->"
                    modified = True
                    print(f"  📝 Modified prompt to force regeneration")

    if not modified:
        # Fallback: modify latent batch size slightly
        for node_id, node in workflow.items():
            if isinstance(node, dict):
                if node.get("class_type") == "EmptyLatentImage":
                    # Toggle batch size between current and current+1 to invalidate cache
                    current = node["inputs"].get("batch_size", 1)
                    # For video workflows, ensure we have enough frames
                    if current > 1:
                        node["inputs"]["batch_size"] = current  # Keep same for video
                    else:
                        node["inputs"]["batch_size"] = 1  # Ensure at least 1
                    # But change dimensions slightly to invalidate cache
                    node["inputs"]["width"] = 512
                    node["inputs"]["height"] = 512
                    modified = True
                    print(f"  🖼️  Reset dimensions to 512x512 to invalidate cache")
                    break

    return workflow, modified

def process_workflow(workflow_path):
    """Load, modify, and save workflow"""

    print(f"\n📄 Processing {workflow_path}")

    try:
        with open(workflow_path, 'r') as f:
            workflow = json.load(f)

        workflow, was_modified = force_regenerate(workflow)

        if was_modified:
            # Save modified workflow
            output_path = workflow_path.replace('.json', '_regenerate.json')
            with open(output_path, 'w') as f:
                json.dump(workflow, f, indent=2)

            print(f"  ✅ Saved to {output_path}")
            return output_path
        else:
            print(f"  ⚠️  No modifications made")
            return None

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Process the problematic ACTION_combat_workflow
        default_workflow = "/opt/tower-anime-production/workflows/comfyui/ACTION_combat_workflow.json"
        print(f"No workflow specified, using default: {default_workflow}")
        process_workflow(default_workflow)
    else:
        for workflow_path in sys.argv[1:]:
            process_workflow(workflow_path)