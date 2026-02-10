#!/usr/bin/env python3
"""
Simple Cyberpunk Generation Test
=================================
Tests generation with a basic workflow that doesn't require AnimateDiff.
"""

import json
import time
import urllib.request

def test_simple_generation():
    # Use the ACTION_combat workflow which doesn't have RIFE or AnimateDiff
    workflow_path = "/opt/tower-anime-production/workflows/comfyui/ACTION_combat_workflow.json"

    with open(workflow_path) as f:
        workflow = json.load(f)

    # Patch the workflow with cyberpunk settings
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue

        ct = node.get("class_type", "")

        # Set checkpoint
        if ct == "CheckpointLoaderSimple":
            node["inputs"]["ckpt_name"] = "cyberrealistic_v9.safetensors"

        # Set positive prompt
        if ct == "CLIPTextEncode":
            title = node.get("_meta", {}).get("title", "").lower()
            if "positive" in title:
                node["inputs"]["text"] = (
                    "Kai Nakamura, cyberpunk warrior, male, spiky black hair, "
                    "cybernetic eye, fighting goblins, neon Tokyo alley, "
                    "action pose, dramatic lighting, rain, reflections, "
                    "masterpiece, best quality, detailed"
                )
            elif "negative" in title:
                node["inputs"]["text"] = (
                    "lowres, bad anatomy, bad hands, text, error, "
                    "missing fingers, worst quality, low quality"
                )

        # Set dimensions
        if ct == "EmptyLatentImage":
            node["inputs"]["width"] = 512
            node["inputs"]["height"] = 768
            node["inputs"]["batch_size"] = 1

        # Reduce steps for speed
        if ct == "KSampler":
            node["inputs"]["steps"] = 15

    # Submit to ComfyUI
    print("Submitting to ComfyUI...")
    data = json.dumps({"prompt": workflow}).encode()
    req = urllib.request.Request(
        "http://localhost:8188/prompt",
        data=data,
        headers={"Content-Type": "application/json"}
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode())
    except Exception as e:
        print(f"❌ Submission failed: {e}")
        return

    prompt_id = result.get("prompt_id")
    if not prompt_id:
        print(f"❌ ComfyUI rejected workflow: {result}")
        return

    print(f"✅ Submitted with prompt_id: {prompt_id}")
    print("Waiting for generation to complete...")

    # Poll for completion
    start = time.time()
    while time.time() - start < 120:
        try:
            req = urllib.request.Request(f"http://localhost:8188/history/{prompt_id}")
            with urllib.request.urlopen(req, timeout=5) as resp:
                history = json.loads(resp.read().decode())

            if prompt_id in history:
                status = history[prompt_id].get("status", {})
                if status.get("completed"):
                    elapsed = round(time.time() - start, 1)
                    outputs = history[prompt_id].get("outputs", {})

                    # Count images
                    image_count = 0
                    for node_outputs in outputs.values():
                        if isinstance(node_outputs, dict):
                            images = node_outputs.get("images", [])
                            image_count += len(images)

                    print(f"✅ Generation complete in {elapsed}s")
                    print(f"   Generated {image_count} image(s)")

                    # Show first image if any
                    for node_id, node_outputs in outputs.items():
                        if isinstance(node_outputs, dict):
                            images = node_outputs.get("images", [])
                            if images:
                                img = images[0]
                                print(f"   First image: {img.get('filename', 'unknown')}")
                                break
                    return

                elif status.get("status_str") == "error":
                    print(f"❌ Generation failed: {status}")
                    return

        except Exception:
            pass

        time.sleep(2)

    print("❌ Generation timed out after 120s")

if __name__ == "__main__":
    print("=" * 60)
    print("  CYBERPUNK GENERATION TEST")
    print("=" * 60)
    print()
    test_simple_generation()
    print()
    print("=" * 60)