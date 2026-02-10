#!/usr/bin/env python3
"""
Test Full SSOT Generation Pipeline
===================================
Tests the complete flow: Search → SSOT Fetch → Generate
"""

import json
import time
import urllib.request
import sys
import os

sys.path.append('/opt/tower-echo-brain/scripts')
from ssot_generation_orchestrator import SSOTOrchestrator

def test_cyberpunk_generation():
    print("=" * 60)
    print("  FULL PIPELINE TEST: Cyberpunk Kai")
    print("=" * 60)
    print()

    # Use the SSOT orchestrator
    orch = SSOTOrchestrator()

    # Plan generation
    print("1. Planning generation with SSOT orchestrator...")
    plan = orch.plan_generation("Generate Kai Nakamura fighting cyberpunk goblins in neon Tokyo")

    print(f"\n   Found {len(plan.references)} references from Qdrant")
    print(f"   Fetched {len(plan.fresh_data)} fresh records from PostgreSQL")
    print(f"   Selected: {plan.resources.checkpoint}")
    print(f"   LoRAs: {[l['name'] for l in plan.resources.loras]}")

    # Override to use our new production workflow
    plan.resources.workflow_file = "cyberpunk_character_production.json"

    # Execute
    print("\n2. Submitting to ComfyUI...")
    result = orch.execute(plan)

    if "error" in result:
        print(f"   ❌ Error: {result['error']}")
        return False
    else:
        print(f"   ✅ Generated in {result.get('elapsed_seconds', '?')}s")
        print(f"   Images: {result.get('images', [])}")
        print(f"   SSOT sources: {result.get('ssot_sources', [])}")
        return True

def test_simple_anime():
    print("\n" + "=" * 60)
    print("  SIMPLE ANIME TEST")
    print("=" * 60)
    print()

    # Load simple workflow
    workflow_path = "/opt/tower-anime-production/workflows/comfyui/anime_character_simple.json"
    with open(workflow_path) as f:
        workflow = json.load(f)

    # Set a test prompt
    for node in workflow.values():
        if isinstance(node, dict) and node.get("class_type") == "CLIPTextEncode":
            if "positive" in node.get("_meta", {}).get("title", "").lower():
                node["inputs"]["text"] = "Mei Kobayashi, anime girl, pink hair, school uniform, cherry blossoms, spring, happy, smiling, detailed, masterpiece"

    # Submit
    print("Submitting simple anime workflow...")
    req = urllib.request.Request(
        "http://localhost:8188/prompt",
        data=json.dumps({"prompt": workflow}).encode(),
        headers={"Content-Type": "application/json"}
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode())
    except Exception as e:
        print(f"   ❌ Submission failed: {e}")
        return False

    prompt_id = result.get("prompt_id")
    if not prompt_id:
        print(f"   ❌ ComfyUI rejected workflow")
        return False

    print(f"   Submitted: {prompt_id}")

    # Wait for completion
    for i in range(30):
        time.sleep(2)
        req = urllib.request.Request(f"http://localhost:8188/history/{prompt_id}")
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                history = json.loads(resp.read().decode())
        except:
            continue

        if prompt_id in history:
            status = history[prompt_id].get("status", {})
            if status.get("completed"):
                print(f"   ✅ Generated successfully!")
                return True
            elif status.get("status_str") == "error":
                print(f"   ❌ Generation failed: {status}")
                return False

    print("   ❌ Timed out")
    return False

if __name__ == "__main__":
    # Test both pipelines
    results = []

    # Test SSOT pipeline with cyberpunk
    try:
        results.append(("SSOT Cyberpunk", test_cyberpunk_generation()))
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        results.append(("SSOT Cyberpunk", False))

    # Test simple anime workflow
    try:
        results.append(("Simple Anime", test_simple_anime()))
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        results.append(("Simple Anime", False))

    # Summary
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}: {name}")
    print("=" * 60)