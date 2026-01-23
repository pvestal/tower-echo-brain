#!/usr/bin/env python3
"""
Anime content pipeline using local Ollama models + ComfyUI.
Run: python scripts/anime_pipeline.py "scene description"
"""
import sys
import json
import httpx
import asyncio
from pathlib import Path

OLLAMA_URL = "http://localhost:11434"
ANIME_API = "http://localhost:8328"  # Updated port based on service status
COMFYUI_URL = "http://localhost:8188"

# Model assignments
NARRATION_MODEL = "gemma2:9b"
REASONING_MODEL = "deepseek-r1:8b"

async def develop_scene(concept: str, style: str = "photorealistic") -> dict:
    """Use LLM to develop scene details."""
    prompt = f"""You are an anime director. Create a detailed scene breakdown.

CONCEPT: {concept}
STYLE: {style}

Return ONLY valid JSON (no markdown):
{{
    "scene_title": "Short title",
    "visual_description": "Detailed description of what we see",
    "camera": "Camera angle and movement",
    "lighting": "Lighting description",
    "mood": "Emotional tone",
    "comfyui_prompt": "Detailed prompt for image generation, include style tags",
    "negative_prompt": "What to avoid in generation"
}}"""

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": NARRATION_MODEL, "prompt": prompt, "stream": False}
        )
        raw = response.json().get("response", "{}")
        
        # Try to extract JSON from response
        try:
            # Handle markdown code blocks
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0]
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0]
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            return {"raw_response": raw, "error": "Failed to parse JSON"}

async def check_services() -> dict:
    """Check if required services are running."""
    status = {}
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            r = await client.get(f"{OLLAMA_URL}/api/tags")
            status["ollama"] = "running"
            status["models"] = [m["name"] for m in r.json().get("models", [])]
        except:
            status["ollama"] = "DOWN"
        
        # Try both anime API ports
        for port in [8328, 8305]:
            try:
                r = await client.get(f"http://localhost:{port}/health")
                status["anime_api"] = f"running on port {port}"
                break
            except:
                continue
        if "anime_api" not in status:
            status["anime_api"] = "DOWN"
        
        try:
            r = await client.get(f"{COMFYUI_URL}/system_stats")
            status["comfyui"] = "running"
        except:
            status["comfyui"] = "DOWN"
    
    return status

async def generate_image(prompt: str, negative: str = "", project: str = "tokyo_debt_desire") -> dict:
    """Submit generation to anime API."""
    async with httpx.AsyncClient(timeout=300.0) as client:
        # Try both ports
        for port in [8328, 8305]:
            try:
                response = await client.post(
                    f"http://localhost:{port}/api/anime/generate",
                    json={
                        "prompt": prompt,
                        "negative_prompt": negative,
                        "project_name": project,
                    }
                )
                return response.json()
            except Exception as e:
                continue
        return {"error": "Anime API not available"}

async def full_pipeline(concept: str, style: str = "photorealistic", generate: bool = False):
    """Run full anime pipeline: concept -> scene details -> (optional) image."""
    print("=" * 60)
    print(f"ANIME PIPELINE: {concept[:50]}...")
    print("=" * 60)
    
    # Check services
    print("\n[1/4] Checking services...")
    status = await check_services()
    print(json.dumps(status, indent=2))
    
    if status.get("ollama") == "DOWN":
        print("ERROR: Ollama not running!")
        return
    
    # Develop scene
    print(f"\n[2/4] Developing scene with {NARRATION_MODEL}...")
    scene = await develop_scene(concept, style)
    print(json.dumps(scene, indent=2))
    
    if "error" in scene:
        print(f"ERROR: {scene['error']}")
        return scene
    
    # Save scene data
    output_file = Path("/tmp/anime_scene.json")
    with open(output_file, "w") as f:
        json.dump(scene, f, indent=2)
    print(f"\n[3/4] Scene saved to {output_file}")
    
    # Generate image if requested
    if generate:
        if status.get("anime_api") == "DOWN":
            print("WARNING: Anime API down, skipping generation")
        else:
            print("\n[4/4] Submitting to ComfyUI...")
            result = await generate_image(
                scene.get("comfyui_prompt", concept),
                scene.get("negative_prompt", ""),
            )
            print(json.dumps(result, indent=2))
            return {"scene": scene, "generation": result}
    
    return scene

def main():
    if len(sys.argv) < 2:
        print("Usage: python anime_pipeline.py 'scene concept' [--generate]")
        print("\nExamples:")
        print('  python anime_pipeline.py "Mei walks through neon-lit streets"')
        print('  python anime_pipeline.py "Confrontation in the rain" --generate')
        sys.exit(1)
    
    concept = sys.argv[1]
    generate = "--generate" in sys.argv
    
    result = asyncio.run(full_pipeline(concept, generate=generate))
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
