#!/usr/bin/env python3
"""
Test Echo Brain's anime production integration
"""

import asyncio
import httpx
import json
from datetime import datetime

async def test_echo_brain_anime():
    """Test all anime endpoints through Echo Brain"""
    
    base_url = "http://localhost:8309/api/echo/anime"
    
    async with httpx.AsyncClient() as client:
        print("🧠 Testing Echo Brain Anime Integration")
        print("=" * 50)
        
        # 1. Check status
        print("\n✅ Testing /status endpoint...")
        resp = await client.get(f"{base_url}/status")
        if resp.status_code == 200:
            data = resp.json()
            print(f"  Enabled: {data['enabled']}")
            print(f"  Total Scenes: {data['stats']['total_scenes']}")
            print(f"  Available LoRAs: {len(data['loras'])}")
        
        # 2. Test scene interpretation
        print("\n✅ Testing scene interpretation...")
        resp = await client.post(
            f"{base_url}/interpret",
            params={"prompt": "epic martial arts fight scene"}
        )
        if resp.status_code == 200:
            data = resp.json()
            print(f"  Input: {data['prompt']}")
            print(f"  Interpreted: {data['interpreted_type']}")
        
        # 3. Generate a test scene
        print("\n✅ Testing scene generation...")
        resp = await client.post(
            f"{base_url}/generate",
            json={
                "prompt": "Mei transforming into magical girl with sparkles",
                "episode_id": 99  # Test episode
            }
        )
        if resp.status_code == 200:
            data = resp.json()
            print(f"  Prompt ID: {data['prompt_id']}")
            print(f"  Scene Type: {data['echo_brain']['interpreted_type']}")
            print(f"  Character: {data['echo_brain']['detected_character']}")
            print(f"  Status: {data['status']}")
        
        # 4. Get recent generations
        print("\n✅ Testing recent generations...")
        resp = await client.get(f"{base_url}/recent/3")
        if resp.status_code == 200:
            data = resp.json()
            print(f"  Found {len(data)} recent generations")
            for gen in data:
                print(f"    - Episode {gen['episode_id']}, Scene {gen['scene_number']}: {gen['scene_type']}")
        
        print("\n" + "=" * 50)
        print("✅ Echo Brain anime integration test complete!")

if __name__ == "__main__":
    asyncio.run(test_echo_brain_anime())
