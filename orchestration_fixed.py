#!/usr/bin/env python3
import asyncio
import httpx
import logging
from datetime import datetime
from typing import Dict, Any

class EchoOrchestrator:
    def __init__(self):
        self.services = {
            "comfyui": {"url": "http://127.0.0.1:8188", "health": "/api/system_stats"},
            "voice": {"url": "http://127.0.0.1:8312", "health": "/api/health"},  
            "kb": {"url": "http://127.0.0.1:8307", "health": "/"},
        }
        self.auth_tokens = {}
        
    async def test_service_connectivity(self) -> Dict[str, bool]:
        results = {}
        async with httpx.AsyncClient() as client:
            for name, config in self.services.items():
                try:
                    response = await client.get(f"{config[\"url\"]}{config[\"health\"]}", timeout=5.0)
                    results[name] = response.status_code == 200
                    print(f"{name}: {\"OK\" if results[name] else \"FAIL\"}")
                except Exception as e:
                    results[name] = False
                    print(f"{name}: ERROR - {e}")
        return results
    
    async def get_voice_token(self):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(f"{self.services[\"voice\"][\"url\"]}/api/auth/token", 
                    json={"username": "echo", "purpose": "orchestration"}, timeout=5.0)
                if response.status_code == 200:
                    data = response.json()
                    return data.get("access_token")
        except Exception as e:
            print(f"Auth error: {e}")
        return None
        
    async def generate_voice(self, text: str):
        token = await self.get_voice_token()
        if not token:
            return {"success": False, "error": "No auth token"}
            
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(f"{self.services[\"voice\"][\"url\"]}/api/tts",
                    json={"text": text, "voice": "echo_default"},
                    headers={"Authorization": f"Bearer {token}"}, timeout=15.0)
                return {"success": response.status_code == 200, "response": response.json()}
        except Exception as e:
            return {"success": False, "error": str(e)}

async def main():
    orch = EchoOrchestrator()
    print("=== SERVICE CONNECTIVITY ===")
    await orch.test_service_connectivity()
    print("\\n=== VOICE AUTH TEST ===")
    token = await orch.get_voice_token()
    print(f"Token obtained: {token is not None}")
    print("\\n=== VOICE GENERATION TEST ===")
    result = await orch.generate_voice("Hello, this is Echo testing orchestration")
    print(f"Voice test result: {result}")

if __name__ == "__main__":
    asyncio.run(main())
