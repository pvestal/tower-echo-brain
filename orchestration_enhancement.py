#!/usr/bin/env python3
"""
Echo Brain Real Orchestration Enhancement
Based on KB Article #144 - Cognitive Architecture
Based on KB Article #145 - Resilience Architecture
"""

import asyncio
import httpx
import json
import subprocess
import logging
from datetime import datetime
from typing import Dict, Any, Optional

class EchoOrchestrator:
    """Real service orchestration for Echo Brain"""
    
    def __init__(self):
        # Correct service endpoints from our discovery
        self.services = {
            "comfyui": {"url": "http://127.0.0.1:8188", "health": "/api/system_stats"},
            "voice": {"url": "http://127.0.0.1:8312", "health": "/api/health"},
            "kb": {"url": "http://127.0.0.1:8307", "health": "/"},
            "anime": {"url": "http://127.0.0.1:8328", "health": "/health"},
            "music": {"url": "http://127.0.0.1:8315", "health": "/health"}
        }
        self.auth_tokens = {}
        self.logger = logging.getLogger(__name__)
        
    async def get_voice_token(self) -> str:
        """Get authentication token for voice service"""
        if "voice" not in self.auth_tokens:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.services[voice][url]}/api/auth/token",
                        json={"username": "echo", "purpose": "orchestration"}
                    )
                    if response.status_code == 200:
                        data = response.json()
                        self.auth_tokens["voice"] = data.get("access_token")
                    else:
                        self.logger.error(f"Voice auth failed: {response.status_code}")
                        return None
            except Exception as e:
                self.logger.error(f"Voice auth error: {e}")
                return None
        return self.auth_tokens.get("voice")
    
    async def test_service_connectivity(self) -> Dict[str, bool]:
        """Test all service endpoints"""
        results = {}
        async with httpx.AsyncClient() as client:
            for name, config in self.services.items():
                try:
                    response = await client.get(
                        f"{config[url]}{config[health]}", 
                        timeout=5.0
                    )
                    results[name] = response.status_code == 200
                except Exception as e:
                    self.logger.error(f"Service {name} failed: {e}")
                    results[name] = False
        return results
    
    async def generate_image(self, prompt: str) -> Dict[str, Any]:
        """Generate image using ComfyUI"""
        try:
            async with httpx.AsyncClient() as client:
                # Simple workflow for ComfyUI
                workflow = {
                    "prompt": {
                        "1": {
                            "inputs": {"text": prompt},
                            "class_type": "CLIPTextEncode"
                        }
                    }
                }
                
                response = await client.post(
                    f"{self.services[comfyui][url]}/api/prompt",
                    json=workflow,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    return {
                        "success": True,
                        "message": f"✅ Image generated: {prompt[:50]}...",
                        "service": "comfyui",
                        "response": response.json()
                    }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "service": "comfyui"
            }
    
    async def generate_voice(self, text: str, voice: str = "echo_default") -> Dict[str, Any]:
        """Generate voice using authenticated voice service"""
        try:
            token = await self.get_voice_token()
            if not token:
                return {"success": False, "error": "Authentication failed"}
                
            async with httpx.AsyncClient() as client:
                headers = {"Authorization": f"Bearer {token}"}
                response = await client.post(
                    f"{self.services[voice][url]}/api/tts",
                    json={"text": text, "voice": voice},
                    headers=headers,
                    timeout=15.0
                )
                
                if response.status_code == 200:
                    return {
                        "success": True,
                        "message": f"✅ Voice generated: {text[:50]}...",
                        "service": "voice",
                        "voice": voice
                    }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "service": "voice"
            }
    
    async def orchestrate_content_creation(self, request: str) -> Dict[str, Any]:
        """Orchestrate multi-service content creation"""
        req_lower = request.lower()
        results = []
        
        # Test connectivity first
        connectivity = await self.test_service_connectivity()
        available_services = [k for k, v in connectivity.items() if v]
        
        orchestration_result = {
            "request": request,
            "timestamp": datetime.now().isoformat(),
            "available_services": available_services,
            "actions_taken": []
        }
        
        # Image generation
        if any(word in req_lower for word in ["image", "picture", "visual", "generate", "create"]):
            if "comfyui" in available_services:
                image_result = await self.generate_image(request)
                orchestration_result["actions_taken"].append(image_result)
        
        # Voice generation
        if any(word in req_lower for word in ["voice", "speak", "say", "narrate"]):
            if "voice" in available_services:
                voice_result = await self.generate_voice(request)
                orchestration_result["actions_taken"].append(voice_result)
        
        # Trailer creation (combines multiple services)
        if any(word in req_lower for word in ["trailer", "video", "movie"]):
            if "comfyui" in available_services:
                image_result = await self.generate_image(f"Movie trailer scene: {request}")
                orchestration_result["actions_taken"].append(image_result)
            
            if "voice" in available_services:
                voice_result = await self.generate_voice(f"Coming soon: {request}")
                orchestration_result["actions_taken"].append(voice_result)
        
        return orchestration_result

# Integration function for Echo Brain
async def handle_orchestration_request(message: str) -> Dict[str, Any]:
    """Main function to handle orchestration requests from Echo Brain"""
    orchestrator = EchoOrchestrator()
    return await orchestrator.orchestrate_content_creation(message)

