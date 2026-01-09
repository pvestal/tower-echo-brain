#!/usr/bin/env python3
"""
Echo Brain - Goblin Slayer Cyberpunk Generation
"""

import asyncio
import json
import logging
import aiohttp
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import random

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Goblin Slayer cyberpunk character definition
GOBLIN_SLAYER_CYBERPUNK = {
    "name": "Goblin Slayer",
    "style": "cyberpunk",
    "description": "armored warrior, cyberpunk style, futuristic armor, glowing red visor, tactical gear, neon lights, dystopian city background, night scene, rain, blade runner aesthetic",
    "negative": "fantasy, medieval, goblins, cute, kawaii, magical girl, pink, flowers"
}

class GenerateRequest(BaseModel):
    prompt: str
    character: Optional[str] = "goblin_slayer"
    style: Optional[str] = "cyberpunk"
    quality: Optional[str] = "production"

@app.post("/api/echo/generate")
async def generate_with_echo(request: GenerateRequest):
    """Generate image through Echo Brain"""
    
    logger.info(f"Echo Brain generating: {request.prompt}")
    
    # Build prompt based on character
    if "goblin" in request.character.lower() or "slayer" in request.prompt.lower():
        character_prompt = GOBLIN_SLAYER_CYBERPUNK["description"]
        negative_prompt = GOBLIN_SLAYER_CYBERPUNK["negative"]
    else:
        # Default to cyberpunk style
        character_prompt = f"{request.character}, cyberpunk style, futuristic"
        negative_prompt = "cute, kawaii, fantasy, medieval"
    
    # Build complete prompt
    full_prompt = f"{character_prompt}, {request.prompt}, masterpiece, best quality, ultra detailed"
    
    # Create Article 71 compliant workflow
    workflow = {
        "3": {
            "inputs": {
                "seed": random.randint(1, 999999),
                "steps": 40,  # Production quality
                "cfg": 10.0,
                "sampler_name": "dpmpp_2m_sde",
                "scheduler": "karras",
                "denoise": 1.0,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0]
            },
            "class_type": "KSampler"
        },
        "4": {
            "inputs": {"ckpt_name": "deliberate_v2.safetensors"},  # Good for cyberpunk
            "class_type": "CheckpointLoaderSimple"
        },
        "5": {
            "inputs": {"width": 1920, "height": 1080, "batch_size": 1},
            "class_type": "EmptyLatentImage"
        },
        "6": {
            "inputs": {
                "text": full_prompt,
                "clip": ["4", 1]
            },
            "class_type": "CLIPTextEncode"
        },
        "7": {
            "inputs": {
                "text": negative_prompt,
                "clip": ["4", 1]
            },
            "class_type": "CLIPTextEncode"
        },
        "8": {
            "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
            "class_type": "VAEDecode"
        },
        "9": {
            "inputs": {
                "images": ["8", 0],
                "filename_prefix": f"echo_goblin_slayer_{request.style}"
            },
            "class_type": "SaveImage"
        }
    }
    
    # Submit to ComfyUI
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8188/api/prompt",
            json={"prompt": workflow}
        ) as resp:
            if resp.status != 200:
                raise HTTPException(status_code=500, detail="ComfyUI error")
            
            result = await resp.json()
            prompt_id = result.get("prompt_id")
    
    return {
        "success": True,
        "message": "Echo Brain generating Goblin Slayer cyberpunk",
        "prompt_id": prompt_id,
        "character": "Goblin Slayer",
        "style": "cyberpunk",
        "quality": "production (40 steps, 10.0 CFG)"
    }

@app.get("/api/echo/status")
async def get_status():
    """Echo Brain status"""
    return {
        "status": "online",
        "service": "Echo Brain - Goblin Slayer Edition",
        "character": "Goblin Slayer Cyberpunk",
        "ready": True
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8309)
