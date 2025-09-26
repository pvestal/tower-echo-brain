#!/usr/bin/env python3
"""Echo Brain with Video Generation"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import json
import os
import subprocess
import time
from echo_video_module import EchoVideoGenerator

app = FastAPI()

class VideoRequest(BaseModel):
    prompt: str
    character: str = None
    duration: int = 30

@app.get("/api/echo/health")
async def health():
    return {"status": "healthy", "service": "Echo Brain with Video"}

@app.post("/api/echo/generate_video")
async def generate_video(request: VideoRequest):
    """Generate video based on prompt"""
    
    try:
        generator = EchoVideoGenerator()
        
        # Use FFmpeg method for now (it works)
        if request.character == "goblin slayer":
            base_image = "/home/patrick/ComfyUI/output/echo_goblin_slayer_cyberpunk_00001_.png"
        else:
            # Generate new image first
            base_image = generator._generate_base_image(request.prompt, request.character)
        
        if base_image and os.path.exists(base_image):
            video_path = generator._generate_ffmpeg_video(base_image, request.duration)
            
            if video_path:
                return {
                    "status": "success",
                    "video_path": video_path,
                    "duration": request.duration,
                    "message": f"Generated {request.duration}s video"
                }
        
        return {"status": "error", "message": "Failed to generate video"}
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/echo/chat")
async def chat(request: dict):
    """Basic chat endpoint"""
    message = request.get("message", "")
    
    if "video" in message.lower() or "generate" in message.lower():
        return {
            "response": "I can generate videos! Use /api/echo/generate_video endpoint.",
            "capabilities": ["video_generation", "image_generation", "chat"]
        }
    
    return {"response": f"Echo heard: {message}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8309)
