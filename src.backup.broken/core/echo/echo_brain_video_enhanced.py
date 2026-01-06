#!/usr/bin/env python3
"""
Echo Brain - Enhanced with Video Generation Capabilities
Professional-grade anime video generation with quality assessment
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List
import asyncio
import json
import time
from pathlib import Path

# Import our video generation system
import sys
sys.path.append('/opt/tower-echo-brain')
from src.modules.generation.video.video_generation_system import VideoGenerationSystem, EchoBrainVideoIntegration

app = FastAPI(title="Echo Brain Video API", version="2.0")

# Initialize systems
video_system = VideoGenerationSystem()
echo_integration = EchoBrainVideoIntegration()

class VideoRequest(BaseModel):
    prompt: str
    duration: Optional[int] = 30
    style: Optional[str] = "anime"
    quality_preset: Optional[str] = "high"
    method: Optional[str] = "auto"

class QualityCheckRequest(BaseModel):
    video_path: str

class VideoResponse(BaseModel):
    success: bool
    video_path: Optional[str]
    quality_score: Optional[int]
    quality_feedback: Optional[List[str]]
    generation_time: Optional[float]
    method_used: Optional[str]

@app.get("/api/echo/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Echo Brain Video Generation",
        "capabilities": ["animatediff", "svd", "ffmpeg_effects", "quality_assessment"]
    }

@app.post("/api/echo/generate_video", response_model=VideoResponse)
async def generate_video(request: VideoRequest):
    """Generate high-quality anime video"""
    start_time = time.time()
    
    try:
        # Log the request
        print(f"Generating video: {request.prompt}")
        
        # Generate video
        result = video_system.generate_video(
            prompt=request.prompt,
            method=request.method,
            duration=request.duration
        )
        
        generation_time = time.time() - start_time
        
        # Prepare response
        response = VideoResponse(
            success=result['success'],
            video_path=result.get('video_path'),
            quality_score=result.get('quality_assessment', {}).get('score', 0),
            quality_feedback=result.get('quality_assessment', {}).get('feedback', []),
            generation_time=generation_time,
            method_used=result.get('method_used')
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/echo/check_quality")
async def check_video_quality(request: QualityCheckRequest):
    """Check quality of existing video"""
    try:
        quality_result = video_system.assess_quality(request.video_path)
        return quality_result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/echo/generation_status")
async def get_generation_status():
    """Get current generation status and available models"""
    import os
    
    models_status = {
        "animatediff": os.path.exists("/home/{os.getenv("TOWER_USER", "patrick")}/Projects/ComfyUI-Working/custom_nodes/ComfyUI-AnimateDiff-Evolved/models/mm_sd_v15_v3.ckpt"),
        "svd": os.path.exists("/home/{os.getenv("TOWER_USER", "patrick")}/Projects/ComfyUI-Working/models/checkpoints/svd_xt.safetensors"),
        "upscaling": os.path.exists("/home/{os.getenv("TOWER_USER", "patrick")}/Projects/ComfyUI-Working/models/upscale_models/4x-UltraSharp.pth"),
        "comfyui": True  # Assume ComfyUI is running
    }
    
    return {
        "models_available": models_status,
        "recommended_method": "animatediff" if models_status["animatediff"] else "ffmpeg_effects",
        "quality_standards": {
            "resolution": "1920x1080 minimum",
            "fps": "24fps minimum",
            "duration": "30 seconds for trailers",
            "motion": "Smooth, not slideshow"
        }
    }

@app.post("/api/echo/batch_generate")
async def batch_generate_videos(requests: List[VideoRequest]):
    """Generate multiple videos in batch"""
    results = []
    
    for req in requests:
        try:
            result = await generate_video(req)
            results.append(result)
        except Exception as e:
            results.append({
                "success": False,
                "error": str(e),
                "prompt": req.prompt
            })
    
    return {"batch_results": results, "total": len(requests)}

@app.get("/api/echo/video_tips")
async def get_video_generation_tips():
    """Get tips for better video generation"""
    return {
        "prompt_tips": [
            "Include 'masterpiece, best quality' for better results",
            "Specify style: 'anime style', 'studio ghibli style', etc.",
            "Add motion descriptions: 'walking', 'dancing', 'fighting'",
            "Include scene details: 'cherry blossoms', 'cyberpunk city'"
        ],
        "quality_tips": [
            "Use AnimateDiff for motion-heavy scenes",
            "Use SVD for converting single images to video",
            "Generate at lower resolution then upscale for efficiency",
            "Test with short clips before long videos"
        ],
        "available_styles": [
            "anime", "realistic", "cyberpunk", "fantasy", "ghibli"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting Echo Brain Video Generation API...")
    uvicorn.run(app, host="0.0.0.0", port=8309)
