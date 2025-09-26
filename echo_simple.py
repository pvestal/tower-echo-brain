#!/usr/bin/env python3
"""Simple Echo Brain with basic video support"""

from fastapi import FastAPI
import uvicorn
import os
import subprocess
import time

app = FastAPI()

@app.get("/api/echo/health")
async def health():
    return {"status": "healthy", "service": "Echo Brain"}

@app.post("/api/echo/generate_video")
async def generate_video(request: dict):
    """Simple video generation using FFmpeg"""
    
    prompt = request.get("prompt", "")
    duration = request.get("duration", 30)
    
    # Use existing Goblin Slayer image
    base_image = "/home/patrick/ComfyUI/output/echo_goblin_slayer_cyberpunk_00001_.png"
    
    if os.path.exists(base_image):
        output_path = f"***REMOVED***/echo_generated_{int(time.time())}.mp4"
        
        cmd = f'''ffmpeg -loop 1 -i {base_image} -c:v libx264 -t {duration} \
        -pix_fmt yuv420p -vf "scale=1920:1080,zoompan=z='min(zoom+0.0015,1.5)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={duration*24}:s=1920x1080:fps=24" \
        -preset fast -crf 23 {output_path} -y'''
        
        result = subprocess.run(cmd, shell=True, capture_output=True)
        
        if result.returncode == 0:
            return {
                "status": "success",
                "video_path": output_path,
                "duration": duration,
                "message": f"Generated {duration}s video"
            }
    
    return {"status": "error", "message": "Failed to generate video"}

@app.post("/api/echo/chat")
async def chat(request: dict):
    return {"response": "Echo Brain is working with video generation!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8309)
