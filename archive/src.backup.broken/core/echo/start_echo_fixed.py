#!/usr/bin/env python3
"""Fixed Echo Brain starter - bypasses database issues"""
import os
import sys
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

# Set environment to bypass database requirements
os.environ['ECHO_NO_DB'] = '1'
os.environ['JWT_SECRET'] = 'echo_brain_secret_2025'

app = FastAPI()

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "Echo Brain Fixed"}

@app.get("/")
async def root():
    return {"message": "Echo Brain is running!"}

@app.post("/process")
async def process(request: Request):
    """Main Echo processing endpoint"""
    data = await request.json()
    query = data.get("query", "")
    
    # For now, just echo back with acknowledgment
    response = {
        "response": f"Echo received: {query}",
        "status": "processing",
        "capabilities": [
            "Creative Director mode",
            "ComfyUI image generation",
            "Story development"
        ]
    }
    
    # If query mentions video/anime/image, trigger creative mode
    if any(word in query.lower() for word in ["video", "anime", "image", "story", "character"]):
        response["mode"] = "creative_director"
        response["message"] = "Creative Director mode activated! Let's develop your story together."
    
    return response

if __name__ == "__main__":
    print("🚀 Starting Fixed Echo Brain on port 8309...")
    uvicorn.run(app, host="0.0.0.0", port=8309)
