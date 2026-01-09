#!/usr/bin/env python3
"""Echo Brain - Integrated with Creative Director and ComfyUI"""
import os
import sys
import json
import time
import asyncio
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

# Add the modules to path
sys.path.insert(0, '/home/{os.getenv("TOWER_USER", "patrick")}/Documents')

app = FastAPI()

# Story tracking
current_stories = {}

class EchoIntegrated:
    def __init__(self):
        self.comfyui_url = "http://localhost:8188"
        
    def generate_image(self, prompt):
        """Actually generate an image using ComfyUI"""
        workflow = {
            "3": {"class_type": "KSampler",
                  "inputs": {"cfg": 7.5, "denoise": 1, "latent_image": ["5", 0],
                           "model": ["4", 0], "negative": ["7", 0], "positive": ["6", 0],
                           "sampler_name": "euler", "scheduler": "normal",
                           "seed": int(time.time()), "steps": 20}},
            "4": {"class_type": "CheckpointLoaderSimple",
                  "inputs": {"ckpt_name": "deliberate_v2.safetensors"}},
            "5": {"class_type": "EmptyLatentImage",
                  "inputs": {"batch_size": 1, "height": 512, "width": 512}},
            "6": {"class_type": "CLIPTextEncode",
                  "inputs": {"clip": ["4", 1], "text": prompt}},
            "7": {"class_type": "CLIPTextEncode",
                  "inputs": {"clip": ["4", 1], "text": "blurry, bad quality"}},
            "8": {"class_type": "VAEDecode",
                  "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
            "9": {"class_type": "SaveImage",
                  "inputs": {"filename_prefix": "echo_generated", "images": ["8", 0]}}
        }
        
        try:
            response = requests.post(f"{self.comfyui_url}/prompt", 
                                    json={"prompt": workflow}, timeout=10)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return None

echo = EchoIntegrated()

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "Echo Brain Integrated"}

@app.post("/process")
async def process(request: Request):
    """Main Echo processing with Creative Director mode"""
    data = await request.json()
    query = data.get("query", "").lower()
    user_id = data.get("user_id", "default")
    
    # Check for creative/video keywords
    creative_words = ["video", "anime", "story", "character", "movie", "film", "scene"]
    if any(word in query for word in creative_words):
        # Creative Director Mode
        if user_id not in current_stories:
            current_stories[user_id] = {
                "character_name": None,
                "setting": None,
                "genre": None,
                "phase": "concept"
            }
        
        story = current_stories[user_id]
        response_text = ""
        questions = []
        
        # Parse user input
        if "name is" in query or "named" in query:
            for word in query.split():
                if word[0].isupper() and len(word) > 2:
                    story["character_name"] = word
                    response_text = f"Great name! {word} sounds interesting. "
                    break
        
        if any(loc in query for loc in ["tokyo", "city", "space", "forest", "school"]):
            story["setting"] = query.split()[-1]
            response_text += f"Love the {story['setting']} setting! "
        
        # Ask next questions based on what we need
        if not story["character_name"]:
            questions.append("What's your main character's name?")
        if not story["setting"]:
            questions.append("Where does the story take place?")
        if story["character_name"] and story["setting"]:
            questions.append("What's the main conflict or challenge?")
            
            # Try to generate an image
            prompt = f"anime character {story['character_name']} in {story['setting']}, detailed"
            result = echo.generate_image(prompt)
            if result:
                response_text += f"🎨 Generating image for {story['character_name']}..."
        
        return {
            "response": response_text + " ".join(questions),
            "mode": "creative_director",
            "story_progress": story,
            "next_questions": questions
        }
    
    # Normal mode
    return {"response": f"Echo processed: {query}", "mode": "normal"}

if __name__ == "__main__":
    print("🚀 Starting Integrated Echo Brain...")
    uvicorn.run(app, host="0.0.0.0", port=8309)
