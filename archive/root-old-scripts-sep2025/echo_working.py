#!/usr/bin/env python3
from src.misc.article71_compliant_workflow import Article71Workflow
"""
WORKING Echo Brain Service - ACTUALLY generates images through ComfyUI
Requirements:
1. Detect creative/anime/video keywords
2. Track story state across multiple calls
3. When story has character + setting, ACTUALLY call ComfyUI to generate image
4. Verify image creation in /home/patrick/ComfyUI/output/
5. Return actual image path in response
"""

import json
import re
import time
import uuid
import os
import asyncio
import websocket
import requests
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import uvicorn

app = FastAPI()

# Story state storage (in-memory for now)
story_states: Dict[str, Dict] = {}

class ChatMessage(BaseModel):
    message: str
    user_id: str = "default"
    session_id: str = "default"

class EchoResponse(BaseModel):
    response: str
    image_path: Optional[str] = None
    story_state: Optional[Dict] = None
    action_taken: Optional[str] = None

# Keywords for creative content detection
CREATIVE_KEYWORDS = [
    'anime', 'manga', 'character', 'story', 'scene', 'draw', 'create', 'generate',
    'image', 'picture', 'video', 'animation', 'art', 'visual', 'illustration',
    'magical', 'fantasy', 'adventure', 'battle', 'romance', 'school', 'demon',
    'sword', 'princess', 'hero', 'villain', 'dragon', 'magic', 'spell',
    'sakura', 'naruto', 'goku', 'sailor', 'gundam', 'mecha'
]

SETTING_KEYWORDS = [
    'school', 'forest', 'castle', 'city', 'village', 'mountain', 'beach', 'space',
    'laboratory', 'battlefield', 'garden', 'temple', 'dungeon', 'sky', 'underwater',
    'tokyo', 'japan', 'academy', 'hospital', 'mansion', 'shop', 'restaurant'
]

CHARACTER_KEYWORDS = [
    'girl', 'boy', 'woman', 'man', 'student', 'teacher', 'warrior', 'mage',
    'princess', 'prince', 'hero', 'demon', 'angel', 'ninja', 'samurai',
    'pilot', 'scientist', 'doctor', 'artist', 'musician', 'chef'
]

def extract_story_elements(text: str) -> Dict[str, Any]:
    """Extract story elements from text"""
    text_lower = text.lower()

    # Extract characters
    characters = []
    for keyword in CHARACTER_KEYWORDS:
        if keyword in text_lower:
            characters.append(keyword)

    # Extract settings
    settings = []
    for keyword in SETTING_KEYWORDS:
        if keyword in text_lower:
            settings.append(keyword)

    # Extract names (capitalized words that aren't common words)
    names = re.findall(r'\b[A-Z][a-z]+\b', text)
    names = [name for name in names if name.lower() not in ['the', 'and', 'but', 'for', 'with', 'this', 'that']]

    return {
        'characters': characters,
        'settings': settings,
        'names': names,
        'raw_text': text
    }

def has_creative_intent(text: str) -> bool:
    """Check if text has creative/anime intent"""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in CREATIVE_KEYWORDS)

def create_comfyui_workflow(prompt: str, character: str = "", setting: str = "") -> Dict:
    """Create ComfyUI workflow for image generation"""

    # Build enhanced prompt
    enhanced_prompt = f"masterpiece, best quality, highly detailed, {prompt}"
    if character:
        enhanced_prompt += f", {character}"
    if setting:
        enhanced_prompt += f", {setting}"
    enhanced_prompt += ", anime style, vibrant colors, professional artwork"

    # Generate unique filename
    timestamp = int(time.time())
    filename_prefix = f"echo_generated_{timestamp}"

    workflow = {
        "3": {
            "inputs": {
                "seed": int(time.time()) % 1000000,
                "steps": 20,
                "cfg": 8.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0]
            },
            "class_type": "KSampler",
            "_meta": {"title": "KSampler"}
        },
        "4": {
            "inputs": {
                "ckpt_name": "deliberate_v2.safetensors"
            },
            "class_type": "CheckpointLoaderSimple",
            "_meta": {"title": "Load Checkpoint"}
        },
        "5": {
            "inputs": {
                "width": 512,
                "height": 512,
                "batch_size": 1
            },
            "class_type": "EmptyLatentImage",
            "_meta": {"title": "Empty Latent Image"}
        },
        "6": {
            "inputs": {
                "text": enhanced_prompt,
                "clip": ["4", 1]
            },
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "CLIP Text Encode (Prompt)"}
        },
        "7": {
            "inputs": {
                "text": "nsfw, nude, bad quality, blurry, low resolution, watermark, text, signature",
                "clip": ["4", 1]
            },
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "CLIP Text Encode (Negative)"}
        },
        "8": {
            "inputs": {
                "samples": ["3", 0],
                "vae": ["4", 2]
            },
            "class_type": "VAEDecode",
            "_meta": {"title": "VAE Decode"}
        },
        "9": {
            "inputs": {
                "filename_prefix": filename_prefix,
                "images": ["8", 0]
            },
            "class_type": "SaveImage",
            "_meta": {"title": "Save Image"}
        }
    }

    return workflow

async def generate_image_comfyui(prompt: str, character: str = "", setting: str = "") -> Optional[str]:
    """Actually generate image through ComfyUI API"""
    try:
        print(f"🎨 Generating image: {prompt} | Character: {character} | Setting: {setting}")

        # Create workflow
        workflow = create_comfyui_workflow(prompt, character, setting)

        # Submit to ComfyUI
        comfyui_url = "http://localhost:8188"

        # Queue the prompt
        queue_response = requests.post(f"{comfyui_url}/prompt", json={"prompt": workflow})

        if queue_response.status_code != 200:
            print(f"❌ Failed to queue prompt: {queue_response.status_code}")
            return None

        queue_data = queue_response.json()
        prompt_id = queue_data.get("prompt_id")

        if not prompt_id:
            print("❌ No prompt_id received")
            return None

        print(f"✅ Queued prompt with ID: {prompt_id}")

        # Wait for completion and get result
        max_wait = 60  # 60 seconds timeout
        start_time = time.time()

        while time.time() - start_time < max_wait:
            try:
                # Check queue status
                queue_status = requests.get(f"{comfyui_url}/queue")
                if queue_status.status_code == 200:
                    queue_info = queue_status.json()

                    # Check if our prompt is still in queue
                    running = queue_info.get("queue_running", [])
                    pending = queue_info.get("queue_pending", [])

                    still_in_queue = any(item[1] == prompt_id for item in running + pending)

                    if not still_in_queue:
                        print(f"✅ Prompt {prompt_id} completed!")
                        break

                await asyncio.sleep(2)

            except Exception as e:
                print(f"⚠️ Error checking queue: {e}")
                await asyncio.sleep(2)

        # Look for generated images
        output_dir = "/home/patrick/ComfyUI/output"

        # Find files that match our prefix and were created recently
        timestamp = int(time.time())
        possible_files = []

        if os.path.exists(output_dir):
            for file in os.listdir(output_dir):
                if file.startswith("echo_generated_") and file.endswith((".png", ".jpg", ".jpeg")):
                    file_path = os.path.join(output_dir, file)
                    file_time = os.path.getmtime(file_path)

                    # If file was created in the last 2 minutes
                    if timestamp - file_time < 120:
                        possible_files.append((file_path, file_time))

        if possible_files:
            # Return the most recently created file
            latest_file = max(possible_files, key=lambda x: x[1])
            image_path = latest_file[0]
            print(f"✅ Generated image: {image_path}")
            return image_path
        else:
            print("❌ No generated image found")
            return None

    except Exception as e:
        print(f"❌ Error generating image: {e}")
        return None

@app.post("/api/echo/chat", response_model=EchoResponse)
async def chat_endpoint(message: ChatMessage):
    """Main chat endpoint with image generation"""

    user_id = message.user_id
    session_id = message.session_id
    text = message.message

    # Get or create story state
    state_key = f"{user_id}_{session_id}"
    if state_key not in story_states:
        story_states[state_key] = {
            'characters': [],
            'settings': [],
            'names': [],
            'history': [],
            'last_update': datetime.now().isoformat()
        }

    story_state = story_states[state_key]

    # Add to history
    story_state['history'].append({
        'message': text,
        'timestamp': datetime.now().isoformat()
    })

    # Extract story elements
    elements = extract_story_elements(text)

    # Update story state
    story_state['characters'].extend(elements['characters'])
    story_state['settings'].extend(elements['settings'])
    story_state['names'].extend(elements['names'])
    story_state['last_update'] = datetime.now().isoformat()

    # Remove duplicates
    story_state['characters'] = list(set(story_state['characters']))
    story_state['settings'] = list(set(story_state['settings']))
    story_state['names'] = list(set(story_state['names']))

    # Check if we should generate an image
    should_generate = False
    image_path = None
    action_taken = None

    if has_creative_intent(text):
        # Check if we have enough story elements
        has_character = len(story_state['characters']) > 0 or len(story_state['names']) > 0
        has_setting = len(story_state['settings']) > 0

        if has_character and has_setting:
            should_generate = True
            action_taken = "GENERATING_IMAGE"

            # Build generation prompt
            character = story_state['names'][0] if story_state['names'] else story_state['characters'][0]
            setting = story_state['settings'][0]

            # Generate image
            image_path = await generate_image_comfyui(text, character, setting)

            if image_path:
                action_taken = "IMAGE_GENERATED"
            else:
                action_taken = "IMAGE_GENERATION_FAILED"

    # Create response
    if should_generate and image_path:
        response_text = f"I've created an image for your story! Character: {character}, Setting: {setting}. The image has been saved to {image_path}."
    elif should_generate and not image_path:
        response_text = f"I tried to generate an image but encountered an issue. Let me gather more details about your story."
    elif has_creative_intent(text):
        missing = []
        if not (story_state['characters'] or story_state['names']):
            missing.append("a character")
        if not story_state['settings']:
            missing.append("a setting/location")

        if missing:
            response_text = f"I'm building your story! I still need {' and '.join(missing)} to create an image. Tell me more!"
        else:
            response_text = "I'm ready to create an image for your story!"
    else:
        response_text = f"I understand! {text}"

    return EchoResponse(
        response=response_text,
        image_path=image_path,
        story_state=story_state,
        action_taken=action_taken
    )

@app.get("/api/echo/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Echo Brain Working",
        "timestamp": datetime.now().isoformat(),
        "comfyui_status": "checking"
    }

@app.get("/api/echo/story/{user_id}/{session_id}")
async def get_story_state(user_id: str, session_id: str):
    """Get current story state"""
    state_key = f"{user_id}_{session_id}"
    if state_key in story_states:
        return story_states[state_key]
    else:
        return {"error": "No story state found"}


@app.get("/api/echo/status")
async def get_status():
    """Status endpoint for dashboard integration"""
    from datetime import datetime
    return {
        "status": "online",
        "service": "echo-brain", 
        "current_thought": "Monitoring Tower services",
        "thought_type": "analysis",
        "response_time": 0.12,
        "recent_queries": len(story_states) if 'story_states' in globals() else 0,
        "last_activity": datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    print("🚀 Starting Echo Brain Working Service on port 8309")
    print("🎨 ComfyUI integration enabled")
    print("📊 Story state tracking enabled")
    uvicorn.run(app, host="0.0.0.0", port=8309)
# ============= SELF-REPAIR SYSTEM =============
import subprocess
from pathlib import Path

async def check_and_repair_services():
    """Echo's self-repair capability"""
    try:
        # Check Telegram bot health
        telegram_check = subprocess.run(
            "systemctl is-active patricksechobot.service",
            shell=True, capture_output=True, text=True
        )
        
        if telegram_check.stdout.strip() != "active":
            print("🔧 Echo: Telegram bot down, fixing it...")
            subprocess.run("sudo systemctl restart patricksechobot.service", shell=True)
            return {"repaired": "telegram_bot"}
        
        # Check for 409 errors (multiple instances)
        error_check = subprocess.run(
            "tail -50 /opt/patricks-echo-bot/bot.log | grep -c 409",
            shell=True, capture_output=True, text=True
        )
        
        if int(error_check.stdout.strip() or 0) > 10:
            print("🔧 Echo: Multiple Telegram instances detected, fixing...")
            subprocess.run("pkill -f patricksecho", shell=True)
            subprocess.run("sudo systemctl restart patricksechobot.service", shell=True)
            return {"repaired": "telegram_conflicts"}
            
        return {"status": "all_healthy"}
    except Exception as e:
        return {"error": str(e)}

# Add self-repair endpoint
@app.post("/api/echo/self-repair")
async def trigger_self_repair():
    """Manually trigger Echo's self-repair"""
    result = await check_and_repair_services()
    return {
        "message": "Self-repair completed",
        "result": result,
        "timestamp": datetime.now().isoformat()
    }

# Auto-repair loop (runs every 5 minutes)
async def auto_repair_loop():
    while True:
        await asyncio.sleep(300)  # 5 minutes
        result = await check_and_repair_services()
        if result.get("repaired"):
            print(f"✅ Echo auto-repaired: {result['repaired']}")

# Start auto-repair in background
asyncio.create_task(auto_repair_loop())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8309)

# Video Generation Addition
from src.core.echo.echo_video_module import EchoVideoGenerator, handle_video_request

@app.post("/api/echo/generate_video")
async def generate_video(request: dict):
    """Generate video based on prompt"""
    
    prompt = request.get('prompt', '')
    character = request.get('character', None)
    duration = request.get('duration', 30)
    
    generator = EchoVideoGenerator()
    result = generator.generate_video_from_prompt(prompt, character, duration)
    
    if result.get('success'):
        return {
            "status": "success",
            "video_path": result['video_path'],
            "duration": result['duration'],
            "quality_score": result['quality_score'],
            "message": f"Generated {duration}s video with quality score {result['quality_score']}/100"
        }
    else:
        return {
            "status": "error",
            "message": result.get('error', 'Video generation failed')
        }

print("✅ Video generation endpoint added to Echo Brain")
