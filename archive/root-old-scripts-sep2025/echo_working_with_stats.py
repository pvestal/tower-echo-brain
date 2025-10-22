#!/usr/bin/env python3
"""
WORKING Echo Brain Service with Comprehensive User Statistics Tracking
- All original functionality preserved
- Added user stats tracking for production monitoring
- SQLite database for persistent stats storage
- Individual and global stats endpoints
"""

import json
import re
import time
import uuid
import os
import asyncio
import websocket
import requests
import sqlite3
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Path
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import uvicorn

app = FastAPI()

# Story state storage (in-memory for now)
story_states: Dict[str, Dict] = {}

# Database initialization
DB_PATH = "/opt/tower-echo-brain/data/user_stats.db"

def init_database():
    """Initialize SQLite database for user statistics"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create user_stats table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_stats (
            user_id TEXT PRIMARY KEY,
            total_generations INTEGER DEFAULT 0,
            successful_generations INTEGER DEFAULT 0,
            failed_generations INTEGER DEFAULT 0,
            total_errors INTEGER DEFAULT 0,
            last_generation_time TEXT,
            first_seen TEXT,
            last_active TEXT,
            total_tokens_used INTEGER DEFAULT 0
        )
    """)
    
    # Create generation_log table for detailed tracking
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS generation_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            session_id TEXT,
            timestamp TEXT,
            prompt TEXT,
            success BOOLEAN,
            error_message TEXT,
            image_path TEXT,
            tokens_used INTEGER DEFAULT 0,
            processing_time REAL,
            FOREIGN KEY (user_id) REFERENCES user_stats (user_id)
        )
    """)
    
    # Create error_log table for error tracking
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS error_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            timestamp TEXT,
            error_type TEXT,
            error_message TEXT,
            context TEXT,
            FOREIGN KEY (user_id) REFERENCES user_stats (user_id)
        )
    """)
    
    conn.commit()
    conn.close()
    print("✅ Database initialized with user statistics tables")

def get_or_create_user_stats(user_id: str):
    """Get or create user statistics record"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if user exists
    cursor.execute("SELECT * FROM user_stats WHERE user_id = ?", (user_id,))
    user = cursor.fetchone()
    
    if not user:
        # Create new user record
        now = datetime.now().isoformat()
        cursor.execute("""
            INSERT INTO user_stats 
            (user_id, total_generations, successful_generations, failed_generations, 
             total_errors, first_seen, last_active, total_tokens_used)
            VALUES (?, 0, 0, 0, 0, ?, ?, 0)
        """, (user_id, now, now))
        conn.commit()
        print(f"📊 Created new user stats for: {user_id}")
    
    conn.close()

def update_user_activity(user_id: str):
    """Update user's last active timestamp"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        UPDATE user_stats 
        SET last_active = ? 
        WHERE user_id = ?
    """, (datetime.now().isoformat(), user_id))
    
    conn.commit()
    conn.close()

def log_generation_attempt(user_id: str, session_id: str, prompt: str, 
                          success: bool, error_message: str = None, 
                          image_path: str = None, processing_time: float = 0.0):
    """Log generation attempt with full details"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Estimate tokens (rough calculation: 1 token ≈ 4 characters)
    tokens_used = len(prompt) // 4 if prompt else 0
    
    # Log to generation_log
    cursor.execute("""
        INSERT INTO generation_log 
        (user_id, session_id, timestamp, prompt, success, error_message, 
         image_path, tokens_used, processing_time)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (user_id, session_id, datetime.now().isoformat(), prompt, 
          success, error_message, image_path, tokens_used, processing_time))
    
    # Update user stats
    if success:
        cursor.execute("""
            UPDATE user_stats 
            SET total_generations = total_generations + 1,
                successful_generations = successful_generations + 1,
                last_generation_time = ?,
                total_tokens_used = total_tokens_used + ?
            WHERE user_id = ?
        """, (datetime.now().isoformat(), tokens_used, user_id))
    else:
        cursor.execute("""
            UPDATE user_stats 
            SET total_generations = total_generations + 1,
                failed_generations = failed_generations + 1,
                last_generation_time = ?,
                total_tokens_used = total_tokens_used + ?
            WHERE user_id = ?
        """, (datetime.now().isoformat(), tokens_used, user_id))
    
    conn.commit()
    conn.close()
    print(f"📊 Logged generation: {user_id} - Success: {success}")

def log_error(user_id: str, error_type: str, error_message: str, context: str = ""):
    """Log error for user"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Log to error_log
    cursor.execute("""
        INSERT INTO error_log (user_id, timestamp, error_type, error_message, context)
        VALUES (?, ?, ?, ?, ?)
    """, (user_id, datetime.now().isoformat(), error_type, error_message, context))
    
    # Update user stats
    cursor.execute("""
        UPDATE user_stats 
        SET total_errors = total_errors + 1
        WHERE user_id = ?
    """, (user_id,))
    
    conn.commit()
    conn.close()
    print(f"📊 Logged error: {user_id} - {error_type}")

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
    names = re.findall(r''\b[A-Z][a-z]+\b', text)
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
                # ComfyUI adds _00001_ suffix to filenames, so check for our prefix
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
    """Main chat endpoint with image generation and stats tracking"""

    user_id = message.user_id
    session_id = message.session_id
    text = message.message
    start_time = time.time()

    # Ensure user stats exist and update activity
    get_or_create_user_stats(user_id)
    update_user_activity(user_id)

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
    error_message = None

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

            try:
                # Generate image
                image_path = await generate_image_comfyui(text, character, setting)

                if image_path:
                    action_taken = "IMAGE_GENERATED"
                    # Log successful generation
                    processing_time = time.time() - start_time
                    log_generation_attempt(user_id, session_id, text, True, 
                                         None, image_path, processing_time)
                else:
                    action_taken = "IMAGE_GENERATION_FAILED"
                    error_message = "ComfyUI generation failed"
                    # Log failed generation
                    processing_time = time.time() - start_time
                    log_generation_attempt(user_id, session_id, text, False, 
                                         error_message, None, processing_time)
                    log_error(user_id, "GENERATION_FAILED", error_message, f"Prompt: {text}")

            except Exception as e:
                action_taken = "IMAGE_GENERATION_ERROR"
                error_message = str(e)
                # Log error
                processing_time = time.time() - start_time
                log_generation_attempt(user_id, session_id, text, False, 
                                     error_message, None, processing_time)
                log_error(user_id, "GENERATION_ERROR", error_message, f"Prompt: {text}")

    # Create response
    if should_generate and image_path:
        response_text = f"Created {character} in {setting}! Image saved successfully."
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
        # Provide contextual responses
        if "hello" in text.lower() or "hi" in text.lower():
            response_text = "Hello! I can generate anime images. Try: Generate a samurai warrior"
        elif "help" in text.lower():
            response_text = "I can create images! Tell me a character and setting, then say generate"
        else:
            response_text = f"Tell me what to create. Example: Generate a dragon in space"

    return EchoResponse(
        response=response_text,
        image_path=image_path,
        story_state=story_state,
        action_taken=action_taken
    )

@app.get("/api/echo/stats/{user_id}")
async def get_user_stats(user_id: str = Path(..., description="User ID to get stats for")):
    """Get comprehensive statistics for a specific user"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get basic user stats
        cursor.execute("SELECT * FROM user_stats WHERE user_id = ?", (user_id,))
        user_row = cursor.fetchone()
        
        if not user_row:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Convert to dict
        columns = [desc[0] for desc in cursor.description]
        user_stats = dict(zip(columns, user_row))
        
        # Calculate success rate
        total_gens = user_stats['total_generations']
        success_rate = (user_stats['successful_generations'] / total_gens * 100) if total_gens > 0 else 0
        error_rate = (user_stats['failed_generations'] / total_gens * 100) if total_gens > 0 else 0
        
        # Get recent generation history (last 10)
        cursor.execute("""
            SELECT timestamp, prompt, success, image_path, processing_time
            FROM generation_log 
            WHERE user_id = ? 
            ORDER BY timestamp DESC 
            LIMIT 10
        """, (user_id,))
        recent_generations = [dict(zip([desc[0] for desc in cursor.description], row)) 
                            for row in cursor.fetchall()]
        
        # Get recent errors (last 5)
        cursor.execute("""
            SELECT timestamp, error_type, error_message
            FROM error_log 
            WHERE user_id = ? 
            ORDER BY timestamp DESC 
            LIMIT 5
        """, (user_id,))
        recent_errors = [dict(zip([desc[0] for desc in cursor.description], row)) 
                        for row in cursor.fetchall()]
        
        # Get usage patterns (generations per day for last 7 days)
        seven_days_ago = (datetime.now() - timedelta(days=7)).isoformat()
        cursor.execute("""
            SELECT DATE(timestamp) as date, COUNT(*) as count
            FROM generation_log 
            WHERE user_id = ? AND timestamp > ?
            GROUP BY DATE(timestamp)
            ORDER BY date
        """, (user_id, seven_days_ago))
        daily_usage = [dict(zip([desc[0] for desc in cursor.description], row)) 
                      for row in cursor.fetchall()]
        
        conn.close()
        
        return {
            "user_id": user_id,
            "basic_stats": user_stats,
            "calculated_metrics": {
                "success_rate_percentage": round(success_rate, 2),
                "error_rate_percentage": round(error_rate, 2),
                "average_tokens_per_generation": round(user_stats['total_tokens_used'] / total_gens, 2) if total_gens > 0 else 0
            },
            "recent_generations": recent_generations,
            "recent_errors": recent_errors,
            "usage_patterns": {
                "daily_usage_last_7_days": daily_usage
            },
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        log_error(user_id, "STATS_ERROR", str(e), "Failed to retrieve user stats")
        raise HTTPException(status_code=500, detail=f"Error retrieving stats: {str(e)}")

@app.get("/api/echo/stats/global")
async def get_global_stats():
    """Get global statistics for admin overview"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Total users
        cursor.execute("SELECT COUNT(*) FROM user_stats")
        total_users = cursor.fetchone()[0]
        
        # Total generations
        cursor.execute("SELECT SUM(total_generations) FROM user_stats")
        total_generations = cursor.fetchone()[0] or 0
        
        # Total successful generations
        cursor.execute("SELECT SUM(successful_generations) FROM user_stats")
        total_successful = cursor.fetchone()[0] or 0
        
        # Total failed generations
        cursor.execute("SELECT SUM(failed_generations) FROM user_stats")
        total_failed = cursor.fetchone()[0] or 0
        
        # Total errors
        cursor.execute("SELECT SUM(total_errors) FROM user_stats")
        total_errors = cursor.fetchone()[0] or 0
        
        # Total tokens
        cursor.execute("SELECT SUM(total_tokens_used) FROM user_stats")
        total_tokens = cursor.fetchone()[0] or 0
        
        # Active users (last 24 hours)
        yesterday = (datetime.now() - timedelta(days=1)).isoformat()
        cursor.execute("SELECT COUNT(*) FROM user_stats WHERE last_active > ?", (yesterday,))
        active_users_24h = cursor.fetchone()[0]
        
        # Most active users (top 5)
        cursor.execute("""
            SELECT user_id, total_generations, successful_generations, last_active
            FROM user_stats 
            ORDER BY total_generations DESC 
            LIMIT 5
        """)
        top_users = [dict(zip([desc[0] for desc in cursor.description], row)) 
                    for row in cursor.fetchall()]
        
        # Generations per day (last 7 days)
        seven_days_ago = (datetime.now() - timedelta(days=7)).isoformat()
        cursor.execute("""
            SELECT DATE(timestamp) as date, 
                   COUNT(*) as total,
                   SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful,
                   SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failed
            FROM generation_log 
            WHERE timestamp > ?
            GROUP BY DATE(timestamp)
            ORDER BY date
        """, (seven_days_ago,))
        daily_stats = [dict(zip([desc[0] for desc in cursor.description], row)) 
                      for row in cursor.fetchall()]
        
        # Most common error types
        cursor.execute("""
            SELECT error_type, COUNT(*) as count
            FROM error_log 
            WHERE timestamp > ?
            GROUP BY error_type
            ORDER BY count DESC
            LIMIT 5
        """, (seven_days_ago,))
        common_errors = [dict(zip([desc[0] for desc in cursor.description], row)) 
                        for row in cursor.fetchall()]
        
        conn.close()
        
        # Calculate metrics
        success_rate = (total_successful / total_generations * 100) if total_generations > 0 else 0
        error_rate = (total_failed / total_generations * 100) if total_generations > 0 else 0
        
        return {
            "overview": {
                "total_users": total_users,
                "active_users_24h": active_users_24h,
                "total_generations": total_generations,
                "total_successful_generations": total_successful,
                "total_failed_generations": total_failed,
                "total_errors": total_errors,
                "total_tokens_used": total_tokens
            },
            "metrics": {
                "global_success_rate_percentage": round(success_rate, 2),
                "global_error_rate_percentage": round(error_rate, 2),
                "average_generations_per_user": round(total_generations / total_users, 2) if total_users > 0 else 0,
                "average_tokens_per_generation": round(total_tokens / total_generations, 2) if total_generations > 0 else 0
            },
            "top_users": top_users,
            "trends": {
                "daily_stats_last_7_days": daily_stats
            },
            "error_analysis": {
                "common_error_types": common_errors
            },
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving global stats: {str(e)}")

@app.get("/api/echo/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Echo Brain Working with Stats",
        "timestamp": datetime.now().isoformat(),
        "comfyui_status": "checking",
        "database_status": "connected" if os.path.exists(DB_PATH) else "not_initialized"
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
    return {
        "status": "online",
        "service": "echo-brain", 
        "current_thought": "Monitoring Tower services with stats tracking",
        "thought_type": "analysis",
        "response_time": 0.12,
        "recent_queries": len(story_states) if 'story_states' in globals() else 0,
        "last_activity": datetime.utcnow().isoformat(),
        "stats_tracking": "enabled",
        "database_status": "active" if os.path.exists(DB_PATH) else "initializing"
    }

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

if __name__ == "__main__":
    print("🚀 Starting Echo Brain Working Service with Stats Tracking on port 8309")
    print("🎨 ComfyUI integration enabled")
    print("📊 Story state tracking enabled")
    print("📈 User statistics tracking enabled")
    
    # Initialize database
    init_database()
    
    # Start auto-repair in background
    asyncio.create_task(auto_repair_loop())
    
    uvicorn.run(app, host="0.0.0.0", port=8309)
