#!/usr/bin/env python3
"""
Echo Voice Interface Service - WebSocket-enabled FastAPI service for real-time voice communication
Integrates STT, TTS, voice cloning, and conversation logging with existing Echo infrastructure
"""

import os
import sys
import json
import logging
import aiohttp
import asyncio
import time
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from datetime import datetime
import subprocess
from typing import Optional, Dict, List, Any
from pathlib import Path
import psycopg2
from contextlib import asynccontextmanager

# Add current directory to Python path for local imports
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))
sys.path.append(str(current_dir.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Connection manager for WebSocket clients
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: str):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                disconnected.append(connection)
        
        for conn in disconnected:
            self.disconnect(conn)

# Pydantic Models for Request/Response Validation
class ChatRequest(BaseModel):
    message: str
    context: Optional[Dict] = None

class GenerationRequest(BaseModel):
    character: str
    num_images: Optional[int] = 1

class LLMRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    intelligence_level: Optional[str] = "medium"  # quick, medium, high, genius

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    timestamp: str
    character_trainer_available: bool
    comfyui_accessible: bool
    orchestrator_available: bool
    ollama_accessible: bool
    vault_available: bool

class GenerationResponse(BaseModel):
    success: bool
    message: str
    character: Optional[str] = None
    images_generated: Optional[int] = 0
    file_paths: Optional[List[str]] = []
    error: Optional[str] = None

# Initialize FastAPI app
app = FastAPI(title="Echo Voice Interface Service", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (the HTML interface)
static_dir = Path(__file__).parent.parent
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Initialize connection manager
manager = ConnectionManager()

class EchoVoiceService:
    """Echo voice service with WebSocket support and conversation logging"""
    
    def __init__(self):
        self.intelligence_level = 85
        self.conversation_history = []
        self.llm_routing = {
            "quick": "tinyllama:latest",
            "medium": "qwen2.5-coder:7b", 
            "high": "qwen2.5-coder:7b",
            "genius": "llama3.1:70b"
        }
        self.db_config = {
            "host": "localhost",
            "database": "tower_consolidated",
            "user": "patrick",
            "password": "admin123"
        }
    
    async def log_conversation(self, user_message: str, echo_response: str, 
                             intelligence_level: str = "medium", context: Dict = None):
        """Log conversation to PostgreSQL database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO echo_conversations 
                (message, response, hemisphere, model_used, created_at, session_id)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                user_message, 
                echo_response, 
                intelligence_level,  # Using hemisphere field for intelligence level
                context.get("model", "unknown") if context else "unknown",
                datetime.now(),
                f"voice_{datetime.now().strftime('%Y%m%d_%H')}"  # Session ID by hour
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
            logger.debug(f"Conversation logged to database")
            
        except Exception as e:
            logger.error(f"Failed to log conversation: {e}")
    
    async def get_conversation_context(self, limit: int = 5) -> List[Dict]:
        """Get recent conversation context from database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT message, response, hemisphere, created_at, model_used 
                FROM echo_conversations 
                ORDER BY created_at DESC 
                LIMIT %s
            """, (limit,))
            
            conversations = []
            for row in cursor.fetchall():
                conversations.append({
                    "user_message": row[0],
                    "echo_response": row[1], 
                    "intelligence_level": row[2],
                    "timestamp": row[3].isoformat(),
                    "model_used": row[4]
                })
            
            cursor.close()
            conn.close()
            return list(reversed(conversations))  # Return chronological order
            
        except Exception as e:
            logger.error(f"Failed to get conversation context: {e}")
            return []
    
    def check_ollama(self) -> bool:
        """Check if Ollama is accessible"""
        try:
            import requests
            response = requests.get("http://localhost:11434/", timeout=3)
            return response.status_code == 200
        except:
            return False
    
    def check_music_service(self) -> bool:
        """Check if music service is available"""
        try:
            import requests
            response = requests.get("http://localhost:8080/api/services", timeout=3)
            services = response.json()
            return any("music" in service.get("name", "").lower() for service in services)
        except:
            return False
    
    def check_database(self) -> bool:
        """Check database connectivity"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            conn.close()
            return True
        except:
            return False

    async def route_llm_request(self, prompt: str, intelligence_level: str = "medium", context: List[Dict] = None):
        """Route request to appropriate LLM with conversation context"""
        model = self.llm_routing.get(intelligence_level, "qwen2.5-coder:7b")
        
        # Build context-aware prompt
        contextual_prompt = self._build_contextual_prompt(prompt, context or [])
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": model,
                        "prompt": contextual_prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "max_tokens": 500
                        }
                    }
                ) as response:
                    result = await response.json()
                    response_text = result.get("response", "I'm having trouble responding right now.")
                    
                    return {
                        "model_used": model,
                        "intelligence_level": intelligence_level,
                        "response": response_text,
                        "success": True
                    }
        except Exception as e:
            logger.error(f"LLM routing error: {e}")
            fallback_response = self._get_fallback_response(prompt)
            return {
                "model_used": "fallback",
                "intelligence_level": intelligence_level,
                "response": fallback_response,
                "success": False,
                "error": str(e)
            }
    
    def _build_contextual_prompt(self, current_prompt: str, context: List[Dict]) -> str:
        """Build context-aware prompt with conversation history"""
        system_prompt = """You are Echo, an advanced AI assistant with voice interface capabilities. 
You are helpful, knowledgeable, and can control various systems like music playback and home automation.
Provide natural, conversational responses suitable for voice interaction. Keep responses concise but informative."""
        
        if not context:
            return f"{system_prompt}\n\nUser: {current_prompt}\nEcho:"
        
        conversation_context = "\n".join([
            f"User: {msg['user_message']}\nEcho: {msg['echo_response']}"
            for msg in context[-3:]  # Last 3 exchanges for context
        ])
        
        return f"{system_prompt}\n\nConversation history:\n{conversation_context}\n\nUser: {current_prompt}\nEcho:"
    
    def _get_fallback_response(self, prompt: str) -> str:
        """Provide fallback responses when LLM is unavailable"""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['hello', 'hi', 'hey']):
            return "Hello! I'm Echo. How can I help you today?"
        elif any(word in prompt_lower for word in ['music', 'play', 'song']):
            return "I can help with music playback, but I'm having trouble connecting to my AI brain right now. Please try again in a moment."
        elif any(word in prompt_lower for word in ['time', 'date']):
            return f"The current time is {datetime.now().strftime('%I:%M %p on %B %d, %Y')}."
        elif any(word in prompt_lower for word in ['status', 'health']):
            return "I'm experiencing some connectivity issues with my AI models, but my core systems are operational."
        else:
            return "I hear you, but I'm having trouble processing that request right now. My AI systems may be busy. Could you try again in a moment?"

# Initialize Echo service
echo = EchoVoiceService()

# Serve the main voice interface
@app.get("/")
async def serve_voice_interface():
    """Serve the main voice interface HTML"""
    return FileResponse(str(static_dir / "echo_voice_interface.html"))

# WebSocket endpoint for real-time voice communication
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for voice communication"""
    await manager.connect(websocket)
    try:
        await websocket.send_text(json.dumps({
            "type": "connected",
            "message": "Connected to Echo voice interface",
            "services": {
                "ollama": echo.check_ollama(),
                "database": echo.check_database(),
                "music": echo.check_music_service()
            }
        }))
        
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data.get("type") == "chat":
                await handle_voice_message(websocket, message_data)
            elif message_data.get("type") == "music":
                await handle_music_command(websocket, message_data)
            elif message_data.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

async def handle_voice_message(websocket: WebSocket, message_data: Dict):
    """Handle voice chat messages through WebSocket"""
    try:
        user_message = message_data.get("message", "")
        intelligence_level = message_data.get("intelligence_level", "medium")
        
        if not user_message:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "No message received"
            }))
            return
        
        # Get conversation context
        context = await echo.get_conversation_context(limit=3)
        
        # Route to LLM
        llm_result = await echo.route_llm_request(
            user_message, 
            intelligence_level,
            context
        )
        
        echo_response = llm_result["response"]
        
        # Log conversation to database
        await echo.log_conversation(
            user_message, 
            echo_response, 
            intelligence_level,
            {"interface": "voice_websocket", "model": llm_result["model_used"]}
        )
        
        # Send response back to client
        response_data = {
            "type": "chat_response",
            "user_message": user_message,
            "echo_response": echo_response,
            "model_used": llm_result["model_used"],
            "intelligence_level": intelligence_level,
            "success": llm_result["success"]
        }
        
        if not llm_result["success"]:
            response_data["error"] = llm_result.get("error", "Unknown error")
        
        await websocket.send_text(json.dumps(response_data))
        
    except Exception as e:
        logger.error(f"Error handling voice message: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Failed to process message: {str(e)}"
        }))

async def handle_music_command(websocket: WebSocket, message_data: Dict):
    """Handle music commands through WebSocket"""
    try:
        command = message_data.get("command", "")
        
        # Forward music commands to the music service or handle directly
        if "playlist" in command.lower():
            # Example music response - integrate with actual music service
            response = {
                "type": "music_response",
                "command": command,
                "message": "Music playlists feature coming soon. Currently checking available music services...",
                "available": echo.check_music_service()
            }
        else:
            response = {
                "type": "music_response", 
                "command": command,
                "message": f"Music command '{command}' received. Integration with music services in progress.",
                "available": echo.check_music_service()
            }
        
        await websocket.send_text(json.dumps(response))
        
    except Exception as e:
        logger.error(f"Error handling music command: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Music command failed: {str(e)}"
        }))

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Echo Voice Interface Service",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "ollama_accessible": echo.check_ollama(),
        "database_connected": echo.check_database(),
        "music_service_available": echo.check_music_service(),
        "active_connections": len(manager.active_connections)
    }

@app.post("/api/chat")
async def chat_with_echo(request: ChatRequest):
    """HTTP endpoint for chat (fallback when WebSocket unavailable)"""
    try:
        user_message = request.message
        context = request.context or {}
        intelligence_level = context.get("intelligence_level", "medium")
        
        # Get conversation context
        conversation_context = await echo.get_conversation_context(limit=3)
        
        # Route to LLM
        llm_result = await echo.route_llm_request(
            user_message, 
            intelligence_level,
            conversation_context
        )
        
        echo_response = llm_result["response"]
        
        # Log conversation
        await echo.log_conversation(
            user_message, 
            echo_response, 
            intelligence_level,
            {"interface": "http_fallback", "model": llm_result["model_used"]}
        )
        
        return {
            "user_message": user_message,
            "echo_response": echo_response,
            "model_used": llm_result["model_used"],
            "intelligence_level": intelligence_level,
            "success": llm_result["success"]
        }
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/conversation/history")
async def get_conversation_history(limit: int = 10):
    """Get recent conversation history"""
    try:
        history = await echo.get_conversation_context(limit=limit)
        return {
            "conversations": history,
            "count": len(history),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
async def status_endpoint():
    """Detailed status endpoint"""
    return {
        "service": "Echo Voice Interface Service",
        "version": "1.0.0",
        "intelligence_level": echo.intelligence_level,
        "ollama_accessible": echo.check_ollama(),
        "database_connected": echo.check_database(),
        "music_service_available": echo.check_music_service(),
        "active_connections": len(manager.active_connections),
        "port": 8310,  # Use different port to avoid conflict
        "timestamp": datetime.now().isoformat(),
        "llm_routing": echo.llm_routing
    }

@app.get("/api/models")
async def models_endpoint():
    """List available LLM models and routing"""
    return {
        "llm_routing": echo.llm_routing,
        "intelligence_levels": ["quick", "medium", "high", "genius"],
        "services": {
            "ollama": echo.check_ollama(),
            "database": echo.check_database(),
            "music": echo.check_music_service()
        }
    }

# Specialized Anime Integration Endpoints
@app.post("/api/anime/enhance-prompt")
async def enhance_anime_prompt(request: LLMRequest):
    """Enhance user prompt with anime-specific details"""
    try:
        anime_system_prompt = """You are an expert anime story creator and visual director. 
        Take the user's basic prompt and enhance it with rich anime-style details including:
        - Visual composition (camera angles, lighting, mood)
        - Character expressions and body language  
        - Background and setting details
        - Anime art style specifications (cel-shading, color palette)
        - Emotional undertones and atmosphere
        
        Transform simple prompts into detailed scene descriptions suitable for image generation.
        Keep the core idea but make it cinematically compelling."""
        
        enhanced_prompt = f"{anime_system_prompt}\n\nUser prompt: {request.prompt}\n\nEnhanced anime scene description:"
        
        # Get conversation context for continuity
        context = await echo.get_conversation_context(limit=3)
        
        # Route to appropriate LLM
        llm_result = await echo.route_llm_request(
            enhanced_prompt, 
            request.intelligence_level,
            context
        )
        
        # Log this as anime enhancement conversation
        await echo.log_conversation(
            f"ANIME_ENHANCE: {request.prompt}", 
            llm_result["response"], 
            request.intelligence_level,
            {"service": "anime_enhancement", "model": llm_result["model_used"]}
        )
        
        return {
            "original_prompt": request.prompt,
            "enhanced_prompt": llm_result["response"],
            "model_used": llm_result["model_used"],
            "intelligence_level": request.intelligence_level,
            "success": llm_result["success"]
        }
        
    except Exception as e:
        logger.error(f"Anime prompt enhancement error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/anime/create-story")
async def create_anime_story(request: LLMRequest):
    """Generate anime story structure from concept"""
    try:
        story_system_prompt = """You are a master anime storyteller specializing in creating compelling narratives.
        Given a story concept, create a structured anime story with:
        
        1. **Title**: Catchy anime-style title
        2. **Genre**: Primary and secondary genres
        3. **Setting**: Time period, location, world-building details
        4. **Main Characters**: 3-5 key characters with names, roles, personalities
        5. **Plot Structure**: 
           - Setup/Inciting Incident
           - Rising Action (2-3 major plot points)
           - Climax
           - Resolution
        6. **Key Themes**: Underlying messages or themes
        7. **Visual Style**: Art direction and aesthetic choices
        
        Format as JSON structure for easy parsing."""
        
        story_prompt = f"{story_system_prompt}\n\nStory concept: {request.prompt}\n\nAnime story structure (JSON format):"
        
        # Get context for story continuity
        context = await echo.get_conversation_context(limit=2)
        
        # Use higher intelligence for story creation
        intelligence = request.intelligence_level if request.intelligence_level in ["high", "genius"] else "high"
        
        llm_result = await echo.route_llm_request(
            story_prompt,
            intelligence, 
            context
        )
        
        # Log story creation
        await echo.log_conversation(
            f"ANIME_STORY: {request.prompt}",
            llm_result["response"],
            intelligence,
            {"service": "anime_story_creation", "model": llm_result["model_used"]}
        )
        
        return {
            "story_concept": request.prompt,
            "story_structure": llm_result["response"],
            "model_used": llm_result["model_used"],
            "intelligence_level": intelligence,
            "success": llm_result["success"]
        }
        
    except Exception as e:
        logger.error(f"Anime story creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/anime/create-character")
async def create_anime_character(request: LLMRequest):
    """Generate detailed anime character from description"""
    try:
        character_system_prompt = """You are an expert anime character designer and writer.
        Create a detailed anime character based on the given description. Include:
        
        1. **Basic Info**: Name (Japanese-style), age, role in story
        2. **Physical Appearance**: 
           - Height, build, distinctive features
           - Hair color/style, eye color
           - Clothing style and signature outfit
           - Any unique markings, accessories
        3. **Personality**: 
           - Core personality traits (3-5 key traits)
           - Strengths and flaws
           - Mannerisms and speech patterns
           - Fears and motivations
        4. **Background**: Brief backstory that shapes their character
        5. **Abilities/Skills**: Any special talents, powers, or expertise
        6. **Relationships**: How they interact with others
        7. **Character Arc Potential**: Growth opportunities in story
        8. **Voice Profile**: Speaking style for voice synthesis
        
        Format as structured JSON for easy parsing."""
        
        character_prompt = f"{character_system_prompt}\n\nCharacter concept: {request.prompt}\n\nDetailed anime character (JSON format):"
        
        # Use medium-high intelligence for character depth
        intelligence = request.intelligence_level if request.intelligence_level in ["medium", "high", "genius"] else "medium"
        
        llm_result = await echo.route_llm_request(
            character_prompt,
            intelligence,
            []  # No context needed for character creation
        )
        
        # Log character creation
        await echo.log_conversation(
            f"ANIME_CHARACTER: {request.prompt}",
            llm_result["response"],
            intelligence,
            {"service": "anime_character_creation", "model": llm_result["model_used"]}
        )
        
        return {
            "character_concept": request.prompt,
            "character_details": llm_result["response"],
            "model_used": llm_result["model_used"],
            "intelligence_level": intelligence,
            "success": llm_result["success"]
        }
        
    except Exception as e:
        logger.error(f"Anime character creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/anime/scene-dialogue")
async def create_scene_dialogue(request: LLMRequest):
    """Generate realistic dialogue for anime scene"""
    try:
        dialogue_system_prompt = """You are an expert anime scriptwriter and dialogue specialist.
        Create natural, engaging dialogue for anime scenes that:
        
        1. **Sounds Authentic**: Natural speech patterns for anime characters
        2. **Shows Character**: Reveals personality through word choice and manner
        3. **Advances Plot**: Moves the story forward or reveals information
        4. **Emotional Resonance**: Conveys appropriate emotions for the scene
        5. **Cultural Context**: Appropriate for anime setting and characters
        
        Include:
        - Character names and dialogue
        - Basic action descriptions in brackets [like this]
        - Emotional cues for voice synthesis (happy, sad, excited, etc.)
        - Pacing notes for dramatic effect
        
        Format as a simple script that can be easily parsed."""
        
        dialogue_prompt = f"{dialogue_system_prompt}\n\nScene description: {request.prompt}\n\nAnime scene dialogue:"
        
        # Get recent conversation context for consistency
        context = await echo.get_conversation_context(limit=4)
        
        llm_result = await echo.route_llm_request(
            dialogue_prompt,
            request.intelligence_level,
            context
        )
        
        # Log dialogue creation
        await echo.log_conversation(
            f"ANIME_DIALOGUE: {request.prompt}",
            llm_result["response"],
            request.intelligence_level,
            {"service": "anime_dialogue_creation", "model": llm_result["model_used"]}
        )
        
        return {
            "scene_description": request.prompt,
            "dialogue_script": llm_result["response"],
            "model_used": llm_result["model_used"],
            "intelligence_level": request.intelligence_level,
            "success": llm_result["success"]
        }
        
    except Exception as e:
        logger.error(f"Anime dialogue creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Echo Voice Interface Service on port 8311")
    logger.info(f"Ollama: {'Accessible' if echo.check_ollama() else 'Not Accessible'}")
    logger.info(f"Database: {'Connected' if echo.check_database() else 'Not Connected'}")
    logger.info(f"Music Service: {'Available' if echo.check_music_service() else 'Not Available'}")
    logger.info("Voice interface: http://localhost:8311/")
    logger.info("WebSocket: ws://localhost:8311/ws")
    uvicorn.run(app, host="0.0.0.0", port=8311, log_level="info")