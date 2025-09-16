#!/usr/bin/env python3
"""
Echo Brain Unified Service with Voice Integration
Adds voice awareness and voice notification capabilities to Echo
"""

import asyncio
import aiohttp
import logging
import json
import psycopg2
import uuid
import re
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Voice Integration Class
class EchoVoiceIntegration:
    """Voice integration for Echo Brain - connects to unified voice service"""
    
    def __init__(self):
        self.voice_service_url = "http://localhost:8331"
        self.enabled = True
        self.character_voices = {
            "echo_default": "neutral",  # Default Echo voice
            "tokyo_debt_desire": "professional",  # Business context
            "sakura": "friendly"  # General interactions
        }
        
    async def send_voice_notification(self, message: str, mood: str = "neutral", character: str = "echo_default"):
        """Send a voice notification using the unified voice service"""
        if not self.enabled:
            logger.debug("Voice notifications disabled")
            return False
            
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "text": message,
                    "character": character,
                    "mood": mood,
                    "pitch": 1.0,
                    "speed": 1.0
                }
                
                voice_url = f"{self.voice_service_url}/api/voice/speak"
                async with session.post(voice_url, json=payload, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"🔊 Voice notification sent: {character} - {mood}")
                        return True
                    else:
                        logger.warning(f"Voice service returned {response.status}")
                        return False
                        
        except Exception as e:
            logger.warning(f"Voice notification failed: {e}")
            return False
    
    def select_character_for_context(self, query: str, context: Dict) -> str:
        """Select appropriate character voice based on context"""
        query_lower = query.lower()
        
        # Business/Professional context
        if any(term in query_lower for term in ['business', 'finance', 'debt', 'money', 'professional']):
            return "tokyo_debt_desire"
        
        # Friendly/Creative context  
        if any(term in query_lower for term in ['creative', 'story', 'friendly', 'help']):
            return "sakura"
            
        # Default Echo voice
        return "echo_default"
    
    def select_mood_for_response(self, response_type: str, success: bool = True) -> str:
        """Select appropriate mood based on response type"""
        if not success:
            return "concerned"
        
        if response_type in ["startup", "success"]:
            return "helpful"
        elif response_type in ["error", "failure"]:
            return "concerned"  
        elif response_type in ["thinking", "processing"]:
            return "neutral"
        else:
            return "friendly"

# Request/Response Models
class QueryRequest(BaseModel):
    query: str
    context: Optional[Dict] = {}
    intelligence_level: Optional[str] = "auto"
    user_id: Optional[str] = "default"
    conversation_id: Optional[str] = None
    voice_enabled: Optional[bool] = True  # NEW: Voice notification control

class QueryResponse(BaseModel):
    response: str
    model_used: str
    intelligence_level: str
    processing_time: float
    escalation_path: List[str]
    requires_clarification: bool = False
    clarifying_questions: List[str] = []
    conversation_id: str
    intent: Optional[str] = None
    confidence: float = 0.0
    voice_notification_sent: bool = False  # NEW: Voice notification status

class VoiceNotificationRequest(BaseModel):
    message: str
    character: Optional[str] = "echo_default"
    mood: Optional[str] = "neutral"

class EchoIntelligenceRouter:
    """Core intelligence routing system for Echo Brain with Voice Integration"""
    
    def __init__(self):
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model_hierarchy = {
            "quick": "tinyllama:latest",        # 1B parameters
            "standard": "llama3.2:3b",          # 3B parameters
            "professional": "mistral:7b",       # 7B parameters
            "expert": "qwen2.5-coder:32b",     # 32B parameters  
            "genius": "llama3.1:70b",          # 70B parameters
        }
        self.specialized_models = {
            "coding": "deepseek-coder-v2:16b",
            "creative": "mixtral:8x7b", 
            "analysis": "codellama:70b"
        }
        self.escalation_history = []
        self.voice = EchoVoiceIntegration()  # NEW: Voice integration
        
    def analyze_complexity(self, query: str, context: Dict) -> str:
        """Analyze query complexity to determine optimal model"""
        complexity_score = 0.0
        
        # Basic query analysis
        complexity_score += len(query.split()) * 0.3
        complexity_score += query.count('?') * 3
        complexity_score += query.count('.') * 1
        
        # Technical complexity indicators
        technical_terms = [
            'database', 'architecture', 'algorithm', 'implementation',
            'refactor', 'optimization', 'integration', 'system'
        ]
        complexity_score += sum(8 for term in technical_terms if term.lower() in query.lower())
        
        # Programming language detection
        code_terms = ['python', 'javascript', 'sql', 'function', 'class', 'async']
        if any(term in query.lower() for term in code_terms):
            complexity_score += 15
            
        # Context complexity
        if context.get('previous_failures', 0) > 0:
            complexity_score += 20  # Escalate if previous attempts failed
            
        if context.get('user_expertise') == 'expert':
            complexity_score += 10
            
        # Route based on score (adjusted for better escalation)
        if complexity_score < 8:
            return "quick"
        elif complexity_score < 25: 
            return "standard"
        elif complexity_score < 40:
            return "professional"
        elif complexity_score < 60:
            return "expert" 
        else:
            return "genius"
    
    def detect_specialization(self, query: str) -> Optional[str]:
        """Detect if query requires specialized model"""
        query_lower = query.lower()
        
        if any(term in query_lower for term in ['code', 'program', 'function', 'debug']):
            return "coding"
        elif any(term in query_lower for term in ['creative', 'story', 'write', 'imagine']):
            return "creative"  
        elif any(term in query_lower for term in ['analyze', 'data', 'research', 'study']):
            return "analysis"
        return None
    
    async def query_model(self, model: str, prompt: str, max_tokens: int = 2048) -> Dict:
        """Query specific Ollama model"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": 0.7
                    }
                }
                
                start_time = asyncio.get_event_loop().time()
                async with session.post(self.ollama_url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        processing_time = asyncio.get_event_loop().time() - start_time
                        return {
                            "success": True,
                            "response": result.get("response", ""),
                            "processing_time": processing_time,
                            "model": model
                        }
                    else:
                        return {"success": False, "error": f"HTTP {response.status}"}
                        
        except Exception as e:
            logger.error(f"Model query failed for {model}: {e}")
            return {"success": False, "error": str(e)}

    async def progressive_escalation_with_voice(self, query: str, context: Dict, voice_enabled: bool = True) -> Dict:
        """Enhanced progressive escalation with voice notifications"""
        escalation_path = []
        
        # Voice notification for start of processing
        if voice_enabled:
            character = self.voice.select_character_for_context(query, context)
            await self.voice.send_voice_notification(
                "Processing your request", 
                "neutral", 
                character
            )
        
        # Start with complexity-based model selection
        initial_level = self.analyze_complexity(query, context)
        
        # Check for specialization override
        specialization = self.detect_specialization(query)
        if specialization:
            model = self.specialized_models[specialization]
            escalation_path.append(f"specialized_{specialization}")
        else:
            model = self.model_hierarchy[initial_level]
            escalation_path.append(initial_level)
        
        # Try the selected model
        logger.info(f"🧠 Echo routing to {model} for query complexity analysis")
        result = await self.query_model(model, query)
        
        if result["success"]:
            # Voice notification for successful processing
            if voice_enabled:
                character = self.voice.select_character_for_context(query, context)
                await self.voice.send_voice_notification(
                    "Request completed successfully", 
                    "helpful", 
                    character
                )
            return {
                "response": result["response"],
                "model_used": model,
                "intelligence_level": specialization or initial_level,
                "processing_time": result["processing_time"],
                "escalation_path": escalation_path,
                "voice_notification_sent": voice_enabled
            }
        else:
            # Escalation needed - try next higher model
            hierarchy_levels = list(self.model_hierarchy.keys())
            if initial_level in hierarchy_levels:
                current_index = hierarchy_levels.index(initial_level)
                if current_index < len(hierarchy_levels) - 1:
                    next_level = hierarchy_levels[current_index + 1]
                    next_model = self.model_hierarchy[next_level]
                    escalation_path.append(f"escalated_to_{next_level}")
                    
                    logger.info(f"🧠 Echo escalating to {next_model}")
                    result = await self.query_model(next_model, query)
                    
                    if result["success"]:
                        # Voice notification for escalated success
                        if voice_enabled:
                            character = self.voice.select_character_for_context(query, context)
                            await self.voice.send_voice_notification(
                                "Request completed after escalation", 
                                "helpful", 
                                character
                            )
                        return {
                            "response": result["response"],
                            "model_used": next_model,
                            "intelligence_level": next_level,
                            "processing_time": result["processing_time"],
                            "escalation_path": escalation_path,
                            "voice_notification_sent": voice_enabled
                        }
            
            # All models failed
            if voice_enabled:
                await self.voice.send_voice_notification(
                    "Unable to process request, all models failed", 
                    "concerned", 
                    "echo_default"
                )
            
            return {
                "response": "I apologize, but I'm unable to process your request at this time due to technical difficulties.",
                "model_used": "fallback",
                "intelligence_level": "error",
                "processing_time": 0.0,
                "escalation_path": escalation_path + ["all_models_failed"],
                "voice_notification_sent": voice_enabled
            }

# Initialize FastAPI and router
app = FastAPI(
    title="Echo Brain Unified with Voice Integration",
    description="Intelligent routing system with voice notifications",
    version="1.1.0"
)

router = EchoIntelligenceRouter()

@app.on_event("startup")
async def startup_event():
    """Startup event with voice notification"""
    logger.info("🧠 Echo Brain starting up with voice integration...")
    
    # Send startup voice notification
    await router.voice.send_voice_notification(
        "Echo Brain service starting up",
        "helpful",
        "echo_default"
    )

@app.get("/api/echo/health")
async def health():
    """Health check with voice capability status"""
    return {
        "status": "healthy",
        "service": "Echo Brain Unified with Voice",
        "voice_integration": router.voice.enabled,
        "voice_service_url": router.voice.voice_service_url,
        "available_characters": list(router.voice.character_voices.keys()),
        "intelligence_levels": list(router.model_hierarchy.keys()),
        "specialized_models": list(router.specialized_models.keys()),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/echo/query", response_model=QueryResponse)
async def query_echo(request: QueryRequest):
    """Enhanced query endpoint with voice notifications"""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    conversation_id = request.conversation_id or str(uuid.uuid4())
    
    try:
        # Process with voice integration
        result = await router.progressive_escalation_with_voice(
            request.query, 
            request.context, 
            request.voice_enabled
        )
        
        return QueryResponse(
            response=result["response"],
            model_used=result["model_used"],
            intelligence_level=result["intelligence_level"],
            processing_time=result["processing_time"],
            escalation_path=result["escalation_path"],
            conversation_id=conversation_id,
            voice_notification_sent=result.get("voice_notification_sent", False)
        )
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        
        # Voice notification for error
        if request.voice_enabled:
            await router.voice.send_voice_notification(
                "An error occurred while processing your request",
                "concerned",
                "echo_default"
            )
        
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/echo/voice/notify")
async def send_voice_notification(request: VoiceNotificationRequest):
    """Direct voice notification endpoint"""
    success = await router.voice.send_voice_notification(
        request.message,
        request.mood,
        request.character
    )
    
    return {
        "success": success,
        "message": request.message,
        "character": request.character,
        "mood": request.mood,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/echo/voice/status")
async def voice_status():
    """Get voice integration status"""
    return {
        "voice_enabled": router.voice.enabled,
        "voice_service_url": router.voice.voice_service_url,
        "available_characters": router.voice.character_voices,
        "supported_moods": ["neutral", "helpful", "friendly", "concerned", "professional"],
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/echo/voice/toggle")
async def toggle_voice(enabled: bool):
    """Toggle voice notifications on/off"""
    router.voice.enabled = enabled
    
    if enabled:
        await router.voice.send_voice_notification(
            "Voice notifications enabled",
            "helpful",
            "echo_default"
        )
    
    return {
        "voice_enabled": router.voice.enabled,
        "message": f"Voice notifications {'enabled' if enabled else 'disabled'}",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == '__main__':
    logger.info("🚀 Starting Echo Brain Unified Service with Voice Integration")
    uvicorn.run(
        "echo_voice_enhanced:app", 
        host="0.0.0.0",  # Listen on all interfaces 
        port=8309, 
        reload=False,
        workers=1
    )