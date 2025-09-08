#!/usr/bin/env python3
"""
Simple Echo Voice Service - Working implementation without heavy LLM dependencies
"""

import logging
import json
import psycopg2
from datetime import datetime
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatRequest(BaseModel):
    message: str
    context: Optional[dict] = {}

app = FastAPI(title="Echo Simple Voice Service")

class EchoSimpleService:
    def __init__(self):
        self.db_config = {
            "host": "192.168.50.135",
            "database": "tower_consolidated", 
            "user": "patrick",
            "password": "admin123"
        }
        self.responses = {
            "hello": "Hello! I'm Echo, your voice assistant.",
            "how are you": "I'm doing great! How can I help you today?",
            "what time is it": f"The current time is {datetime.now().strftime('%I:%M %p')}",
            "what can you do": "I can have conversations, answer questions, and help with various tasks. Try asking me something!",
            "test": "Test successful! I can hear you clearly.",
            "default": "I heard you say: '{message}'. I'm currently in simple mode. How can I help you?"
        }
    
    def get_response(self, message: str) -> str:
        message_lower = message.lower().strip()
        
        # Check for exact matches first
        for key, response in self.responses.items():
            if key in message_lower:
                if key == "default":
                    return response.format(message=message)
                return response
        
        # Default response
        return self.responses["default"].format(message=message)
    
    def log_conversation(self, user_message: str, echo_response: str):
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
                "simple",
                "echo_simple_v1",
                datetime.now(),
                f"voice_{datetime.now().strftime('%Y%m%d_%H')}"
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
            logger.debug("Conversation logged successfully")
            
        except Exception as e:
            logger.error(f"Failed to log conversation: {e}")

echo_service = EchoSimpleService()

@app.get("/", response_class=HTMLResponse)
async def get_interface():
    with open("/opt/tower-echo-brain/static/echo_voice_interface.html", "r") as f:
        return f.read()

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Echo Simple Voice Service",
        "timestamp": datetime.now().isoformat(),
        "database_connected": True
    }

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        user_message = request.message.strip()
        if not user_message:
            raise HTTPException(status_code=400, detail="Empty message")
        
        echo_response = echo_service.get_response(user_message)
        
        # Log conversation
        echo_service.log_conversation(user_message, echo_response)
        
        return {
            "user_message": user_message,
            "echo_response": echo_response,
            "model": "echo_simple_v1",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            user_message = message_data.get("message", "")
            if not user_message:
                continue
            
            echo_response = echo_service.get_response(user_message)
            echo_service.log_conversation(user_message, echo_response)
            
            await websocket.send_text(json.dumps({
                "user_message": user_message,
                "echo_response": echo_response,
                "model": "echo_simple_v1",
                "timestamp": datetime.now().isoformat()
            }))
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        logger.info("WebSocket connection closed")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Echo Tower Voice Service on port 8309")
    uvicorn.run(app, host="0.0.0.0", port=8309, log_level="info")