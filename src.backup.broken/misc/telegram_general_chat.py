#!/usr/bin/env python3
"""
Telegram General Chat Integration for Echo Brain
Provides full Echo Brain capabilities via Telegram with conversation persistence
"""

import os
import json
import logging
import requests
from typing import Dict, Optional
from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI router for general Telegram chat
general_telegram_router = APIRouter(prefix="/api/telegram/general", tags=["telegram-general"])

# Configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET", "telegram_webhook_secret_2025")
ECHO_API_URL = "http://localhost:8309/api/echo/chat"

class TelegramMessage(BaseModel):
    chat_id: int
    text: str
    message_id: Optional[int] = None
    from_user: Optional[Dict] = None

async def send_telegram_message(chat_id: int, text: str, reply_to_message_id: Optional[int] = None):
    """Send a message via Telegram Bot API"""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown"
    }
    
    if reply_to_message_id:
        payload["reply_to_message_id"] = reply_to_message_id
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")
        raise

async def query_echo_brain(message: str, conversation_id: str, user_id: str = "telegram_user") -> Dict:
    """Query Echo Brain's main chat endpoint with conversation context and execution support"""
    try:
        # Check if this is an execution command
        request_type = "conversation"  # Default

        # Detect execution commands
        execution_keywords = [
            "/execute", "/exec", "/run", "/shell", "/bash",
            "/test", "/unittest", "/refactor", "/fix", "/repair",
            "/monitor", "/analyze"
        ]

        if any(message.startswith(cmd) for cmd in execution_keywords):
            request_type = "system_command"
            logger.info(f"🔨 Detected execution command: {message[:50]}")

        payload = {
            "query": message,
            "conversation_id": conversation_id,
            "user_id": user_id,
            "intelligence_level": "auto" if request_type == "conversation" else "system",
            "request_type": request_type,
            "enable_board_consultation": True  # Enable Board of Directors if needed
        }
        
        response = requests.post(ECHO_API_URL, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error querying Echo Brain: {e}")
        raise

@general_telegram_router.post("/webhook/{secret}")
async def general_chat_webhook(
    secret: str,
    request: Request,
    background_tasks: BackgroundTasks
):
    """
    Handle incoming Telegram messages for general Echo Brain chat
    Provides full Echo capabilities with conversation persistence
    """
    # Verify webhook secret
    if secret != TELEGRAM_WEBHOOK_SECRET:
        logger.warning(f"Invalid webhook secret attempted: {secret}")
        raise HTTPException(status_code=403, detail="Invalid webhook secret")
    
    try:
        # Parse Telegram update
        update = await request.json()
        logger.info(f"Received Telegram update for general chat: {json.dumps(update, indent=2)}")
        
        # Check if this is a message update
        if 'message' not in update:
            return {"status": "ignored", "reason": "not a message"}
        
        message = update['message']
        
        # Extract message details
        chat_id = message['chat']['id']
        message_text = message.get('text', '')
        message_id = message.get('message_id')
        from_user = message.get('from', {})
        user_id = str(from_user.get('id', 'unknown'))
        username = from_user.get('username', from_user.get('first_name', 'User'))
        
        # Ignore empty messages
        if not message_text.strip():
            return {"status": "ignored", "reason": "empty message"}
        
        logger.info(f"📱 Message from {username} (ID: {user_id}, Chat: {chat_id}): {message_text}")
        
        # Use chat_id as conversation_id for persistent per-chat conversations
        conversation_id = f"telegram_chat_{chat_id}"
        
        # Process message in background
        background_tasks.add_task(
            process_general_message,
            message_text,
            conversation_id,
            chat_id,
            message_id,
            user_id,
            username
        )
        
        return {"status": "accepted", "processing": True}
        
    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

async def process_general_message(
    message_text: str,
    conversation_id: str,
    chat_id: int,
    message_id: Optional[int],
    user_id: str,
    username: str
):
    """Process general chat message through Echo Brain with full capabilities"""
    try:
        logger.info(f"🧠 Processing message through Echo Brain for {username}...")
        
        # Query Echo Brain with conversation context
        echo_response = await query_echo_brain(message_text, conversation_id, user_id)
        
        # Extract response details
        response_text = echo_response.get('response', 'I encountered an issue processing your request.')
        model_used = echo_response.get('model_used', 'unknown')
        intelligence_level = echo_response.get('intelligence_level', 'unknown')
        processing_time = echo_response.get('processing_time', 0)

        # Format execution results better
        if intelligence_level == 'system_command' or 'direct_executor' in model_used:
            # Format as code block for execution results
            if not response_text.startswith('```'):
                response_text = f"⚙️ **Execution Result**\n```\n{response_text}\n```"
            footer = f"\n_Executed in {processing_time:.2f}s_"
        else:
            # Add metadata footer for normal responses
            footer = f"\n\n_Model: {model_used} ({intelligence_level}) | {processing_time:.2f}s_"

        full_response = response_text + footer
        
        # Send response via Telegram
        await send_telegram_message(chat_id, full_response, message_id)
        
        logger.info(f"✅ Response sent to {username} via Telegram (model: {model_used}, time: {processing_time:.2f}s)")
        
    except Exception as e:
        logger.error(f"Error processing general message: {e}")
        try:
            await send_telegram_message(
                chat_id,
                "I encountered an error processing your message. Please try again.",
                message_id
            )
        except:
            logger.error("Failed to send error message to user")

@general_telegram_router.get("/health")
async def general_chat_health():
    """Health check for general Telegram chat integration"""
    bot_configured = bool(TELEGRAM_BOT_TOKEN)
    echo_api_available = False
    
    try:
        response = requests.get("http://localhost:8309/api/echo/health", timeout=5)
        echo_api_available = response.status_code == 200
    except:
        pass
    
    return {
        "status": "healthy" if (bot_configured and echo_api_available) else "degraded",
        "service": "Telegram General Chat",
        "bot_configured": bot_configured,
        "echo_api_available": echo_api_available,
        "timestamp": datetime.utcnow().isoformat()
    }

# Export router for inclusion in main app
__all__ = ['general_telegram_router']
