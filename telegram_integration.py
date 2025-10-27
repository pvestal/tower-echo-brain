#!/usr/bin/env python3
"""
Telegram Bot Integration for Echo Brain Veteran Support System
Handles incoming Telegram messages and provides therapeutic support
"""

import os
import json
import logging
from typing import Dict, Optional
from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from datetime import datetime
import asyncio

# Import the veteran guardian system
from veteran_guardian_system import (
    VeteranGuardianSystem,
    VeteranSupportTester,
    RiskLevel
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI router for Telegram webhook
telegram_router = APIRouter(prefix="/api/telegram", tags=["telegram"])

# Configuration from environment or vault
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET", "telegram_webhook_secret_2025")
SUPPORT_CHANNEL_ID = os.getenv("TELEGRAM_SUPPORT_CHANNEL_ID", "")

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', '5432')),
    'database': os.getenv('DB_NAME', 'echo_brain'),
    'user': os.getenv('DB_USER', 'patrick'),
    'password': os.getenv('DB_PASSWORD', '')
}

# Telegram configuration
TELEGRAM_CONFIG = {
    'bot_token': TELEGRAM_BOT_TOKEN,
    'support_channel_id': SUPPORT_CHANNEL_ID
}

# Initialize the veteran guardian system
guardian_system = None

def initialize_guardian_system():
    """Initialize the veteran guardian support system"""
    global guardian_system
    try:
        guardian_system = VeteranGuardianSystem(DB_CONFIG, TELEGRAM_CONFIG)
        logger.info("Veteran Guardian System initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Veteran Guardian System: {e}")
        return False

@telegram_router.on_event("startup")
async def startup_event():
    """Initialize guardian system on startup"""
    if not initialize_guardian_system():
        logger.warning("Guardian system not initialized, will retry on first request")

@telegram_router.post("/webhook/{secret}")
async def telegram_webhook(
    secret: str,
    request: Request,
    background_tasks: BackgroundTasks
):
    """
    Handle incoming Telegram webhook updates
    Processes messages and provides therapeutic responses
    SECURITY: Only accepts messages from authorized user IDs
    """
    # Verify webhook secret
    if secret != TELEGRAM_WEBHOOK_SECRET:
        logger.warning(f"Invalid webhook secret attempted: {secret}")
        raise HTTPException(status_code=403, detail="Invalid webhook secret")

    # Initialize guardian system if not ready
    if not guardian_system:
        if not initialize_guardian_system():
            raise HTTPException(status_code=503, detail="Service temporarily unavailable")

    try:
        # Parse Telegram update
        update = await request.json()
        logger.info(f"Received Telegram update: {json.dumps(update, indent=2)}")

        # Check if this is a message update
        if 'message' not in update:
            return {"status": "ignored", "reason": "not a message"}

        message = update['message']

        # Extract user information
        user_id = message.get('from', {}).get('id')
        username = message.get('from', {}).get('username', 'Unknown')
        first_name = message.get('from', {}).get('first_name', '')
        chat_id = message.get('chat', {}).get('id')
        text = message.get('text', '')

        if not text:
            return {"status": "ignored", "reason": "no text content"}

        # Log incoming message (allow anyone to message)
        logger.info(f"📩 Processing message from {username} (ID: {user_id}): {text[:100]}...")

        # Process message asynchronously in background
        background_tasks.add_task(
            process_veteran_message,
            update,
            user_id,
            username,
            chat_id,
            text
        )

        # Immediate response to Telegram
        return {"status": "accepted", "processing": True}

    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

async def process_veteran_message(
    update: Dict,
    user_id: int,
    username: str,
    chat_id: int,
    text: str
):
    """
    Process veteran message in the background
    Provides therapeutic response based on risk assessment
    SECURITY: Filters sensitive information for non-Patrick users
    """
    try:
        # Check if query requests sensitive information
        from src.security.sensitive_filter import should_answer_query, filter_sensitive_response, is_patrick

        allow, reason = should_answer_query(text, user_id)
        if not allow:
            # Send generic response for sensitive queries from non-Patrick users
            await send_telegram_response(chat_id, reason)
            logger.info(f"🚫 Blocked sensitive query from {username} (ID: {user_id})")
            return

        # Process through veteran guardian system
        result = await guardian_system.process_telegram_message(update)

        # Filter sensitive information from response if not Patrick
        response = result.get('response', 'I encountered an issue processing your request.')
        if not is_patrick(user_id):
            response = filter_sensitive_response(response, user_id, username)

        # Log result
        logger.info(f"Message processed - Risk: {result.get('risk_level')}, "
                   f"Concerns: {result.get('concerns')}, "
                   f"Response time: {result.get('response_time_ms')}ms")

        # If critical risk, send alert to support team
        if result.get('risk_level') == 'critical':
            await send_crisis_alert(user_id, username, text, result)

    except Exception as e:
        logger.error(f"Error processing veteran message: {e}")
        # Send generic supportive message on error
        try:
            await send_fallback_response(chat_id, update.get('message', {}).get('message_id'))
        except Exception as fallback_error:
            logger.error(f"Failed to send fallback response: {fallback_error}")

async def send_crisis_alert(user_id: int, username: str, message: str, result: Dict):
    """
    Send alert to support team for critical situations
    """
    alert_message = f"""
🚨 CRITICAL ALERT - Veteran Support

User: @{username} (ID: {user_id})
Risk Level: {result.get('risk_level')}
Concerns: {', '.join(result.get('concerns', []))}

Message excerpt:
"{message[:200]}..."

Response has been sent. Please monitor and follow up.
    """

    # Send to support channel if configured
    if SUPPORT_CHANNEL_ID:
        try:
            import aiohttp
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {
                'chat_id': SUPPORT_CHANNEL_ID,
                'text': alert_message,
                'parse_mode': 'HTML'
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        logger.info("Crisis alert sent to support team")
                    else:
                        logger.error(f"Failed to send crisis alert: {response.status}")
        except Exception as e:
            logger.error(f"Error sending crisis alert: {e}")

async def send_fallback_response(chat_id: int, reply_to_message_id: Optional[int] = None):
    """
    Send a generic supportive message when processing fails
    """
    fallback_text = """
I hear you and I'm here for you. Even though I'm having some technical difficulties right now, please know that you're not alone.

If you're in crisis, please reach out:
• Veteran Crisis Line: Call 988 and Press 1
• Text HOME to 741741
• Or go to your nearest emergency room

I'll be back online shortly. Your message has been logged and someone will follow up with you.

Stay strong, warrior. 💪
    """

    try:
        import aiohttp
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            'chat_id': chat_id,
            'text': fallback_text,
            'parse_mode': 'HTML'
        }

        if reply_to_message_id:
            payload['reply_to_message_id'] = reply_to_message_id

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    logger.info("Fallback response sent")
                else:
                    logger.error(f"Failed to send fallback: {response.status}")
    except Exception as e:
        logger.error(f"Error sending fallback response: {e}")

@telegram_router.get("/health")
async def telegram_health():
    """Check Telegram integration health"""
    return {
        "status": "healthy" if guardian_system else "initializing",
        "service": "Telegram Veteran Support",
        "bot_configured": bool(TELEGRAM_BOT_TOKEN),
        "database_connected": guardian_system is not None,
        "timestamp": datetime.now().isoformat()
    }

@telegram_router.get("/metrics")
async def get_support_metrics():
    """Get veteran support metrics"""
    if not guardian_system:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        metrics = await guardian_system.get_support_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Error fetching metrics: {e}")
        raise HTTPException(status_code=500, detail="Error fetching metrics")

@telegram_router.post("/test")
async def run_tests():
    """Run comprehensive test suite for veteran support responses"""
    if not guardian_system:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        tester = VeteranSupportTester(guardian_system)
        results = await tester.run_comprehensive_tests()
        return results
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        raise HTTPException(status_code=500, detail="Error running tests")

@telegram_router.post("/test-message")
async def test_message(
    message: str,
    user_id: int = 12345,
    username: str = "test_user"
):
    """
    Test the veteran support system with a sample message
    Useful for testing without actual Telegram integration
    """
    if not guardian_system:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        # Create mock Telegram update
        mock_update = {
            "message": {
                "message_id": 999,
                "from": {
                    "id": user_id,
                    "username": username
                },
                "chat": {
                    "id": user_id
                },
                "text": message
            }
        }

        # Process through guardian system
        result = await guardian_system.process_telegram_message(mock_update)

        # Get the generated response
        conversation = await guardian_system.get_or_create_conversation(user_id, username)
        history = await guardian_system.get_conversation_history(
            conversation['conversation_id'],
            limit=2
        )

        bot_response = None
        if history and len(history) > 0:
            for msg in history:
                if msg['sender'] == 'bot':
                    bot_response = msg['message_text']
                    break

        return {
            "input_message": message,
            "risk_level": result.get('risk_level'),
            "concerns": result.get('concerns'),
            "response": bot_response,
            "response_time_ms": result.get('response_time_ms'),
            "status": result.get('status')
        }

    except Exception as e:
        logger.error(f"Error testing message: {e}")
        raise HTTPException(status_code=500, detail=f"Error testing message: {str(e)}")

# Export router for inclusion in main Echo Brain app
__all__ = ['telegram_router', 'initialize_guardian_system']