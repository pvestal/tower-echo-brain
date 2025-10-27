#!/usr/bin/env python3
"""
Dedicated Veteran Guardian Bot Endpoints
Separate API endpoints specifically for veteran support bot functionality
"""

import os
import json
import logging
from typing import Dict, Optional
from fastapi import APIRouter, HTTPException, Request, BackgroundTasks, Depends
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

# Create dedicated FastAPI router for Veteran Guardian Bot
veteran_router = APIRouter(prefix="/api/veteran-guardian", tags=["veteran-guardian"])

# Configuration from environment or vault
VETERAN_BOT_TOKEN = os.getenv("VETERAN_BOT_TOKEN", "")
VETERAN_WEBHOOK_SECRET = os.getenv("VETERAN_WEBHOOK_SECRET", "veteran_guardian_secret_2025")
VETERAN_SUPPORT_CHANNEL_ID = os.getenv("VETERAN_SUPPORT_CHANNEL_ID", "")
VETERAN_ALERT_CHANNEL_ID = os.getenv("VETERAN_ALERT_CHANNEL_ID", "")

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', '5432')),
    'database': os.getenv('DB_NAME', 'echo_brain'),
    'user': os.getenv('DB_USER', 'patrick'),
    'password': os.getenv('DB_PASSWORD', 'patrick123')
}

# Veteran Bot configuration
VETERAN_BOT_CONFIG = {
    'bot_token': VETERAN_BOT_TOKEN,
    'support_channel_id': VETERAN_SUPPORT_CHANNEL_ID,
    'alert_channel_id': VETERAN_ALERT_CHANNEL_ID
}

# Initialize the veteran guardian system
veteran_guardian = None

def initialize_veteran_guardian():
    """Initialize the veteran guardian support system"""
    global veteran_guardian
    try:
        veteran_guardian = VeteranGuardianSystem(DB_CONFIG, VETERAN_BOT_CONFIG)
        logger.info("🎖️ Veteran Guardian System initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Veteran Guardian System: {e}")
        return False

@veteran_router.on_event("startup")
async def startup_veteran_guardian():
    """Initialize guardian system on startup"""
    if not initialize_veteran_guardian():
        logger.warning("Veteran Guardian system not initialized, will retry on first request")

@veteran_router.post("/webhook/{secret}")
async def veteran_guardian_webhook(
    secret: str,
    request: Request,
    background_tasks: BackgroundTasks
):
    """
    Dedicated webhook for Veteran Guardian Bot
    Processes messages specifically for veteran mental health support
    """
    # Verify webhook secret
    if secret != VETERAN_WEBHOOK_SECRET:
        logger.warning(f"Invalid veteran webhook secret attempted: {secret}")
        raise HTTPException(status_code=403, detail="Invalid webhook secret")

    # Initialize guardian system if not ready
    if not veteran_guardian:
        if not initialize_veteran_guardian():
            raise HTTPException(status_code=503, detail="Veteran Guardian service temporarily unavailable")

    try:
        # Parse Telegram update
        update = await request.json()
        logger.info(f"🎖️ Veteran Guardian - Received update from user: {update.get('message', {}).get('from', {}).get('username', 'Unknown')}")

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

        # Log incoming veteran message
        logger.info(f"🎖️ Processing veteran message from {username} (ID: {user_id}): {text[:100]}...")

        # Process message asynchronously in background for veterans
        background_tasks.add_task(
            process_veteran_support_message,
            update,
            user_id,
            username,
            chat_id,
            text
        )

        # Immediate response to Telegram
        return {"status": "accepted", "processing": True, "service": "veteran_guardian"}

    except Exception as e:
        logger.error(f"🎖️ Error processing veteran webhook: {e}")
        # Send emergency fallback message for veterans
        try:
            await send_veteran_emergency_response(update.get('message', {}).get('chat', {}).get('id'))
        except:
            pass
        raise HTTPException(status_code=500, detail="Veteran support service error")

async def process_veteran_support_message(
    update: Dict,
    user_id: int,
    username: str,
    chat_id: int,
    text: str
):
    """
    Process veteran support message with specialized care
    Enhanced processing for military-specific mental health support
    """
    try:
        start_time = datetime.now()

        # Process through veteran guardian system
        result = await veteran_guardian.process_telegram_message(update)

        # Enhanced logging for veteran support
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"🎖️ Veteran message processed - User: {username}, "
                   f"Risk: {result.get('risk_level')}, "
                   f"Concerns: {result.get('concerns')}, "
                   f"Response time: {processing_time:.0f}ms")

        # Enhanced crisis alerting for veterans
        if result.get('risk_level') in ['critical', 'high']:
            await send_veteran_crisis_alert(user_id, username, text, result, processing_time)

        # Log successful veteran support interaction
        await log_veteran_interaction(user_id, username, text, result, processing_time)

    except Exception as e:
        logger.error(f"🎖️ Error processing veteran support message: {e}")
        # Send veteran-specific fallback response
        try:
            await send_veteran_fallback_response(chat_id, update.get('message', {}).get('message_id'))
        except Exception as fallback_error:
            logger.error(f"🎖️ Failed to send veteran fallback response: {fallback_error}")

async def send_veteran_crisis_alert(user_id: int, username: str, message: str, result: Dict, processing_time: float):
    """
    Send enhanced crisis alert specifically for veteran support team
    """
    risk_level = result.get('risk_level', 'unknown')
    concerns = result.get('concerns', [])

    # Create detailed alert for veteran support team
    alert_message = f"""
🚨 **VETERAN CRISIS ALERT** 🎖️

**Veteran**: @{username} (ID: {user_id})
**Risk Level**: {risk_level.upper()} ⚠️
**Concerns**: {', '.join(concerns)}
**Processing Time**: {processing_time:.0f}ms

**Message Preview**:
"{message[:300]}{'...' if len(message) > 300 else ''}"

**Immediate Actions Taken**:
✅ Therapeutic response sent
✅ Crisis resources provided
✅ Conversation logged for follow-up

**Next Steps**:
• Monitor for follow-up messages
• Consider direct outreach if no response in 1 hour
• Review conversation history for patterns

*This is an automated alert from the Veteran Guardian Support System*
    """

    # Send to veteran support channel if configured
    if VETERAN_SUPPORT_CHANNEL_ID:
        try:
            import aiohttp
            url = f"https://api.telegram.org/bot{VETERAN_BOT_TOKEN}/sendMessage"
            payload = {
                'chat_id': VETERAN_SUPPORT_CHANNEL_ID,
                'text': alert_message,
                'parse_mode': 'Markdown'
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        logger.info("🎖️ Veteran crisis alert sent to support team")
                    else:
                        logger.error(f"🎖️ Failed to send veteran crisis alert: {response.status}")
        except Exception as e:
            logger.error(f"🎖️ Error sending veteran crisis alert: {e}")

async def send_veteran_fallback_response(chat_id: int, reply_to_message_id: Optional[int] = None):
    """
    Send veteran-specific emergency fallback message
    """
    veteran_fallback_text = """
🎖️ **Brother/Sister, I hear you.**

Even though I'm having some technical difficulties right now, please know that **you are not alone** and **your service matters**.

**IMMEDIATE SUPPORT AVAILABLE**:
• **Veteran Crisis Line**: Call **988** and Press **1**
• **Text Support**: Text **HOME** to **741741**
• **Emergency**: **911** or go to your nearest VA or emergency room

**You've survived 100% of your worst days. You'll get through this one too.**

I'll be back online shortly. Your message has been logged and a veteran support specialist will follow up with you.

**Stay strong, warrior.** 🇺🇸
    """

    try:
        import aiohttp
        url = f"https://api.telegram.org/bot{VETERAN_BOT_TOKEN}/sendMessage"
        payload = {
            'chat_id': chat_id,
            'text': veteran_fallback_text,
            'parse_mode': 'Markdown'
        }

        if reply_to_message_id:
            payload['reply_to_message_id'] = reply_to_message_id

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    logger.info("🎖️ Veteran fallback response sent")
                else:
                    logger.error(f"🎖️ Failed to send veteran fallback: {response.status}")
    except Exception as e:
        logger.error(f"🎖️ Error sending veteran fallback response: {e}")

async def send_veteran_emergency_response(chat_id: int):
    """
    Send immediate emergency response for system failures
    """
    emergency_text = """
🚨 **EMERGENCY VETERAN SUPPORT** 🎖️

If you're in immediate danger:
• **Call 988, Press 1** (Veteran Crisis Line)
• **Text HOME to 741741**
• **Call 911**

**You matter. Your life has value. Help is available.**
    """

    try:
        import aiohttp
        url = f"https://api.telegram.org/bot{VETERAN_BOT_TOKEN}/sendMessage"
        payload = {
            'chat_id': chat_id,
            'text': emergency_text,
            'parse_mode': 'Markdown'
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                logger.info(f"🎖️ Emergency response sent: {response.status}")
    except Exception as e:
        logger.error(f"🎖️ Failed to send emergency response: {e}")

async def log_veteran_interaction(user_id: int, username: str, message: str, result: Dict, processing_time: float):
    """
    Log veteran interaction for analytics and improvement
    """
    try:
        interaction_log = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'username': username,
            'message_length': len(message),
            'risk_level': result.get('risk_level'),
            'concerns': result.get('concerns', []),
            'processing_time_ms': processing_time,
            'status': result.get('status'),
            'service': 'veteran_guardian'
        }

        logger.info(f"🎖️ Veteran interaction logged: {json.dumps(interaction_log)}")
    except Exception as e:
        logger.error(f"🎖️ Failed to log veteran interaction: {e}")

@veteran_router.get("/health")
async def veteran_guardian_health():
    """Check Veteran Guardian Bot health"""
    return {
        "status": "healthy" if veteran_guardian else "initializing",
        "service": "Veteran Guardian Bot",
        "specialization": "Military Mental Health & Crisis Support",
        "bot_configured": bool(VETERAN_BOT_TOKEN),
        "database_connected": veteran_guardian is not None,
        "crisis_resources": ["988 Press 1", "Text HOME to 741741", "911"],
        "timestamp": datetime.now().isoformat()
    }

@veteran_router.get("/metrics")
async def get_veteran_support_metrics():
    """Get veteran-specific support metrics"""
    if not veteran_guardian:
        raise HTTPException(status_code=503, detail="Veteran Guardian service not initialized")

    try:
        metrics = await veteran_guardian.get_support_metrics()
        # Add veteran-specific metadata
        metrics['service'] = 'veteran_guardian'
        metrics['specialization'] = 'military_mental_health'
        return metrics
    except Exception as e:
        logger.error(f"🎖️ Error fetching veteran metrics: {e}")
        raise HTTPException(status_code=500, detail="Error fetching veteran metrics")

@veteran_router.post("/crisis-test")
async def run_veteran_crisis_tests():
    """Run comprehensive crisis intervention tests for veteran scenarios"""
    if not veteran_guardian:
        raise HTTPException(status_code=503, detail="Veteran Guardian service not initialized")

    try:
        tester = VeteranSupportTester(veteran_guardian)
        results = await tester.run_comprehensive_tests()

        # Add veteran-specific test metadata
        results['service'] = 'veteran_guardian'
        results['test_type'] = 'crisis_intervention'
        results['military_specific'] = True

        return results
    except Exception as e:
        logger.error(f"🎖️ Error running veteran crisis tests: {e}")
        raise HTTPException(status_code=500, detail="Error running veteran crisis tests")

@veteran_router.post("/support-message")
async def test_veteran_support_message(
    message: str,
    user_id: int = 12345,
    username: str = "test_veteran"
):
    """
    Test the veteran support system with a sample message
    Specialized endpoint for testing veteran-specific scenarios
    """
    if not veteran_guardian:
        raise HTTPException(status_code=503, detail="Veteran Guardian service not initialized")

    try:
        # Create mock Telegram update for veteran testing
        mock_update = {
            "message": {
                "message_id": 999,
                "from": {
                    "id": user_id,
                    "username": username,
                    "first_name": "Test"
                },
                "chat": {
                    "id": user_id,
                    "type": "private"
                },
                "text": message,
                "date": int(datetime.now().timestamp())
            }
        }

        start_time = datetime.now()

        # Process through veteran guardian system
        result = await veteran_guardian.process_telegram_message(mock_update)

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        # Get the generated response
        conversation = await veteran_guardian.get_or_create_conversation(user_id, username)
        history = await veteran_guardian.get_conversation_history(
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
            "service": "veteran_guardian",
            "input_message": message,
            "risk_level": result.get('risk_level'),
            "concerns": result.get('concerns'),
            "therapeutic_response": bot_response,
            "processing_time_ms": processing_time,
            "status": result.get('status'),
            "crisis_resources_included": "988" in (bot_response or ""),
            "military_context_aware": any(word in (bot_response or "").lower()
                                        for word in ["warrior", "service", "brother", "sister", "battle"]),
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "test_user": f"{username} ({user_id})",
                "specialized_for": "veteran_mental_health"
            }
        }

    except Exception as e:
        logger.error(f"🎖️ Error testing veteran support message: {e}")
        raise HTTPException(status_code=500, detail=f"Error testing veteran support message: {str(e)}")

@veteran_router.get("/status")
async def veteran_guardian_status():
    """Get detailed status of Veteran Guardian Bot system"""
    try:
        status = {
            "service": "Veteran Guardian Bot",
            "status": "operational" if veteran_guardian else "initializing",
            "specialization": "Military Mental Health & Crisis Intervention",
            "capabilities": [
                "PTSD Crisis Support",
                "Addiction Counseling",
                "Suicide Prevention",
                "Combat Trauma Support",
                "Military Cultural Competence",
                "24/7 Crisis Intervention"
            ],
            "configuration": {
                "bot_token_set": bool(VETERAN_BOT_TOKEN),
                "support_channel_configured": bool(VETERAN_SUPPORT_CHANNEL_ID),
                "alert_channel_configured": bool(VETERAN_ALERT_CHANNEL_ID),
                "database_connected": veteran_guardian is not None
            },
            "endpoints": {
                "webhook": f"/api/veteran-guardian/webhook/{VETERAN_WEBHOOK_SECRET}",
                "health": "/api/veteran-guardian/health",
                "metrics": "/api/veteran-guardian/metrics",
                "test": "/api/veteran-guardian/support-message",
                "crisis_test": "/api/veteran-guardian/crisis-test"
            },
            "emergency_resources": {
                "veteran_crisis_line": "988 Press 1",
                "crisis_text": "Text HOME to 741741",
                "emergency": "911"
            },
            "timestamp": datetime.now().isoformat()
        }

        if veteran_guardian:
            try:
                metrics = await veteran_guardian.get_support_metrics()
                status["current_metrics"] = {
                    "total_veterans_supported": metrics.get('overall', {}).get('total_veterans', 0),
                    "total_conversations": metrics.get('overall', {}).get('total_conversations', 0),
                    "crisis_interventions": metrics.get('overall', {}).get('total_interventions', 0)
                }
            except:
                status["current_metrics"] = "metrics_unavailable"

        return status

    except Exception as e:
        logger.error(f"🎖️ Error getting veteran guardian status: {e}")
        raise HTTPException(status_code=500, detail="Error getting status")

# Export router for inclusion in main Echo Brain app
__all__ = ['veteran_router', 'initialize_veteran_guardian']