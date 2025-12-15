#!/usr/bin/env python3
"""
Testing and debugging API routes for Echo Brain
"""
import asyncio
import json
import time
import uuid
import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from src.api.models import ExecuteRequest, ExecuteResponse, TestRequest, VoiceNotificationRequest, VoiceStatusRequest
from src.db.database import database
from src.services.testing import testing_framework
from src.utils.helpers import safe_executor, tower_orchestrator
from src.core.echo.echo_brain_thoughts import echo_brain

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/api/echo/execute")
async def execute_command(request: ExecuteRequest):
    """Execute shell commands safely"""
    start_time = time.time()

    if not request.conversation_id:
        request.conversation_id = str(uuid.uuid4())

    logger.info(f"🔧 Execute command: {request.command}")

    try:
        result = await safe_executor.execute(request.command, allow_all=not request.safe_mode)

        response = ExecuteResponse(
            command=request.command,
            success=result["success"],
            output=result["output"],
            error=result.get("error"),
            exit_code=result["exit_code"],
            processing_time=result["processing_time"],
            conversation_id=request.conversation_id,
            safety_checks=result["safety_checks"]
        )

        return response

    except Exception as e:
        logger.error(f"Command execution failed: {e}")
        processing_time = time.time() - start_time

        response = ExecuteResponse(
            command=request.command,
            success=False,
            output="",
            error=str(e),
            exit_code=-1,
            processing_time=processing_time,
            conversation_id=request.conversation_id,
            safety_checks={"passed_safety_check": False, "safe_mode_enabled": request.safe_mode, "safety_message": str(e)}
        )

        return response

@router.get("/api/echo/stream")
async def stream_brain_activity():
    """Stream real-time brain activity"""
    async def generate():
        try:
            while True:
                brain_state = echo_brain.get_brain_state()
                data = json.dumps({
                    "timestamp": datetime.now().isoformat(),
                    "brain_state": brain_state
                })
                yield f"data: {data}\n\n"
                await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/plain")

@router.post("/api/echo/voice/notify")
async def voice_notify(request: VoiceNotificationRequest):
    """Send REAL voice notification using Tower voice service"""
    logger.info(f"🗣️ Voice notification: {request.message[:50]}...")

    try:
        # Use REAL Tower orchestrator for voice generation
        result = await tower_orchestrator.generate_voice(
            text=request.message,
            character=request.character
        )

        if result["success"]:
            return {
                "success": True,
                "message": "Voice notification sent successfully",
                "character": request.character,
                "priority": request.priority,
                "voice_result": result
            }
        else:
            return {
                "success": False,
                "error": f"Voice generation failed: {result.get('error', 'Unknown error')}"
            }
    except Exception as e:
        logger.error(f"Voice notification failed: {e}")
        return {"success": False, "error": str(e)}

@router.post("/api/echo/voice/status")
async def voice_status(request: VoiceStatusRequest):
    """Get voice system status"""
    try:
        return {
            "status": "ready",
            "user_id": request.user_id,
            "available_characters": ["yukiko", "akira", "system"],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@router.get("/api/echo/voice/characters")
async def get_voice_characters():
    """Get available voice characters"""
    return {
        "characters": [
            {"name": "yukiko", "description": "Calm, analytical assistant"},
            {"name": "akira", "description": "Energetic, creative assistant"},
            {"name": "system", "description": "System announcements"}
        ]
    }

@router.post("/api/echo/test/{target}")
async def test_service(target: str, request: TestRequest):
    """Test a Tower service"""
    logger.info(f"🧪 Testing service: {target}")

    try:
        if request.test_type == "debug":
            result = await testing_framework.run_debug_analysis(target)
        else:
            result = await testing_framework.run_universal_test(target)

        return result
    except Exception as e:
        logger.error(f"Service test failed: {e}")
        return {"success": False, "error": str(e)}

@router.post("/api/echo/debug/{service}")
async def debug_service(service: str):
    """Debug a specific Tower service with comprehensive analysis"""
    logger.info(f"🐛 Debugging service: {service}")

    try:
        debug_result = await testing_framework.run_debug_analysis(service)
        return {
            "service": service,
            "debug_result": debug_result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Service debug failed: {e}")
        return {"success": False, "error": str(e)}

@router.get("/api/echo/testing/capabilities")
async def get_testing_capabilities():
    """Get available testing capabilities"""
    try:
        capabilities = await testing_framework.get_testing_capabilities()
        return {
            "capabilities": capabilities,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get testing capabilities: {e}")
        return {"error": str(e)}