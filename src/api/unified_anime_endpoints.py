#!/usr/bin/env python3
"""
Unified Anime API Endpoints
Exposes the unified anime interface for cross-platform coordination.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/echo/unified", tags=["Unified Anime Interface"])

class TelegramCommandRequest(BaseModel):
    """Telegram command processing request"""
    message_text: str = Field(..., description="Full Telegram message text")
    user_id: str = Field(..., description="Telegram user ID")
    chat_id: int = Field(..., description="Telegram chat ID")
    message_id: int = Field(..., description="Telegram message ID")

class BrowserGenerationRequest(BaseModel):
    """Browser generation request"""
    prompt: str = Field(..., description="Generation prompt")
    project: Optional[str] = Field(None, description="Project name")
    character: Optional[str] = Field(None, description="Character name")
    style: Optional[str] = Field(None, description="Style preference")
    session_id: Optional[str] = Field(None, description="Browser session ID")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Additional parameters")

class ContextContinuityRequest(BaseModel):
    """Cross-platform context continuity request"""
    user_id: str = Field(..., description="User ID")
    target_platform: str = Field(..., description="Target platform (telegram, browser, echo_brain)")

@router.post("/telegram/command")
async def process_telegram_command(
    request: TelegramCommandRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Process Telegram command through unified interface.

    Handles /generate, /character, /project commands and routes them
    through Echo's intelligent coordination system.
    """
    try:
        logger.info(f"📱 Processing Telegram command: {request.message_text[:50]}...")

        # Import here to avoid circular imports
        from integrations.unified_anime_interface import process_telegram_generate

        result = await process_telegram_generate(
            request.message_text,
            request.user_id,
            request.chat_id,
            request.message_id
        )

        logger.info(f"📱 Telegram command processed: {result.get('success', False)}")

        return {
            "success": True,
            "platform": "telegram",
            "command_result": result,
            "telegram_response": result.get("platform_response", {}),
            "processing_info": {
                "message_id": request.message_id,
                "chat_id": request.chat_id,
                "user_id": request.user_id
            }
        }

    except Exception as e:
        logger.error(f"Telegram command processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/browser/generate")
async def process_browser_generation(
    request: BrowserGenerationRequest,
    user_id: str = "patrick",
    session_token: str = "default",
    background_tasks: BackgroundTasks = None
) -> Dict[str, Any]:
    """
    Process browser generation request through unified interface.

    Provides seamless browser-based anime generation with full
    Echo coordination and context management.
    """
    try:
        logger.info(f"🌐 Processing browser generation: {request.prompt[:50]}...")

        # Import here to avoid circular imports
        from integrations.unified_anime_interface import process_browser_generation

        request_data = {
            "prompt": request.prompt,
            "project": request.project,
            "character": request.character,
            "style": request.style,
            "session_id": request.session_id,
            "parameters": request.parameters or {}
        }

        result = await process_browser_generation(request_data, user_id, session_token)

        logger.info(f"🌐 Browser generation processed: {result.get('success', False)}")

        return {
            "success": True,
            "platform": "browser",
            "generation_result": result,
            "browser_response": result.get("platform_response", {}),
            "processing_info": {
                "user_id": user_id,
                "session_token": session_token,
                "session_id": request.session_id
            }
        }

    except Exception as e:
        logger.error(f"Browser generation processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/context/continuity")
async def get_context_continuity(
    user_id: str = "patrick",
    target_platform: str = "echo_brain"
) -> Dict[str, Any]:
    """
    Get cross-platform context continuity data.

    Enables seamless transitions between Telegram, browser, and
    Echo Brain interfaces with preserved context.
    """
    try:
        logger.info(f"🔄 Getting context continuity for {user_id} -> {target_platform}")

        # Import here to avoid circular imports
        from integrations.unified_anime_interface import get_cross_platform_context

        continuity = await get_cross_platform_context(user_id, target_platform)

        if continuity:
            return {
                "has_continuity": True,
                "continuity_data": continuity,
                "user_id": user_id,
                "target_platform": target_platform,
                "recommendations": {
                    "should_continue_session": continuity.get("session_age_minutes", 0) < 30,
                    "context_strength": "strong" if continuity.get("can_continue") else "weak",
                    "suggested_prompt": f"Continue with {continuity.get('context', {}).get('last_character', 'previous work')}"
                }
            }
        else:
            return {
                "has_continuity": False,
                "user_id": user_id,
                "target_platform": target_platform,
                "message": "No recent session context found"
            }

    except Exception as e:
        logger.error(f"Context continuity retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions/active")
async def get_active_sessions(user_id: str = "patrick") -> Dict[str, Any]:
    """
    Get active sessions across all platforms for a user.

    Shows current anime generation sessions and their contexts.
    """
    try:
        # Import here to avoid circular imports
        from integrations.unified_anime_interface import unified_interface

        sessions = []
        for key, session in unified_interface.session_manager.active_sessions.items():
            if session["user_id"] == user_id:
                sessions.append({
                    "session_id": session["session_id"],
                    "platform": session["platform"].value,
                    "created_at": session["created_at"].isoformat(),
                    "last_activity": session["last_activity"].isoformat(),
                    "context": session["context"]
                })

        return {
            "user_id": user_id,
            "active_sessions": sessions,
            "session_count": len(sessions),
            "platforms": list(set(s["platform"] for s in sessions)),
            "most_recent": max(sessions, key=lambda x: x["last_activity"]) if sessions else None
        }

    except Exception as e:
        logger.error(f"Active sessions retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/session/migrate")
async def migrate_session_context(
    request: ContextContinuityRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Migrate session context between platforms.

    Enables smooth transitions from Telegram to browser or vice versa
    with full context preservation.
    """
    try:
        logger.info(f"🔄 Migrating session context: {request.user_id} -> {request.target_platform}")

        # Import here to avoid circular imports
        from integrations.unified_anime_interface import unified_interface
        from integrations.unified_anime_interface import Platform

        # Determine source platform (find most recent session)
        recent_session = None
        for session in unified_interface.session_manager.active_sessions.values():
            if session["user_id"] == request.user_id:
                if recent_session is None or session["last_activity"] > recent_session["last_activity"]:
                    recent_session = session

        if not recent_session:
            raise HTTPException(status_code=404, detail="No active session found for migration")

        # Perform migration
        migrated_context = await unified_interface.session_manager.migrate_session_context(
            recent_session["platform"],
            Platform(request.target_platform),
            request.user_id
        )

        return {
            "migration_successful": migrated_context is not None,
            "from_platform": recent_session["platform"].value,
            "to_platform": request.target_platform,
            "user_id": request.user_id,
            "migrated_context": migrated_context,
            "migration_timestamp": unified_interface.session_manager.platform_bridges.get(
                f"bridge_{recent_session['platform'].value}_{request.target_platform}_{request.user_id}",
                {}
            ).get("created_at")
        }

    except Exception as e:
        logger.error(f"Session migration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_unified_interface_status() -> Dict[str, Any]:
    """
    Get status of the unified anime interface system.

    Shows system health, active sessions, and platform availability.
    """
    try:
        # Import here to avoid circular imports
        from integrations.unified_anime_interface import unified_interface

        total_sessions = len(unified_interface.session_manager.active_sessions)
        platforms_in_use = set()
        active_users = set()

        for session in unified_interface.session_manager.active_sessions.values():
            platforms_in_use.add(session["platform"].value)
            active_users.add(session["user_id"])

        return {
            "system_status": "operational",
            "unified_interface_active": True,
            "statistics": {
                "total_active_sessions": total_sessions,
                "platforms_in_use": list(platforms_in_use),
                "active_users": len(active_users),
                "platform_bridges": len(unified_interface.session_manager.platform_bridges)
            },
            "capabilities": {
                "telegram_commands": True,
                "browser_generation": True,
                "cross_platform_continuity": True,
                "session_migration": True,
                "intent_classification": True
            },
            "supported_commands": [
                "/generate <prompt>",
                "/character <name> <prompt>",
                "/project <name> <action>",
                "Browser generation with full context"
            ]
        }

    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

logger.info("🌐 Unified Anime API endpoints loaded")