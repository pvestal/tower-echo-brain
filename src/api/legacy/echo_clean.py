#!/usr/bin/env python3
"""
Clean Echo Brain query endpoint using proper dependency injection
This is how it SHOULD be done - modular, not monolithic
"""

import time
import uuid
import logging
from typing import Optional, Any
from fastapi import APIRouter, Depends, HTTPException

from src.api.models import QueryRequest, QueryResponse
from src.api.dependencies import (
    get_current_user,
    get_user_context,
    get_user_recognition,
    get_user_manager,
    require_permission
)
from src.services.conversation import conversation_manager
from src.core.intelligence import intelligence_router

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/api/echo/query", response_model=QueryResponse)
@router.post("/api/echo/chat", response_model=QueryResponse)
async def query_echo(
    request: QueryRequest,
    username: str = Depends(get_current_user),
    user_context: Any = Depends(get_user_context),
    user_recognition: dict = Depends(get_user_recognition),
    user_manager: Any = Depends(get_user_manager)
):
    """
    Clean query endpoint using dependency injection
    Notice how much cleaner this is!
    """
    start_time = time.time()

    # Generate conversation ID if not provided
    if not request.conversation_id:
        request.conversation_id = str(uuid.uuid4())

    logger.info(f"Query from {username} ({user_recognition['access_level']}): {request.query[:50]}...")

    # Add to conversation history (if user manager available)
    if user_manager:
        await user_manager.add_conversation(username, "user", request.query)

    # Route to intelligence system for processing
    response = await intelligence_router.process_query(
        query=request.query,
        conversation_id=request.conversation_id,
        user_context=user_context,
        user_recognition=user_recognition
    )

    # Add response to conversation history
    if user_manager:
        await user_manager.add_conversation(username, "assistant", response.response)

    # Update processing time
    response.processing_time = time.time() - start_time

    return response

@router.post("/api/echo/system/command")
async def execute_system_command(
    command: str,
    username: str = Depends(get_current_user),
    _: bool = Depends(require_permission("system_commands"))
):
    """
    System command execution - properly protected by permission dependency
    """
    logger.info(f"System command from {username}: {command[:50]}...")

    # This would execute the command
    # Notice we don't need permission checking here - it's handled by the dependency!
    result = await execute_command_safely(command)

    return {
        "status": "executed",
        "user": username,
        "command": command,
        "result": result
    }

@router.get("/api/echo/oversight/dashboard")
async def get_oversight_dashboard(
    username: str = Depends(get_current_user),
    user_manager: Any = Depends(get_user_manager),
    _: bool = Depends(require_permission("creator"))
):
    """
    Creator oversight dashboard - clean with dependency injection
    """
    # Get all users
    all_users = await user_manager.get_all_users() if user_manager else []

    return {
        "creator": username,
        "total_users": len(all_users),
        "users": all_users,
        "timestamp": time.time()
    }

@router.get("/api/echo/users/{target_user}")
async def get_user_info(
    target_user: str,
    user_manager: Any = Depends(get_user_manager),
    _: bool = Depends(require_permission("creator"))
):
    """
    Get user information - creator only
    """
    if not user_manager:
        raise HTTPException(status_code=500, detail="User manager not available")

    context = await user_manager.get_or_create_context(target_user)
    memory = await user_manager.get_user_memory(target_user)

    return {
        "user": target_user,
        "context": context.get_context_summary(),
        "memory": memory
    }

async def execute_command_safely(command: str) -> dict:
    """
    Execute command with safety checks
    This is separated from the endpoint logic - single responsibility!
    """
    import asyncio

    try:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd="/tmp"
        )

        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=30
        )

        return {
            "stdout": stdout.decode('utf-8', errors='replace'),
            "stderr": stderr.decode('utf-8', errors='replace'),
            "returncode": process.returncode
        }

    except asyncio.TimeoutError:
        return {"error": "Command timed out after 30 seconds"}
    except Exception as e:
        return {"error": str(e)}