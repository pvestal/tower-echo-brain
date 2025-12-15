#!/usr/bin/env python3
"""
Creator Oversight Dashboard for Echo Brain
Provides complete visibility into Echo's activities for Patrick
"""

from fastapi import APIRouter, HTTPException, Request
from typing import Dict, Any, List
import logging
from datetime import datetime, timedelta
from src.core.echo_identity import get_echo_identity
from src.core.user_context_manager import get_user_context_manager
from src.integrations.vault_manager import get_vault_manager
from src.tasks.task_queue import TaskQueue
from src.db.database import database
import psutil
import asyncio

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/echo/oversight", tags=["oversight"])

async def verify_creator(request: Request) -> bool:
    """Verify the request is from the creator"""
    # Check various authentication methods
    username = request.headers.get("X-Username", "")
    if username == "patrick":
        return True

    # Check query params
    if request.query_params.get("user") == "patrick":
        return True

    # Check authorization header
    auth_header = request.headers.get("Authorization", "")
    if "creator" in auth_header.lower() or "patrick" in auth_header.lower():
        return True

    return False

@router.get("/dashboard")
async def get_oversight_dashboard(request: Request) -> Dict[str, Any]:
    """Get complete oversight dashboard (creator only)"""
    if not await verify_creator(request):
        raise HTTPException(status_code=403, detail="Creator access required")

    echo_identity = get_echo_identity()
    user_manager = await get_user_context_manager()
    vault_manager = await get_vault_manager()

    # Get system metrics
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')

    # Get recent tasks from database
    recent_tasks_query = """
        SELECT task_type, status, priority, created_at, completed_at, result
        FROM echo_tasks
        ORDER BY created_at DESC
        LIMIT 20
    """
    recent_tasks = await database.fetch_all(recent_tasks_query)

    # Get active users
    all_users = await user_manager.get_all_users()

    # Get recent conversations
    recent_conversations_query = """
        SELECT user_id, message, response, timestamp
        FROM echo_conversations
        ORDER BY timestamp DESC
        LIMIT 10
    """
    recent_conversations = await database.fetch_all(recent_conversations_query)

    # Get service health
    services = {
        "echo_brain": "running" if cpu_percent > 0 else "stopped",
        "background_worker": await check_service_health("echo-worker"),
        "ollama": await check_service_health("ollama"),
        "comfyui": await check_service_health("comfyui"),
        "vault": "connected" if vault_manager.is_initialized else "disconnected",
        "telegram": await check_telegram_status(),
        "database": "connected"
    }

    # Get learning metrics
    learning_metrics_query = """
        SELECT COUNT(*) as total_learnings,
               COUNT(DISTINCT user_id) as unique_users,
               COUNT(CASE WHEN created_at > NOW() - INTERVAL '24 hours' THEN 1 END) as today_learnings
        FROM echo_learning
    """
    learning = await database.fetch_one(learning_metrics_query)

    # Get autonomous behavior stats
    autonomous_stats_query = """
        SELECT behavior_type, COUNT(*) as count,
               COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed
        FROM echo_autonomous_actions
        WHERE created_at > NOW() - INTERVAL '7 days'
        GROUP BY behavior_type
    """
    autonomous_stats = await database.fetch_all(autonomous_stats_query)

    dashboard = {
        "identity": echo_identity.get_status_report(),
        "system_metrics": {
            "cpu_percent": cpu_percent,
            "memory": {
                "total_gb": round(memory.total / 1024**3, 2),
                "used_gb": round(memory.used / 1024**3, 2),
                "percent": memory.percent
            },
            "disk": {
                "total_gb": round(disk.total / 1024**3, 2),
                "used_gb": round(disk.used / 1024**3, 2),
                "percent": disk.percent
            },
            "uptime": await get_uptime()
        },
        "services": services,
        "active_users": {
            "total": len(all_users),
            "users": all_users
        },
        "recent_tasks": [
            {
                "type": task["task_type"],
                "status": task["status"],
                "priority": task["priority"],
                "created": task["created_at"].isoformat() if task["created_at"] else None,
                "completed": task["completed_at"].isoformat() if task["completed_at"] else None,
                "result": task["result"][:100] if task["result"] else None
            }
            for task in recent_tasks
        ],
        "recent_conversations": [
            {
                "user": conv["user_id"],
                "message": conv["message"][:100] if conv["message"] else "",
                "response": conv["response"][:100] if conv["response"] else "",
                "timestamp": conv["timestamp"].isoformat() if conv["timestamp"] else None
            }
            for conv in recent_conversations
        ],
        "learning_metrics": {
            "total_learnings": learning["total_learnings"] if learning else 0,
            "unique_users": learning["unique_users"] if learning else 0,
            "today_learnings": learning["today_learnings"] if learning else 0
        },
        "autonomous_behaviors": [
            {
                "type": stat["behavior_type"],
                "total": stat["count"],
                "completed": stat["completed"],
                "success_rate": round(stat["completed"] / stat["count"] * 100, 1) if stat["count"] > 0 else 0
            }
            for stat in autonomous_stats
        ],
        "vault_access_log": vault_manager.get_access_log("patrick")[-10:],  # Last 10 vault accesses
        "capabilities": echo_identity.capabilities,
        "directives": echo_identity.directives
    }

    return dashboard

@router.get("/users/{username}")
async def get_user_details(username: str, request: Request) -> Dict[str, Any]:
    """Get detailed information about a specific user (creator only)"""
    if not await verify_creator(request):
        raise HTTPException(status_code=403, detail="Creator access required")

    user_manager = await get_user_context_manager()
    context = await user_manager.get_or_create_context(username)

    # Get user-specific tasks
    user_tasks_query = """
        SELECT task_type, status, COUNT(*) as count
        FROM echo_tasks
        WHERE user_id = %s
        GROUP BY task_type, status
    """
    user_tasks = await database.fetch_all(user_tasks_query, (username,))

    # Get user's recent activity
    activity_query = """
        SELECT timestamp, action_type, details
        FROM echo_user_activity
        WHERE user_id = %s
        ORDER BY timestamp DESC
        LIMIT 50
    """
    activities = await database.fetch_all(activity_query, (username,))

    return {
        "context": context.get_context_summary(),
        "memory": await user_manager.get_user_memory(username),
        "task_summary": [
            {"type": task["task_type"], "status": task["status"], "count": task["count"]}
            for task in user_tasks
        ],
        "recent_activity": [
            {
                "timestamp": act["timestamp"].isoformat() if act["timestamp"] else None,
                "action": act["action_type"],
                "details": act["details"]
            }
            for act in activities
        ]
    }

@router.post("/users/{username}/preferences")
async def update_user_preference(username: str, key: str, value: Any, request: Request) -> Dict[str, Any]:
    """Update user preferences (creator only)"""
    if not await verify_creator(request):
        raise HTTPException(status_code=403, detail="Creator access required")

    user_manager = await get_user_context_manager()
    success = await user_manager.update_preference(username, key, value)

    return {"success": success, "username": username, "preference": {key: value}}

@router.post("/users/{username}/permissions")
async def update_user_permissions(username: str, permissions: Dict[str, bool], request: Request) -> Dict[str, Any]:
    """Update user permissions (creator only)"""
    if not await verify_creator(request):
        raise HTTPException(status_code=403, detail="Creator access required")

    user_manager = await get_user_context_manager()
    context = await user_manager.get_or_create_context(username)

    # Update permissions
    context.permissions.update(permissions)
    await user_manager._save_context(context)

    return {"success": True, "username": username, "permissions": context.permissions}

@router.get("/vault/credentials")
async def get_all_credentials(request: Request) -> Dict[str, Any]:
    """Get all stored credentials (creator only)"""
    if not await verify_creator(request):
        raise HTTPException(status_code=403, detail="Creator access required")

    vault_manager = await get_vault_manager()
    credentials = vault_manager.get_all_credentials("patrick")

    # Mask sensitive values for display
    masked_creds = {}
    for service, creds in credentials.items():
        if isinstance(creds, dict):
            masked_creds[service] = {
                key: f"***{str(value)[-4:]}" if "password" in key.lower() or "token" in key.lower() or "key" in key.lower()
                else value
                for key, value in creds.items()
            }
        else:
            masked_creds[service] = creds

    return {
        "credentials": masked_creds,
        "vault_connected": vault_manager.is_initialized,
        "services_configured": list(credentials.keys())
    }

@router.post("/test/capability")
async def test_echo_capability(capability: str, test_data: Dict[str, Any], request: Request) -> Dict[str, Any]:
    """Test a specific Echo capability (creator only)"""
    if not await verify_creator(request):
        raise HTTPException(status_code=403, detail="Creator access required")

    results = {"capability": capability, "test_data": test_data, "results": {}}

    try:
        if capability == "telegram":
            from src.integrations.telegram_client import get_telegram_client
            client = await get_telegram_client()
            success = await client.send_notification("Test", "Testing Echo Telegram capability")
            results["results"] = {"telegram_sent": success}

        elif capability == "ollama":
            from src.integrations.ollama_client import get_ollama_client
            client = await get_ollama_client()
            response = await client.generate("Say 'Echo test successful' in 5 words or less")
            results["results"] = {"ollama_response": response}

        elif capability == "comfyui":
            from src.integrations.comfyui_client import get_comfyui_client
            client = await get_comfyui_client()
            status = await client.get_queue_status()
            results["results"] = {"comfyui_status": status}

        elif capability == "identity":
            echo = get_echo_identity()
            user_check = echo.recognize_user("patrick")
            results["results"] = {"creator_recognized": user_check}

        else:
            results["results"] = {"error": f"Unknown capability: {capability}"}

    except Exception as e:
        results["results"] = {"error": str(e)}

    return results

# Helper functions
async def check_service_health(service_name: str) -> str:
    """Check if a systemd service is running"""
    try:
        proc = await asyncio.create_subprocess_exec(
            "systemctl", "is-active", service_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL
        )
        stdout, _ = await proc.communicate()
        return "running" if stdout.decode().strip() == "active" else "stopped"
    except:
        return "unknown"

async def check_telegram_status() -> str:
    """Check Telegram bot status"""
    try:
        vault_manager = await get_vault_manager()
        creds = vault_manager.get_telegram_credentials()
        return "configured" if creds.get("bot_token") else "not configured"
    except:
        return "error"

async def get_uptime() -> str:
    """Get system uptime"""
    try:
        with open('/proc/uptime', 'r') as f:
            uptime_seconds = float(f.readline().split()[0])
            days = int(uptime_seconds // 86400)
            hours = int((uptime_seconds % 86400) // 3600)
            return f"{days}d {hours}h"
    except:
        return "unknown"