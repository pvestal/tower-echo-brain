#!/usr/bin/env python3
"""
Enhanced Telegram-Echo Execution Bridge
Enables full execution capabilities, unit testing, and autonomous actions via Telegram
"""

import os
import json
import logging
import requests
import asyncio
from typing import Dict, Optional, List
from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create enhanced router
telegram_executor_router = APIRouter(prefix="/api/telegram/executor", tags=["telegram-executor"])

# Configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET", "telegram_webhook_secret_2025")
ECHO_BASE_URL = "http://localhost:8309"

# Command mappings
EXECUTION_COMMANDS = {
    "/execute": "system_command",
    "/exec": "system_command",
    "/run": "system_command",
    "/shell": "system_command",
    "/bash": "system_command",
    "/test": "unit_test",
    "/unittest": "unit_test",
    "/refactor": "code_refactor",
    "/fix": "auto_fix",
    "/repair": "service_repair",
    "/monitor": "system_monitor",
    "/analyze": "code_analyze"
}

async def send_telegram_message(chat_id: int, text: str, parse_mode: str = "Markdown") -> Dict:
    """Send a message via Telegram Bot API"""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    payload = {
        "chat_id": chat_id,
        "text": text[:4096],  # Telegram message limit
        "parse_mode": parse_mode
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")
        raise

async def execute_via_echo(command: str, action_type: str, conversation_id: str) -> Dict:
    """Execute command through Echo's verified execution system"""
    try:
        if action_type == "system_command":
            # Direct system command execution
            payload = {
                "query": command,
                "conversation_id": conversation_id,
                "request_type": "system_command",
                "intelligence_level": "system"
            }
            response = requests.post(f"{ECHO_BASE_URL}/api/echo/query", json=payload, timeout=60)

        elif action_type == "unit_test":
            # Run unit tests
            test_cmd = command.replace("/test", "").replace("/unittest", "").strip()
            if not test_cmd:
                test_cmd = "cd /opt/tower-echo-brain && python -m pytest tests/ -v"

            payload = {
                "action_template": "RUN_TESTS",
                "parameters": {
                    "test_command": test_cmd,
                    "project_path": "/opt/tower-echo-brain"
                },
                "description": f"Running unit tests: {test_cmd}"
            }
            response = requests.post(f"{ECHO_BASE_URL}/api/echo/verified/execute", json=payload, timeout=120)

        elif action_type == "code_refactor":
            # Trigger code refactoring
            target = command.replace("/refactor", "").strip() or "/opt/tower-echo-brain/src"
            payload = {
                "action": "analyze_and_refactor",
                "target_path": target,
                "auto_fix": True
            }
            response = requests.post(f"{ECHO_BASE_URL}/api/echo/autonomous/execute", json=payload, timeout=60)

        elif action_type == "auto_fix":
            # Auto-fix code issues
            target = command.replace("/fix", "").strip() or "/opt/tower-echo-brain"
            payload = {
                "action": "fix_code_issues",
                "project_path": target,
                "languages": ["python", "javascript", "typescript"]
            }
            response = requests.post(f"{ECHO_BASE_URL}/api/echo/autonomous/execute", json=payload, timeout=60)

        elif action_type == "service_repair":
            # Repair services
            service = command.replace("/repair", "").strip()
            if not service:
                # Repair all broken services
                payload = {
                    "action": "repair_all_services"
                }
            else:
                payload = {
                    "service": service,
                    "issue": "service_down",
                    "description": f"Repairing service: {service}"
                }
                response = requests.post(f"{ECHO_BASE_URL}/api/echo/verified/repair", json=payload, timeout=30)

        elif action_type == "system_monitor":
            # Get system monitoring status
            response = requests.get(f"{ECHO_BASE_URL}/api/echo/system/metrics", timeout=10)

        elif action_type == "code_analyze":
            # Analyze code quality
            target = command.replace("/analyze", "").strip() or "/opt/tower-echo-brain"
            payload = {
                "action": "analyze_code_quality",
                "project_path": target
            }
            response = requests.post(f"{ECHO_BASE_URL}/api/echo/autonomous/execute", json=payload, timeout=60)

        else:
            # Default to chat
            payload = {
                "query": command,
                "conversation_id": conversation_id
            }
            response = requests.post(f"{ECHO_BASE_URL}/api/echo/query", json=payload, timeout=60)

        response.raise_for_status()
        return response.json()

    except Exception as e:
        logger.error(f"Error executing via Echo: {e}")
        return {
            "error": str(e),
            "success": False,
            "response": f"❌ Execution failed: {str(e)}"
        }

def format_execution_response(result: Dict, action_type: str) -> str:
    """Format execution result for Telegram"""
    if result.get("error"):
        return f"❌ **Execution Failed**\n```\n{result.get('error')}\n```"

    if action_type == "system_command":
        response = result.get("response", "No output")
        return f"✅ **Command Executed**\n```\n{response[:3500]}\n```"

    elif action_type == "unit_test":
        if result.get("success"):
            return f"✅ **Tests Passed**\n```\n{result.get('stdout', '')[:3500]}\n```"
        else:
            return f"❌ **Tests Failed**\n```\n{result.get('stderr', result.get('stdout', ''))[:3500]}\n```"

    elif action_type in ["code_refactor", "auto_fix"]:
        files_fixed = result.get("files_fixed", 0)
        issues_found = result.get("issues_found", 0)
        return f"🔧 **Code {action_type.replace('_', ' ').title()}**\n" \
               f"Files analyzed: {result.get('files_analyzed', 0)}\n" \
               f"Issues found: {issues_found}\n" \
               f"Files fixed: {files_fixed}\n" \
               f"Score: {result.get('average_score', 'N/A')}/10"

    elif action_type == "service_repair":
        if result.get("success"):
            return f"✅ **Service Repaired**\n{result.get('action_taken', '')}\n" \
                   f"Verification: {result.get('actual_outcome', '')}"
        else:
            return f"❌ **Repair Failed**\n{result.get('actual_outcome', result.get('response', ''))}"

    elif action_type == "system_monitor":
        metrics = result
        return f"📊 **System Metrics**\n" \
               f"CPU: {metrics.get('cpu', {}).get('usage_percent', 0):.1f}%\n" \
               f"Memory: {metrics.get('memory', {}).get('percent', 0):.1f}%\n" \
               f"Disk: {metrics.get('disk', {}).get('percent', 0):.1f}%\n" \
               f"Services: {metrics.get('services', {}).get('healthy', 0)}/{metrics.get('services', {}).get('total', 0)}"

    else:
        return result.get("response", "No response")

@telegram_executor_router.post("/webhook/{secret}")
async def executor_webhook(
    secret: str,
    request: Request,
    background_tasks: BackgroundTasks
):
    """
    Enhanced Telegram webhook for Echo execution capabilities
    Handles system commands, unit tests, refactoring, and repairs
    """
    if secret != TELEGRAM_WEBHOOK_SECRET:
        logger.warning(f"Invalid webhook secret attempted: {secret}")
        raise HTTPException(status_code=403, detail="Invalid webhook secret")

    try:
        update = await request.json()

        if 'message' not in update:
            return {"status": "ignored", "reason": "not a message"}

        message = update['message']
        chat_id = message['chat']['id']
        message_text = message.get('text', '')
        message_id = message.get('message_id')
        from_user = message.get('from', {})
        user_id = str(from_user.get('id', 'unknown'))

        if not message_text.strip():
            return {"status": "ignored", "reason": "empty message"}

        logger.info(f"🚀 Executor received: {message_text} from user {user_id}")

        # Check if this is an execution command
        action_type = None
        for cmd_prefix, cmd_type in EXECUTION_COMMANDS.items():
            if message_text.startswith(cmd_prefix):
                action_type = cmd_type
                break

        # If not an execution command, check for inline execution request
        if not action_type and any(keyword in message_text.lower() for keyword in
                                   ["execute", "run", "test", "refactor", "fix", "repair"]):
            # Infer action type from content
            if "test" in message_text.lower():
                action_type = "unit_test"
            elif "refactor" in message_text.lower():
                action_type = "code_refactor"
            elif "fix" in message_text.lower():
                action_type = "auto_fix"
            elif "repair" in message_text.lower():
                action_type = "service_repair"
            else:
                action_type = "system_command"

        conversation_id = f"telegram_executor_{chat_id}"

        # Process in background
        background_tasks.add_task(
            process_execution,
            message_text,
            action_type or "chat",
            conversation_id,
            chat_id,
            message_id
        )

        return {"status": "accepted", "processing": True}

    except Exception as e:
        logger.error(f"Webhook error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

async def process_execution(
    command: str,
    action_type: str,
    conversation_id: str,
    chat_id: int,
    message_id: Optional[int]
):
    """Process execution command through Echo"""
    try:
        # Send typing indicator
        await send_telegram_message(chat_id, "⚙️ Executing...")

        # Execute through Echo
        result = await execute_via_echo(command, action_type, conversation_id)

        # Format and send response
        response_text = format_execution_response(result, action_type)
        await send_telegram_message(chat_id, response_text)

    except Exception as e:
        logger.error(f"Execution processing error: {e}")
        await send_telegram_message(
            chat_id,
            f"❌ **Error**\n```\n{str(e)}\n```"
        )

# Quick test endpoint
@telegram_executor_router.get("/test")
async def test_executor():
    """Test if executor is working"""
    return {
        "status": "operational",
        "capabilities": list(EXECUTION_COMMANDS.keys()),
        "autonomous_behaviors": [
            "service_monitoring",
            "code_refactoring",
            "auto_fixing",
            "unit_testing"
        ]
    }