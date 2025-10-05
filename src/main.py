#!/usr/bin/env python3
"""
Echo Brain Unified Service - Refactored Main Entry Point with Autonomous Task System
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
import asyncio
import logging
import uvicorn
import psycopg2
import json
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, WebSocket, BackgroundTasks, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# Import modular components
from src.db.models import (
    QueryRequest, QueryResponse, ExecuteRequest, ExecuteResponse,
    TestRequest, VoiceNotificationRequest, VoiceStatusRequest
)
from src.db.database import database
from src.core.intelligence import intelligence_router
# Board of Directors Integration
from src.board_integration import board, BoardIntegration
from src.services.conversation import conversation_manager
from src.services.testing import testing_framework
from src.utils.helpers import safe_executor, tower_orchestrator

# Import task system components
from src.tasks import TaskQueue, BackgroundWorker, AutonomousBehaviors
from src.tasks.task_queue import Task, TaskType, TaskPriority, TaskStatus

# Import existing modules that remain external
from echo_brain_thoughts import echo_brain
from routing.service_registry import ServiceRegistry
from routing.request_logger import RequestLogger
from routing.feedback_system import FeedbackProcessor, UserFeedback, FeedbackType
from routing.user_preferences import UserPreferences, PreferenceType
from routing.knowledge_manager import KnowledgeManager, create_simple_knowledge_manager
from routing.sandbox_executor import SandboxExecutor, create_strict_sandbox
from board_api import create_board_api
from model_manager import (
    get_model_manager, ModelManagementRequest, ModelManagementResponse,
    ModelOperation, ModelInfo
)
from routing.auth_middleware import get_current_user
from model_decision_engine import get_decision_engine
from telegram_integration import telegram_router
from veteran_guardian_endpoints import veteran_router
from agent_development_endpoints import agent_dev_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import cognitive model selector for intelligent model selection
model_selector = None
try:
    from fixed_model_selector import ModelSelector
    model_selector = ModelSelector()
    logger.info("✅ Cognitive model selector loaded")
except ImportError:
    logger.warning("⚠️ Cognitive model selector not available")

# Initialize FastAPI application

# Ollama Integration for NVIDIA GPU
import requests
from managers.dynamic_escalation_manager import DynamicEscalationManager

def query_ollama(prompt: str, model: str = "qwen2.5-coder:7b"):
    """Query Ollama running on NVIDIA GPU"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            },
            timeout=30
        )
        if response.status_code == 200:
            return response.json().get("response", "No response")
    except Exception as e:
        print(f"Ollama error: {e}")
        return None

async def query_echo_dynamic(message: str, context: dict = None):
    """Query Echo with dynamic persona-driven model selection"""
    global escalation_manager
    if escalation_manager is None:
        escalation_manager = DynamicEscalationManager()
        await escalation_manager.initialize()
    
    result = await escalation_manager.process_message(message, context)
    return result
    return None

async def query_echo_dynamic(message: str, context: dict = None):
    """Query Echo with dynamic persona-driven model selection"""
    global escalation_manager
    if escalation_manager is None:
        escalation_manager = DynamicEscalationManager()
        await escalation_manager.initialize()
    
    result = await escalation_manager.process_message(message, context)
    return result

from src.photo_comparison import router as photo_router
# Dynamic Escalation Manager
escalation_manager = None

app = FastAPI(
    title="Echo Brain Unified Service with Autonomous Task System",
