#!/usr/bin/env python3
"""
Model management API routes for Echo Brain
"""
import subprocess
import uuid
import logging
import time
from datetime import datetime
from fastapi import APIRouter, HTTPException, BackgroundTasks
from src.db.models import QueryRequest
from model_manager import get_model_manager

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/api/echo/models/list")
async def list_models():
    """List all available Ollama models"""
    try:
        # Use direct ollama command as fallback
        from src.api.dependencies import execute_ollama_command

        # Direct ollama list command
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            models = []
            for line in lines:
                if line:
                    parts = line.split()
                    if len(parts) >= 4:
                        models.append({
                            "name": parts[0],
                            "size": f"{parts[2]} {parts[3]}",
                            "modified": " ".join(parts[4:]) if len(parts) > 4 else ""
                        })
            return models
        return {"error": "Failed to list models"}
    except Exception as e:
        logger.error(f"Model listing failed: {e}")
        return {"error": str(e)}

@router.post("/api/echo/models/pull/{model_name}")
async def pull_model(model_name: str, background_tasks: BackgroundTasks):
    """Pull a specific model"""
    logger.info(f"📥 Pulling model: {model_name}")

    try:
        from src.api.dependencies import execute_ollama_command

        async def pull_model_async(model: str):
            result = await execute_ollama_command(["ollama", "pull", model])
            if result["success"]:
                logger.info(f"✅ Model {model} pulled successfully")
            else:
                logger.error(f"❌ Failed to pull {model}: {result.get('stderr', result.get('error'))}")
            return result

        task_id = str(uuid.uuid4())
        background_tasks.add_task(pull_model_async, model_name)

        return {
            "success": True,
            "message": f"Model pull started for {model_name}",
            "task_id": task_id
        }
    except Exception as e:
        logger.error(f"Model pull failed: {e}")
        return {"success": False, "error": str(e)}

@router.delete("/api/echo/models/{model_name}")
async def remove_model(model_name: str):
    """Remove a specific model"""
    logger.info(f"🗑️ Removing model: {model_name}")

    try:
        from src.api.dependencies import execute_ollama_command

        result = await execute_ollama_command(["ollama", "rm", model_name])

        if result["success"]:
            return {"success": True, "message": f"Model {model_name} removed successfully"}
        else:
            return {"success": False, "error": result.get("stderr", "Failed to remove model")}
    except Exception as e:
        logger.error(f"Model removal failed: {e}")
        return {"success": False, "error": str(e)}

@router.get("/api/echo/models/status/{request_id}")
async def get_model_operation_status(request_id: str):
    """Get status of a model operation"""
    try:
        # Initialize dependencies if not available
        from routing.service_registry import ServiceRegistry
        from routing.request_logger import RequestLogger
        board_registry = ServiceRegistry()
        request_logger = RequestLogger()
        model_manager = get_model_manager(board_registry, request_logger)
        status = await model_manager.get_operation_status(request_id)
        return status
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return {"error": str(e)}

@router.post("/api/echo/code")
async def generate_code_only(request: QueryRequest):
    """Code generation endpoint - forces code output"""
    from .echo import query_echo

    # Modify query to force code generation
    code_request = QueryRequest(
        query=f"Write only code, no explanations: {request.query}",
        conversation_id=request.conversation_id,
        user_id=request.user_id,
        intelligence_level="small"  # Use code model
    )
    # Call the regular query handler with modified request
    response = await query_echo(code_request)
    response.mode = "code_only"
    return response