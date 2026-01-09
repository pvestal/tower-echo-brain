#!/usr/bin/env python3
"""
Delegation routes for Echo Brain to use Tower LLMs
Saves Opus tokens by delegating heavy tasks to local models
"""

import logging
from typing import Dict, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.core.tower_llm_executor import tower_executor

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/delegate", tags=["delegation"])


class DelegationRequest(BaseModel):
    task: str
    context: Optional[Dict] = None
    model: Optional[str] = "qwen2.5-coder:7b"
    priority: Optional[str] = "normal"


class DelegationResponse(BaseModel):
    success: bool
    task: str
    model: str
    commands_executed: int
    results: list
    execution_history: list
    timestamp: str
    error: Optional[str] = None


@router.post("/to-tower", response_model=DelegationResponse)
async def delegate_to_tower(request: DelegationRequest):
    """
    Delegate a task to Tower LLM with execution capabilities
    This saves Opus tokens by using local compute for heavy tasks
    """
    try:
        logger.info(f"📡 Delegating task to Tower LLM: {request.task[:100]}...")

        # Switch executor model if specified
        if request.model != tower_executor.model:
            tower_executor.model = request.model

        # Delegate the task
        result = await tower_executor.delegate_task(
            task=request.task,
            context=request.context
        )

        if result.get("success"):
            logger.info(f"✅ Tower LLM executed {result.get('commands_executed', 0)} commands")
            return DelegationResponse(**result)
        else:
            logger.error(f"❌ Delegation failed: {result.get('error')}")
            raise HTTPException(status_code=500, detail=result.get("error"))

    except Exception as e:
        logger.error(f"Delegation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/capabilities")
async def get_tower_capabilities():
    """
    Get the current capabilities of Tower LLM executor
    """
    return {
        "model": tower_executor.model,
        "available_models": [
            "qwen2.5-coder:7b",
            "deepseek-coder:latest",
            "codellama:7b"
        ],
        "capabilities": list(tower_executor.capabilities.keys()),
        "operations": {
            category: list(ops.keys())
            for category, ops in tower_executor.capabilities.items()
        },
        "max_commands_per_task": tower_executor.max_commands_per_task
    }


@router.get("/history")
async def get_execution_history(limit: int = 10):
    """
    Get recent execution history from Tower LLM
    """
    return {
        "history": tower_executor.execution_history[-limit:],
        "total_executions": len(tower_executor.execution_history)
    }


@router.post("/test")
async def test_delegation():
    """
    Test the delegation system with a simple task
    """
    test_task = "List all Python files in /opt/tower-echo-brain/src/core and count them"

    result = await tower_executor.delegate_task(
        task=test_task,
        context={"base_path": "/opt/tower-echo-brain"}
    )

    return {
        "test_task": test_task,
        "result": result
    }