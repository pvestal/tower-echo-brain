#!/usr/bin/env python3
"""
Integration layer for resilient model manager with Echo Brain's existing architecture.
"""

import logging
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .resilient_model_manager import (
    ResilientModelManager,
    TaskUrgency,
    ExecutionResult,
    get_resilient_manager
)

logger = logging.getLogger(__name__)

# API Router for resilient model endpoints
router = APIRouter(prefix="/api/echo/resilient", tags=["resilient-models"])


class ModelQueryRequest(BaseModel):
    """Request for model-based query."""
    query: str
    task_type: str = "general"
    urgency: str = "background"  # interactive, background, batch
    system_prompt: Optional[str] = ""
    conversation_id: Optional[str] = None


class ModelHealthRequest(BaseModel):
    """Request for model health check."""
    model_name: str
    force_check: bool = False


class ModelPreloadRequest(BaseModel):
    """Request to preload a model."""
    model_name: str


class TaskTypeClassifier:
    """
    Classifies user queries into task types for model selection.

    This replaces the board of directors voting with evidence-based classification.
    """

    def __init__(self):
        self.patterns = {
            "code_generation": [
                "write", "create", "implement", "build", "generate code",
                "function", "class", "script", "program"
            ],
            "code_review": [
                "review", "check", "analyze code", "find bugs", "security",
                "vulnerability", "audit", "best practices"
            ],
            "reasoning": [
                "why", "explain", "reason", "think", "logic", "deduce",
                "infer", "conclude", "analyze"
            ],
            "complex": [
                "design", "architect", "plan", "strategy", "comprehensive",
                "detailed analysis", "deep dive", "thorough"
            ],
            "analysis": [
                "analyze", "examine", "investigate", "study", "research",
                "evaluate", "assess"
            ],
            "creative": [
                "imagine", "create story", "fiction", "poem", "creative",
                "artistic", "novel", "innovative"
            ],
            "technical": [
                "technical", "engineering", "system", "infrastructure",
                "database", "network", "deployment"
            ],
            "simple": [
                "what is", "define", "simple", "basic", "tell me",
                "list", "name"
            ],
            "fast_response": [
                "quick", "fast", "brief", "short", "summary",
                "tldr", "overview"
            ]
        }

    def classify(self, query: str) -> str:
        """
        Classify query into task type.

        Returns task type string for model selection.
        """
        query_lower = query.lower()

        # Score each task type
        scores = {}
        for task_type, patterns in self.patterns.items():
            score = sum(1 for pattern in patterns if pattern in query_lower)
            if score > 0:
                scores[task_type] = score

        # Return highest scoring type, default to general
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]

        # Check query length for urgency hints
        if len(query) < 50:
            return "simple"
        elif len(query) > 500:
            return "complex"

        return "general"


class EchoBrainIntegration:
    """
    Integrates resilient model manager with Echo Brain's existing systems.

    This replaces:
    - Board of Directors voting (consensus-based) with evidence-based selection
    - Simple Ollama calls with resilient execution
    - Basic retry logic with intelligent circuit breakers
    """

    def __init__(self):
        self.classifier = TaskTypeClassifier()
        self.manager: Optional[ResilientModelManager] = None

    async def initialize(self):
        """Initialize the integration."""
        self.manager = await get_resilient_manager()
        logger.info("✅ Resilient model manager initialized for Echo Brain")

    async def process_query(
        self,
        query: str,
        task_type: Optional[str] = None,
        urgency: TaskUrgency = TaskUrgency.BACKGROUND,
        system_prompt: str = "",
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a query using resilient model selection.

        Args:
            query: User query
            task_type: Optional override for task classification
            urgency: How urgent is this request
            system_prompt: System instructions
            conversation_id: For conversation tracking

        Returns:
            Response dictionary with model output and metadata
        """
        if not self.manager:
            await self.initialize()

        # Classify task if not provided
        if not task_type:
            task_type = self.classifier.classify(query)
            logger.info(f"Classified query as task type: {task_type}")

        # Execute with resilient fallback
        result = await self.manager.complete_with_fallback(
            task_type=task_type,
            prompt=query,
            system=system_prompt or self._get_default_system_prompt(),
            urgency=urgency
        )

        if result.success:
            response = {
                "success": True,
                "response": result.value,
                "model_used": result.model_used,
                "task_type": task_type,
                "fallback_used": result.fallback_used,
                "attempts": result.attempts,
                "latency_ms": result.total_latency_ms,
                "selection_reason": result.selection_reason,
                "conversation_id": conversation_id
            }

            # Log successful execution
            logger.info(
                f"Query processed successfully: "
                f"model={result.model_used}, "
                f"latency={result.total_latency_ms:.0f}ms, "
                f"attempts={result.attempts}"
            )
        else:
            response = {
                "success": False,
                "error": result.error,
                "error_severity": result.error_severity.value if result.error_severity else None,
                "attempts": result.attempts,
                "task_type": task_type,
                "conversation_id": conversation_id
            }

            logger.error(f"Query failed: {result.error}")

        return response

    def _get_default_system_prompt(self) -> str:
        """Get Echo Brain's default system prompt."""
        return """You are Echo Brain, an advanced AI assistant integrated into the Tower ecosystem.
You provide thoughtful, accurate, and helpful responses.
You have access to multiple specialized models and automatically select the best one for each task.
When technical tasks fail, you gracefully degrade to provide the best possible assistance."""

    async def check_health(self, model_name: str, force: bool = False) -> Dict[str, Any]:
        """Check health of a specific model."""
        if not self.manager:
            await self.initialize()

        health = await self.manager.health_checker.check_model(model_name, force)

        return {
            "model": model_name,
            "state": health.state.value,
            "last_error": health.last_error,
            "consecutive_failures": health.consecutive_failures,
            "avg_latency_ms": health.avg_latency_ms,
            "last_check": health.last_health_check.isoformat() if health.last_health_check else None
        }

    async def get_system_status(self) -> Dict[str, Any]:
        """Get complete system status."""
        if not self.manager:
            await self.initialize()

        return await self.manager.get_system_status()

    async def preload_model(self, model_name: str) -> Dict[str, Any]:
        """Preload a model into memory."""
        if not self.manager:
            await self.initialize()

        result = await self.manager.preload_model(model_name)

        return {
            "success": result.success,
            "model": model_name,
            "error": result.error,
            "latency_ms": result.total_latency_ms
        }


# Singleton integration instance
_integration_instance: Optional[EchoBrainIntegration] = None

async def get_echo_integration() -> EchoBrainIntegration:
    """Get or create the singleton Echo Brain integration."""
    global _integration_instance
    if _integration_instance is None:
        _integration_instance = EchoBrainIntegration()
        await _integration_instance.initialize()
    return _integration_instance


# =============================================================================
# API ENDPOINTS
# =============================================================================

@router.post("/query")
async def resilient_query(request: ModelQueryRequest):
    """Process query with resilient model management."""
    integration = await get_echo_integration()

    # Convert urgency string to enum
    urgency_map = {
        "interactive": TaskUrgency.INTERACTIVE,
        "background": TaskUrgency.BACKGROUND,
        "batch": TaskUrgency.BATCH
    }
    urgency = urgency_map.get(request.urgency.lower(), TaskUrgency.BACKGROUND)

    result = await integration.process_query(
        query=request.query,
        task_type=request.task_type if request.task_type != "general" else None,
        urgency=urgency,
        system_prompt=request.system_prompt,
        conversation_id=request.conversation_id
    )

    if not result["success"]:
        raise HTTPException(status_code=503, detail=result)

    return result


@router.get("/status")
async def get_status():
    """Get system status of all models and circuits."""
    integration = await get_echo_integration()
    return await integration.get_system_status()


@router.post("/health")
async def check_model_health(request: ModelHealthRequest):
    """Check health of a specific model."""
    integration = await get_echo_integration()
    return await integration.check_health(request.model_name, request.force_check)


@router.post("/preload")
async def preload_model(request: ModelPreloadRequest):
    """Preload a model into memory."""
    integration = await get_echo_integration()
    result = await integration.preload_model(request.model_name)

    if not result["success"]:
        raise HTTPException(status_code=500, detail=result)

    return result


@router.get("/models")
async def list_models():
    """List all configured models with their profiles."""
    manager = await get_resilient_manager()

    models = []
    for model_id, config in manager.models.items():
        models.append({
            "id": model_id,
            "name": config.name,
            "vram_mb": config.vram_mb,
            "load_time_seconds": config.load_time_seconds,
            "strengths": config.strengths,
            "quality_scores": config.quality_scores
        })

    return {"models": models}


@router.get("/chains")
async def list_fallback_chains():
    """List all configured fallback chains."""
    manager = await get_resilient_manager()

    chains = []
    for task_type, chain in manager.fallback_chains.items():
        chains.append({
            "task_type": task_type,
            "models": chain.models,
            "min_quality": chain.min_quality
        })

    return {"chains": chains}