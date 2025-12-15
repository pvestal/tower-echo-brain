#!/usr/bin/env python3
"""
API routes for conversation memory management.

Provides endpoints for multi-turn conversation context, entity resolution,
and memory-enhanced query processing.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.managers.conversation_memory_manager import (
    get_conversation_memory_manager,
    enhance_query_with_memory,
    record_conversation_turn,
    get_conversation_context_for_execution,
    EntityType
)

logger = logging.getLogger(__name__)

# API Router
router = APIRouter(prefix="/api/echo/memory", tags=["conversation-memory"])


# Request/Response Models
class RecordTurnRequest(BaseModel):
    """Request to record a conversation turn."""
    conversation_id: str
    role: str  # "user" or "assistant"
    content: str
    intent: Optional[str] = None
    execution_results: List[str] = []


class EnhancedQueryRequest(BaseModel):
    """Request for memory-enhanced query processing."""
    query: str
    conversation_id: str
    max_context_turns: int = 5


class ReferenceResolutionRequest(BaseModel):
    """Request to resolve pronouns/references in text."""
    text: str
    conversation_id: str


class ContextSearchRequest(BaseModel):
    """Request to search for relevant context."""
    query: str
    conversation_id: Optional[str] = None
    max_results: int = 5


class SessionSummaryRequest(BaseModel):
    """Request for conversation session summary."""
    conversation_id: str


# =============================================================================
# CONVERSATION RECORDING ENDPOINTS
# =============================================================================

@router.post("/record")
async def record_conversation_turn(request: RecordTurnRequest):
    """
    Record a conversation turn for memory tracking.

    This should be called for both user queries and assistant responses.
    """
    try:
        memory_manager = await get_conversation_memory_manager()

        turn = await memory_manager.add_turn(
            conversation_id=request.conversation_id,
            role=request.role,
            content=request.content,
            intent=request.intent,
            execution_results=request.execution_results
        )

        return {
            "success": True,
            "conversation_id": request.conversation_id,
            "entities_extracted": len(turn.entities),
            "entities": [
                {
                    "name": entity.name,
                    "type": entity.entity_type.value,
                    "value": str(entity.value),
                    "confidence": entity.confidence
                }
                for entity in turn.entities
            ],
            "timestamp": turn.timestamp.isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to record conversation turn: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# MEMORY-ENHANCED QUERY ENDPOINTS
# =============================================================================

@router.post("/enhance")
async def enhance_query(request: EnhancedQueryRequest):
    """
    Enhance a query with conversation memory context.

    Resolves pronouns and provides relevant context for LLM processing.
    """
    try:
        memory_manager = await get_conversation_memory_manager()

        # Resolve references in query
        enhanced_query, resolved_entities = await memory_manager.resolve_reference(
            request.query, request.conversation_id
        )

        # Get context prompt
        context_prompt = await memory_manager.get_context_prompt(
            request.conversation_id, request.max_context_turns
        )

        # Find additional relevant context
        relevant_context = await memory_manager.find_relevant_context(
            request.query, request.conversation_id
        )

        return {
            "success": True,
            "original_query": request.query,
            "enhanced_query": enhanced_query,
            "context_prompt": context_prompt,
            "resolved_entities": [
                {
                    "name": entity.name,
                    "type": entity.entity_type.value,
                    "value": str(entity.value)
                }
                for entity in resolved_entities
            ],
            "relevant_context": relevant_context,
            "conversation_id": request.conversation_id
        }

    except Exception as e:
        logger.error(f"Failed to enhance query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/resolve")
async def resolve_references(request: ReferenceResolutionRequest):
    """
    Resolve pronouns and references in text to actual entities.
    """
    try:
        memory_manager = await get_conversation_memory_manager()

        enhanced_text, resolved_entities = await memory_manager.resolve_reference(
            request.text, request.conversation_id
        )

        return {
            "success": True,
            "original_text": request.text,
            "enhanced_text": enhanced_text,
            "resolved_entities": [
                {
                    "name": entity.name,
                    "type": entity.entity_type.value,
                    "value": str(entity.value),
                    "confidence": entity.confidence
                }
                for entity in resolved_entities
            ]
        }

    except Exception as e:
        logger.error(f"Failed to resolve references: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# CONTEXT SEARCH ENDPOINTS
# =============================================================================

@router.post("/search")
async def search_context(request: ContextSearchRequest):
    """
    Search for relevant historical context based on a query.
    """
    try:
        memory_manager = await get_conversation_memory_manager()

        relevant_context = await memory_manager.find_relevant_context(
            request.query, request.conversation_id
        )

        return {
            "success": True,
            "query": request.query,
            "relevant_context": relevant_context[:request.max_results],
            "total_found": len(relevant_context)
        }

    except Exception as e:
        logger.error(f"Failed to search context: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/context/{conversation_id}")
async def get_conversation_context(conversation_id: str):
    """
    Get the current context for a conversation including active entities.
    """
    try:
        memory_manager = await get_conversation_memory_manager()

        if conversation_id not in memory_manager.active_sessions:
            return {
                "success": False,
                "error": "Conversation not found",
                "conversation_id": conversation_id
            }

        session = memory_manager.active_sessions[conversation_id]

        return {
            "success": True,
            "conversation_id": conversation_id,
            "started_at": session.started_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "turn_count": len(session.turns),
            "active_entities": {
                name: {
                    "type": entity.entity_type.value,
                    "value": str(entity.value),
                    "confidence": entity.confidence,
                    "mention_count": entity.mention_count,
                    "last_mentioned": entity.last_mentioned.isoformat(),
                    "aliases": list(entity.aliases)
                }
                for name, entity in session.active_entities.items()
            },
            "session_summary": session.session_summary
        }

    except Exception as e:
        logger.error(f"Failed to get conversation context: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# EXECUTION INTEGRATION ENDPOINTS
# =============================================================================

@router.get("/execution-context/{conversation_id}")
async def get_execution_context(conversation_id: str):
    """
    Get conversation context relevant for verified execution.

    Returns entities that might be referenced in execution commands.
    """
    try:
        execution_context = await get_conversation_context_for_execution(conversation_id)

        return {
            "success": True,
            "conversation_id": conversation_id,
            "execution_context": execution_context,
            "entity_counts": {
                entity_type: len(entities)
                for entity_type, entities in execution_context.items()
            }
        }

    except Exception as e:
        logger.error(f"Failed to get execution context: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# SESSION MANAGEMENT ENDPOINTS
# =============================================================================

@router.post("/summary")
async def get_session_summary(request: SessionSummaryRequest):
    """
    Generate a summary of the conversation session.
    """
    try:
        memory_manager = await get_conversation_memory_manager()

        summary = await memory_manager.get_session_summary(request.conversation_id)

        return {
            "success": True,
            "conversation_id": request.conversation_id,
            "summary": summary,
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to generate session summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions")
async def list_active_sessions():
    """
    List all active conversation sessions.
    """
    try:
        memory_manager = await get_conversation_memory_manager()

        sessions = []
        for conv_id, session in memory_manager.active_sessions.items():
            sessions.append({
                "conversation_id": conv_id,
                "started_at": session.started_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
                "turn_count": len(session.turns),
                "entity_count": len(session.active_entities),
                "summary": session.session_summary
            })

        # Sort by last activity
        sessions.sort(key=lambda s: s["last_activity"], reverse=True)

        return {
            "success": True,
            "active_sessions": len(sessions),
            "sessions": sessions,
            "global_entity_count": len(memory_manager.global_entities)
        }

    except Exception as e:
        logger.error(f"Failed to list active sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/entities")
async def list_global_entities():
    """
    List all global entities tracked across conversations.
    """
    try:
        memory_manager = await get_conversation_memory_manager()

        entities_by_type = {}
        for entity in memory_manager.global_entities.values():
            entity_type = entity.entity_type.value
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []

            entities_by_type[entity_type].append({
                "name": entity.name,
                "value": str(entity.value),
                "confidence": entity.confidence,
                "mention_count": entity.mention_count,
                "first_mentioned": entity.first_mentioned.isoformat(),
                "last_mentioned": entity.last_mentioned.isoformat(),
                "aliases": list(entity.aliases)
            })

        # Sort each type by mention count
        for entity_type in entities_by_type:
            entities_by_type[entity_type].sort(
                key=lambda e: e["mention_count"],
                reverse=True
            )

        return {
            "success": True,
            "total_entities": len(memory_manager.global_entities),
            "entities_by_type": entities_by_type,
            "entity_type_counts": {
                entity_type: len(entities)
                for entity_type, entities in entities_by_type.items()
            }
        }

    except Exception as e:
        logger.error(f"Failed to list global entities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# HEALTH & MONITORING ENDPOINTS
# =============================================================================

@router.get("/health")
async def memory_system_health():
    """Health check for conversation memory system."""
    try:
        memory_manager = await get_conversation_memory_manager()

        # Calculate some basic statistics
        total_entities = len(memory_manager.global_entities)
        active_sessions = len(memory_manager.active_sessions)

        # Count entities by type
        entity_type_counts = {}
        for entity in memory_manager.global_entities.values():
            entity_type = entity.entity_type.value
            entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1

        return {
            "status": "healthy",
            "service": "conversation-memory",
            "active_sessions": active_sessions,
            "total_entities": total_entities,
            "entity_type_distribution": entity_type_counts,
            "storage_path": str(memory_manager.storage_path),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Conversation memory health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "conversation-memory",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }