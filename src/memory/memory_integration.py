#!/usr/bin/env python3
"""
Integration helper for memory system with database.
"""

import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

async def save_conversation_with_entities(
    database,
    query: str,
    response: str,
    conversation_id: str,
    entities: Dict[str, Any],
    model_used: str = None,
    processing_time: float = 0.0,
    escalation_path: list = None,
    user_id: str = "default",
    intent: str = None,
    confidence: float = 0.0,
    requires_clarification: bool = False,
    clarifying_questions: list = None,
    complexity_score: float = None,
    tier: str = None
):
    """
    Save conversation with extracted entities to database.

    This extends the standard log_interaction to also save entities.
    """
    try:
        # First log the standard interaction
        await database.log_interaction(
            query, response, model_used or "unknown",
            processing_time, escalation_path or [],
            conversation_id, user_id,
            intent, confidence,
            requires_clarification,
            clarifying_questions,
            complexity_score,
            tier
        )

        # Now update with entities if we have any
        if entities:
            try:
                # Connect directly to update entities
                import asyncpg
                conn = await asyncpg.connect(
                    host="localhost",
                    database="echo_brain",
                    user="patrick",
                    password="***REMOVED***"
                )

                await conn.execute("""
                    UPDATE echo_conversations
                    SET entities_mentioned = $1
                    WHERE conversation_id = $2
                    AND created_at = (
                        SELECT MAX(created_at)
                        FROM echo_conversations
                        WHERE conversation_id = $2
                    )
                """, json.dumps(entities), conversation_id)

                await conn.close()
                logger.info(f"✅ Saved entities: {entities}")

            except Exception as e:
                logger.error(f"Failed to save entities: {e}")

    except Exception as e:
        logger.error(f"Failed to save conversation: {e}")