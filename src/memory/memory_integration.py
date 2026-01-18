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

    This saves directly to echo_unified_interactions with entities in metadata.
    """
    try:
        import psycopg2
        import psycopg2.extras

        # Build metadata with entities
        metadata = {"entities": entities} if entities else {}
        if complexity_score is not None:
            metadata["complexity_score"] = complexity_score
        if tier is not None:
            metadata["tier"] = tier

        conn = psycopg2.connect(
            host="localhost",
            database="echo_brain",
            user="patrick",
            password="RP78eIrW7cI2jYvL5akt1yurE"
        )
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO echo_unified_interactions
            (query, response, model_used, processing_time, escalation_path,
             conversation_id, user_id, intent, confidence, requires_clarification,
             clarifying_questions, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (query, response, model_used or "unknown", processing_time,
              psycopg2.extras.Json(escalation_path or []), conversation_id or "", user_id or "default",
              intent or "", confidence, requires_clarification,
              psycopg2.extras.Json(clarifying_questions or []),
              psycopg2.extras.Json(metadata)))

        conn.commit()
        cursor.close()
        conn.close()

        if entities:
            logger.info(f"✅ Saved conversation with entities: {entities}")
        else:
            logger.info(f"✅ Saved conversation for {conversation_id}")

    except Exception as e:
        logger.error(f"Failed to save conversation with entities: {e}")