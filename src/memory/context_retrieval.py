#!/usr/bin/env python3
"""
Retrieves and formats conversation history for context injection.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import asyncpg
import logging
import os
import sys

# Add config path
sys.path.append('/opt/tower-echo-brain')

# Try to load configuration
try:
    from src.config.memory_config import (
        MAX_HISTORY_MESSAGES,
        MAX_HISTORY_AGE_DAYS,
        MAX_ENTITY_LOOKBACK
    )
except ImportError:
    # Fallback defaults if config not found
    MAX_HISTORY_MESSAGES = 50
    MAX_HISTORY_AGE_DAYS = None  # Indefinite retention
    MAX_ENTITY_LOOKBACK = 10

logger = logging.getLogger(__name__)

class ConversationContextRetriever:
    """
    Retrieves recent conversation history for a given conversation_id.

    CRITICAL: This class MUST be called from the main query endpoint.
    If it exists but isn't called, the memory system is theater.
    """

    def __init__(self, db_pool: asyncpg.Pool = None, max_history_messages: int = None, max_history_age_days: int = None):
        self.db_pool = db_pool
        # Use config values or provided overrides
        self.max_history_messages = max_history_messages or MAX_HISTORY_MESSAGES
        self.max_history_age_days = max_history_age_days if max_history_age_days is not None else MAX_HISTORY_AGE_DAYS
        self._connection_string = None

        logger.info(f"ConversationContextRetriever initialized: max_messages={self.max_history_messages}, max_age_days={self.max_history_age_days or 'indefinite'}")

    async def _get_connection(self):
        """Get a database connection from pool or create one"""
        if self.db_pool:
            return self.db_pool.acquire()
        else:
            # Create a single connection if no pool
            if not self._connection_string:
                self._connection_string = "postgresql://patrick:***REMOVED***@localhost/echo_brain"
            conn = await asyncpg.connect(self._connection_string)
            return conn

    async def get_recent_history(
        self,
        conversation_id: str
    ) -> List[Dict[str, Any]]:
        """
        Fetch recent messages for this conversation.

        Returns list of {role, content, entities, timestamp} dicts.
        """
        if not conversation_id:
            return []

        # Build query with optional time constraint
        if self.max_history_age_days:
            query = f"""
                SELECT
                    query AS query_text,
                    response AS response_text,
                    metadata AS entities_mentioned,
                    timestamp AS created_at
                FROM echo_unified_interactions
                WHERE conversation_id = $1
                  AND timestamp > NOW() - INTERVAL '{self.max_history_age_days} days'
                ORDER BY timestamp DESC
                LIMIT $2
            """
        else:
            # No time limit - get all history for this conversation
            query = """
                SELECT
                    query AS query_text,
                    response AS response_text,
                    metadata AS entities_mentioned,
                    timestamp AS created_at
                FROM echo_unified_interactions
                WHERE conversation_id = $1
                ORDER BY timestamp DESC
                LIMIT $2
            """

        try:
            if self.db_pool:
                async with self.db_pool.acquire() as conn:
                    rows = await conn.fetch(
                        query,
                        conversation_id,
                        self.max_history_messages
                    )
            else:
                conn = await self._get_connection()
                try:
                    rows = await conn.fetch(
                        query,
                        conversation_id,
                        self.max_history_messages
                    )
                finally:
                    await conn.close()

            # Convert to message format (oldest first)
            history = []
            for row in reversed(rows):
                if row["query_text"]:
                    history.append({
                        "role": "user",
                        "content": row["query_text"],
                        "timestamp": row["created_at"]
                    })
                if row["response_text"]:
                    history.append({
                        "role": "assistant",
                        "content": row["response_text"],
                        "entities": row["entities_mentioned"] or {},
                        "timestamp": row["created_at"]
                    })

            logger.info(f"Retrieved {len(history)} messages for conversation {conversation_id}")
            return history

        except Exception as e:
            logger.error(f"Failed to retrieve history: {e}")
            return []

    async def get_active_entities(
        self,
        conversation_id: str
    ) -> Dict[str, Any]:
        """
        Get entities mentioned in recent conversation.

        Used for pronoun resolution: "it" → most recent entity
        """
        # Get recent entities - no time limit, just most recent N entries
        query = f"""
            SELECT metadata AS entities_mentioned, timestamp AS created_at
            FROM echo_unified_interactions
            WHERE conversation_id = $1
              AND metadata IS NOT NULL
              AND metadata != '{{}}'::jsonb
            ORDER BY timestamp DESC
            LIMIT {MAX_ENTITY_LOOKBACK}
        """

        try:
            if self.db_pool:
                async with self.db_pool.acquire() as conn:
                    rows = await conn.fetch(query, conversation_id)
            else:
                conn = await self._get_connection()
                try:
                    rows = await conn.fetch(query, conversation_id)
                finally:
                    await conn.close()

            # Merge entities, most recent takes precedence
            entities = {}
            for row in reversed(rows):  # Oldest first
                if row["entities_mentioned"]:
                    # asyncpg returns JSONB as string, need to convert
                    entity_data = row["entities_mentioned"]
                    if isinstance(entity_data, str):
                        import json
                        try:
                            entity_data = json.loads(entity_data)
                        except:
                            continue
                    if isinstance(entity_data, dict):
                        # Extract entities from metadata structure
                        if "entities" in entity_data:
                            entities.update(entity_data["entities"])
                        else:
                            entities.update(entity_data)

            return entities

        except Exception as e:
            logger.error(f"Failed to get entities: {e}")
            return {}

    def format_history_for_prompt(
        self,
        history: List[Dict[str, Any]]
    ) -> str:
        """
        Format conversation history for injection into LLM prompt.
        """
        if not history:
            return ""

        lines = ["<conversation_history>"]
        for msg in history[-6:]:  # Last 3 exchanges
            role = msg["role"].upper()
            content = msg["content"][:500] if msg["content"] else ""  # Truncate long messages
            if content:
                lines.append(f"{role}: {content}")
        lines.append("</conversation_history>")

        return "\n".join(lines)