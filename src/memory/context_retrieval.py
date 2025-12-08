#!/usr/bin/env python3
"""
Retrieves and formats conversation history for context injection.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import asyncpg
import logging

logger = logging.getLogger(__name__)

class ConversationContextRetriever:
    """
    Retrieves recent conversation history for a given conversation_id.

    CRITICAL: This class MUST be called from the main query endpoint.
    If it exists but isn't called, the memory system is theater.
    """

    def __init__(self, db_pool: asyncpg.Pool = None):
        self.db_pool = db_pool
        self.max_history_messages = 10
        self.max_history_age_hours = 24
        self._connection_string = None

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

        query = """
            SELECT
                query_text,
                response_text,
                entities_mentioned,
                created_at
            FROM echo_conversations
            WHERE conversation_id = $1
              AND created_at > NOW() - INTERVAL '24 hours'
            ORDER BY created_at DESC
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
        query = """
            SELECT entities_mentioned, created_at
            FROM echo_conversations
            WHERE conversation_id = $1
              AND entities_mentioned IS NOT NULL
              AND entities_mentioned != '{}'::jsonb
            ORDER BY created_at DESC
            LIMIT 5
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