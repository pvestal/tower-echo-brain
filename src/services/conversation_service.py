"""
Conversation Service for Echo Brain
Handles conversation search, retrieval, and multi-turn session management
"""
import asyncpg
import logging
import os
from typing import Dict, Any, List, Optional
from src.core.pg_reasoning import search_pg

logger = logging.getLogger(__name__)


class ConversationService:
    """Service for searching and managing conversations"""

    def __init__(self):
        self.logger = logger
        self._pg_dsn = None

    def _get_dsn(self) -> str:
        if not self._pg_dsn:
            password = os.environ.get('PGPASSWORD', os.environ.get('DB_PASSWORD', ''))
            self._pg_dsn = f"postgresql://patrick:{password}@localhost/echo_brain"
        return self._pg_dsn

    async def search_conversations(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """
        Search conversations using PostgreSQL full-text search

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            Dictionary with search results
        """
        try:
            # Use the existing PostgreSQL search function
            memories = search_pg(query, limit)

            # Format results for frontend
            results = []
            for memory in memories:
                results.append({
                    "conversation_id": memory.get("conv"),
                    "role": memory.get("role"),
                    "content": memory.get("content", "")[:500],  # Truncate for display
                    "relevance_score": 0.8  # Placeholder score
                })

            return {
                "query": query,
                "results": results,
                "total_found": len(results),
                "search_method": "postgresql_fulltext"
            }

        except Exception as e:
            logger.error(f"Failed to search conversations: {e}")
            return {
                "query": query,
                "results": [],
                "total_found": 0,
                "error": str(e)
            }

    async def get_session_turns(self, session_id: str, limit: int = 10) -> List[Dict]:
        """Get recent conversation turns for a session.

        Returns list of {role, content, timestamp} dicts ordered oldest-first.
        """
        try:
            conn = await asyncpg.connect(self._get_dsn(), timeout=5)
            rows = await conn.fetch("""
                SELECT role, content, timestamp
                FROM conversation_messages
                WHERE conversation_id = $1
                ORDER BY timestamp DESC
                LIMIT $2
            """, session_id, limit)
            await conn.close()

            # Reverse to oldest-first for chat context
            turns = []
            for row in reversed(rows):
                turns.append({
                    "role": row["role"],
                    "content": row["content"],
                    "timestamp": row["timestamp"].isoformat() if row["timestamp"] else None
                })
            return turns

        except Exception as e:
            logger.error(f"Failed to get session turns for {session_id}: {e}")
            return []

    async def store_turn(self, session_id: str, role: str, content: str) -> Optional[int]:
        """Store a conversation turn.

        Args:
            session_id: The conversation/session identifier
            role: 'user' or 'assistant'
            content: The message content

        Returns:
            The inserted row ID, or None on failure
        """
        try:
            conn = await asyncpg.connect(self._get_dsn(), timeout=5)
            row_id = await conn.fetchval("""
                INSERT INTO conversation_messages (conversation_id, role, content)
                VALUES ($1, $2, $3)
                RETURNING id
            """, session_id, role, content)
            await conn.close()
            return row_id
        except Exception as e:
            logger.error(f"Failed to store turn for {session_id}: {e}")
            return None


# Singleton
_conversation_service: Optional[ConversationService] = None

def get_conversation_service() -> ConversationService:
    global _conversation_service
    if not _conversation_service:
        _conversation_service = ConversationService()
    return _conversation_service