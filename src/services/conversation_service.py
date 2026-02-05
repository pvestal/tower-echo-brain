"""
Conversation Service for Echo Brain
Handles conversation search and retrieval
"""
import logging
from typing import Dict, List, Any
from src.core.pg_reasoning import search_pg

logger = logging.getLogger(__name__)

class ConversationService:
    """Service for searching and managing conversations"""

    def __init__(self):
        self.logger = logger

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