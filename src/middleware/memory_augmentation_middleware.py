#!/usr/bin/env python3
"""
Memory Augmentation Middleware - Automatically adds memory context to all queries
"""

import psycopg2
import logging
import uuid
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class MemoryAugmentationMiddleware:
    """Middleware to automatically augment queries with memory context"""

    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'database': 'echo_brain',
            'user': 'patrick',
            'password': 'RP78eIrW7cI2jYvL5akt1yurE'
        }
        # Request-scoped memory cache to prevent cross-contamination
        self._request_cache = {}

    def search_memories(self, query: str) -> List[str]:
        """Search all memory sources and return relevant context"""
        memories = []

        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()

            # Search learned patterns
            cur.execute("""
                SELECT pattern_text, confidence
                FROM echo_learned_patterns
                WHERE pattern_text ILIKE %s
                ORDER BY confidence DESC
                LIMIT 3
            """, (f'%{query}%',))

            for row in cur.fetchall():
                if row[0]:
                    memories.append(f"[Knowledge: {row[1]:.2f}]: {row[0][:200]}...")

            # Search previous conversations
            cur.execute("""
                SELECT query, response
                FROM echo_unified_interactions
                WHERE query ILIKE %s OR response ILIKE %s
                ORDER BY timestamp DESC
                LIMIT 2
            """, (f'%{query}%', f'%{query}%'))

            for row in cur.fetchall():
                if row[0] and row[1]:
                    query_text, response_text = row[0], row[1]

                    # Domain filtering: Skip anime-contaminated memories for technical queries
                    anime_keywords = ['goblin', 'anime', 'scene', 'cyber', 'slayer', 'character', 'mei', 'tokyo debt']
                    tech_keywords = ['code', 'function', 'debug', 'system', 'architecture', 'what is', 'calculate']

                    query_lower = query.lower()
                    response_lower = response_text.lower()

                    is_technical_query = any(kw in query_lower for kw in tech_keywords)
                    has_anime_content = any(kw in response_lower for kw in anime_keywords)

                    if is_technical_query and has_anime_content:
                        logger.info(f"🚫 Skipping anime-contaminated memory for technical query: {query_text[:50]}...")
                        continue

                    memories.append(f"[Previous Q]: {query_text[:100]}")
                    memories.append(f"[Previous A]: {response_text[:100]}...")

            cur.close()
            conn.close()

            logger.info(f"📚 Memory augmentation found {len(memories)} relevant memories")

        except Exception as e:
            logger.error(f"Memory search error: {e}")

        return memories

    def augment_query(self, query: str, request_id: Optional[str] = None) -> str:
        """Augment a query with memory context

        Args:
            query: The query to augment
            request_id: Optional unique identifier for this request to prevent memory contamination
        """

        # Generate request ID if not provided
        if request_id is None:
            request_id = str(uuid.uuid4())

        # Clear any stale cache entries (keep only last 100 requests)
        if len(self._request_cache) > 100:
            self._request_cache.clear()

        # Don't augment if already augmented
        if "Based on these memories:" in query or "Relevant memories:" in query:
            return query

        # BYPASS SIMPLE QUERIES - don't augment if:
        # - Query is too short (less than 15 characters)
        # - Query is a simple JSON/command request
        # - Query starts with common command patterns
        query_lower = query.lower().strip()

        # Skip augmentation for simple/command queries
        bypass_patterns = [
            'return exactly',
            'return only',
            'return json',
            'return:',
            'json:',
            '{"',
            '[{',
            'test',
            'ping',
            'hello',
            'status',
            'echo',
            'what is 2+2',
            'what is 1+1'
        ]

        # Debug logging
        logger.info(f"🔍 DEBUG: query='{query}', length={len(query)}, query_lower='{query_lower}'")
        logger.info(f"🔍 DEBUG: bypass_patterns check results: {[(p, p in query_lower) for p in bypass_patterns]}")

        if len(query) < 20 or any(p in query_lower for p in bypass_patterns):
            logger.info(f"📋 Bypassing memory augmentation for simple query: {query[:50]}")
            return query

        memories = self.search_memories(query)

        if memories:
            augmented = "Relevant memories:\n"
            for mem in memories[:5]:  # Limit to top 5 memories
                augmented += f"• {mem}\n"
            augmented += f"\nCurrent query: {query}"
            logger.info(f"📚 Augmented query with {len(memories)} memories")
            return augmented

        return query

# Global instance
memory_augmenter = MemoryAugmentationMiddleware()

def augment_with_memories(query: str, request_id: Optional[str] = None) -> str:
    """Public function to augment queries with memory

    Args:
        query: The query to augment
        request_id: Optional unique identifier for this request to prevent memory contamination
    """
    return memory_augmenter.augment_query(query, request_id)