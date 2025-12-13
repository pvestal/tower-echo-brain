#!/usr/bin/env python3
"""
Memory Augmentation Middleware - Automatically adds memory context to all queries
"""

import psycopg2
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class MemoryAugmentationMiddleware:
    """Middleware to automatically augment queries with memory context"""

    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'database': 'echo_brain',
            'user': 'patrick',
            'password': 'tower_echo_brain_secret_key_2025'
        }

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
                SELECT user_query, response
                FROM conversations
                WHERE user_query ILIKE %s OR response ILIKE %s
                ORDER BY timestamp DESC
                LIMIT 2
            """, (f'%{query}%', f'%{query}%'))

            for row in cur.fetchall():
                if row[0] and row[1]:
                    memories.append(f"[Previous Q]: {row[0][:100]}")
                    memories.append(f"[Previous A]: {row[1][:100]}...")

            cur.close()
            conn.close()

            logger.info(f"📚 Memory augmentation found {len(memories)} relevant memories")

        except Exception as e:
            logger.error(f"Memory search error: {e}")

        return memories

    def augment_query(self, query: str) -> str:
        """Augment a query with memory context"""

        # Don't augment if already augmented
        if "Based on these memories:" in query or "Relevant memories:" in query:
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

def augment_with_memories(query: str) -> str:
    """Public function to augment queries with memory"""
    return memory_augmenter.augment_query(query)