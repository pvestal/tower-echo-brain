#!/usr/bin/env python3
"""
Memory integration for Echo - connects 134k+ existing memories
"""

import psycopg2
import logging
from typing import Dict, List, Any, Optional
from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/echo/resilient", tags=["resilient-models"])


class MemorySearch:
    """Search Echo's existing memory tables"""

    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'database': 'echo_brain',
            'user': 'patrick',
            'password': '***REMOVED***'
        }

    def search_all(self, query: str) -> Dict[str, List]:
        """Search all memory sources"""
        memories = {}

        # Search conversations
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            cur.execute("""
                SELECT user_query, response, timestamp
                FROM conversations
                WHERE user_query ILIKE %s OR response ILIKE %s
                ORDER BY timestamp DESC
                LIMIT 5
            """, (f'%{query}%', f'%{query}%'))

            memories['conversations'] = []
            for row in cur.fetchall():
                memories['conversations'].append({
                    'query': row[0][:200] if row[0] else '',
                    'response': row[1][:200] if row[1] else '',
                    'date': row[2]
                })
            cur.close()
            conn.close()
        except Exception as e:
            logger.error(f"Conversation search error: {e}")
            memories['conversations'] = []

        # Search learned patterns (Work documents)
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            cur.execute("""
                SELECT pattern_text, confidence
                FROM echo_learned_patterns
                WHERE pattern_text ILIKE %s
                ORDER BY confidence DESC
                LIMIT 5
            """, (f'%{query}%',))

            memories['learned_patterns'] = []
            for row in cur.fetchall():
                memories['learned_patterns'].append({
                    'text': row[0][:500] if row[0] else '',
                    'confidence': row[1]
                })
            cur.close()
            conn.close()
        except Exception as e:
            logger.error(f"Pattern search error: {e}")
            memories['learned_patterns'] = []

        # Search photos
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            cur.execute("""
                SELECT filename, file_path, metadata
                FROM photo_index
                WHERE filename ILIKE %s OR metadata::text ILIKE %s
                LIMIT 5
            """, (f'%{query}%', f'%{query}%'))

            memories['photos'] = []
            for row in cur.fetchall():
                memories['photos'].append({
                    'name': row[0],
                    'path': row[1],
                    'metadata': row[2]
                })
            cur.close()
            conn.close()
        except Exception as e:
            logger.error(f"Photo search error: {e}")
            memories['photos'] = []

        # Search takeout insights
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            cur.execute("""
                SELECT context, insight_type, file_id
                FROM takeout_insights
                WHERE context::text ILIKE %s
                LIMIT 5
            """, (f'%{query}%',))

            memories['takeout'] = []
            for row in cur.fetchall():
                memories['takeout'].append({
                    'content': str(row[0])[:300] if row[0] else '',
                    'type': row[1],
                    'source': str(row[2]) if row[2] else ''
                })
            cur.close()
            conn.close()
        except Exception as e:
            logger.error(f"Takeout search error: {e}")
            memories['takeout'] = []

        return memories


class EchoMemoryIntegration:
    """Echo with memory integration"""

    def __init__(self):
        self.memory_search = MemorySearch()
        self.manager = None
        logger.info("✅ Echo initialized with memory search")

    async def initialize(self):
        """Initialize model manager"""
        from .resilient_model_manager import get_resilient_manager
        self.manager = await get_resilient_manager()

    async def process_with_memory(self, query: str, conversation_id: str = None) -> Dict[str, Any]:
        """Process query with memory search first"""
        print(f"[MEMORY] process_with_memory called for: {query}")

        if not self.manager:
            await self.initialize()

        # SEARCH MEMORIES FIRST
        print(f"[MEMORY] Searching memories for: {query}")
        memories = self.memory_search.search_all(query)
        print(f"[MEMORY] Raw memories found: {memories.keys()}")

        # Count total memories found
        total_memories = sum(len(v) for v in memories.values())
        logger.info(f"📚 Found {total_memories} memories for query")

        # Build augmented query with memory context
        augmented_query = query
        if total_memories > 0:
            context_parts = ["\nRelevant memories:"]

            # Add learned patterns (Work docs)
            if memories.get('learned_patterns'):
                context_parts.append("\n📄 From documents:")
                for pattern in memories['learned_patterns'][:2]:
                    context_parts.append(f"- {pattern['text'][:200]}...")

            # Add previous conversations
            if memories.get('conversations'):
                context_parts.append("\n💬 Previous conversations:")
                for conv in memories['conversations'][:2]:
                    context_parts.append(f"- Q: {conv['query'][:100]}")
                    context_parts.append(f"  A: {conv['response'][:100]}")

            # Add photos if relevant
            if memories.get('photos'):
                context_parts.append(f"\n📸 Found {len(memories['photos'])} related photos")

            # Add takeout insights
            if memories.get('takeout'):
                context_parts.append("\n📧 From personal data:")
                for item in memories['takeout'][:2]:
                    context_parts.append(f"- {item['content'][:150]}...")

            augmented_query = query + "\n".join(context_parts) + f"\n\nNow answering: {query}"

        # Process with model
        from .resilient_model_manager import TaskUrgency
        result = await self.manager.complete_with_fallback(
            task_type="general",
            prompt=augmented_query,
            system="You are Echo Brain with access to stored memories. Use the provided context when relevant.",
            urgency=TaskUrgency.BACKGROUND
        )

        if result.success:
            return {
                "success": True,
                "response": result.value,
                "model_used": result.model_used,
                "memories_retrieved": total_memories,
                "memory_breakdown": {
                    "patterns": len(memories.get('learned_patterns', [])),
                    "conversations": len(memories.get('conversations', [])),
                    "photos": len(memories.get('photos', [])),
                    "takeout": len(memories.get('takeout', []))
                },
                "conversation_id": conversation_id
            }
        else:
            return {
                "success": False,
                "error": result.error,
                "memories_retrieved": total_memories
            }


# Initialize
echo_memory = EchoMemoryIntegration()

@router.post("/query")
async def query_with_memory(request: Dict[str, Any]):
    """Query endpoint with memory search"""
    return await echo_memory.process_with_memory(
        query=request.get("query", ""),
        conversation_id=request.get("conversation_id")
    )