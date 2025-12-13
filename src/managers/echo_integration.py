#!/usr/bin/env python3
"""
MEMORY-INTEGRATED Echo Brain Integration Layer
THIS VERSION ACTUALLY USES MEMORY!
"""

import logging
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# EXISTING IMPORTS
from .resilient_model_manager import (
    ResilientModelManager,
    TaskUrgency,
    ExecutionResult,
    get_resilient_manager
)

# CRITICAL NEW IMPORTS - THE MISSING MEMORY COMPONENTS!
from .knowledge_retrieval import KnowledgeRetrieval
import sys
sys.path.append('/opt/tower-echo-brain/src')
from echo_vector_memory import VectorMemory
from memory.context_retrieval import ConversationContextRetriever

logger = logging.getLogger(__name__)

# API Router
router = APIRouter(prefix="/api/echo/resilient", tags=["resilient-models"])


class MemoryOrchestrator:
    """
    CENTRAL MEMORY SERVICE - The foundation Echo was missing!
    ALL queries go through memory FIRST
    """

    def __init__(self):
        # Initialize ALL memory systems
        self.knowledge_retrieval = KnowledgeRetrieval()  # PostgreSQL semantic search
        self.vector_memory = VectorMemory()  # Qdrant 4096D vectors
        self.context_retriever = ConversationContextRetriever()  # Conversation context

        logger.info("🧠 MEMORY ORCHESTRATOR INITIALIZED - Echo can now remember!")

    async def retrieve_all_memories(self, query: str, conversation_id: str = None) -> Dict[str, Any]:
        """
        Retrieve memories from ALL sources before processing
        THIS IS WHAT WAS MISSING!
        """
        memories = {
            "knowledge_items": [],
            "vector_memories": [],
            "conversation_context": [],
            "learned_patterns": []
        }

        # 1. Search knowledge_items table (PostgreSQL)
        try:
            knowledge = self.knowledge_retrieval.search_knowledge(query, limit=5)
            if knowledge:
                memories["knowledge_items"] = knowledge
                logger.info(f"✅ Retrieved {len(knowledge)} knowledge items")
        except Exception as e:
            logger.warning(f"Knowledge retrieval error: {e}")

        # 2. Search vector memories (Qdrant)
        try:
            vector_results = await self.vector_memory.recall(query, limit=5)
            if vector_results:
                memories["vector_memories"] = vector_results
                logger.info(f"✅ Retrieved {len(vector_results)} vector memories")
        except Exception as e:
            logger.warning(f"Vector memory error: {e}")

        # 3. Get conversation context
        if conversation_id:
            try:
                context = self.context_retriever.get_context(conversation_id)
                if context:
                    memories["conversation_context"] = context
                    logger.info("✅ Retrieved conversation context")
            except Exception as e:
                logger.warning(f"Context retrieval error: {e}")

        # 4. Search learned patterns (direct PostgreSQL)
        try:
            import psycopg2
            conn = psycopg2.connect(
                host='localhost',
                database='echo_brain',
                user='patrick',
                password='***REMOVED***'
            )
            cur = conn.cursor()
            cur.execute("""
                SELECT pattern_text, confidence, frequency
                FROM echo_learned_patterns
                WHERE pattern_text ILIKE %s
                ORDER BY confidence DESC, frequency DESC
                LIMIT 5
            """, (f'%{query[:50]}%',))
            patterns = cur.fetchall()
            if patterns:
                memories["learned_patterns"] = [
                    {"text": p[0], "confidence": p[1], "frequency": p[2]}
                    for p in patterns
                ]
                logger.info(f"✅ Retrieved {len(patterns)} learned patterns")
            cur.close()
            conn.close()
        except Exception as e:
            logger.warning(f"Pattern retrieval error: {e}")

        return memories

    def augment_query_with_memories(self, query: str, memories: Dict[str, Any]) -> str:
        """
        Augment the query with retrieved memories
        This creates the context-aware prompt!
        """
        augmented_parts = []

        # Add knowledge items
        if memories["knowledge_items"]:
            augmented_parts.append("\n📚 Relevant Knowledge:")
            for item in memories["knowledge_items"][:3]:
                augmented_parts.append(f"- {item.get('title', '')}: {item.get('content', '')[:200]}...")

        # Add vector memories
        if memories["vector_memories"]:
            augmented_parts.append("\n🧠 Related Memories:")
            for mem in memories["vector_memories"][:3]:
                augmented_parts.append(f"- {mem.get('text', '')[:200]}... (relevance: {mem.get('score', 0):.2f})")

        # Add learned patterns
        if memories["learned_patterns"]:
            augmented_parts.append("\n📊 Learned Patterns:")
            for pattern in memories["learned_patterns"][:2]:
                augmented_parts.append(f"- {pattern['text'][:150]}... (confidence: {pattern['confidence']})")

        # Add conversation context
        if memories["conversation_context"]:
            augmented_parts.append("\n💬 Previous Context:")
            for ctx in memories["conversation_context"][-2:]:
                augmented_parts.append(f"- {ctx[:150]}...")

        # Combine with original query
        if augmented_parts:
            memory_context = "\n".join(augmented_parts)
            return f"""Based on my memory and knowledge:
{memory_context}

Now answering your query: {query}"""

        return query

    async def store_interaction(self, query: str, response: str, metadata: Dict = None):
        """
        Store the interaction in memory for future recall
        CONTINUOUS LEARNING!
        """
        # Store in vector memory
        try:
            await self.vector_memory.remember(
                f"Q: {query}\nA: {response}",
                metadata=metadata
            )
            logger.info("✅ Stored interaction in vector memory")
        except Exception as e:
            logger.warning(f"Failed to store in vector memory: {e}")


class TaskTypeClassifier:
    """Task classification (existing code)"""
    def __init__(self):
        self.patterns = {
            "code_generation": ["write", "create", "implement", "build", "function"],
            "code_review": ["review", "check", "analyze code", "find bugs"],
            "reasoning": ["why", "explain", "reason", "think", "logic"],
            "complex": ["design", "architect", "plan", "strategy"],
            "analysis": ["analyze", "examine", "investigate", "study"],
            "creative": ["imagine", "create story", "fiction", "poem"],
            "technical": ["technical", "engineering", "system", "database"],
            "simple": ["what is", "define", "simple", "basic"],
            "fast_response": ["quick", "fast", "brief", "short"]
        }

    def classify(self, query: str) -> str:
        query_lower = query.lower()
        scores = {}
        for task_type, patterns in self.patterns.items():
            score = sum(1 for pattern in patterns if pattern in query_lower)
            if score > 0:
                scores[task_type] = score
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        return "general"


class EchoBrainIntegration:
    """
    MEMORY-INTEGRATED Echo Brain
    Now with memory as the foundation!
    """

    def __init__(self):
        self.classifier = TaskTypeClassifier()
        self.manager: Optional[ResilientModelManager] = None

        # INITIALIZE MEMORY ORCHESTRATOR - THE CORE!
        self.memory = MemoryOrchestrator()
        logger.info("✅ Echo Brain initialized WITH MEMORY INTEGRATION")

    async def initialize(self):
        """Initialize the integration"""
        self.manager = await get_resilient_manager()
        logger.info("✅ Resilient model manager initialized")

    async def process_query(
        self,
        query: str,
        task_type: Optional[str] = None,
        urgency: TaskUrgency = TaskUrgency.BACKGROUND,
        system_prompt: str = "",
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process query WITH MEMORY RETRIEVAL FIRST!
        Memory → Context → LLM → Store → Response
        """
        if not self.manager:
            await self.initialize()

        # Classify task
        if not task_type:
            task_type = self.classifier.classify(query)
            logger.info(f"Classified as: {task_type}")

        # ========== MEMORY RETRIEVAL (THE MISSING PIECE!) ==========
        logger.info("🔍 RETRIEVING MEMORIES BEFORE PROCESSING...")
        memories = await self.memory.retrieve_all_memories(query, conversation_id)

        # Count total memories retrieved
        total_memories = sum(len(v) if isinstance(v, list) else 0 for v in memories.values())
        logger.info(f"📚 Retrieved {total_memories} total memories")

        # AUGMENT QUERY WITH MEMORIES
        augmented_query = self.memory.augment_query_with_memories(query, memories)
        if augmented_query != query:
            logger.info("✅ Query augmented with memory context")

        # Execute with augmented query
        result = await self.manager.complete_with_fallback(
            task_type=task_type,
            prompt=augmented_query,  # USE AUGMENTED QUERY!
            system=system_prompt or self._get_memory_aware_system_prompt(),
            urgency=urgency
        )

        # Store interaction for future recall
        if result.success:
            await self.memory.store_interaction(
                query,
                result.value,
                metadata={
                    "model": result.model_used,
                    "task_type": task_type,
                    "conversation_id": conversation_id
                }
            )

            response = {
                "success": True,
                "response": result.value,
                "model_used": result.model_used,
                "task_type": task_type,
                "memories_retrieved": total_memories,  # NEW!
                "memory_sources": {  # NEW!
                    "knowledge_items": len(memories["knowledge_items"]),
                    "vector_memories": len(memories["vector_memories"]),
                    "learned_patterns": len(memories["learned_patterns"]),
                    "conversation_context": len(memories["conversation_context"])
                },
                "fallback_used": result.fallback_used,
                "attempts": result.attempts,
                "latency_ms": result.total_latency_ms,
                "conversation_id": conversation_id
            }

            logger.info(f"✅ Query processed with {total_memories} memories")
        else:
            response = {
                "success": False,
                "error": result.error,
                "memories_retrieved": total_memories
            }

        return response

    def _get_memory_aware_system_prompt(self) -> str:
        """System prompt that acknowledges memory"""
        return """You are Echo Brain, an AI with persistent memory.
I have access to my stored knowledge, learned patterns, and conversation history.
When answering, I will use relevant memories to provide informed, contextual responses.
I continuously learn from our interactions."""


# Initialize integration
echo_integration = EchoBrainIntegration()

# API endpoints
@router.post("/query")
async def query_with_memory(request: Dict[str, Any]):
    """Process query with full memory integration"""
    return await echo_integration.process_query(
        query=request.get("query", ""),
        task_type=request.get("task_type"),
        conversation_id=request.get("conversation_id")
    )

@router.get("/memory/stats")
async def get_memory_stats():
    """Get memory system statistics"""
    # Implementation for memory stats
    return {"status": "Memory system active"}
