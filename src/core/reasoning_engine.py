#!/usr/bin/env python3
"""
Echo Brain Reasoning Engine

The ACTUAL THINKING layer:
  Query → Embed → Search Memories → LLM Synthesis → Intelligent Response
"""

import asyncio
import httpx
import logging
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ReasoningEngine:
    """
    The brain that turns memory retrieval into intelligent responses.
    """

    def __init__(
        self,
        ollama_url: str = None,
        qdrant_url: str = None,
        collection_name: str = None,
        embedding_model: str = None,
        reasoning_model: str = None
    ):
        # Read from environment variables with fallbacks
        self.ollama_url = ollama_url or os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.collection_name = collection_name or os.getenv("QDRANT_COLLECTION", "echo_memory")
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "mxbai-embed-large:latest")
        self.reasoning_model = reasoning_model or os.getenv("OLLAMA_MODEL", "mistral:7b")

        # Available fallback models in priority order (ONLY WHAT'S ACTUALLY INSTALLED)
        self.fallback_models = [
            "mistral:7b",
            "mistral:7b"
        ]

        # Track which model is actually being used
        self.active_model = self.reasoning_model

        logger.info(f"🧠 ReasoningEngine initialized:")
        logger.info(f"  - Ollama: {self.ollama_url}")
        logger.info(f"  - Qdrant: {self.qdrant_url}")
        logger.info(f"  - Collection: {self.collection_name}")
        logger.info(f"  - Embedding Model: {self.embedding_model}")
        logger.info(f"  - Reasoning Model: {self.reasoning_model}")
        
    async def think(
        self,
        query: str,
        max_memories: int = 10,
        min_confidence: float = 0.4
    ) -> Dict[str, Any]:
        """
        Main entry point - think about a query and generate a response.
        
        Returns dict matching test expectations:
        - answer: str
        - confidence: float
        - memories_used: int
        - model_used: str
        - reasoning_time_ms: int
        - sources: List[Dict]
        """
        start_time = datetime.now()

        # Step 1: Retrieve relevant memories
        memory_search_start = datetime.now()
        memories = await self._retrieve_memories(query, max_memories)
        memory_search_time_ms = (datetime.now() - memory_search_start).total_seconds() * 1000

        # Filter by confidence
        relevant_memories = [
            m for m in memories
            if m.get("score", 0) >= min_confidence
        ]

        # Calculate average memory score BEFORE filtering
        avg_memory_score = sum(m.get("score", 0) for m in memories) / len(memories) if memories else 0.0

        # Step 2: Build context from memories
        context = self._build_context(relevant_memories)

        # Step 3: Generate response using LLM
        llm_start = datetime.now()
        answer = await self._generate_response(query, context, relevant_memories)
        llm_generation_time_ms = (datetime.now() - llm_start).total_seconds() * 1000

        # Calculate confidence (average of FILTERED memory scores)
        if relevant_memories:
            confidence = sum(m.get("score", 0) for m in relevant_memories) / len(relevant_memories)
        else:
            confidence = 0.0

        total_time_ms = (datetime.now() - start_time).total_seconds() * 1000

        # Build sources list
        sources = [
            {
                "content": m.get("content", "")[:200],
                "score": round(m.get("score", 0), 4),
                "type": m.get("type", "memory"),
                "source": m.get("source", "echo_memory")
            }
            for m in relevant_memories[:5]
        ]

        return {
            "query": query,
            "answer": answer,
            "confidence": round(confidence, 4),
            "memories_used": len(relevant_memories),
            "memories_searched": len(memories),
            "avg_memory_score": round(avg_memory_score, 4),
            "model_used": self.active_model,
            "reasoning_time_ms": int(total_time_ms),
            "memory_search_time_ms": int(memory_search_time_ms),
            "llm_generation_time_ms": int(llm_generation_time_ms),
            "total_time_ms": int(total_time_ms),
            "embedding_model_used": self.embedding_model,
            "sources": sources
        }
    
    async def think_stream(
        self,
        query: str,
        max_memories: int = 10,
        min_confidence: float = 0.4
    ):
        """
        Streaming version of think() that yields progress updates.
        Yields dicts with 'type' and 'data' for different stages.
        """
        start_time = datetime.now()

        # Yield initial acknowledgment
        yield {
            "type": "status",
            "data": "Searching memories...",
            "timestamp": datetime.now().isoformat()
        }

        # Step 1: Retrieve relevant memories
        memories = await self._retrieve_memories(query, max_memories)

        # Filter by confidence
        relevant_memories = [
            m for m in memories
            if m.get("score", 0) >= min_confidence
        ]

        # Calculate average score of ALL memories searched
        avg_memory_score = sum(m.get("score", 0) for m in memories) / len(memories) if memories else 0
        memories_searched_count = len(memories)  # Store for complete event

        # Yield memory search complete
        yield {
            "type": "memories_found",
            "data": {
                "memories_searched": memories_searched_count,
                "memories_used": len(relevant_memories),
                "avg_memory_score": round(avg_memory_score, 4),
                "avg_filtered_score": sum(m.get("score", 0) for m in relevant_memories) / len(relevant_memories) if relevant_memories else 0
            },
            "timestamp": datetime.now().isoformat()
        }

        # Step 2: Build context from memories
        context = self._build_context(relevant_memories)

        # Yield thinking status
        yield {
            "type": "status",
            "data": f"Thinking with {self.active_model}...",
            "timestamp": datetime.now().isoformat()
        }

        # Step 3: Stream response from LLM
        async for chunk in self._generate_response_stream(query, context, relevant_memories):
            yield {
                "type": "response_chunk",
                "data": chunk,
                "timestamp": datetime.now().isoformat()
            }

        # Calculate final metrics
        elapsed_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        confidence = sum(m.get("score", 0) for m in relevant_memories) / len(relevant_memories) if relevant_memories else 0.0

        # Yield completion with metadata
        yield {
            "type": "complete",
            "data": {
                "confidence": round(confidence, 4),
                "memories_searched": memories_searched_count,
                "memories_used": len(relevant_memories),
                "avg_memory_score": round(avg_memory_score, 4),
                "model_used": self.active_model,
                "reasoning_time_ms": elapsed_ms,
                "sources": [
                    {
                        "content": m.get("content", "")[:200],
                        "score": round(m.get("score", 0), 4),
                        "type": m.get("type", "memory")
                    }
                    for m in relevant_memories[:5]
                ]
            },
            "timestamp": datetime.now().isoformat()
        }

    async def search_only(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Search memories without LLM synthesis."""
        memories = await self._retrieve_memories(query, limit)
        
        return {
            "query": query,
            "count": len(memories),
            "results": [
                {
                    "content": m.get("content", "")[:500],
                    "score": round(m.get("score", 0), 4),
                    "type": m.get("type", "memory"),
                    "source": m.get("source", "echo_memory")
                }
                for m in memories
            ]
        }
    
    async def _retrieve_memories(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search for relevant memories using semantic search."""
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                # Generate embedding for query
                embed_response = await client.post(
                    f"{self.ollama_url}/api/embeddings",
                    json={"model": self.embedding_model, "prompt": query}
                )
                
                if embed_response.status_code != 200:
                    logger.error(f"Embedding failed: {embed_response.status_code}")
                    return []
                    
                embedding = embed_response.json().get("embedding", [])
                
                if not embedding:
                    logger.error("Empty embedding returned")
                    return []
                
                # Search Qdrant
                search_response = await client.post(
                    f"{self.qdrant_url}/collections/{self.collection_name}/points/search",
                    json={
                        "vector": embedding,
                        "limit": limit,
                        "with_payload": True
                    }
                )
                
                if search_response.status_code != 200:
                    logger.error(f"Qdrant search failed: {search_response.status_code}")
                    return []
                    
                results = search_response.json().get("result", [])
                
                # Format results
                memories = []
                for point in results:
                    payload = point.get("payload", {})
                    memories.append({
                        "id": str(point.get("id", "")),
                        "score": float(point.get("score", 0)),
                        "content": payload.get("content", payload.get("text", "")),
                        "type": payload.get("type", "memory"),
                        "source": payload.get("source", "echo_memory"),
                        "timestamp": payload.get("timestamp", payload.get("created_at", ""))
                    })
                    
                return memories
                
        except Exception as e:
            logger.error(f"Memory retrieval failed: {e}")
            return []
    
    def _build_context(self, memories: List[Dict[str, Any]]) -> str:
        """Build a context string from memories for the LLM."""
        if not memories:
            return "No relevant memories found in the knowledge base."
            
        context_parts = []
        for i, mem in enumerate(memories, 1):
            content = mem.get("content", "").strip()
            score = mem.get("score", 0)
            mem_type = mem.get("type", "memory")
            
            # Truncate very long content
            if len(content) > 600:
                content = content[:600] + "..."
                
            context_parts.append(
                f"[Memory {i}] (relevance: {score:.2f}, type: {mem_type})\n{content}"
            )
            
        return "\n\n".join(context_parts)
    
    async def _select_available_model(self) -> Optional[str]:
        """Check which models are actually available and select one."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{self.ollama_url}/api/tags")
                if response.status_code == 200:
                    available = [m["name"] for m in response.json().get("models", [])]

                    # Try preferred model first
                    if self.reasoning_model in available:
                        return self.reasoning_model

                    # Try fallbacks
                    for model in self.fallback_models:
                        if model in available:
                            logger.warning(f"Primary model {self.reasoning_model} not available, using {model}")
                            return model

                    logger.error(f"No suitable models found. Available: {available}")
                    return None
        except Exception as e:
            logger.error(f"Failed to check available models: {e}")
            return self.reasoning_model  # Hope for the best

    async def _generate_response_stream(
        self,
        query: str,
        context: str,
        memories: List[Dict[str, Any]]
    ):
        """Stream the LLM response chunk by chunk."""

        # Select available model
        model_to_use = await self._select_available_model()
        if not model_to_use:
            yield self._fallback_response(query, memories)
            return

        self.active_model = model_to_use

        # Build prompts
        system_prompt = """You are Echo Brain, an intelligent AI assistant with access to Patrick's personal knowledge base containing conversations, code, and documentation from his Tower system.

Your job is to answer questions based on the memories and context provided. Follow these guidelines:

1. SYNTHESIZE - Don't just list memories. Combine information into a coherent, helpful answer.
2. BE SPECIFIC - Reference concrete details from the memories when relevant.
3. BE HONEST - If memories don't contain enough info, say so clearly.
4. BE CONCISE - Get to the point. No unnecessary preamble.
5. STAY GROUNDED - Only use information from the provided memories, don't make things up.

If the memories contain relevant technical details, code patterns, or system architecture info, include those specifics in your answer."""

        user_prompt = f"""Question: {query}

Here are the relevant memories from the knowledge base:

{context}

Based on these memories, provide a helpful and specific answer to the question. If the memories don't contain enough information to fully answer, acknowledge what you do know and what's missing."""

        try:
            async with httpx.AsyncClient(timeout=None) as client:
                # Stream the response
                async with client.stream(
                    "POST",
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": model_to_use,
                        "prompt": user_prompt,
                        "system": system_prompt,
                        "stream": True,
                        "options": {
                            "temperature": 0.7,
                            "num_predict": 800,
                            "top_p": 0.9
                        }
                    }
                ) as response:
                    async for line in response.aiter_lines():
                        if line:
                            import json
                            try:
                                data = json.loads(line)
                                if "response" in data:
                                    yield data["response"]
                            except json.JSONDecodeError:
                                continue

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            yield f"\n\n[Error: {str(e)}]"

    async def _generate_response(
        self,
        query: str,
        context: str,
        memories: List[Dict[str, Any]]
    ) -> str:
        """Use the LLM to generate an intelligent response."""

        # Select available model
        model_to_use = await self._select_available_model()
        if not model_to_use:
            return self._fallback_response(query, memories)

        self.active_model = model_to_use

        # Build the system prompt
        system_prompt = """You are Echo Brain, an intelligent AI assistant with access to Patrick's personal knowledge base containing conversations, code, and documentation from his Tower system.

Your job is to answer questions based on the memories and context provided. Follow these guidelines:

1. SYNTHESIZE - Don't just list memories. Combine information into a coherent, helpful answer.
2. BE SPECIFIC - Reference concrete details from the memories when relevant.
3. BE HONEST - If memories don't contain enough info, say so clearly.
4. BE CONCISE - Get to the point. No unnecessary preamble.
5. STAY GROUNDED - Only use information from the provided memories, don't make things up.

If the memories contain relevant technical details, code patterns, or system architecture info, include those specifics in your answer."""

        # Build the user prompt with context
        user_prompt = f"""Question: {query}

Here are the relevant memories from the knowledge base:

{context}

Based on these memories, provide a helpful and specific answer to the question. If the memories don't contain enough information to fully answer, acknowledge what you do know and what's missing."""

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": model_to_use,
                        "prompt": user_prompt,
                        "system": system_prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "num_predict": 800,
                            "top_p": 0.9
                        }
                    }
                )
                
                if response.status_code != 200:
                    logger.error(f"LLM generation failed: {response.status_code}")
                    return self._fallback_response(query, memories)
                    
                result = response.json()
                generated = result.get("response", "").strip()
                
                if not generated:
                    return self._fallback_response(query, memories)
                    
                return generated
                
        except asyncio.TimeoutError:
            logger.error("LLM generation timed out")
            return self._fallback_response(query, memories)
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return self._fallback_response(query, memories)
    
    def _fallback_response(self, query: str, memories: List[Dict[str, Any]]) -> str:
        """Generate a basic response if LLM fails."""
        if not memories:
            return f"I couldn't find any relevant information about '{query}' in my memory. The knowledge base may not contain information on this topic yet."
            
        # Summarize what we found
        memory_types = set(m.get("type", "memory") for m in memories)
        top_score = memories[0].get("score", 0) if memories else 0
        
        response = f"I found {len(memories)} relevant memories (best match: {top_score:.1%} confidence) but couldn't generate a synthesis. "
        response += f"Memory types: {', '.join(memory_types)}. "
        response += f"Top result: {memories[0].get('content', '')[:200]}..."
        
        return response


# Health check helper
async def check_dependencies() -> Dict[str, Any]:
    """Check if Qdrant and Ollama are available."""
    checks = {
        "qdrant": False,
        "ollama": False,
        "embedding_model": False,
        "reasoning_model": False,
        "vector_count": 0
    }
    models = {}
    
    async with httpx.AsyncClient(timeout=10) as client:
        # Check Qdrant
        try:
            resp = await client.get("http://localhost:6333/collections/echo_memory")
            if resp.status_code == 200:
                info = resp.json()
                checks["qdrant"] = True
                checks["vector_count"] = info.get("result", {}).get("points_count", 0)
        except:
            pass
            
        # Check Ollama
        try:
            resp = await client.get("http://localhost:11434/api/tags")
            if resp.status_code == 200:
                checks["ollama"] = True
                model_list = [m["name"] for m in resp.json().get("models", [])]
                models["available"] = model_list
                
                # Check for specific models
                checks["embedding_model"] = any("mxbai-embed-large" in m for m in model_list)
                checks["reasoning_model"] = any(
                    any(x in m for x in ["llama3", "mistral", "phi"])
                    for m in model_list
                )
        except:
            pass
    
    return {
        "status": "healthy" if all([checks["qdrant"], checks["ollama"], checks["embedding_model"]]) else "degraded",
        "checks": checks,
        "models": models
    }

# Create singleton instance
reasoning_engine = ReasoningEngine()
