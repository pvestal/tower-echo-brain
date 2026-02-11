"""
Voice Conversation Memory — persists voice interactions into Echo Brain.

Each voice turn (user transcript + Echo response) is:
1. Embedded via Ollama (nomic-embed-text, 768-dim) and stored in Qdrant echo_memory
2. Logged to PostgreSQL voice_conversations table for history/analytics

Designed to run fire-and-forget so it never blocks the voice pipeline.
"""
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

import httpx

logger = logging.getLogger("echo.voice_memory")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "echo_memory")
DB_URL = os.getenv("DATABASE_URL", "postgresql://echo:echo_secure_password_123@localhost/echo_brain")

CATEGORY = "voice:conversation"


async def _embed_text(text: str) -> Optional[List[float]]:
    """Get embedding vector from Ollama."""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/embed",
                json={"model": EMBED_MODEL, "input": text},
            )
            if resp.status_code != 200:
                logger.warning(f"Ollama embed returned {resp.status_code}")
                return None
            data = resp.json()
            # Ollama /api/embed returns "embeddings" (list of lists)
            embeddings = data.get("embeddings")
            if embeddings and len(embeddings) > 0:
                return embeddings[0]
            # Fallback for older Ollama versions
            embedding = data.get("embedding")
            if embedding:
                return embedding
            logger.warning(f"Ollama returned no embedding. Keys: {list(data.keys())}")
            return None
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return None


async def _store_vector(point_id: str, vector: List[float], payload: Dict[str, Any]) -> bool:
    """Store a single vector in Qdrant echo_memory."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.put(
                f"{QDRANT_URL}/collections/{COLLECTION}/points",
                json={"points": [{"id": point_id, "vector": vector, "payload": payload}]},
            )
            return resp.status_code in (200, 201)
    except Exception as e:
        logger.error(f"Qdrant store failed: {e}")
        return False


async def _store_postgres(
    session_id: str,
    user_text: str,
    response_text: str,
    metadata: Dict[str, Any],
) -> bool:
    """Insert conversation turn into PostgreSQL."""
    try:
        import asyncpg

        conn = await asyncpg.connect(DB_URL)
        try:
            await conn.execute(
                """
                INSERT INTO voice_conversations
                    (session_id, user_text, response_text, query_type,
                     confidence, sources, stt_time_ms, chat_time_ms, tts_time_ms)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                session_id,
                user_text,
                response_text,
                metadata.get("query_type"),
                metadata.get("confidence"),
                metadata.get("sources"),
                metadata.get("stt_time_ms"),
                metadata.get("chat_time_ms"),
                metadata.get("tts_time_ms"),
            )
            return True
        finally:
            await conn.close()
    except Exception as e:
        logger.error(f"PostgreSQL store failed: {e}")
        return False


async def store_voice_turn(
    session_id: str,
    user_text: str,
    response_text: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Persist a voice conversation turn into Echo Brain memory.

    Stores two vectors (user + assistant) in Qdrant and one row in PostgreSQL.
    This is designed to be called via asyncio.create_task() so it doesn't
    block the voice response pipeline.
    """
    metadata = metadata or {}
    now = datetime.now().isoformat()

    # Common payload fields
    base_payload = {
        "category": CATEGORY,
        "session_id": session_id,
        "timestamp": now,
        "ingested_at": now,
        "query_type": metadata.get("query_type", ""),
        "confidence": metadata.get("confidence", 0.0),
        "sources": metadata.get("sources", []),
    }

    # Embed and store user message
    user_embedding = await _embed_text(user_text)
    if user_embedding:
        user_payload = {
            **base_payload,
            "text": user_text[:10000],
            "role": "user",
            "source": f"voice:{session_id}:{now}:user",
        }
        stored = await _store_vector(str(uuid4()), user_embedding, user_payload)
        if stored:
            logger.info(f"Stored user voice vector: {user_text[:60]}...")

    # Embed and store assistant response
    response_embedding = await _embed_text(response_text)
    if response_embedding:
        response_payload = {
            **base_payload,
            "text": response_text[:10000],
            "role": "assistant",
            "source": f"voice:{session_id}:{now}:assistant",
        }
        stored = await _store_vector(str(uuid4()), response_embedding, response_payload)
        if stored:
            logger.info(f"Stored assistant voice vector: {response_text[:60]}...")

    # Store in PostgreSQL for history
    await _store_postgres(session_id, user_text, response_text, metadata)

    logger.info(f"Voice turn persisted [session={session_id[:8]}]")
