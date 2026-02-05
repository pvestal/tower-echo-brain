#!/usr/bin/env python3
"""
Echo Brain Unified Memory System
Single source of truth for all data ingestion, storage, and retrieval
"""

import os
import asyncio
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import httpx
import asyncpg
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MemoryConfig:
    """Configuration for unified memory system"""
    db_host: str = "localhost"
    db_name: str = "echo_brain"
    db_user: str = "patrick"
    db_password: str = os.getenv("DB_PASSWORD", "")

    qdrant_url: str = "http://localhost:6333"
    collection_name: str = "echo_memory"

    ollama_url: str = "http://localhost:11434"
    embedding_model: str = "mxbai-embed-large:latest"
    embedding_dim: int = 1024

    conversation_dir: str = "/home/patrick/.claude/projects"
    max_text_length: int = 3000  # Safe limit for embeddings

class UnifiedMemorySystem:
    """
    Central memory management for Echo Brain
    Handles all ingestion, storage, and retrieval operations
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self.db_pool = None
        self._running = False
        self.stats = {
            "conversations_processed": 0,
            "embeddings_created": 0,
            "last_ingestion": None,
            "errors": []
        }

    async def initialize(self):
        """Initialize database connection pool and verify systems"""
        try:
            # Create PostgreSQL connection pool
            self.db_pool = await asyncpg.create_pool(
                host=self.config.db_host,
                database=self.config.db_name,
                user=self.config.db_user,
                password=self.config.db_password,
                min_size=2,
                max_size=10
            )

            # Verify Qdrant is accessible
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.config.qdrant_url}/collections/{self.config.collection_name}")
                if response.status_code != 200:
                    logger.warning(f"Qdrant collection {self.config.collection_name} not accessible")

            # Verify Ollama is accessible
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.config.ollama_url}/api/tags")
                if response.status_code != 200:
                    logger.warning("Ollama not accessible")

            self._running = True
            logger.info("Unified Memory System initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize memory system: {e}")
            raise

    async def shutdown(self):
        """Clean shutdown of memory system"""
        self._running = False
        if self.db_pool:
            await self.db_pool.close()
        logger.info("Unified Memory System shut down")

    async def ingest_conversations(self):
        """
        Main ingestion process for Claude conversations
        Stores in both PostgreSQL and Qdrant
        """
        if not self._running:
            logger.error("Memory system not initialized")
            return

        conversation_dir = Path(self.config.conversation_dir)
        if not conversation_dir.exists():
            logger.error(f"Conversation directory not found: {conversation_dir}")
            return

        files = list(conversation_dir.rglob("*.jsonl"))
        logger.info(f"Found {len(files)} conversation files to process")

        for file_path in files:
            try:
                await self._process_conversation_file(file_path)
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                self.stats["errors"].append(str(e))

        self.stats["last_ingestion"] = datetime.now()
        logger.info(f"Ingestion complete: {self.stats['conversations_processed']} conversations, "
                   f"{self.stats['embeddings_created']} embeddings")

    async def _process_conversation_file(self, file_path: Path):
        """Process a single conversation file"""
        import uuid
        conversation_id = hashlib.md5(str(file_path).encode()).hexdigest()
        messages = []
        file_embeddings = 0

        # Parse JSONL file
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if data.get('type') in ['user', 'assistant']:
                        content = self._extract_content(data.get('message', {}))
                        if content and len(content) > 50:
                            messages.append({
                                'role': data['type'],
                                'content': content[:self.config.max_text_length]
                            })
                except:
                    continue

        if not messages:
            return

        # Store in PostgreSQL
        await self._store_conversation_postgres(conversation_id, file_path.name, messages)

        # Create embeddings and store in Qdrant
        for i, msg in enumerate(messages):
            if len(msg['content']) > 100:  # Only embed substantial messages
                embedding = await self._create_embedding(msg['content'])
                if embedding:
                    # Qdrant requires UUID format
                    point_uuid = str(uuid.uuid4())
                    success = await self._store_embedding_qdrant(
                        point_uuid,
                        embedding,
                        {
                            'content': msg['content'][:1000],
                            'role': msg['role'],
                            'conversation_id': conversation_id,
                            'file': file_path.name,
                            'timestamp': datetime.now().isoformat()
                        }
                    )
                    if success:
                        file_embeddings += 1
                        self.stats["embeddings_created"] += 1

        self.stats["conversations_processed"] += 1
        logger.info(f"✓ Processed {file_path.name}: {len(messages)} messages, {file_embeddings} embeddings")

    def _extract_content(self, message) -> str:
        """Extract text from various message formats"""
        if isinstance(message, str):
            return message
        if isinstance(message, dict):
            if 'content' in message:
                content = message['content']
                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    texts = []
                    for item in content:
                        if isinstance(item, dict) and 'text' in item:
                            texts.append(item['text'])
                        elif isinstance(item, str):
                            texts.append(item)
                    return ' '.join(texts)
        return ""

    async def _store_conversation_postgres(self, conv_id: str, filename: str, messages: List[Dict]):
        """Store conversation metadata in PostgreSQL"""
        async with self.db_pool.acquire() as conn:
            # Create summary
            summary = "\n".join([
                f"{msg['role']}: {msg['content'][:100]}..."
                for msg in messages[:3]
            ])[:500]

            # Upsert conversation
            await conn.execute("""
                INSERT INTO conversations (external_id, title, summary, participants, message_count, created_at)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (external_id) DO UPDATE
                SET message_count = $5, updated_at = NOW()
            """, conv_id, f"Conversation: {filename}", summary,
               ["patrick", "claude"], len(messages), datetime.now())

    async def _create_embedding(self, text: str) -> Optional[List[float]]:
        """Create embedding using Ollama"""
        # Ensure text is within limits
        if len(text) > self.config.max_text_length:
            text = text[:self.config.max_text_length]

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    f"{self.config.ollama_url}/api/embeddings",
                    json={"model": self.config.embedding_model, "prompt": text}
                )
                if response.status_code == 200:
                    embedding = response.json().get("embedding")
                    if embedding and len(embedding) == self.config.embedding_dim:
                        return embedding
        except Exception as e:
            logger.debug(f"Embedding failed: {e}")
        return None

    async def _store_embedding_qdrant(self, point_id: str, vector: List[float], payload: Dict):
        """Store embedding in Qdrant"""
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.put(
                    f"{self.config.qdrant_url}/collections/{self.config.collection_name}/points",
                    json={
                        "points": [{
                            "id": point_id,
                            "vector": vector,
                            "payload": payload
                        }]
                    }
                )
                return response.status_code == 200
        except Exception as e:
            logger.debug(f"Qdrant storage failed: {e}")
            return False

    async def search_memory(self, query: str, limit: int = 10) -> List[Dict]:
        """Search memories using semantic similarity"""
        # Create embedding for query
        embedding = await self._create_embedding(query)
        if not embedding:
            return []

        # Search in Qdrant
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    f"{self.config.qdrant_url}/collections/{self.config.collection_name}/points/search",
                    json={
                        "vector": embedding,
                        "limit": limit,
                        "with_payload": True
                    }
                )
                if response.status_code == 200:
                    results = response.json().get("result", [])
                    return [
                        {
                            "score": r["score"],
                            "content": r["payload"].get("content", ""),
                            "metadata": r["payload"]
                        }
                        for r in results
                    ]
        except Exception as e:
            logger.error(f"Search failed: {e}")
        return []

    def get_stats(self) -> Dict[str, Any]:
        """Get current system statistics"""
        return {
            **self.stats,
            "is_running": self._running,
            "config": {
                "conversation_dir": self.config.conversation_dir,
                "collection": self.config.collection_name,
                "embedding_model": self.config.embedding_model
            }
        }

# Global instance
memory_system = UnifiedMemorySystem()