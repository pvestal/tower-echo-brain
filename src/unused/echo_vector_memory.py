#!/usr/bin/env python3
"""
Echo Vector Memory Module - 4096D Enhanced with AMD GPU services
This module adds persistent memory and learning capabilities to Echo
"""

import asyncio
import logging
import httpx
import json
import hashlib
from typing import List, Dict, Optional, Any
from datetime import datetime
import sys
import os
sys.path.append('/opt/tower-echo-brain/src')
from src.config.qdrant_4096d_config import COLLECTION_MAPPING, QDRANT_CONFIG, get_4096d_collection
from src.interfaces.vector_memory import VectorMemoryInterface

logger = logging.getLogger(__name__)


class VectorMemory(VectorMemoryInterface):
    """
    Vector-based memory system for Echo using AMD GPU services
    Integrates with existing Qdrant vector database and embedding service
    """

    def __init__(self):
        # AMD GPU Services running on Tower
        self.vector_db_url = "http://127.0.0.1:6333"
        self.embeddings_url = "http://127.0.0.1:6333"

        # Use Ollama for embeddings (since custom service isn't working)
        self.ollama_url = "http://127.0.0.1:11434"

        # Use 1024D collection with mxbai-embed-large
        base_collection = "echo_memory"  # Using the new collection
        self.collection_name = base_collection
        self.vector_dimensions = 1024  # Using 1024D for better quality
        self.embedding_model = "mxbai-embed-large:latest"  # Superior embedding model

        # Initialize collection on startup
        asyncio.create_task(self._ensure_collection())

    async def _ensure_collection(self):
        """Ensure the vector collection exists"""
        async with httpx.AsyncClient() as client:
            try:
                # Check if collection exists
                response = await client.get(
                    f"{self.vector_db_url}/collections/{self.collection_name}"
                )
                if response.status_code == 404:
                    # Create collection with proper dimensions
                    await client.put(
                        f"{self.vector_db_url}/collections/{self.collection_name}",
                        json={
                            "vectors": {
                                "size": self.vector_dimensions,  # 4096D enhanced
                                "distance": QDRANT_CONFIG['distance']
                            }
                        }
                    )
                    logger.info(f"Created vector collection: {self.collection_name}")
                else:
                    logger.info(f"Vector collection exists: {self.collection_name}")
            except Exception as e:
                logger.error(f"Failed to ensure collection: {e}")

    async def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding using Ollama with mxbai-embed-large model (1024D)"""
        async with httpx.AsyncClient(timeout=30) as client:
            try:
                # Use Ollama's embedding endpoint
                response = await client.post(
                    f"{self.ollama_url}/api/embeddings",
                    json={
                        "model": self.embedding_model,
                        "prompt": text
                    }
                )
                if response.status_code == 200:
                    data = response.json()
                    return data.get("embedding")
                else:
                    logger.error(f"Embedding generation failed: {response.status_code}")
            except Exception as e:
                logger.error(f"Error generating embedding: {e}")
        return None

    async def remember(self, text: str, metadata: Optional[Dict] = None) -> bool:
        """
        Store a memory in the vector database

        Args:
            text: The text to remember
            metadata: Optional metadata about the memory

        Returns:
            Success status
        """
        # Generate embedding
        embedding = await self._generate_embedding(text)
        if not embedding:
            return False

        # Create unique ID
        memory_id = abs(hash(text + str(datetime.now()))) % (2**31)

        # Prepare payload
        payload = {
            "text": text,
            "timestamp": datetime.now().isoformat(),
            "source": "conversation",
            "type": "memory"
        }
        if metadata:
            payload.update(metadata)

        # Store in vector database
        async with httpx.AsyncClient() as client:
            try:
                response = await client.put(
                    f"{self.vector_db_url}/collections/{self.collection_name}/points",
                    json={
                        "points": [{
                            "id": memory_id,
                            "vector": embedding,
                            "payload": payload
                        }]
                    }
                )
                if response.status_code == 200:
                    logger.info(f"Stored memory: {text[:50]}...")
                    return True
            except Exception as e:
                logger.error(f"Failed to store memory: {e}")

        return False

    async def recall(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Search for relevant memories based on a query

        Args:
            query: The search query
            limit: Maximum number of results

        Returns:
            List of relevant memories with scores
        """
        # Generate query embedding
        embedding = await self._generate_embedding(query)
        if not embedding:
            return []

        # Search vector database
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.vector_db_url}/collections/{self.collection_name}/points/search",
                    json={
                        "vector": embedding,
                        "limit": limit,
                        "with_payload": True
                    }
                )
                if response.status_code == 200:
                    results = response.json().get("result", [])
                    memories = []
                    for result in results:
                        memory = result.get("payload", {})
                        memory["score"] = result.get("score", 0)
                        memories.append(memory)
                    return memories
            except Exception as e:
                logger.error(f"Failed to recall memories: {e}")

        return []

    async def learn_from_conversation(self, user_message: str, echo_response: str):
        """
        Learn from a conversation exchange

        Args:
            user_message: What the user said
            echo_response: How Echo responded
        """
        # Store the exchange as a learning example
        conversation = f"User: {user_message}\nEcho: {echo_response}"

        await self.remember(
            conversation,
            metadata={
                "type": "conversation",
                "user_message": user_message,
                "echo_response": echo_response,
                "learned_at": datetime.now().isoformat()
            }
        )

    async def get_context_for_query(self, query: str) -> str:
        """
        Get relevant context from memory for a query

        Args:
            query: The user's query

        Returns:
            Formatted context string for inclusion in prompts
        """
        memories = await self.recall(query, limit=3)

        if not memories:
            return ""

        context = "Relevant memories:\n"
        for memory in memories:
            text = memory.get("text", "")
            score = memory.get("score", 0)
            context += f"- (relevance: {score:.2f}) {text[:200]}...\n"

        return context

    async def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the memory system"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.vector_db_url}/collections/{self.collection_name}"
                )
                if response.status_code == 200:
                    data = response.json().get("result", {})
                    return {
                        "total_memories": data.get("points_count", 0),
                        "indexed_memories": data.get("indexed_vectors_count", 0),
                        "collection_status": data.get("status", "unknown")
                    }
            except Exception as e:
                logger.error(f"Failed to get statistics: {e}")

        return {"error": "Failed to retrieve statistics"}

    # Additional methods required by VectorMemoryInterface

    async def store_memory(self, content: str, metadata: Dict[str, Any]) -> str:
        """
        Store memory and return unique identifier

        Args:
            content: Memory content to store
            metadata: Metadata dictionary including type, source, etc.

        Returns:
            str: Unique identifier for the stored memory
        """
        success = await self.remember(content, metadata)
        if success:
            # Generate memory ID using content hash
            memory_id = abs(hash(content + str(datetime.now()))) % (2**31)
            return str(memory_id)
        else:
            raise Exception("Failed to store memory")

    async def search_memory(self, query: str, limit: int = 10,
                          filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Advanced memory search with optional filtering

        Args:
            query: Search query string
            limit: Maximum results to return
            filters: Optional filters for metadata-based filtering

        Returns:
            List[Dict]: Filtered and scored memory results
        """
        # Use existing recall method as base
        memories = await self.recall(query, limit)

        # Apply filters if provided
        if filters:
            filtered_memories = []
            for memory in memories:
                match = True
                for key, value in filters.items():
                    if memory.get(key) != value:
                        match = False
                        break
                if match:
                    filtered_memories.append(memory)
            return filtered_memories

        return memories

    async def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a specific memory by ID

        Args:
            memory_id: Unique identifier of memory to delete

        Returns:
            bool: True if memory was successfully deleted
        """
        async with httpx.AsyncClient() as client:
            try:
                response = await client.delete(
                    f"{self.vector_db_url}/collections/{self.collection_name}/points",
                    json={"points": [int(memory_id)]}
                )
                return response.status_code == 200
            except Exception as e:
                logger.error(f"Failed to delete memory {memory_id}: {e}")
                return False

    async def update_memory(self, memory_id: str, content: Optional[str] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update existing memory content or metadata

        Args:
            memory_id: Unique identifier of memory to update
            content: New content (optional)
            metadata: New metadata to merge (optional)

        Returns:
            bool: True if memory was successfully updated
        """
        if content is None and metadata is None:
            return False

        try:
            # Get existing memory
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.vector_db_url}/collections/{self.collection_name}/points/{memory_id}"
                )

                if response.status_code != 200:
                    return False

                existing_data = response.json().get("result", {})
                payload = existing_data.get("payload", {})

                # Update content if provided
                if content is not None:
                    payload["text"] = content
                    # Regenerate embedding if content changed
                    embedding = await self._generate_embedding(content)
                    if embedding is None:
                        return False
                else:
                    embedding = existing_data.get("vector", [])

                # Update metadata if provided
                if metadata is not None:
                    payload.update(metadata)

                payload["updated_at"] = datetime.now().isoformat()

                # Update the memory
                update_response = await client.put(
                    f"{self.vector_db_url}/collections/{self.collection_name}/points",
                    json={
                        "points": [{
                            "id": int(memory_id),
                            "vector": embedding,
                            "payload": payload
                        }]
                    }
                )

                return update_response.status_code == 200

        except Exception as e:
            logger.error(f"Failed to update memory {memory_id}: {e}")
            return False

    async def create_collection(self, collection_name: str,
                              vector_size: int = 384) -> bool:
        """
        Create a new memory collection

        Args:
            collection_name: Name of the collection to create
            vector_size: Dimension size for vectors

        Returns:
            bool: True if collection was successfully created
        """
        async with httpx.AsyncClient() as client:
            try:
                response = await client.put(
                    f"{self.vector_db_url}/collections/{collection_name}",
                    json={
                        "vectors": {
                            "size": vector_size,
                            "distance": QDRANT_CONFIG.get('distance', 'Cosine')
                        }
                    }
                )
                return response.status_code == 200
            except Exception as e:
                logger.error(f"Failed to create collection {collection_name}: {e}")
                return False

    async def list_collections(self) -> List[str]:
        """
        List all available memory collections

        Returns:
            List[str]: Names of available collections
        """
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{self.vector_db_url}/collections")
                if response.status_code == 200:
                    data = response.json()
                    collections = data.get("result", {}).get("collections", [])
                    return [col.get("name", "") for col in collections]
            except Exception as e:
                logger.error(f"Failed to list collections: {e}")

        return []

    def get_embedding_model(self) -> str:
        """
        Get the name of the current embedding model

        Returns:
            str: Name/identifier of the embedding model in use
        """
        return self.embedding_model

    def get_vector_dimensions(self) -> int:
        """
        Get the vector dimensions used by this memory system

        Returns:
            int: Number of dimensions in the vector space
        """
        return self.vector_dimensions


# Integration helper for Echo's main service
class EchoWithMemory:
    """
    Wrapper to add memory capabilities to Echo's responses
    """

    def __init__(self, memory: VectorMemory):
        self.memory = memory

    async def process_with_memory(self, user_message: str, base_response_func) -> str:
        """
        Process a user message with memory context

        Args:
            user_message: The user's input
            base_response_func: Echo's original response function

        Returns:
            Enhanced response with memory context
        """
        # Get relevant context from memory
        context = await self.memory.get_context_for_query(user_message)

        # Add context to the prompt for Echo
        enhanced_prompt = user_message
        if context:
            enhanced_prompt = f"{context}\n\nUser query: {user_message}"

        # Get Echo's response
        response = await base_response_func(enhanced_prompt)

        # Learn from this exchange
        await self.memory.learn_from_conversation(user_message, response)

        return response


# Example integration code for Echo's main.py
INTEGRATION_CODE = '''
# Add to Echo's main.py imports:
from src.echo_vector_memory import VectorMemory, EchoWithMemory

# Initialize memory system at startup:
vector_memory = VectorMemory()
echo_with_memory = EchoWithMemory(vector_memory)

# Modify query endpoint to use memory:
@app.post("/query")
async def enhanced_query(request: QueryRequest):
    # Original Echo processing...

    # Enhanced with memory
    response = await echo_with_memory.process_with_memory(
        request.query,
        lambda prompt: query_ollama(prompt, request.model)
    )

    return QueryResponse(response=response)

# Add memory status endpoint:
@app.get("/memory/stats")
async def memory_stats():
    return await vector_memory.get_statistics()
'''


if __name__ == "__main__":
    # Test the memory system
    async def test():
        memory = VectorMemory()

        # Wait for initialization
        await asyncio.sleep(2)

        # Test storing a memory
        success = await memory.remember(
            "Patrick prefers concise answers without unnecessary explanations",
            metadata={"type": "preference", "user": os.getenv("DB_USER", "echo_brain_service")}
        )
        print(f"Stored preference: {success}")

        # Test recalling
        memories = await memory.recall("How does Patrick like his answers?")
        print(f"Recalled {len(memories)} relevant memories")
        for m in memories:
            print(f"  - {m.get('text', '')[:100]}... (score: {m.get('score', 0):.2f})")

        # Get stats
        stats = await memory.get_statistics()
        print(f"Memory stats: {stats}")

    asyncio.run(test())