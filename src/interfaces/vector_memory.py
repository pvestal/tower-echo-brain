#!/usr/bin/env python3
"""
Vector Memory Interface Protocol
Defines the contract for vector-based memory systems in Echo Brain
"""

from typing import Protocol, runtime_checkable, List, Dict, Optional, Any
from datetime import datetime

@runtime_checkable
class VectorMemoryInterface(Protocol):
    """
    Protocol for vector-based memory systems

    Defines standardized methods for storing, retrieving, and managing
    vectorized memories with semantic search capabilities.
    """

    async def remember(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store a memory in the vector database

        Args:
            text: The text content to remember
            metadata: Optional metadata dictionary with additional context

        Returns:
            bool: True if memory was successfully stored, False otherwise
        """
        ...

    async def recall(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant memories based on a query

        Args:
            query: The search query string
            limit: Maximum number of results to return

        Returns:
            List[Dict]: List of relevant memories with scores and metadata
        """
        ...

    async def store_memory(self, content: str, metadata: Dict[str, Any]) -> str:
        """
        Store memory and return unique identifier

        Args:
            content: Memory content to store
            metadata: Metadata dictionary including type, source, etc.

        Returns:
            str: Unique identifier for the stored memory
        """
        ...

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
        ...

    async def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a specific memory by ID

        Args:
            memory_id: Unique identifier of memory to delete

        Returns:
            bool: True if memory was successfully deleted
        """
        ...

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
        ...

    async def learn_from_conversation(self, user_message: str, echo_response: str) -> bool:
        """
        Learn from a conversation exchange

        Args:
            user_message: What the user said
            echo_response: How Echo responded

        Returns:
            bool: True if conversation was successfully stored for learning
        """
        ...

    async def get_context_for_query(self, query: str, max_context_length: int = 2000) -> str:
        """
        Get relevant context from memory for a query

        Args:
            query: The user's query
            max_context_length: Maximum length of context to return

        Returns:
            str: Formatted context string for inclusion in prompts
        """
        ...

    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the memory system

        Returns:
            Dict: Statistics including memory count, collection status, etc.
        """
        ...

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
        ...

    async def list_collections(self) -> List[str]:
        """
        List all available memory collections

        Returns:
            List[str]: Names of available collections
        """
        ...

    def get_embedding_model(self) -> str:
        """
        Get the name of the current embedding model

        Returns:
            str: Name/identifier of the embedding model in use
        """
        ...

    def get_vector_dimensions(self) -> int:
        """
        Get the vector dimensions used by this memory system

        Returns:
            int: Number of dimensions in the vector space
        """
        ...


@runtime_checkable
class MemoryBackendInterface(Protocol):
    """
    Protocol for memory backend implementations (Qdrant, FAISS, etc.)
    """

    async def initialize(self) -> bool:
        """Initialize the memory backend"""
        ...

    async def store_vector(self, vector: List[float], payload: Dict[str, Any],
                         point_id: Optional[str] = None) -> str:
        """Store a vector with payload"""
        ...

    async def search_vectors(self, query_vector: List[float], limit: int = 10,
                           filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        ...

    async def delete_vector(self, point_id: str) -> bool:
        """Delete a vector by ID"""
        ...

    async def get_collection_info(self) -> Dict[str, Any]:
        """Get collection metadata and statistics"""
        ...


@runtime_checkable
class EmbeddingGeneratorInterface(Protocol):
    """
    Protocol for text embedding generation
    """

    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate vector embedding for text

        Args:
            text: Text to embed

        Returns:
            Optional[List[float]]: Vector embedding or None if failed
        """
        ...

    async def generate_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts

        Args:
            texts: List of texts to embed

        Returns:
            List[Optional[List[float]]]: List of embeddings, None for failed items
        """
        ...

    def get_model_info(self) -> Dict[str, Any]:
        """Get embedding model information"""
        ...

    def get_embedding_dimensions(self) -> int:
        """Get embedding vector dimensions"""
        ...