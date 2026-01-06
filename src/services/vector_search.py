#!/usr/bin/env python3
"""
Unified Vector Search Service
Uses OpenAI embeddings (1536D) and Qdrant for semantic search.
"""
import os
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import asyncio
import sys
sys.path.insert(0, '/opt/tower-echo-brain')

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import SearchParams

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

@dataclass
class SearchResult:
    """A single search result."""
    id: str
    score: float
    collection: str
    payload: dict

@dataclass
class SearchResponse:
    """Search response with results and metadata."""
    query: str
    results: List[SearchResult]
    collections_searched: List[str]


class VectorSearchService:
    """
    Unified vector search across all Echo Brain collections.
    """

    COLLECTIONS = ["documents", "conversations", "facts", "code"]

    def __init__(self):
        self.client = AsyncQdrantClient(url=QDRANT_URL)
        self._embedding_service = None

    async def _get_embedder(self):
        if self._embedding_service is None:
            from src.services.embedding_service import create_embedding_service
            self._embedding_service = await create_embedding_service()
        return self._embedding_service

    async def search(
        self,
        query: str,
        collections: Optional[List[str]] = None,
        limit: int = 10,
        score_threshold: float = 0.7,
        filters: Optional[Dict] = None
    ) -> SearchResponse:
        """
        Search across collections for relevant content.

        Args:
            query: Natural language query
            collections: List of collections to search (default: all)
            limit: Max results per collection
            score_threshold: Minimum similarity score
            filters: Optional payload filters

        Returns:
            SearchResponse with ranked results
        """
        collections = collections or self.COLLECTIONS
        embedder = await self._get_embedder()

        # Generate query embedding
        query_vector = await embedder.embed_single(query)

        # Search each collection
        all_results = []
        for collection in collections:
            try:
                from qdrant_client.models import ScoredPoint
                points = await self.client.query_points(
                    collection_name=collection,
                    query=query_vector,
                    limit=limit,
                    score_threshold=score_threshold,
                    search_params=SearchParams(hnsw_ef=128, exact=False)
                )
                results = points.points if hasattr(points, 'points') else []

                for r in results:
                    all_results.append(SearchResult(
                        id=str(r.id),
                        score=r.score,
                        collection=collection,
                        payload=r.payload or {}
                    ))
            except Exception as e:
                print(f"Error searching {collection}: {e}")

        # Sort by score descending
        all_results.sort(key=lambda x: x.score, reverse=True)

        return SearchResponse(
            query=query,
            results=all_results[:limit],
            collections_searched=collections
        )

    async def add_document(
        self,
        collection: str,
        doc_id: str,
        text: str,
        metadata: Dict
    ) -> bool:
        """
        Add a document to a collection.

        Args:
            collection: Target collection name
            doc_id: Unique document ID
            text: Text content to embed
            metadata: Payload metadata

        Returns:
            True if successful
        """
        embedder = await self._get_embedder()
        embedding = await embedder.embed_single(text)

        await self.client.upsert(
            collection_name=collection,
            points=[{
                "id": doc_id,
                "vector": embedding,
                "payload": {**metadata, "text": text[:1000]}  # Store preview
            }]
        )
        return True

    async def add_documents_batch(
        self,
        collection: str,
        documents: List[Dict]
    ) -> int:
        """
        Add multiple documents efficiently.

        Args:
            collection: Target collection
            documents: List of {"id": str, "text": str, "metadata": dict}

        Returns:
            Number of documents added
        """
        if not documents:
            return 0

        embedder = await self._get_embedder()

        texts = [d["text"] for d in documents]
        embeddings = await embedder.embed_batch(texts)

        points = []
        for doc, embedding in zip(documents, embeddings):
            points.append({
                "id": doc["id"],
                "vector": embedding,
                "payload": {**doc.get("metadata", {}), "text": doc["text"][:1000]}
            })

        await self.client.upsert(
            collection_name=collection,
            points=points,
            wait=True
        )
        return len(points)

    async def close(self):
        """Clean up resources."""
        if self._embedding_service:
            await self._embedding_service.close()


# Singleton
_service: Optional[VectorSearchService] = None

async def get_vector_search() -> VectorSearchService:
    global _service
    if _service is None:
        _service = VectorSearchService()
    return _service