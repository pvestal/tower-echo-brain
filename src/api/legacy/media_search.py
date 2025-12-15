#!/usr/bin/env python3
"""
Media Search API endpoints for Echo Brain
Provides direct access to photo, video, and document search
"""

import logging
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Query
import httpx
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/search", tags=["media"])

# Configuration
QDRANT_URL = "http://127.0.0.1:6333"
OLLAMA_URL = "http://127.0.0.1:11434"


class MediaSearchEngine:
    """Engine for searching personal media in vector database"""

    def __init__(self):
        self.qdrant_url = QDRANT_URL
        self.ollama_url = OLLAMA_URL

    async def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for search query"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.ollama_url}/api/embeddings",
                    json={"model": "nomic-embed-text", "prompt": text},
                    timeout=30.0
                )
                if response.status_code == 200:
                    return response.json()["embedding"]
        except Exception as e:
            logger.error(f"Embedding error: {e}")
        return []

    async def search_collection(self, collection: str, embedding: List[float],
                                 vector_size: int, filters: Optional[Dict] = None,
                                 limit: int = 10) -> List[Dict]:
        """Search a specific Qdrant collection"""
        try:
            async with httpx.AsyncClient() as client:
                # Resize embedding to match collection
                search_vector = embedding[:vector_size] if len(embedding) > vector_size else embedding

                search_body = {
                    "vector": search_vector,
                    "limit": limit,
                    "with_payload": True
                }

                if filters:
                    search_body["filter"] = filters

                response = await client.post(
                    f"{self.qdrant_url}/collections/{collection}/points/search",
                    json=search_body
                )

                if response.status_code == 200:
                    return response.json()["result"]
        except Exception as e:
            logger.error(f"Search error in {collection}: {e}")
        return []


# Initialize search engine
search_engine = MediaSearchEngine()


@router.get("/photos")
async def search_photos(
    query: str = Query(..., description="Search query for photos"),
    year: Optional[int] = Query(None, description="Filter by year"),
    limit: int = Query(20, description="Maximum results to return")
) -> Dict[str, Any]:
    """
    Search for photos in Google Takeout and personal collections
    """
    try:
        # Get embedding for query
        embedding = await search_engine.get_embedding(query)
        if not embedding:
            raise HTTPException(status_code=500, detail="Could not generate embedding")

        results = {
            "query": query,
            "photos": [],
            "collections_searched": [],
            "timestamp": datetime.now().isoformat()
        }

        # Search Google media collections
        collections_to_search = [
            ("google_media_memory", 512),
            ("google_media_memory_4096d", 4096),
            ("unified_media_memory", 512),
            ("gpu_accelerated_media", 512)
        ]

        for collection_name, vector_size in collections_to_search:
            # Build filters
            filters = None
            if year:
                filters = {
                    "must": [
                        {"key": "file_path", "match": {"text": f"/{year}/"}}
                    ]
                }

            # Add photo extension filter
            photo_filter = {
                "should": [
                    {"key": "file_type", "match": {"value": ".jpg"}},
                    {"key": "file_type", "match": {"value": ".jpeg"}},
                    {"key": "file_type", "match": {"value": ".png"}},
                    {"key": "file_type", "match": {"value": ".gif"}}
                ]
            }

            if filters:
                filters["must"].append(photo_filter)
            else:
                filters = photo_filter

            # Search collection
            search_results = await search_engine.search_collection(
                collection_name, embedding, vector_size, filters, limit
            )

            if search_results:
                results["collections_searched"].append(collection_name)
                for item in search_results:
                    payload = item.get("payload", {})
                    results["photos"].append({
                        "path": payload.get("file_path", ""),
                        "score": item.get("score", 0),
                        "size": payload.get("file_size", 0),
                        "type": payload.get("file_type", ""),
                        "collection": collection_name
                    })

        # Sort by relevance score
        results["photos"].sort(key=lambda x: x["score"], reverse=True)
        results["total_found"] = len(results["photos"])

        # Limit results
        results["photos"] = results["photos"][:limit]

        return results

    except Exception as e:
        logger.error(f"Photo search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/videos")
async def search_videos(
    query: str = Query(..., description="Search query for videos"),
    source: Optional[str] = Query(None, description="Filter by source (comfyui, google, etc)"),
    limit: int = Query(10, description="Maximum results to return")
) -> Dict[str, Any]:
    """
    Search for videos including AI-generated content from ComfyUI
    """
    try:
        # Get embedding for query
        embedding = await search_engine.get_embedding(query)
        if not embedding:
            raise HTTPException(status_code=500, detail="Could not generate embedding")

        results = {
            "query": query,
            "videos": [],
            "collections_searched": [],
            "timestamp": datetime.now().isoformat()
        }

        # Video extensions
        video_extensions = [".mp4", ".avi", ".mkv", ".webm", ".mov", ".m4v"]

        # Collections to search
        collections = [
            ("anime_generations", 512),  # ComfyUI outputs
            ("unified_media_memory", 512),
            ("google_media_memory", 512)
        ]

        for collection_name, vector_size in collections:
            # Skip if source filter doesn't match
            if source:
                if source == "comfyui" and collection_name != "anime_generations":
                    continue
                elif source == "google" and "google" not in collection_name:
                    continue

            # Build video filter
            filters = {
                "should": [
                    {"key": "file_type", "match": {"value": ext}}
                    for ext in video_extensions
                ]
            }

            # Add path filter for ComfyUI
            if collection_name == "anime_generations" or source == "comfyui":
                filters["should"].append(
                    {"key": "file_path", "match": {"text": "/ComfyUI/output/"}}
                )

            # Search
            search_results = await search_engine.search_collection(
                collection_name, embedding, vector_size, filters, limit
            )

            if search_results:
                results["collections_searched"].append(collection_name)
                for item in search_results:
                    payload = item.get("payload", {})
                    video_info = {
                        "path": payload.get("file_path", ""),
                        "score": item.get("score", 0),
                        "type": payload.get("file_type", ""),
                        "collection": collection_name,
                        "source": "comfyui" if "ComfyUI" in payload.get("file_path", "") else "personal"
                    }

                    # Add metadata if available
                    if "metadata" in payload:
                        video_info["metadata"] = payload["metadata"]

                    results["videos"].append(video_info)

        # Sort by relevance
        results["videos"].sort(key=lambda x: x["score"], reverse=True)
        results["total_found"] = len(results["videos"])
        results["videos"] = results["videos"][:limit]

        return results

    except Exception as e:
        logger.error(f"Video search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents")
async def search_documents(
    query: str = Query(..., description="Search query for documents"),
    file_type: Optional[str] = Query(None, description="Filter by type (pdf, txt, md, etc)"),
    limit: int = Query(20, description="Maximum results to return")
) -> Dict[str, Any]:
    """
    Search for documents and text files
    """
    try:
        embedding = await search_engine.get_embedding(query)
        if not embedding:
            raise HTTPException(status_code=500, detail="Could not generate embedding")

        results = {
            "query": query,
            "documents": [],
            "timestamp": datetime.now().isoformat()
        }

        # Document extensions
        doc_extensions = [".pdf", ".txt", ".md", ".doc", ".docx", ".json", ".yaml", ".yml"]

        # Filter by specific type if requested
        if file_type:
            doc_extensions = [f".{file_type.lower()}"]

        # Search collections
        for collection_name, vector_size in [
            ("unified_media_memory", 512),
            ("claude_conversations", 768)
        ]:
            filters = {
                "should": [
                    {"key": "file_type", "match": {"value": ext}}
                    for ext in doc_extensions
                ]
            }

            search_results = await search_engine.search_collection(
                collection_name, embedding, vector_size, filters, limit
            )

            for item in search_results:
                payload = item.get("payload", {})
                results["documents"].append({
                    "path": payload.get("file_path", ""),
                    "content_preview": payload.get("content", "")[:200],
                    "score": item.get("score", 0),
                    "type": payload.get("file_type", ""),
                    "collection": collection_name
                })

        results["documents"].sort(key=lambda x: x["score"], reverse=True)
        results["total_found"] = len(results["documents"])
        results["documents"] = results["documents"][:limit]

        return results

    except Exception as e:
        logger.error(f"Document search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/media/stats")
async def get_media_stats() -> Dict[str, Any]:
    """
    Get statistics about indexed media
    """
    try:
        stats = {
            "collections": {},
            "total_items": 0,
            "timestamp": datetime.now().isoformat()
        }

        async with httpx.AsyncClient() as client:
            # Get all collections
            response = await client.get(f"{QDRANT_URL}/collections")
            if response.status_code == 200:
                collections = response.json()["result"]["collections"]

                # Filter media collections
                media_keywords = ["media", "google", "photo", "video", "anime", "gpu"]

                for collection in collections:
                    name = collection["name"]
                    if any(keyword in name.lower() for keyword in media_keywords):
                        # Get collection info
                        info_response = await client.get(f"{QDRANT_URL}/collections/{name}")
                        if info_response.status_code == 200:
                            info = info_response.json()["result"]
                            count = info.get("points_count", 0)
                            stats["collections"][name] = {
                                "count": count,
                                "vector_size": info["config"]["params"]["vectors"]["size"],
                                "status": info["status"]
                            }
                            stats["total_items"] += count

        return stats

    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))