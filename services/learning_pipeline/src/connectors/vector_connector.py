"""
Qdrant vector database connector for the learning pipeline.
Handles embeddings, vector storage, and semantic search.
"""

import asyncio
import logging
import numpy as np
import json
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from qdrant_client.http import models
import aiohttp
from sentence_transformers import SentenceTransformer

from ..config.settings import VectorDatabaseConfig
from ..models.learning_item import LearningItem, LearningItemType

logger = logging.getLogger(__name__)


class VectorConnector:
    """
    Async Qdrant vector database connector.

    Features:
    - Automatic embedding generation
    - Batch processing for performance
    - Collection management
    - Semantic search capabilities
    - Fallback embedding generation
    """

    def __init__(self, config: VectorDatabaseConfig):
        self.config = config
        self.client: Optional[QdrantClient] = None
        self.is_connected = False
        self._embedding_model = None
        self._semantic_service_url = None

    async def connect(self) -> None:
        """Establish connection to Qdrant server."""
        try:
            logger.info(f"Connecting to Qdrant at {self.config.host}:{self.config.port}")

            # Create Qdrant client
            self.client = QdrantClient(
                host=self.config.host,
                port=self.config.port,
                timeout=self.config.timeout
            )

            # Test connection
            collections = self.client.get_collections()
            logger.info(f"Connected to Qdrant. Found {len(collections.collections)} collections")

            # Ensure our collection exists
            await self.ensure_collection_exists()

            self.is_connected = True
            logger.info("Qdrant vector database connection established")

        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            self.is_connected = False
            raise

    async def disconnect(self) -> None:
        """Close connection to Qdrant."""
        if self.client:
            logger.info("Closing Qdrant connection")
            # Note: QdrantClient doesn't have explicit close method in sync version
            self.client = None
            self.is_connected = False

    async def ensure_collection_exists(self) -> bool:
        """Ensure the collection exists, create if it doesn't."""
        try:
            # Check if collection exists
            try:
                collection_info = self.client.get_collection(self.config.collection_name)
                logger.info(f"Collection '{self.config.collection_name}' already exists")
                return True
            except Exception:
                # Collection doesn't exist, create it
                logger.info(f"Creating collection '{self.config.collection_name}'")

                self.client.create_collection(
                    collection_name=self.config.collection_name,
                    vectors_config=VectorParams(
                        size=self.config.embedding_dimension,
                        distance=Distance.COSINE
                    ),
                    optimizers_config=models.OptimizersConfig(
                        default_segment_number=2,
                        max_segment_size=20000,
                        memmap_threshold=20000,
                        indexing_threshold=20000,
                        flush_interval_sec=5,
                        max_optimization_threads=1
                    )
                )

                # Create payload indexes for faster filtering
                self.client.create_payload_index(
                    collection_name=self.config.collection_name,
                    field_name="item_type",
                    field_schema=models.KeywordIndexParams()
                )

                self.client.create_payload_index(
                    collection_name=self.config.collection_name,
                    field_name="source_file",
                    field_schema=models.KeywordIndexParams()
                )

                logger.info(f"Created collection '{self.config.collection_name}' successfully")
                return True

        except Exception as e:
            logger.error(f"Failed to ensure collection exists: {e}")
            raise

    async def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information."""
        try:
            info = self.client.get_collection(self.config.collection_name)
            return {
                'name': self.config.collection_name,
                'dimension': info.config.params.vectors.size,
                'distance': info.config.params.vectors.distance,
                'points_count': info.points_count,
                'status': info.status
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            raise

    async def add_learning_items(self, learning_items: List[LearningItem]) -> int:
        """Add learning items to vector database with embeddings."""
        if not learning_items:
            return 0

        logger.info(f"Adding {len(learning_items)} learning items to vector database")

        try:
            # Process in batches
            total_added = 0
            batch_size = min(self.config.batch_size, len(learning_items))

            for i in range(0, len(learning_items), batch_size):
                batch = learning_items[i:i + batch_size]
                added_count = await self._process_batch(batch)
                total_added += added_count

                logger.info(f"Processed batch {i//batch_size + 1}, added {added_count} vectors")

            logger.info(f"Successfully added {total_added} vectors to Qdrant")
            return total_added

        except Exception as e:
            logger.error(f"Failed to add learning items to vector database: {e}")
            raise

    async def _process_batch(self, learning_items: List[LearningItem]) -> int:
        """Process a batch of learning items."""
        # Extract text content for embedding
        texts = [item.content for item in learning_items]

        # Generate embeddings
        embeddings = await self._generate_embeddings(texts)

        # Validate embedding dimensions
        for embedding in embeddings:
            if len(embedding) != self.config.embedding_dimension:
                raise ValueError(
                    f"Embedding dimension mismatch. Expected {self.config.embedding_dimension}, "
                    f"got {len(embedding)}"
                )

        # Create points for Qdrant
        points = []
        for idx, (item, embedding) in enumerate(zip(learning_items, embeddings)):
            point_id = hash(f"{item.content}_{item.source_file}_{idx}") % (2**31)

            payload = {
                'content': item.content,
                'item_type': item.item_type.value,
                'title': item.title,
                'source_file': getattr(item, 'source_file', ''),
                'confidence_score': getattr(item, 'confidence_score', 0.0),
                'tags': getattr(item, 'tags', []),
                'category': getattr(item, 'category', ''),
                'created_at': getattr(item, 'created_at', None),
                'metadata': getattr(item, 'metadata', {})
            }

            # Clean up None values
            payload = {k: v for k, v in payload.items() if v is not None}

            points.append(PointStruct(
                id=point_id,
                vector=embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                payload=payload
            ))

        # Upload to Qdrant
        operation_info = self.client.upsert(
            collection_name=self.config.collection_name,
            points=points,
            wait=True
        )

        return len(points)

    async def _generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for texts with fallback."""
        try:
            # Try semantic service first
            if self._semantic_service_url:
                return await self._generate_embeddings_via_service(texts)
        except Exception as e:
            logger.warning(f"Semantic service failed: {e}. Using fallback embedding.")

        # Fallback to local embedding model
        return await self._generate_embeddings_local(texts)

    async def _generate_embeddings_via_service(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings via semantic memory service."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self._semantic_service_url}/embeddings",
                json={'texts': texts},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status != 200:
                    raise Exception(f"Semantic service returned {response.status}")

                data = await response.json()
                return [np.array(emb) for emb in data['embeddings']]

    async def _generate_embeddings_local(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings using local model."""
        if self._embedding_model is None:
            logger.info("Loading local embedding model")
            # Use a model that produces 384-dimensional embeddings
            self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            self._embedding_model.encode,
            texts
        )

        return [np.array(emb) for emb in embeddings]

    async def search_similar(
        self,
        query_text: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar content."""
        try:
            # Generate embedding for query
            query_embedding = (await self._generate_embeddings([query_text]))[0]

            # Build filter if provided
            qdrant_filter = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))

                if conditions:
                    qdrant_filter = Filter(must=conditions)

            # Search in Qdrant
            search_results = self.client.search(
                collection_name=self.config.collection_name,
                query_vector=query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding,
                query_filter=qdrant_filter,
                limit=limit,
                with_payload=True
            )

            # Format results
            results = []
            for result in search_results:
                results.append({
                    'id': result.id,
                    'score': result.score,
                    'content': result.payload.get('content', ''),
                    'title': result.payload.get('title', ''),
                    'item_type': result.payload.get('item_type', ''),
                    'source_file': result.payload.get('source_file', ''),
                    'confidence_score': result.payload.get('confidence_score', 0.0),
                    'tags': result.payload.get('tags', []),
                    'category': result.payload.get('category', ''),
                    'metadata': result.payload.get('metadata', {})
                })

            return results

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise

    async def search_by_type(self, item_type: LearningItemType, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for items of specific type."""
        return await self.search_similar(
            query_text="",  # Empty query to get all
            limit=limit,
            filters={"item_type": item_type.value}
        )

    async def update_learning_item(self, learning_item: LearningItem) -> int:
        """Update an existing learning item."""
        if not hasattr(learning_item, 'unique_id') or not learning_item.unique_id:
            raise ValueError("Learning item must have unique_id for updates")

        # Delete existing points with this unique_id
        deleted_count = await self.delete_by_unique_id(learning_item.unique_id)

        # Add updated item
        added_count = await self.add_learning_items([learning_item])

        return added_count

    async def delete_by_source_file(self, source_file: str) -> int:
        """Delete all vectors from a specific source file."""
        try:
            # Search for points with this source file
            search_results = self.client.scroll(
                collection_name=self.config.collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="source_file", match=MatchValue(value=source_file))]
                ),
                limit=10000,  # Large limit to get all matching points
                with_payload=False
            )

            if not search_results[0]:  # No points found
                return 0

            # Extract point IDs
            point_ids = [point.id for point in search_results[0]]

            # Delete points
            self.client.delete(
                collection_name=self.config.collection_name,
                points_selector=models.PointIdsList(points=point_ids)
            )

            logger.info(f"Deleted {len(point_ids)} vectors from source file: {source_file}")
            return len(point_ids)

        except Exception as e:
            logger.error(f"Failed to delete vectors by source file: {e}")
            raise

    async def delete_by_unique_id(self, unique_id: str) -> int:
        """Delete vectors by unique ID."""
        try:
            # Search for points with this unique_id
            search_results = self.client.scroll(
                collection_name=self.config.collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="unique_id", match=MatchValue(value=unique_id))]
                ),
                limit=100,
                with_payload=False
            )

            if not search_results[0]:
                return 0

            # Extract point IDs
            point_ids = [point.id for point in search_results[0]]

            # Delete points
            self.client.delete(
                collection_name=self.config.collection_name,
                points_selector=models.PointIdsList(points=point_ids)
            )

            return len(point_ids)

        except Exception as e:
            logger.error(f"Failed to delete vectors by unique_id: {e}")
            raise

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            info = self.client.get_collection(self.config.collection_name)
            return {
                'vectors_count': info.points_count,
                'indexed_vectors_count': info.points_count,  # Qdrant doesn't separate this
                'memory_usage': 'N/A',  # Would need additional API calls
                'collection_size': info.points_count * self.config.embedding_dimension * 4,  # Rough estimate
                'status': info.status
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            raise

    async def health_check(self) -> bool:
        """Check Qdrant connection health."""
        if not self.is_connected or not self.client:
            return False

        try:
            # Simple health check
            collections = self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False

    async def get_health_info(self) -> Dict[str, Any]:
        """Get detailed health information."""
        try:
            collections = self.client.get_collections()
            collection_info = await self.get_collection_info()

            return {
                'status': 'healthy',
                'version': 'N/A',  # Would need specific API call
                'collections': {
                    'total': len(collections.collections),
                    'target_collection': collection_info
                },
                'connected': self.is_connected
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'connected': False
            }