#!/usr/bin/env python3
"""
Mock vector store implementation for testing Echo Brain without Qdrant dependencies.
Provides in-memory vector storage with realistic behavior.
Patrick Vestal - December 9, 2025
"""

import asyncio
import json
import random
import time
from typing import List, Dict, Any, Optional, AsyncIterable
from datetime import datetime
import numpy as np

from ..interfaces.vector_store_interface import (
    VectorStoreInterface, VectorPoint, SearchResult, CollectionInfo,
    IndexConfig, DistanceMetric, IndexType,
    AdvancedVectorStoreInterface, SemanticVectorStoreInterface
)


class MockVectorStore(VectorStoreInterface):
    """Mock vector store implementation using in-memory storage."""

    def __init__(self):
        """Initialize mock vector store."""
        self._collections: Dict[str, Dict[str, Any]] = {}
        self._collection_configs: Dict[str, IndexConfig] = {}

    async def create_collection(self, collection_name: str, dimension: int, index_config: IndexConfig, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Create mock collection."""
        await asyncio.sleep(0.1)  # Simulate creation time

        if collection_name in self._collections:
            return False

        self._collections[collection_name] = {
            "points": {},  # point_id -> VectorPoint
            "dimension": dimension,
            "created_at": datetime.now(),
            "last_updated": datetime.now(),
            "metadata": metadata or {}
        }
        self._collection_configs[collection_name] = index_config

        return True

    async def delete_collection(self, collection_name: str) -> bool:
        """Delete mock collection."""
        await asyncio.sleep(0.05)

        if collection_name not in self._collections:
            return False

        del self._collections[collection_name]
        if collection_name in self._collection_configs:
            del self._collection_configs[collection_name]

        return True

    async def list_collections(self) -> List[str]:
        """List mock collections."""
        await asyncio.sleep(0.01)
        return list(self._collections.keys())

    async def get_collection_info(self, collection_name: str) -> Optional[CollectionInfo]:
        """Get mock collection info."""
        await asyncio.sleep(0.01)

        if collection_name not in self._collections:
            return None

        collection = self._collections[collection_name]
        config = self._collection_configs.get(collection_name)

        return CollectionInfo(
            name=collection_name,
            dimension=collection["dimension"],
            vector_count=len(collection["points"]),
            index_type=config.index_type if config else IndexType.FLAT,
            distance_metric=config.distance_metric if config else DistanceMetric.COSINE,
            created_at=collection["created_at"],
            last_updated=collection["last_updated"],
            metadata=collection["metadata"]
        )

    async def insert_vectors(self, collection_name: str, vectors: List[VectorPoint], batch_size: int = 100) -> bool:
        """Insert mock vectors."""
        if collection_name not in self._collections:
            return False

        await asyncio.sleep(len(vectors) * 0.001)  # Simulate insertion time

        collection = self._collections[collection_name]

        for vector in vectors:
            # Validate dimension
            if len(vector.vector) != collection["dimension"]:
                continue

            # Add timestamp if not present
            if vector.timestamp is None:
                vector.timestamp = datetime.now()

            collection["points"][vector.id] = vector

        collection["last_updated"] = datetime.now()
        return True

    async def upsert_vectors(self, collection_name: str, vectors: List[VectorPoint], batch_size: int = 100) -> bool:
        """Upsert mock vectors (same as insert for mock)."""
        return await self.insert_vectors(collection_name, vectors, batch_size)

    async def delete_vectors(self, collection_name: str, point_ids: List[str]) -> bool:
        """Delete mock vectors."""
        if collection_name not in self._collections:
            return False

        await asyncio.sleep(len(point_ids) * 0.001)

        collection = self._collections[collection_name]
        deleted_count = 0

        for point_id in point_ids:
            if point_id in collection["points"]:
                del collection["points"][point_id]
                deleted_count += 1

        if deleted_count > 0:
            collection["last_updated"] = datetime.now()

        return deleted_count > 0

    async def search_similar(self, collection_name: str, query_vector: np.ndarray, top_k: int = 10, score_threshold: Optional[float] = None, metadata_filter: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search mock similar vectors."""
        if collection_name not in self._collections:
            return []

        await asyncio.sleep(random.uniform(0.01, 0.05))  # Simulate search time

        collection = self._collections[collection_name]
        config = self._collection_configs.get(collection_name)

        distance_metric = config.distance_metric if config else DistanceMetric.COSINE

        results = []
        for point_id, point in collection["points"].items():
            # Apply metadata filter
            if metadata_filter:
                if not self._matches_filter(point.metadata, metadata_filter):
                    continue

            # Compute similarity
            similarity, distance = self._compute_similarity(
                query_vector,
                point.vector,
                distance_metric
            )

            # Apply threshold
            if score_threshold is not None and similarity < score_threshold:
                continue

            results.append(SearchResult(
                point_id=point_id,
                vector=point.vector,
                score=similarity,
                metadata=point.metadata,
                distance=distance
            ))

        # Sort by similarity (highest first)
        results.sort(key=lambda x: x.score, reverse=True)

        return results[:top_k]

    async def batch_search(self, collection_name: str, query_vectors: List[np.ndarray], top_k: int = 10, score_threshold: Optional[float] = None, metadata_filter: Optional[Dict[str, Any]] = None) -> List[List[SearchResult]]:
        """Batch search mock vectors."""
        results = []
        for query_vector in query_vectors:
            search_results = await self.search_similar(
                collection_name, query_vector, top_k, score_threshold, metadata_filter
            )
            results.append(search_results)
        return results

    async def get_vector(self, collection_name: str, point_id: str) -> Optional[VectorPoint]:
        """Get mock vector by ID."""
        if collection_name not in self._collections:
            return None

        await asyncio.sleep(0.001)

        collection = self._collections[collection_name]
        return collection["points"].get(point_id)

    async def get_vectors_batch(self, collection_name: str, point_ids: List[str]) -> List[Optional[VectorPoint]]:
        """Get mock vectors batch."""
        results = []
        for point_id in point_ids:
            vector = await self.get_vector(collection_name, point_id)
            results.append(vector)
        return results

    async def update_metadata(self, collection_name: str, point_id: str, metadata: Dict[str, Any]) -> bool:
        """Update mock metadata."""
        if collection_name not in self._collections:
            return False

        await asyncio.sleep(0.001)

        collection = self._collections[collection_name]
        if point_id not in collection["points"]:
            return False

        collection["points"][point_id].metadata.update(metadata)
        collection["last_updated"] = datetime.now()
        return True

    async def count_vectors(self, collection_name: str, metadata_filter: Optional[Dict[str, Any]] = None) -> int:
        """Count mock vectors."""
        if collection_name not in self._collections:
            return 0

        await asyncio.sleep(0.01)

        collection = self._collections[collection_name]

        if metadata_filter is None:
            return len(collection["points"])

        # Count with filter
        count = 0
        for point in collection["points"].values():
            if self._matches_filter(point.metadata, metadata_filter):
                count += 1

        return count

    def _compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray, metric: DistanceMetric) -> tuple:
        """Compute similarity and distance between vectors."""
        if metric == DistanceMetric.COSINE:
            # Cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm_a = np.linalg.norm(vec1)
            norm_b = np.linalg.norm(vec2)

            if norm_a == 0 or norm_b == 0:
                similarity = 0.0
            else:
                similarity = dot_product / (norm_a * norm_b)

            distance = 1.0 - similarity
            return similarity, distance

        elif metric == DistanceMetric.EUCLIDEAN:
            # Euclidean distance
            distance = np.linalg.norm(vec1 - vec2)
            # Convert to similarity (higher is better)
            similarity = 1.0 / (1.0 + distance)
            return similarity, distance

        elif metric == DistanceMetric.DOT_PRODUCT:
            # Dot product similarity
            similarity = np.dot(vec1, vec2)
            distance = -similarity  # Negative because higher dot product = smaller distance
            return similarity, distance

        elif metric == DistanceMetric.MANHATTAN:
            # Manhattan distance
            distance = np.sum(np.abs(vec1 - vec2))
            similarity = 1.0 / (1.0 + distance)
            return similarity, distance

        else:
            # Default to cosine
            return self._compute_similarity(vec1, vec2, DistanceMetric.COSINE)

    def _matches_filter(self, metadata: Dict[str, Any], filter_criteria: Dict[str, Any]) -> bool:
        """Check if metadata matches filter criteria."""
        for key, value in filter_criteria.items():
            if key not in metadata:
                return False

            if isinstance(value, dict):
                # Handle nested filters like {"$gte": 10}
                if "$gte" in value:
                    if metadata[key] < value["$gte"]:
                        return False
                elif "$lte" in value:
                    if metadata[key] > value["$lte"]:
                        return False
                elif "$eq" in value:
                    if metadata[key] != value["$eq"]:
                        return False
                elif "$in" in value:
                    if metadata[key] not in value["$in"]:
                        return False
            else:
                # Direct value comparison
                if metadata[key] != value:
                    return False

        return True


class MockAdvancedVectorStore(MockVectorStore, AdvancedVectorStoreInterface):
    """Mock advanced vector store with additional features."""

    async def create_index(self, collection_name: str, index_config: IndexConfig) -> bool:
        """Create/update mock index."""
        if collection_name not in self._collections:
            return False

        await asyncio.sleep(0.5)  # Simulate index creation time
        self._collection_configs[collection_name] = index_config
        return True

    async def search_with_context(self, collection_name: str, query_vector: np.ndarray, context_vectors: List[np.ndarray], top_k: int = 10, context_weight: float = 0.3, metadata_filter: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Mock context-aware search."""
        await asyncio.sleep(0.1)

        # Combine query vector with context
        if context_vectors:
            avg_context = np.mean(context_vectors, axis=0)
            combined_query = (1 - context_weight) * query_vector + context_weight * avg_context
        else:
            combined_query = query_vector

        # Use regular search with combined query
        results = await self.search_similar(
            collection_name, combined_query, top_k, None, metadata_filter
        )

        # Add context information to metadata
        for result in results:
            result.metadata["context_aware"] = True
            result.metadata["context_weight"] = context_weight

        return results

    async def cluster_vectors(self, collection_name: str, num_clusters: int, method: str = "kmeans", sample_size: Optional[int] = None, metadata_filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Mock vector clustering."""
        if collection_name not in self._collections:
            return {"error": "Collection not found"}

        await asyncio.sleep(0.5)

        collection = self._collections[collection_name]
        points = list(collection["points"].values())

        # Apply metadata filter
        if metadata_filter:
            points = [p for p in points if self._matches_filter(p.metadata, metadata_filter)]

        # Apply sampling
        if sample_size and len(points) > sample_size:
            points = random.sample(points, sample_size)

        if len(points) < num_clusters:
            num_clusters = len(points)

        # Generate mock cluster assignments
        cluster_assignments = {}
        clusters = {}

        for i, point in enumerate(points):
            cluster_id = random.randint(0, num_clusters - 1)
            cluster_assignments[point.id] = cluster_id

            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(point.id)

        # Generate mock centroids
        centroids = []
        for cluster_id in range(num_clusters):
            centroid = np.random.normal(0, 1, collection["dimension"])
            centroid = centroid / np.linalg.norm(centroid)  # Normalize
            centroids.append(centroid.tolist())

        return {
            "num_clusters": num_clusters,
            "cluster_assignments": cluster_assignments,
            "clusters": clusters,
            "centroids": centroids,
            "method": method,
            "silhouette_score": random.uniform(0.3, 0.8),
            "points_clustered": len(points),
            "metadata": {
                "collection": collection_name,
                "sample_size": sample_size
            }
        }

    async def get_vector_stats(self, collection_name: str, sample_size: int = 1000) -> Dict[str, Any]:
        """Get mock vector statistics."""
        if collection_name not in self._collections:
            return {"error": "Collection not found"}

        await asyncio.sleep(0.2)

        collection = self._collections[collection_name]
        points = list(collection["points"].values())

        if len(points) == 0:
            return {"error": "No vectors in collection"}

        # Sample points if needed
        if len(points) > sample_size:
            points = random.sample(points, sample_size)

        # Mock statistics
        vectors = np.array([p.vector for p in points])

        return {
            "total_vectors": len(collection["points"]),
            "sampled_vectors": len(points),
            "dimension": collection["dimension"],
            "mean_magnitude": float(np.mean([np.linalg.norm(v) for v in vectors])),
            "std_magnitude": float(np.std([np.linalg.norm(v) for v in vectors])),
            "mean_vector": np.mean(vectors, axis=0).tolist(),
            "density": random.uniform(0.1, 0.9),
            "sparsity": random.uniform(0.1, 0.5),
            "distribution": {
                "min": float(np.min(vectors)),
                "max": float(np.max(vectors)),
                "mean": float(np.mean(vectors)),
                "std": float(np.std(vectors))
            }
        }

    async def stream_vectors(self, collection_name: str, batch_size: int = 100, metadata_filter: Optional[Dict[str, Any]] = None) -> AsyncIterable[List[VectorPoint]]:
        """Stream mock vectors."""
        if collection_name not in self._collections:
            return

        collection = self._collections[collection_name]
        points = list(collection["points"].values())

        # Apply filter
        if metadata_filter:
            points = [p for p in points if self._matches_filter(p.metadata, metadata_filter)]

        # Stream in batches
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            await asyncio.sleep(0.01)  # Simulate streaming delay
            yield batch

    async def backup_collection(self, collection_name: str, backup_path: str, include_vectors: bool = True, include_metadata: bool = True) -> bool:
        """Mock collection backup."""
        if collection_name not in self._collections:
            return False

        await asyncio.sleep(0.5)  # Simulate backup time

        collection = self._collections[collection_name]
        backup_data = {
            "collection_name": collection_name,
            "dimension": collection["dimension"],
            "created_at": collection["created_at"].isoformat(),
            "backup_timestamp": datetime.now().isoformat(),
            "points": []
        }

        for point_id, point in collection["points"].items():
            point_data = {
                "id": point_id,
                "timestamp": point.timestamp.isoformat() if point.timestamp else None
            }

            if include_vectors:
                point_data["vector"] = point.vector.tolist()

            if include_metadata:
                point_data["metadata"] = point.metadata

            backup_data["points"].append(point_data)

        # Mock file writing
        try:
            with open(backup_path, 'w') as f:
                json.dump(backup_data, f, indent=2, default=str)
            return True
        except Exception:
            return False

    async def restore_collection(self, backup_path: str, collection_name: Optional[str] = None) -> bool:
        """Mock collection restore."""
        await asyncio.sleep(0.5)  # Simulate restore time

        try:
            with open(backup_path, 'r') as f:
                backup_data = json.load(f)

            restored_name = collection_name or backup_data["collection_name"]

            # Create collection
            config = IndexConfig(
                index_type=IndexType.FLAT,
                distance_metric=DistanceMetric.COSINE
            )

            await self.create_collection(
                restored_name,
                backup_data["dimension"],
                config
            )

            # Restore points
            points = []
            for point_data in backup_data["points"]:
                if "vector" in point_data:
                    vector = np.array(point_data["vector"])
                    timestamp = datetime.fromisoformat(point_data["timestamp"]) if point_data.get("timestamp") else None

                    point = VectorPoint(
                        id=point_data["id"],
                        vector=vector,
                        metadata=point_data.get("metadata", {}),
                        timestamp=timestamp
                    )
                    points.append(point)

            await self.insert_vectors(restored_name, points)
            return True

        except Exception:
            return False


class MockSemanticVectorStore(MockAdvancedVectorStore, SemanticVectorStoreInterface):
    """Mock semantic-aware vector store."""

    def __init__(self):
        """Initialize mock semantic vector store."""
        super().__init__()
        self._embedder = None  # Would normally be an embedding model

    async def search_by_text(self, collection_name: str, query_text: str, top_k: int = 10, score_threshold: Optional[float] = None, metadata_filter: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Mock text search (with simulated embedding)."""
        await asyncio.sleep(0.05)

        # Generate mock embedding for text
        query_embedding = self._generate_text_embedding(query_text)

        # Use regular vector search
        return await self.search_similar(
            collection_name, query_embedding, top_k, score_threshold, metadata_filter
        )

    async def semantic_clustering(self, collection_name: str, cluster_descriptions: List[str], metadata_filter: Optional[Dict[str, Any]] = None) -> Dict[str, List[str]]:
        """Mock semantic clustering."""
        await asyncio.sleep(0.3)

        if collection_name not in self._collections:
            return {}

        collection = self._collections[collection_name]
        points = list(collection["points"].values())

        # Apply filter
        if metadata_filter:
            points = [p for p in points if self._matches_filter(p.metadata, metadata_filter)]

        # Generate mock cluster assignments based on descriptions
        result = {}
        for description in cluster_descriptions:
            # Randomly assign some points to each cluster
            num_points = random.randint(0, len(points) // len(cluster_descriptions) + 2)
            assigned_points = random.sample(points, min(num_points, len(points)))
            result[description] = [p.id for p in assigned_points]

        return result

    async def add_semantic_tags(self, collection_name: str, point_id: str, tags: List[str], confidence_threshold: float = 0.7) -> bool:
        """Mock semantic tagging."""
        await asyncio.sleep(0.05)

        point = await self.get_vector(collection_name, point_id)
        if not point:
            return False

        # Generate mock confidence scores for tags
        tag_confidences = {}
        for tag in tags:
            confidence = random.uniform(0.4, 1.0)
            if confidence >= confidence_threshold:
                tag_confidences[tag] = confidence

        # Update metadata with tags
        if tag_confidences:
            point.metadata["semantic_tags"] = tag_confidences
            return await self.update_metadata(collection_name, point_id, point.metadata)

        return False

    async def find_semantic_outliers(self, collection_name: str, outlier_threshold: float = 0.1, sample_size: Optional[int] = None) -> List[str]:
        """Mock outlier detection."""
        await asyncio.sleep(0.2)

        if collection_name not in self._collections:
            return []

        collection = self._collections[collection_name]
        points = list(collection["points"].values())

        if sample_size and len(points) > sample_size:
            points = random.sample(points, sample_size)

        # Mock outlier detection - randomly select some points as outliers
        num_outliers = max(1, int(len(points) * outlier_threshold))
        outliers = random.sample(points, min(num_outliers, len(points)))

        return [p.id for p in outliers]

    def _generate_text_embedding(self, text: str) -> np.ndarray:
        """Generate mock embedding for text."""
        # Simple mock embedding based on text hash
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()
        seed = int(text_hash[:8], 16) % (2**32)
        np.random.seed(seed)

        # Generate normalized random vector
        embedding = np.random.normal(0, 1, 384)  # Common embedding dimension
        embedding = embedding / np.linalg.norm(embedding)

        return embedding