#!/usr/bin/env python3
"""
Abstract interface for vector database/storage systems in Echo Brain.
Provides standardized vector operations and similarity search.
Patrick Vestal - December 9, 2025
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple, AsyncIterable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np


class DistanceMetric(Enum):
    """Distance metrics for vector similarity."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"


class IndexType(Enum):
    """Vector index types."""
    FLAT = "flat"
    HNSW = "hnsw"
    IVF = "ivf"
    LSH = "lsh"


@dataclass
class VectorPoint:
    """Single vector point with metadata."""
    id: str
    vector: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None


@dataclass
class SearchResult:
    """Result from vector similarity search."""
    point_id: str
    vector: np.ndarray
    score: float
    metadata: Dict[str, Any]
    distance: float


@dataclass
class CollectionInfo:
    """Information about a vector collection."""
    name: str
    dimension: int
    vector_count: int
    index_type: IndexType
    distance_metric: DistanceMetric
    created_at: datetime
    last_updated: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IndexConfig:
    """Configuration for vector index."""
    index_type: IndexType
    distance_metric: DistanceMetric
    parameters: Dict[str, Any] = field(default_factory=dict)


class VectorStoreInterface(ABC):
    """Abstract interface for vector storage systems."""

    @abstractmethod
    async def create_collection(
        self,
        collection_name: str,
        dimension: int,
        index_config: IndexConfig,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Create a new vector collection.

        Args:
            collection_name: Name of the collection
            dimension: Vector dimension size
            index_config: Index configuration
            metadata: Optional collection metadata

        Returns:
            True if created successfully, False otherwise
        """
        pass

    @abstractmethod
    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a vector collection.

        Args:
            collection_name: Name of collection to delete

        Returns:
            True if deleted successfully, False otherwise
        """
        pass

    @abstractmethod
    async def list_collections(self) -> List[str]:
        """List all available collections.

        Returns:
            List of collection names
        """
        pass

    @abstractmethod
    async def get_collection_info(self, collection_name: str) -> Optional[CollectionInfo]:
        """Get information about a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            CollectionInfo object or None if collection doesn't exist
        """
        pass

    @abstractmethod
    async def insert_vectors(
        self,
        collection_name: str,
        vectors: List[VectorPoint],
        batch_size: int = 100
    ) -> bool:
        """Insert vectors into collection.

        Args:
            collection_name: Name of the collection
            vectors: List of VectorPoint objects to insert
            batch_size: Batch size for insertion

        Returns:
            True if inserted successfully, False otherwise
        """
        pass

    @abstractmethod
    async def upsert_vectors(
        self,
        collection_name: str,
        vectors: List[VectorPoint],
        batch_size: int = 100
    ) -> bool:
        """Insert or update vectors in collection.

        Args:
            collection_name: Name of the collection
            vectors: List of VectorPoint objects to upsert
            batch_size: Batch size for operation

        Returns:
            True if upserted successfully, False otherwise
        """
        pass

    @abstractmethod
    async def delete_vectors(
        self,
        collection_name: str,
        point_ids: List[str]
    ) -> bool:
        """Delete vectors from collection.

        Args:
            collection_name: Name of the collection
            point_ids: List of point IDs to delete

        Returns:
            True if deleted successfully, False otherwise
        """
        pass

    @abstractmethod
    async def search_similar(
        self,
        collection_name: str,
        query_vector: np.ndarray,
        top_k: int = 10,
        score_threshold: Optional[float] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar vectors.

        Args:
            collection_name: Name of the collection
            query_vector: Query vector for similarity search
            top_k: Number of top results to return
            score_threshold: Optional minimum score threshold
            metadata_filter: Optional metadata filter

        Returns:
            List of SearchResult objects sorted by similarity
        """
        pass

    @abstractmethod
    async def batch_search(
        self,
        collection_name: str,
        query_vectors: List[np.ndarray],
        top_k: int = 10,
        score_threshold: Optional[float] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[List[SearchResult]]:
        """Batch search for similar vectors.

        Args:
            collection_name: Name of the collection
            query_vectors: List of query vectors
            top_k: Number of top results per query
            score_threshold: Optional minimum score threshold
            metadata_filter: Optional metadata filter

        Returns:
            List of search results for each query
        """
        pass

    @abstractmethod
    async def get_vector(
        self,
        collection_name: str,
        point_id: str
    ) -> Optional[VectorPoint]:
        """Get vector by ID.

        Args:
            collection_name: Name of the collection
            point_id: ID of the point to retrieve

        Returns:
            VectorPoint object or None if not found
        """
        pass

    @abstractmethod
    async def get_vectors_batch(
        self,
        collection_name: str,
        point_ids: List[str]
    ) -> List[Optional[VectorPoint]]:
        """Get multiple vectors by ID.

        Args:
            collection_name: Name of the collection
            point_ids: List of point IDs to retrieve

        Returns:
            List of VectorPoint objects (None for missing points)
        """
        pass

    @abstractmethod
    async def update_metadata(
        self,
        collection_name: str,
        point_id: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """Update metadata for a vector point.

        Args:
            collection_name: Name of the collection
            point_id: ID of the point to update
            metadata: New metadata to set

        Returns:
            True if updated successfully, False otherwise
        """
        pass

    @abstractmethod
    async def count_vectors(
        self,
        collection_name: str,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> int:
        """Count vectors in collection.

        Args:
            collection_name: Name of the collection
            metadata_filter: Optional metadata filter

        Returns:
            Number of vectors matching filter
        """
        pass


class AdvancedVectorStoreInterface(VectorStoreInterface):
    """Advanced vector store interface with additional features."""

    @abstractmethod
    async def create_index(
        self,
        collection_name: str,
        index_config: IndexConfig
    ) -> bool:
        """Create or update vector index.

        Args:
            collection_name: Name of the collection
            index_config: Index configuration

        Returns:
            True if created successfully, False otherwise
        """
        pass

    @abstractmethod
    async def search_with_context(
        self,
        collection_name: str,
        query_vector: np.ndarray,
        context_vectors: List[np.ndarray],
        top_k: int = 10,
        context_weight: float = 0.3,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search with contextual information.

        Args:
            collection_name: Name of the collection
            query_vector: Primary query vector
            context_vectors: Context vectors for search
            top_k: Number of top results to return
            context_weight: Weight for context (0.0 to 1.0)
            metadata_filter: Optional metadata filter

        Returns:
            List of SearchResult objects with context-aware scoring
        """
        pass

    @abstractmethod
    async def cluster_vectors(
        self,
        collection_name: str,
        num_clusters: int,
        method: str = "kmeans",
        sample_size: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Cluster vectors in collection.

        Args:
            collection_name: Name of the collection
            num_clusters: Number of clusters
            method: Clustering method (kmeans, dbscan, etc.)
            sample_size: Optional sample size for large collections
            metadata_filter: Optional metadata filter

        Returns:
            Dictionary with cluster information and assignments
        """
        pass

    @abstractmethod
    async def get_vector_stats(
        self,
        collection_name: str,
        sample_size: int = 1000
    ) -> Dict[str, Any]:
        """Get statistical information about vectors.

        Args:
            collection_name: Name of the collection
            sample_size: Sample size for statistics

        Returns:
            Dictionary with vector statistics
        """
        pass

    @abstractmethod
    async def stream_vectors(
        self,
        collection_name: str,
        batch_size: int = 100,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> AsyncIterable[List[VectorPoint]]:
        """Stream vectors from collection.

        Args:
            collection_name: Name of the collection
            batch_size: Batch size for streaming
            metadata_filter: Optional metadata filter

        Yields:
            Batches of VectorPoint objects
        """
        pass

    @abstractmethod
    async def backup_collection(
        self,
        collection_name: str,
        backup_path: str,
        include_vectors: bool = True,
        include_metadata: bool = True
    ) -> bool:
        """Backup collection to file.

        Args:
            collection_name: Name of the collection
            backup_path: Path for backup file
            include_vectors: Whether to include vector data
            include_metadata: Whether to include metadata

        Returns:
            True if backed up successfully, False otherwise
        """
        pass

    @abstractmethod
    async def restore_collection(
        self,
        backup_path: str,
        collection_name: Optional[str] = None
    ) -> bool:
        """Restore collection from backup.

        Args:
            backup_path: Path to backup file
            collection_name: Optional new collection name

        Returns:
            True if restored successfully, False otherwise
        """
        pass


class SemanticVectorStoreInterface(AdvancedVectorStoreInterface):
    """Semantic-aware vector store interface."""

    @abstractmethod
    async def search_by_text(
        self,
        collection_name: str,
        query_text: str,
        top_k: int = 10,
        score_threshold: Optional[float] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search vectors using text query (with automatic embedding).

        Args:
            collection_name: Name of the collection
            query_text: Text query to search for
            top_k: Number of top results to return
            score_threshold: Optional minimum score threshold
            metadata_filter: Optional metadata filter

        Returns:
            List of SearchResult objects
        """
        pass

    @abstractmethod
    async def semantic_clustering(
        self,
        collection_name: str,
        cluster_descriptions: List[str],
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[str]]:
        """Cluster vectors based on semantic descriptions.

        Args:
            collection_name: Name of the collection
            cluster_descriptions: List of textual cluster descriptions
            metadata_filter: Optional metadata filter

        Returns:
            Dictionary mapping cluster descriptions to point IDs
        """
        pass

    @abstractmethod
    async def add_semantic_tags(
        self,
        collection_name: str,
        point_id: str,
        tags: List[str],
        confidence_threshold: float = 0.7
    ) -> bool:
        """Add semantic tags to vector points.

        Args:
            collection_name: Name of the collection
            point_id: ID of the point
            tags: List of semantic tags
            confidence_threshold: Minimum confidence for tagging

        Returns:
            True if tags added successfully, False otherwise
        """
        pass

    @abstractmethod
    async def find_semantic_outliers(
        self,
        collection_name: str,
        outlier_threshold: float = 0.1,
        sample_size: Optional[int] = None
    ) -> List[str]:
        """Find semantically outlying vectors.

        Args:
            collection_name: Name of the collection
            outlier_threshold: Threshold for outlier detection
            sample_size: Optional sample size for analysis

        Returns:
            List of point IDs for outlier vectors
        """
        pass