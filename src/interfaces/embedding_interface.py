#!/usr/bin/env python3
"""
Abstract interface for text embedding models in Echo Brain.
Provides standardized embedding generation and similarity computation.
Patrick Vestal - December 9, 2025
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np

from .ml_model_interface import MLModelInterface, ModelType, ModelPrediction


@dataclass
class EmbeddingResult:
    """Result from embedding generation."""
    embeddings: np.ndarray
    input_text: str
    model_name: str
    dimensions: int
    processing_time: float
    metadata: Dict[str, Any]


@dataclass
class SimilarityResult:
    """Result from similarity computation."""
    similarity_score: float
    text_a: str
    text_b: str
    similarity_type: str  # cosine, euclidean, etc.
    metadata: Dict[str, Any]


class EmbeddingInterface(MLModelInterface):
    """Abstract interface for text embedding models."""

    def __init__(self, model_name: str):
        """Initialize embedding interface."""
        super().__init__(model_name, ModelType.EMBEDDING)

    @abstractmethod
    async def encode(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
        context: Optional[Dict[str, Any]] = None
    ) -> EmbeddingResult:
        """Generate embeddings for text(s).

        Args:
            texts: Single text string or list of texts
            normalize: Whether to normalize embeddings
            context: Optional context information

        Returns:
            EmbeddingResult with generated embeddings
        """
        pass

    @abstractmethod
    async def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True,
        context: Optional[Dict[str, Any]] = None
    ) -> List[EmbeddingResult]:
        """Generate embeddings for batch of texts.

        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            normalize: Whether to normalize embeddings
            context: Optional context information

        Returns:
            List of EmbeddingResult objects
        """
        pass

    @abstractmethod
    async def compute_similarity(
        self,
        text_a: str,
        text_b: str,
        similarity_type: str = "cosine",
        context: Optional[Dict[str, Any]] = None
    ) -> SimilarityResult:
        """Compute similarity between two texts.

        Args:
            text_a: First text string
            text_b: Second text string
            similarity_type: Type of similarity (cosine, euclidean, dot_product)
            context: Optional context information

        Returns:
            SimilarityResult with similarity score
        """
        pass

    @abstractmethod
    async def find_similar(
        self,
        query_text: str,
        candidate_texts: List[str],
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float]]:
        """Find most similar texts to query.

        Args:
            query_text: Query text string
            candidate_texts: List of candidate texts
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity threshold
            context: Optional context information

        Returns:
            List of (text, similarity_score) tuples sorted by similarity
        """
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension size.

        Returns:
            Dimension size of embeddings
        """
        pass

    @abstractmethod
    def get_max_sequence_length(self) -> int:
        """Get maximum sequence length supported.

        Returns:
            Maximum sequence length in tokens
        """
        pass

    async def validate_text_input(self, text: str) -> bool:
        """Validate text input for embedding.

        Args:
            text: Text to validate

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(text, str):
            return False

        if len(text.strip()) == 0:
            return False

        # Check if text exceeds maximum length
        if len(text) > self.get_max_sequence_length() * 4:  # Rough token estimation
            return False

        return True


class SemanticEmbeddingInterface(EmbeddingInterface):
    """Interface for semantic embedding models with advanced features."""

    @abstractmethod
    async def encode_with_context(
        self,
        text: str,
        context: str,
        context_weight: float = 0.3,
        normalize: bool = True
    ) -> EmbeddingResult:
        """Generate embeddings with contextual information.

        Args:
            text: Main text to encode
            context: Contextual information
            context_weight: Weight given to context (0.0 to 1.0)
            normalize: Whether to normalize embeddings

        Returns:
            EmbeddingResult with context-aware embeddings
        """
        pass

    @abstractmethod
    async def encode_conversation(
        self,
        messages: List[Dict[str, str]],
        aggregate: bool = True,
        normalize: bool = True
    ) -> Union[EmbeddingResult, List[EmbeddingResult]]:
        """Encode conversation messages.

        Args:
            messages: List of message dicts with 'role' and 'content'
            aggregate: Whether to return aggregated embedding or individual
            normalize: Whether to normalize embeddings

        Returns:
            Single EmbeddingResult (if aggregate=True) or list of results
        """
        pass

    @abstractmethod
    async def cluster_embeddings(
        self,
        embeddings: np.ndarray,
        num_clusters: int,
        method: str = "kmeans",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Cluster embeddings into groups.

        Args:
            embeddings: Array of embeddings to cluster
            num_clusters: Number of clusters
            method: Clustering method (kmeans, dbscan, etc.)
            context: Optional context information

        Returns:
            Dictionary with cluster assignments and centroids
        """
        pass

    @abstractmethod
    async def dimension_reduction(
        self,
        embeddings: np.ndarray,
        target_dimensions: int,
        method: str = "pca",
        context: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """Reduce embedding dimensions.

        Args:
            embeddings: Array of embeddings to reduce
            target_dimensions: Target dimension size
            method: Reduction method (pca, tsne, umap)
            context: Optional context information

        Returns:
            Reduced-dimension embeddings array
        """
        pass


class MultilingualEmbeddingInterface(EmbeddingInterface):
    """Interface for multilingual embedding models."""

    @abstractmethod
    async def encode_multilingual(
        self,
        texts: List[str],
        languages: Optional[List[str]] = None,
        normalize: bool = True,
        align_languages: bool = True
    ) -> List[EmbeddingResult]:
        """Generate embeddings for multilingual texts.

        Args:
            texts: List of texts in different languages
            languages: Optional list of language codes
            normalize: Whether to normalize embeddings
            align_languages: Whether to align cross-lingual embeddings

        Returns:
            List of EmbeddingResult objects
        """
        pass

    @abstractmethod
    async def cross_lingual_similarity(
        self,
        text_a: str,
        text_b: str,
        lang_a: Optional[str] = None,
        lang_b: Optional[str] = None
    ) -> SimilarityResult:
        """Compute cross-lingual similarity.

        Args:
            text_a: Text in first language
            text_b: Text in second language
            lang_a: Optional language code for text_a
            lang_b: Optional language code for text_b

        Returns:
            SimilarityResult with cross-lingual similarity
        """
        pass

    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes.

        Returns:
            List of ISO language codes
        """
        pass

    @abstractmethod
    async def detect_language(self, text: str) -> str:
        """Detect language of input text.

        Args:
            text: Text to analyze

        Returns:
            Detected language code
        """
        pass