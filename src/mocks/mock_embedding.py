#!/usr/bin/env python3
"""
Mock embedding implementation for testing Echo Brain without ML dependencies.
Returns realistic fake embeddings and similarity computations.
Patrick Vestal - December 9, 2025
"""

import asyncio
import hashlib
import random
import time
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
import numpy as np

from ..interfaces.embedding_interface import (
    EmbeddingInterface, EmbeddingResult, SimilarityResult,
    SemanticEmbeddingInterface, MultilingualEmbeddingInterface
)
from ..interfaces.ml_model_interface import ModelStatus, ModelMetadata, ModelType


class MockEmbedding(EmbeddingInterface):
    """Mock embedding implementation for testing."""

    def __init__(self, model_name: str = "mock-embedding", dimensions: int = 384):
        """Initialize mock embedding model."""
        super().__init__(model_name)
        self._dimensions = dimensions
        self._max_sequence_length = 512
        self._status = ModelStatus.READY

        # Mock model metadata
        self._metadata = ModelMetadata(
            name=model_name,
            version="1.0.0-mock",
            model_type=ModelType.EMBEDDING,
            parameters=110000000,  # 110M parameters
            memory_usage=1024,  # 1GB
            compute_requirements={"gpu_memory": "1GB", "cpu_cores": 2},
            capabilities=["text_embedding", "similarity", "clustering"],
            created_at=datetime.now()
        )

    async def load(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Mock model loading."""
        self._status = ModelStatus.LOADING
        await asyncio.sleep(0.1)
        self._status = ModelStatus.READY
        return True

    async def unload(self) -> bool:
        """Mock model unloading."""
        self._status = ModelStatus.UNLOADED
        return True

    async def predict(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> Any:
        """Mock prediction - delegates to encode."""
        if isinstance(input_data, str):
            return await self.encode(input_data, context=context)
        elif isinstance(input_data, list):
            return await self.encode_batch(input_data, context=context)
        else:
            raise ValueError("Input must be string or list of strings")

    async def batch_predict(self, inputs: List[Any], context: Optional[Dict[str, Any]] = None) -> List[Any]:
        """Mock batch prediction."""
        results = []
        for input_data in inputs:
            result = await self.predict(input_data, context)
            results.append(result)
        return results

    def validate_input(self, input_data: Any) -> bool:
        """Validate input data."""
        if isinstance(input_data, str):
            return self.validate_text_input(input_data)
        elif isinstance(input_data, list):
            return all(self.validate_text_input(text) for text in input_data if isinstance(text, str))
        return False

    async def health_check(self) -> Dict[str, Any]:
        """Mock health check."""
        return {
            "status": "healthy",
            "model_loaded": self._status == ModelStatus.READY,
            "dimensions": self._dimensions,
            "max_sequence_length": self._max_sequence_length,
            "memory_usage": 512,  # MB
            "embeddings_generated": random.randint(1000, 10000)
        }

    async def encode(self, texts: Union[str, List[str]], normalize: bool = True, context: Optional[Dict[str, Any]] = None) -> EmbeddingResult:
        """Generate mock embeddings."""
        start_time = time.time()
        await asyncio.sleep(random.uniform(0.01, 0.05))  # Simulate processing time

        if isinstance(texts, str):
            texts = [texts]

        # Generate deterministic but realistic embeddings based on text hash
        embeddings_list = []
        for text in texts:
            embedding = self._generate_mock_embedding(text, normalize)
            embeddings_list.append(embedding)

        embeddings_array = np.array(embeddings_list)
        processing_time = time.time() - start_time

        return EmbeddingResult(
            embeddings=embeddings_array,
            input_text=texts[0] if len(texts) == 1 else f"{len(texts)} texts",
            model_name=self.name,
            dimensions=self._dimensions,
            processing_time=processing_time,
            metadata={
                "num_texts": len(texts),
                "normalized": normalize,
                "avg_text_length": sum(len(t) for t in texts) / len(texts),
                "context": context
            }
        )

    async def encode_batch(self, texts: List[str], batch_size: int = 32, normalize: bool = True, context: Optional[Dict[str, Any]] = None) -> List[EmbeddingResult]:
        """Generate mock batch embeddings."""
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            result = await self.encode(batch, normalize, context)

            # Split batch result into individual results
            for j, text in enumerate(batch):
                individual_result = EmbeddingResult(
                    embeddings=result.embeddings[j:j+1],
                    input_text=text,
                    model_name=self.name,
                    dimensions=self._dimensions,
                    processing_time=result.processing_time / len(batch),
                    metadata={
                        "batch_index": i + j,
                        "batch_size": batch_size,
                        "normalized": normalize
                    }
                )
                results.append(individual_result)

        return results

    async def compute_similarity(self, text_a: str, text_b: str, similarity_type: str = "cosine", context: Optional[Dict[str, Any]] = None) -> SimilarityResult:
        """Compute mock similarity between texts."""
        start_time = time.time()
        await asyncio.sleep(0.01)

        # Generate embeddings for both texts
        embedding_a = self._generate_mock_embedding(text_a, normalize=True)
        embedding_b = self._generate_mock_embedding(text_b, normalize=True)

        # Compute similarity based on type
        if similarity_type == "cosine":
            similarity = np.dot(embedding_a, embedding_b)
        elif similarity_type == "euclidean":
            similarity = 1.0 / (1.0 + np.linalg.norm(embedding_a - embedding_b))
        elif similarity_type == "dot_product":
            similarity = np.dot(embedding_a, embedding_b)
        else:
            similarity = np.dot(embedding_a, embedding_b)  # Default to cosine

        # Add some realistic variation
        similarity = max(0.0, min(1.0, similarity + random.uniform(-0.05, 0.05)))

        processing_time = time.time() - start_time

        return SimilarityResult(
            similarity_score=similarity,
            text_a=text_a,
            text_b=text_b,
            similarity_type=similarity_type,
            metadata={
                "processing_time": processing_time,
                "embedding_dimensions": self._dimensions,
                "context": context
            }
        )

    async def find_similar(self, query_text: str, candidate_texts: List[str], top_k: int = 5, similarity_threshold: float = 0.7, context: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float]]:
        """Find mock similar texts."""
        await asyncio.sleep(random.uniform(0.05, 0.15))

        # Compute similarities with all candidates
        similarities = []
        for candidate in candidate_texts:
            similarity_result = await self.compute_similarity(query_text, candidate, context=context)
            if similarity_result.similarity_score >= similarity_threshold:
                similarities.append((candidate, similarity_result.similarity_score))

        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension size."""
        return self._dimensions

    def get_max_sequence_length(self) -> int:
        """Get maximum sequence length."""
        return self._max_sequence_length

    def _generate_mock_embedding(self, text: str, normalize: bool = True) -> np.ndarray:
        """Generate deterministic mock embedding based on text."""
        # Use text hash as seed for reproducible embeddings
        text_hash = hashlib.md5(text.encode()).hexdigest()
        seed = int(text_hash[:8], 16) % (2**32)
        np.random.seed(seed)

        # Generate embedding with some text-based characteristics
        embedding = np.random.normal(0, 1, self._dimensions).astype(np.float32)

        # Add text-based features
        text_length_factor = min(len(text) / 100, 2.0)
        embedding[0] = text_length_factor  # First dimension represents text length

        # Add word count influence
        word_count = len(text.split())
        embedding[1] = word_count / 50.0  # Second dimension represents word density

        # Add character diversity
        char_diversity = len(set(text.lower())) / 26.0
        embedding[2] = char_diversity  # Third dimension represents character diversity

        if normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

        return embedding


class MockSemanticEmbedding(MockEmbedding, SemanticEmbeddingInterface):
    """Mock semantic embedding with advanced features."""

    def __init__(self, model_name: str = "mock-semantic-embedding"):
        """Initialize mock semantic embedding."""
        super().__init__(model_name, dimensions=768)  # Larger for semantic

    async def encode_with_context(self, text: str, context: str, context_weight: float = 0.3, normalize: bool = True) -> EmbeddingResult:
        """Generate mock context-aware embeddings."""
        start_time = time.time()
        await asyncio.sleep(0.02)

        # Generate embeddings for text and context
        text_embedding = self._generate_mock_embedding(text, normalize=False)
        context_embedding = self._generate_mock_embedding(context, normalize=False)

        # Combine with weighted average
        combined_embedding = (1 - context_weight) * text_embedding + context_weight * context_embedding

        if normalize:
            norm = np.linalg.norm(combined_embedding)
            if norm > 0:
                combined_embedding = combined_embedding / norm

        processing_time = time.time() - start_time

        return EmbeddingResult(
            embeddings=combined_embedding.reshape(1, -1),
            input_text=text,
            model_name=self.name,
            dimensions=self._dimensions,
            processing_time=processing_time,
            metadata={
                "context": context,
                "context_weight": context_weight,
                "context_aware": True
            }
        )

    async def encode_conversation(self, messages: List[Dict[str, str]], aggregate: bool = True, normalize: bool = True) -> Union[EmbeddingResult, List[EmbeddingResult]]:
        """Encode mock conversation messages."""
        await asyncio.sleep(0.05)

        results = []
        for i, message in enumerate(messages):
            content = message.get('content', '')
            role = message.get('role', 'user')

            # Add role information to embedding generation
            role_text = f"[{role}]: {content}"
            embedding = self._generate_mock_embedding(role_text, normalize)

            result = EmbeddingResult(
                embeddings=embedding.reshape(1, -1),
                input_text=content,
                model_name=self.name,
                dimensions=self._dimensions,
                processing_time=0.01,
                metadata={
                    "message_index": i,
                    "role": role,
                    "conversation_encoding": True
                }
            )
            results.append(result)

        if aggregate:
            # Average all embeddings
            all_embeddings = np.array([r.embeddings.flatten() for r in results])
            avg_embedding = np.mean(all_embeddings, axis=0)

            if normalize:
                norm = np.linalg.norm(avg_embedding)
                if norm > 0:
                    avg_embedding = avg_embedding / norm

            return EmbeddingResult(
                embeddings=avg_embedding.reshape(1, -1),
                input_text=f"Conversation with {len(messages)} messages",
                model_name=self.name,
                dimensions=self._dimensions,
                processing_time=sum(r.processing_time for r in results),
                metadata={
                    "aggregated": True,
                    "num_messages": len(messages),
                    "roles": list(set(msg.get('role', 'user') for msg in messages))
                }
            )
        else:
            return results

    async def cluster_embeddings(self, embeddings: np.ndarray, num_clusters: int, method: str = "kmeans", context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Mock embedding clustering."""
        await asyncio.sleep(0.3)

        n_samples = embeddings.shape[0]
        if num_clusters > n_samples:
            num_clusters = n_samples

        # Generate mock cluster assignments
        cluster_assignments = np.random.randint(0, num_clusters, n_samples)

        # Generate mock centroids
        centroids = []
        for cluster_id in range(num_clusters):
            centroid = np.random.normal(0, 1, embeddings.shape[1])
            centroid = centroid / np.linalg.norm(centroid)  # Normalize
            centroids.append(centroid)

        # Calculate mock cluster statistics
        cluster_sizes = [np.sum(cluster_assignments == i) for i in range(num_clusters)]

        return {
            "cluster_assignments": cluster_assignments.tolist(),
            "centroids": np.array(centroids),
            "num_clusters": num_clusters,
            "cluster_sizes": cluster_sizes,
            "method": method,
            "silhouette_score": random.uniform(0.3, 0.7),
            "inertia": random.uniform(100, 1000),
            "metadata": {
                "n_samples": n_samples,
                "n_features": embeddings.shape[1],
                "context": context
            }
        }

    async def dimension_reduction(self, embeddings: np.ndarray, target_dimensions: int, method: str = "pca", context: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Mock dimension reduction."""
        await asyncio.sleep(0.2)

        n_samples, current_dims = embeddings.shape
        target_dimensions = min(target_dimensions, current_dims)

        # Generate mock reduced embeddings
        np.random.seed(42)  # For reproducibility
        reduced_embeddings = np.random.normal(0, 1, (n_samples, target_dimensions))

        # Normalize each embedding
        for i in range(n_samples):
            norm = np.linalg.norm(reduced_embeddings[i])
            if norm > 0:
                reduced_embeddings[i] = reduced_embeddings[i] / norm

        return reduced_embeddings.astype(np.float32)


class MockMultilingualEmbedding(MockEmbedding, MultilingualEmbeddingInterface):
    """Mock multilingual embedding model."""

    def __init__(self, model_name: str = "mock-multilingual-embedding"):
        """Initialize mock multilingual embedding."""
        super().__init__(model_name, dimensions=512)
        self._supported_languages = [
            "en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko", "ar", "hi"
        ]

    async def encode_multilingual(self, texts: List[str], languages: Optional[List[str]] = None, normalize: bool = True, align_languages: bool = True) -> List[EmbeddingResult]:
        """Generate mock multilingual embeddings."""
        await asyncio.sleep(0.1)

        if languages is None:
            languages = ["en"] * len(texts)

        results = []
        for text, lang in zip(texts, languages):
            # Add language code influence to embedding
            lang_text = f"[{lang}] {text}"
            embedding = self._generate_mock_embedding(lang_text, normalize)

            # If aligning languages, add small language-invariant component
            if align_languages:
                # Generate base semantic embedding
                base_embedding = self._generate_mock_embedding(text, normalize=False)
                # Blend with language-specific embedding
                embedding = 0.8 * embedding + 0.2 * base_embedding
                if normalize:
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm

            result = EmbeddingResult(
                embeddings=embedding.reshape(1, -1),
                input_text=text,
                model_name=self.name,
                dimensions=self._dimensions,
                processing_time=0.02,
                metadata={
                    "language": lang,
                    "multilingual": True,
                    "aligned": align_languages
                }
            )
            results.append(result)

        return results

    async def cross_lingual_similarity(self, text_a: str, text_b: str, lang_a: Optional[str] = None, lang_b: Optional[str] = None) -> SimilarityResult:
        """Compute mock cross-lingual similarity."""
        await asyncio.sleep(0.02)

        # Generate aligned embeddings for cross-lingual comparison
        lang_a = lang_a or "en"
        lang_b = lang_b or "en"

        # For cross-lingual similarity, use aligned embeddings
        embedding_a = self._generate_mock_embedding(f"[aligned] {text_a}", normalize=True)
        embedding_b = self._generate_mock_embedding(f"[aligned] {text_b}", normalize=True)

        similarity = np.dot(embedding_a, embedding_b)

        # Reduce similarity slightly for different languages
        if lang_a != lang_b:
            similarity *= random.uniform(0.85, 0.95)

        similarity = max(0.0, min(1.0, similarity))

        return SimilarityResult(
            similarity_score=similarity,
            text_a=text_a,
            text_b=text_b,
            similarity_type="cross_lingual_cosine",
            metadata={
                "language_a": lang_a,
                "language_b": lang_b,
                "cross_lingual": True,
                "aligned": True
            }
        )

    def get_supported_languages(self) -> List[str]:
        """Get mock supported languages."""
        return self._supported_languages.copy()

    async def detect_language(self, text: str) -> str:
        """Mock language detection."""
        await asyncio.sleep(0.01)

        # Simple mock language detection based on text characteristics
        text_lower = text.lower()

        # Very basic heuristics for demo
        if any(word in text_lower for word in ["the", "and", "is", "are", "this"]):
            return "en"
        elif any(word in text_lower for word in ["el", "la", "es", "son", "este"]):
            return "es"
        elif any(word in text_lower for word in ["le", "la", "est", "sont", "ce"]):
            return "fr"
        elif any(word in text_lower for word in ["der", "die", "das", "ist", "sind"]):
            return "de"
        elif any(char in text for char in "你好世界"):
            return "zh"
        elif any(char in text for char in "こんにちは"):
            return "ja"
        else:
            # Default to English
            return "en"