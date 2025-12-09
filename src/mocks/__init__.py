#!/usr/bin/env python3
"""
Echo Brain Mock ML Implementations

This package contains mock implementations of all ML interfaces that return
realistic fake data without requiring any ML libraries. Use for testing
and development when ML models are not available.
"""

from .mock_llm import (
    MockLLM,
    MockOllamaLLM,
    MockCodeLLM,
    MockConversationalLLM
)

from .mock_embedding import (
    MockEmbedding,
    MockSemanticEmbedding,
    MockMultilingualEmbedding
)

from .mock_vision import (
    MockVision,
    MockImageClassification,
    MockObjectDetection,
    MockImageQuality,
    MockImageGeneration,
    MockImageEnhancement
)

from .mock_vector_store import (
    MockVectorStore,
    MockAdvancedVectorStore,
    MockSemanticVectorStore
)

__all__ = [
    # Mock LLM implementations
    'MockLLM',
    'MockOllamaLLM',
    'MockCodeLLM',
    'MockConversationalLLM',

    # Mock embedding implementations
    'MockEmbedding',
    'MockSemanticEmbedding',
    'MockMultilingualEmbedding',

    # Mock vision implementations
    'MockVision',
    'MockImageClassification',
    'MockObjectDetection',
    'MockImageQuality',
    'MockImageGeneration',
    'MockImageEnhancement',

    # Mock vector store implementations
    'MockVectorStore',
    'MockAdvancedVectorStore',
    'MockSemanticVectorStore',
]