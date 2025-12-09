#!/usr/bin/env python3
"""
Echo Brain ML Interface Abstractions

This package contains abstract interfaces that separate business logic from ML implementations.
All ML components should implement these interfaces to ensure testability and flexibility.
"""

from .ml_model_interface import (
    MLModelInterface,
    StreamingMLModelInterface,
    TrainableMLModelInterface,
    MultiModalMLModelInterface,
    ModelType,
    ModelStatus,
    ModelMetadata,
    ModelPrediction
)

from .llm_interface import (
    LLMInterface,
    OllamaLLMInterface,
    CodeLLMInterface,
    ConversationalLLMInterface,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    StreamChunk,
    MessageRole,
    ResponseFormat
)

from .embedding_interface import (
    EmbeddingInterface,
    SemanticEmbeddingInterface,
    MultilingualEmbeddingInterface,
    EmbeddingResult,
    SimilarityResult
)

from .vision_interface import (
    VisionInterface,
    ImageClassificationInterface,
    ObjectDetectionInterface,
    ImageQualityInterface,
    ImageGenerationInterface,
    ImageEnhancementInterface,
    VisionTask,
    VisionResult,
    ImageInput,
    BoundingBox,
    DetectionResult
)

from .vector_store_interface import (
    VectorStoreInterface,
    AdvancedVectorStoreInterface,
    SemanticVectorStoreInterface,
    VectorPoint,
    SearchResult,
    CollectionInfo,
    IndexConfig,
    DistanceMetric,
    IndexType
)

__all__ = [
    # Base ML interfaces
    'MLModelInterface',
    'StreamingMLModelInterface',
    'TrainableMLModelInterface',
    'MultiModalMLModelInterface',
    'ModelType',
    'ModelStatus',
    'ModelMetadata',
    'ModelPrediction',

    # LLM interfaces
    'LLMInterface',
    'OllamaLLMInterface',
    'CodeLLMInterface',
    'ConversationalLLMInterface',
    'ChatMessage',
    'ChatRequest',
    'ChatResponse',
    'StreamChunk',
    'MessageRole',
    'ResponseFormat',

    # Embedding interfaces
    'EmbeddingInterface',
    'SemanticEmbeddingInterface',
    'MultilingualEmbeddingInterface',
    'EmbeddingResult',
    'SimilarityResult',

    # Vision interfaces
    'VisionInterface',
    'ImageClassificationInterface',
    'ObjectDetectionInterface',
    'ImageQualityInterface',
    'ImageGenerationInterface',
    'ImageEnhancementInterface',
    'VisionTask',
    'VisionResult',
    'ImageInput',
    'BoundingBox',
    'DetectionResult',

    # Vector store interfaces
    'VectorStoreInterface',
    'AdvancedVectorStoreInterface',
    'SemanticVectorStoreInterface',
    'VectorPoint',
    'SearchResult',
    'CollectionInfo',
    'IndexConfig',
    'DistanceMetric',
    'IndexType',
]