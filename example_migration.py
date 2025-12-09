#!/usr/bin/env python3
"""
Example showing how to migrate existing Echo Brain components to use the new interface abstraction layer.

This demonstrates how to update components like learning_system.py, quality_assessment.py,
and ai_testing_framework.py to use dependency injection instead of direct ML imports.
"""

import asyncio
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.container import get_container
from src.interfaces.llm_interface import ChatRequest, ChatMessage, MessageRole
from src.interfaces.embedding_interface import EmbeddingInterface
from src.interfaces.vision_interface import ImageInput, ImageQualityInterface
from src.interfaces.vector_store_interface import VectorPoint, IndexConfig, IndexType, DistanceMetric


class RefactoredLearningSystem:
    """
    Example refactored learning system using dependency injection.

    BEFORE: Direct imports of torch, sklearn, PIL, transformers
    AFTER: Uses abstract interfaces with automatic mock/real switching
    """

    def __init__(self):
        """Initialize learning system with dependency injection."""
        # Get dependencies through container instead of direct imports
        self.container = get_container()
        self.llm = self.container.get_llm()
        self.embedding = self.container.get(EmbeddingInterface)
        self.vector_store = self.container.get_vector_store()

        # Initialize vector collection for learned patterns
        self.collection_name = "learned_patterns"
        self.pattern_dimension = 384

    async def initialize(self):
        """Initialize learning system dependencies."""
        print("🧠 Initializing Refactored Learning System")

        # Create vector collection for storing learned patterns
        config = IndexConfig(
            index_type=IndexType.HNSW,
            distance_metric=DistanceMetric.COSINE
        )

        await self.vector_store.create_collection(
            self.collection_name,
            self.pattern_dimension,
            config
        )

        print("✅ Learning system initialized with dependency injection")

    async def learn_from_conversation(self, conversation_text: str) -> dict:
        """
        Learn patterns from conversation using injected dependencies.

        BEFORE: Direct sklearn.cluster.KMeans, torch operations
        AFTER: Uses abstract interfaces that work in testing/production
        """
        print(f"📚 Learning from conversation: {conversation_text[:50]}...")

        # Extract topics using LLM interface
        topic_prompt = f"Extract key topics from this conversation:\n{conversation_text}"
        topics_response = await self.llm.simple_completion(topic_prompt)

        # Generate embedding for conversation
        embedding_result = await self.embedding.encode(conversation_text)

        # Analyze sentiment
        sentiment = await self.llm.analyze_sentiment(conversation_text)

        # Store learned pattern in vector store
        pattern_id = f"conversation_{hash(conversation_text) % 10000}"

        pattern = VectorPoint(
            id=pattern_id,
            vector=embedding_result.embeddings[0],
            metadata={
                "type": "conversation",
                "topics": topics_response,
                "sentiment": sentiment,
                "length": len(conversation_text),
                "learned_at": "2025-12-09T12:00:00Z"
            }
        )

        await self.vector_store.insert_vectors(self.collection_name, [pattern])

        return {
            "pattern_id": pattern_id,
            "topics": topics_response,
            "sentiment": sentiment,
            "embedding_dimension": embedding_result.dimensions,
            "stored": True
        }

    async def find_similar_patterns(self, query_text: str, top_k: int = 5) -> list:
        """
        Find similar learned patterns using vector similarity.

        BEFORE: Manual similarity calculations with numpy
        AFTER: Uses vector store interface with proper indexing
        """
        print(f"🔍 Finding patterns similar to: {query_text[:30]}...")

        # Generate query embedding
        query_embedding = await self.embedding.encode(query_text)

        # Search for similar patterns
        results = await self.vector_store.search_similar(
            self.collection_name,
            query_embedding.embeddings[0],
            top_k=top_k,
            score_threshold=0.7
        )

        return [
            {
                "pattern_id": result.point_id,
                "similarity": result.score,
                "metadata": result.metadata
            }
            for result in results
        ]

    async def generate_insight(self, user_query: str) -> str:
        """
        Generate insights based on learned patterns.

        BEFORE: Complex manual model orchestration
        AFTER: Clean interface-based approach
        """
        print(f"💡 Generating insight for: {user_query}")

        # Find relevant patterns
        similar_patterns = await self.find_similar_patterns(user_query)

        # Create context from patterns
        context_parts = []
        for pattern in similar_patterns:
            context_parts.append(f"- {pattern['metadata'].get('topics', 'No topics')}")

        context = "\n".join(context_parts)

        # Generate insight using LLM with context
        insight_prompt = f"""Based on these learned patterns:
{context}

User query: {user_query}

Provide a personalized insight:"""

        insight = await self.llm.simple_completion(insight_prompt, temperature=0.7)

        return insight


class RefactoredQualityAssessment:
    """
    Example refactored quality assessment using dependency injection.

    BEFORE: Direct imports of cv2, PIL, torch, transformers
    AFTER: Uses vision interface for all image operations
    """

    def __init__(self):
        """Initialize quality assessor with dependency injection."""
        self.container = get_container()
        self.quality_assessor = self.container.get(ImageQualityInterface)
        self.llm = self.container.get_llm()

    async def assess_comprehensive_quality(self, image_path: str) -> dict:
        """
        Assess image quality using abstract interfaces.

        BEFORE: Direct cv2.imread, PIL.Image.open, manual algorithms
        AFTER: Uses standardized ImageInput and quality interfaces
        """
        print(f"📸 Assessing quality for: {image_path}")

        # Create image input (in real implementation, would load from file)
        # For demo, create mock image data
        import numpy as np
        mock_image_data = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)

        image_input = ImageInput(
            image_data=mock_image_data,
            format="jpg",
            width=1024,
            height=1024,
            channels=3,
            metadata={"source_path": image_path}
        )

        # Assess technical quality
        technical_metrics = await self.quality_assessor.assess_technical_quality(image_input)

        # Assess artistic quality
        artistic_metrics = await self.quality_assessor.assess_artistic_quality(image_input)

        # Calculate overall score
        overall_score = await self.quality_assessor.calculate_overall_score(image_input)

        # Detect artifacts
        artifacts = await self.quality_assessor.detect_artifacts(image_input)

        # Generate quality report using LLM
        quality_data = {
            "technical": technical_metrics,
            "artistic": artistic_metrics,
            "overall": overall_score,
            "artifacts": artifacts
        }

        report_prompt = f"""Analyze this image quality assessment data:
{quality_data}

Provide a concise quality report with recommendations:"""

        quality_report = await self.llm.simple_completion(report_prompt, temperature=0.3)

        return {
            "technical_metrics": technical_metrics,
            "artistic_metrics": artistic_metrics,
            "overall_score": overall_score,
            "artifacts_detected": artifacts,
            "quality_report": quality_report,
            "assessment_successful": True
        }


class RefactoredAITestingFramework:
    """
    Example refactored AI testing framework using dependency injection.

    BEFORE: Direct sklearn imports for accuracy metrics
    AFTER: Uses interfaces that work with mocks for fast testing
    """

    def __init__(self):
        """Initialize testing framework with dependency injection."""
        self.container = get_container()

    async def test_llm_accuracy(self, test_cases: list) -> dict:
        """
        Test LLM accuracy using interface abstraction.

        BEFORE: Direct model.predict() calls with manual accuracy calculation
        AFTER: Uses LLM interface that works with mocks or real models
        """
        print(f"🧪 Testing LLM accuracy with {len(test_cases)} test cases")

        llm = self.container.get_llm()

        correct_predictions = 0
        total_tests = len(test_cases)
        test_results = []

        for i, test_case in enumerate(test_cases):
            query = test_case.get("input", "")
            expected = test_case.get("expected", "")

            # Get LLM response
            response = await llm.simple_completion(query)

            # Simple accuracy check (in real implementation, would be more sophisticated)
            is_correct = expected.lower() in response.lower()
            if is_correct:
                correct_predictions += 1

            test_results.append({
                "test_id": i,
                "query": query,
                "expected": expected,
                "actual": response,
                "correct": is_correct
            })

            print(f"  Test {i+1}/{total_tests}: {'✅' if is_correct else '❌'}")

        accuracy = correct_predictions / total_tests if total_tests > 0 else 0

        return {
            "total_tests": total_tests,
            "correct_predictions": correct_predictions,
            "accuracy": accuracy,
            "test_results": test_results,
            "framework": "dependency_injection"
        }

    async def test_embedding_consistency(self, text_pairs: list) -> dict:
        """
        Test embedding consistency using interface abstraction.

        BEFORE: Direct transformer model operations
        AFTER: Uses embedding interface with consistent API
        """
        print(f"🔢 Testing embedding consistency with {len(text_pairs)} pairs")

        embedding = self.container.get(EmbeddingInterface)

        consistency_scores = []

        for i, (text1, text2) in enumerate(text_pairs):
            # Test that similar texts have high similarity
            similarity = await embedding.compute_similarity(text1, text2)
            consistency_scores.append(similarity.similarity_score)

            print(f"  Pair {i+1}: {similarity.similarity_score:.3f}")

        avg_consistency = sum(consistency_scores) / len(consistency_scores)

        return {
            "total_pairs": len(text_pairs),
            "consistency_scores": consistency_scores,
            "average_consistency": avg_consistency,
            "min_consistency": min(consistency_scores),
            "max_consistency": max(consistency_scores)
        }


async def demonstrate_migration():
    """Demonstrate the refactored components in action."""
    print("🔄 Echo Brain Component Migration Demonstration")
    print("=" * 60)
    print("Shows how to migrate from direct ML imports to interface abstraction")

    # Set environment for testing
    os.environ['ECHO_ENVIRONMENT'] = 'testing'

    print("\n1️⃣ Testing Refactored Learning System")
    print("-" * 40)

    learning_system = RefactoredLearningSystem()
    await learning_system.initialize()

    # Learn from a conversation
    conversation = "I really enjoyed learning about machine learning today. The neural network concepts were fascinating and I can see how they apply to real-world problems."

    learned_pattern = await learning_system.learn_from_conversation(conversation)
    print(f"Learned pattern: {learned_pattern['pattern_id']}")

    # Find similar patterns
    similar = await learning_system.find_similar_patterns("tell me about AI")
    print(f"Found {len(similar)} similar patterns")

    # Generate insight
    insight = await learning_system.generate_insight("What should I learn next about AI?")
    print(f"Generated insight: {insight[:100]}...")

    print("\n2️⃣ Testing Refactored Quality Assessment")
    print("-" * 40)

    quality_assessor = RefactoredQualityAssessment()

    quality_result = await quality_assessor.assess_comprehensive_quality("/path/to/image.jpg")
    print(f"Overall quality score: {quality_result['overall_score']:.3f}")
    print(f"Quality report: {quality_result['quality_report'][:100]}...")

    print("\n3️⃣ Testing Refactored AI Testing Framework")
    print("-" * 40)

    testing_framework = RefactoredAITestingFramework()

    # Test LLM accuracy
    llm_test_cases = [
        {"input": "What is 2+2?", "expected": "4"},
        {"input": "What color is the sky?", "expected": "blue"},
        {"input": "What is Python?", "expected": "programming"}
    ]

    llm_results = await testing_framework.test_llm_accuracy(llm_test_cases)
    print(f"LLM Accuracy: {llm_results['accuracy']:.1%}")

    # Test embedding consistency
    text_pairs = [
        ("The cat sat on the mat", "A feline rested on the rug"),
        ("Machine learning is powerful", "AI technology is amazing"),
        ("Python is great", "I love programming in Python")
    ]

    embedding_results = await testing_framework.test_embedding_consistency(text_pairs)
    print(f"Average embedding consistency: {embedding_results['average_consistency']:.3f}")

    print("\n✅ Migration demonstration completed!")
    print("\n💡 Migration Benefits:")
    print("   • Eliminates direct ML library dependencies in business logic")
    print("   • Enables fast testing with realistic mock data")
    print("   • Provides clean separation of concerns")
    print("   • Allows easy switching between test/production environments")
    print("   • Improves testability and maintainability")
    print("   • Reduces coupling between components")


if __name__ == "__main__":
    asyncio.run(demonstrate_migration())