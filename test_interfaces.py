#!/usr/bin/env python3
"""
Test script to demonstrate Echo Brain interface abstraction layer.
Shows how ML components can be used without any ML library dependencies.
"""

import asyncio
import os
import sys
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.container import get_container, Environment
from src.interfaces.llm_interface import ChatRequest, ChatMessage, MessageRole
from src.interfaces.embedding_interface import EmbeddingInterface
from src.interfaces.vision_interface import ImageInput
from src.interfaces.vector_store_interface import VectorPoint, IndexConfig, IndexType, DistanceMetric


async def test_llm_interface():
    """Test LLM interface with mock implementation."""
    print("\n🤖 Testing LLM Interface")
    print("=" * 50)

    container = get_container()
    llm = container.get_llm()

    # Test simple completion
    print("📝 Simple completion:")
    response = await llm.simple_completion("What is artificial intelligence?")
    print(f"Response: {response[:100]}...")

    # Test chat completion
    print("\n💬 Chat completion:")
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful AI assistant."),
        ChatMessage(role=MessageRole.USER, content="Explain quantum computing in simple terms.")
    ]

    chat_request = ChatRequest(
        messages=messages,
        model_name="mock-llama",
        temperature=0.7
    )

    chat_response = await llm.chat_completion(chat_request)
    print(f"Chat response: {chat_response.content[:100]}...")
    print(f"Tokens used: {chat_response.usage}")

    # Test streaming
    print("\n🌊 Streaming response:")
    print("Tokens: ", end="")
    async for chunk in llm.chat_completion_stream(chat_request):
        print(chunk.delta, end="", flush=True)
        if chunk.finish_reason:
            break
    print()

    # Test sentiment analysis
    print("\n😊 Sentiment analysis:")
    sentiment = await llm.analyze_sentiment("I love this new technology!")
    print(f"Sentiment: {sentiment}")


async def test_embedding_interface():
    """Test embedding interface with mock implementation."""
    print("\n🔢 Testing Embedding Interface")
    print("=" * 50)

    container = get_container()
    embedding = container.get(EmbeddingInterface)

    # Test single text encoding
    print("📊 Text encoding:")
    result = await embedding.encode("Hello, this is a test sentence.")
    print(f"Embedding shape: {result.embeddings.shape}")
    print(f"Dimensions: {result.dimensions}")
    print(f"Processing time: {result.processing_time:.3f}s")

    # Test batch encoding
    print("\n📦 Batch encoding:")
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming technology.",
        "Python is a great programming language."
    ]
    batch_results = await embedding.encode_batch(texts)
    print(f"Batch size: {len(batch_results)}")
    for i, result in enumerate(batch_results):
        print(f"  Text {i+1}: {result.embeddings.shape}")

    # Test similarity computation
    print("\n🔍 Similarity computation:")
    similarity = await embedding.compute_similarity(
        "Machine learning is powerful.",
        "AI technology is amazing."
    )
    print(f"Similarity score: {similarity.similarity_score:.3f}")

    # Test finding similar texts
    print("\n🎯 Finding similar texts:")
    candidates = [
        "Deep learning neural networks",
        "Cooking delicious pasta",
        "Machine learning algorithms",
        "Playing guitar music",
        "Artificial intelligence research"
    ]

    similar_texts = await embedding.find_similar(
        "AI and machine learning",
        candidates,
        top_k=3
    )

    print("Most similar texts:")
    for text, score in similar_texts:
        print(f"  {score:.3f}: {text}")


async def test_vision_interface():
    """Test vision interface with mock implementation."""
    print("\n👁️ Testing Vision Interface")
    print("=" * 50)

    container = get_container()

    # Test image classification
    from src.interfaces.vision_interface import ImageClassificationInterface
    classifier = container.get(ImageClassificationInterface)

    # Create mock image
    mock_image = ImageInput(
        image_data=np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8),
        format="png",
        width=512,
        height=512,
        channels=3
    )

    print("🖼️ Image classification:")
    result = await classifier.classify_image(mock_image, top_k=5)
    print(f"Predictions: {len(result.predictions)}")
    for pred in result.predictions[:3]:
        print(f"  {pred['class']}: {pred['confidence']:.3f}")

    # Test image quality assessment
    print("\n✨ Image quality assessment:")
    from src.interfaces.vision_interface import ImageQualityInterface
    quality_assessor = container.get(ImageQualityInterface)

    overall_score = await quality_assessor.calculate_overall_score(mock_image)
    print(f"Overall quality score: {overall_score:.3f}")

    technical_quality = await quality_assessor.assess_technical_quality(mock_image)
    print("Technical quality metrics:")
    for metric, value in technical_quality.items():
        print(f"  {metric}: {value:.3f}")


async def test_vector_store_interface():
    """Test vector store interface with mock implementation."""
    print("\n🗂️ Testing Vector Store Interface")
    print("=" * 50)

    container = get_container()
    vector_store = container.get_vector_store()

    collection_name = "test_collection"
    dimension = 384

    # Create collection
    print("📁 Creating collection:")
    config = IndexConfig(
        index_type=IndexType.FLAT,
        distance_metric=DistanceMetric.COSINE
    )

    success = await vector_store.create_collection(collection_name, dimension, config)
    print(f"Collection created: {success}")

    # Insert vectors
    print("\n📥 Inserting vectors:")
    vectors = []
    for i in range(10):
        vector = np.random.normal(0, 1, dimension)
        vector = vector / np.linalg.norm(vector)  # Normalize

        point = VectorPoint(
            id=f"vector_{i}",
            vector=vector,
            metadata={"index": i, "category": "test"}
        )
        vectors.append(point)

    insert_success = await vector_store.insert_vectors(collection_name, vectors)
    print(f"Vectors inserted: {insert_success}")

    # Search similar vectors
    print("\n🔍 Searching similar vectors:")
    query_vector = np.random.normal(0, 1, dimension)
    query_vector = query_vector / np.linalg.norm(query_vector)

    search_results = await vector_store.search_similar(
        collection_name, query_vector, top_k=3
    )

    print(f"Found {len(search_results)} similar vectors:")
    for result in search_results:
        print(f"  ID: {result.point_id}, Score: {result.score:.3f}")

    # Get collection info
    print("\n📊 Collection information:")
    info = await vector_store.get_collection_info(collection_name)
    if info:
        print(f"  Name: {info.name}")
        print(f"  Dimension: {info.dimension}")
        print(f"  Vector count: {info.vector_count}")
        print(f"  Distance metric: {info.distance_metric.value}")


async def test_dependency_injection():
    """Test dependency injection and environment switching."""
    print("\n⚙️ Testing Dependency Injection")
    print("=" * 50)

    # Show current container diagnostics
    container = get_container()
    diagnostics = container.get_diagnostics()

    print("Container diagnostics:")
    print(f"  Environment: {diagnostics['environment']}")
    print(f"  Registered components: {diagnostics['registered_components']}")
    print(f"  Active singletons: {diagnostics['active_singletons']}")

    print("\nRegistered components:")
    for interface, config in diagnostics['components'].items():
        print(f"  {interface}: {config['implementation']} ({config['lifecycle']})")

    # Test scoped dependencies
    print("\n🔄 Testing scoped dependencies:")
    with container.create_scope("test_scope") as scoped_container:
        llm1 = scoped_container.get_llm()
        llm2 = scoped_container.get_llm()
        print(f"Same LLM instance in scope: {llm1 is llm2}")

    print("Scope disposed automatically")


async def main():
    """Run all interface tests."""
    print("🧪 Echo Brain Interface Abstraction Layer Test")
    print("=" * 60)
    print("This test demonstrates ML interfaces without ML dependencies!")

    # Set testing environment
    os.environ['ECHO_ENVIRONMENT'] = 'testing'

    try:
        await test_dependency_injection()
        await test_llm_interface()
        await test_embedding_interface()
        await test_vision_interface()
        await test_vector_store_interface()

        print("\n✅ All tests completed successfully!")
        print("\n💡 Key Benefits:")
        print("   • No ML library imports required")
        print("   • Fast test execution with realistic mock data")
        print("   • Type safety with abstract interfaces")
        print("   • Easy switching between mock/real implementations")
        print("   • Dependency injection for flexible testing")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())