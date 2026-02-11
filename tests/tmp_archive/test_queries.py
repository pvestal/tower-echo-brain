#!/usr/bin/env python3
"""
Test the 5 verification queries
"""

import ollama
from qdrant_client import QdrantClient

# Configuration
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "echo_memory"
EMBEDDING_MODEL = "nomic-embed-text"

# Test queries
test_queries = [
    ("What vehicles does Patrick own?", "2022 Toyota Tundra 1794"),
    ("What GPUs are in Tower?", "RTX 3060 12GB AND RX 9070 XT 16GB"),
    ("What embedding model does Echo Brain use?", "nomic-embed-text 768D"),
    ("What anime projects is Patrick working on?", "Tokyo Debt Desire or Cyberpunk Goblin Slayer"),
    ("What RV does Patrick have?", "Sundowner Trailblazer")
]

def test_retrieval():
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    print("=" * 70)
    print("TESTING RETRIEVAL QUALITY WITH 5 QUERIES")
    print("=" * 70)

    for query, expected in test_queries:
        print(f"\nQuery: {query}")
        print(f"Expected: {expected}")
        print("-" * 50)

        # Create embedding with search_query prefix
        embedding = ollama.embeddings(
            model=EMBEDDING_MODEL,
            prompt=f"search_query: {query}"
        )

        # Search
        from qdrant_client.models import SearchRequest
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding['embedding'],
            limit=5
        ).points

        print("Top 5 results:")
        found_answer = False

        for i, result in enumerate(results, 1):
            content = result.payload.get('content', '')
            score = result.score

            print(f"{i}. Score: {score:.3f}")
            print(f"   Content: {content[:200]}...")

            # Check if expected answer is found
            if any(part.lower() in content.lower() for part in expected.lower().split(" and ")):
                found_answer = True

        if found_answer:
            print("✅ PASSED - Found expected information")
        else:
            print("❌ FAILED - Did not find expected information")

    print("\n" + "=" * 70)
    print("TESTING COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    test_retrieval()