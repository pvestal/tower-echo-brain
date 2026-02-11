#!/usr/bin/env python3
"""
Test Qdrant search for architecture documents
"""
import json
import urllib.request

def generate_embedding(text):
    """Generate embedding using Ollama nomic-embed-text model"""
    req = urllib.request.Request(
        "http://localhost:11434/api/embed",
        data=json.dumps({
            "model": "nomic-embed-text",
            "input": text
        }).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )

    try:
        resp = urllib.request.urlopen(req, timeout=30)
        data = json.load(resp)
        embeddings = data.get('embeddings', [])
        return embeddings[0] if embeddings else None
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def search_qdrant(query, limit=5):
    """Search Qdrant for similar vectors"""
    print(f"Searching for: {query}")
    print("=" * 70)

    # Generate embedding
    embedding = generate_embedding(query)
    if not embedding:
        print("Failed to generate embedding")
        return

    print(f"Generated embedding with {len(embedding)} dimensions")

    # Search Qdrant
    req = urllib.request.Request(
        "http://localhost:6333/collections/echo_memory/points/search",
        data=json.dumps({
            "vector": embedding,
            "limit": limit,
            "with_payload": True,
            "score_threshold": 0.3
        }).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )

    try:
        resp = urllib.request.urlopen(req)
        data = json.load(resp)
        results = data.get('result', [])

        print(f"\nFound {len(results)} results:\n")

        for i, result in enumerate(results, 1):
            score = result.get('score', 0)
            payload = result.get('payload', {})
            source = payload.get('source', 'unknown')
            source_path = payload.get('source_path', '')
            section_title = payload.get('section_title', '')
            priority = payload.get('priority', 0)
            authoritative = payload.get('authoritative', False)
            content = payload.get('content', payload.get('text', ''))[:200]

            print(f"{i}. Score: {score:.4f}")
            print(f"   Source: {source}")
            if source_path:
                print(f"   Path: {source_path}")
            if section_title:
                print(f"   Section: {section_title}")
            if priority > 0:
                print(f"   Priority: {priority}")
            if authoritative:
                print(f"   Authoritative: YES")
            print(f"   Content: {content}...")
            print()

    except Exception as e:
        print(f"Error searching Qdrant: {e}")

# Test queries
queries = [
    "What databases does Echo Brain use?",
    "PostgreSQL Qdrant echo_brain database",
    "Echo Brain architecture system configuration",
    "108 modules 29 directories",
]

for query in queries:
    search_qdrant(query, limit=3)
    print("\n" + "=" * 70 + "\n")