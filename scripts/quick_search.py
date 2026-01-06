#!/usr/bin/env python3
"""Quick search test using direct Qdrant API."""
import asyncio
import sys
sys.path.insert(0, '/opt/tower-echo-brain')
import httpx

from src.services.embedding_service import create_embedding_service

async def search(query: str, collection: str = "documents", limit: int = 5):
    """Direct search using Qdrant HTTP API."""
    # Generate embedding
    service = await create_embedding_service()
    query_vector = await service.embed_single(query)
    await service.close()

    # Search
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"http://localhost:6333/collections/{collection}/points/search",
            json={
                "vector": query_vector,
                "limit": limit,
                "with_payload": True
            }
        )
        data = response.json()

        print(f"\n🔍 Query: '{query}'")
        print(f"📚 Collection: {collection}")
        print("=" * 60)

        if 'result' in data and data['result']:
            print(f"✅ Found {len(data['result'])} results:\n")
            for i, r in enumerate(data['result'], 1):
                payload = r.get('payload', {})

                # Get display text based on collection
                if collection == "code":
                    title = payload.get('entity_name', 'Unknown')
                    file = payload.get('file_path', 'Unknown')[:50]
                    print(f"{i}. Score: {r['score']:.3f}")
                    print(f"   Entity: {title}")
                    print(f"   File: {file}")
                elif collection == "conversations":
                    preview = payload.get('text_preview', 'N/A')[:100]
                    print(f"{i}. Score: {r['score']:.3f}")
                    print(f"   Message: {preview}...")
                else:  # documents
                    title = payload.get('title', 'Unknown')[:60]
                    print(f"{i}. Score: {r['score']:.3f}")
                    print(f"   Title: {title}")
                print()
        else:
            print("❌ No results found")

# Test queries
async def main():
    print("=" * 70)
    print("ECHO BRAIN QUICK SEARCH TEST")
    print("=" * 70)

    # Test each collection
    await search("Echo Brain architecture", "documents")
    await search("embedding service", "code")
    await search("OpenAI embeddings upgrade", "conversations")
    await search("anime workflow ComfyUI", "documents")
    await search("vector search Qdrant", "code")

if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else None
    collection = sys.argv[2] if len(sys.argv) > 2 else "documents"

    if query:
        asyncio.run(search(query, collection))
    else:
        asyncio.run(main())