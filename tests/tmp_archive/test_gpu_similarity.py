#!/usr/bin/env python3
import asyncio
import httpx

async def test_similarity():
    async with httpx.AsyncClient(timeout=30) as client:
        # Embed the user's question
        query = "What GPUs are in Tower?"
        print(f"Query: {query}\n")

        embed_resp = await client.post(
            'http://localhost:11434/api/embed',
            json={'model': 'nomic-embed-text', 'input': query}
        )

        if embed_resp.status_code != 200:
            print("Failed to embed query")
            return

        query_vector = embed_resp.json()['embeddings'][0]

        # Test similarity against specific vectors
        test_ids = [
            'eca5d438-0d38-4e1e-a9e9-0df3a6cf1bc0',  # GPU fact 1
            'ecd02416-3bcc-4ffe-b753-1cbc0b858bee',  # GPU fact 2
            '9dd79347-4401-4c29-b400-87384f6a9641'   # GPU fact 3
        ]

        print("Testing similarity scores for GPU fact vectors:")
        print("-" * 50)

        for vec_id in test_ids:
            # Get the point
            get_resp = await client.post(
                'http://localhost:6333/collections/echo_memory/points',
                json={'ids': [vec_id], 'with_payload': True}
            )

            if get_resp.status_code != 200:
                print(f"Failed to get point {vec_id}")
                continue

            points = get_resp.json()['result']
            if not points:
                print(f"Point {vec_id} not found")
                continue

            point = points[0]
            content = point['payload']['content'][:80]

            # Calculate similarity manually (cosine similarity)
            import numpy as np
            vec1 = np.array(query_vector)
            vec2 = np.array(point['vector'])
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

            print(f"ID: {vec_id[:8]}...")
            print(f"Content: {content}...")
            print(f"Similarity: {similarity:.4f}\n")

        # Now search and see what actually gets returned
        print("\nActual search results:")
        print("-" * 50)

        search_resp = await client.post(
            'http://localhost:6333/collections/echo_memory/points/search',
            json={
                'vector': query_vector,
                'limit': 5,
                'with_payload': True,
                'score_threshold': 0.0  # Get all results
            }
        )

        results = search_resp.json()['result']
        for i, point in enumerate(results):
            content = point['payload']['content'][:80]
            print(f"{i+1}. Score: {point['score']:.4f} | ID: {point['id'][:8]}...")
            print(f"   Content: {content}...")

if __name__ == '__main__':
    asyncio.run(test_similarity())