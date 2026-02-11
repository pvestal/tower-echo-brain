#!/usr/bin/env python3
import asyncio
import httpx
import json

async def test_gpu_search():
    async with httpx.AsyncClient(timeout=30) as client:
        # Step 1: Embed the exact query
        query = 'What GPUs are in Tower?'
        print(f'Embedding query: {query}')

        embed_resp = await client.post(
            'http://localhost:11434/api/embed',
            json={'model': 'nomic-embed-text', 'input': query}
        )

        if embed_resp.status_code != 200:
            print(f'Embedding failed: {embed_resp.status_code}')
            return

        query_vector = embed_resp.json()['embeddings'][0]
        print(f'Query embedded successfully (dim={len(query_vector)})')

        # Step 2: Search Qdrant with this vector
        search_resp = await client.post(
            'http://localhost:6333/collections/echo_memory/points/search',
            json={
                'vector': query_vector,
                'limit': 10,
                'with_payload': True
            }
        )

        if search_resp.status_code != 200:
            print(f'Search failed: {search_resp.status_code}')
            return

        results = search_resp.json()['result']

        print(f'\nTop 10 search results for "What GPUs are in Tower?":')
        print('=' * 70)

        for i, point in enumerate(results):
            score = point['score']
            content = point['payload']['content']
            ptype = point['payload'].get('type', 'unknown')
            source = point['payload'].get('source', 'unknown')

            # Extract first line or first 100 chars
            preview = content.split('\n')[0][:100]

            print(f'{i+1}. Score: {score:.4f} | Type: {ptype} | Source: {source}')
            print(f'   Content: {preview}...')

            # Check if this contains GPU info
            if any(gpu in content for gpu in ['RTX 3060', 'RX 9070', 'GPU', 'Tower GPU']):
                print(f'   ✓ CONTAINS GPU INFO')
            print()

if __name__ == '__main__':
    asyncio.run(test_gpu_search())