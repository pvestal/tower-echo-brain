#!/usr/bin/env python3
"""
Index ALL 90,235 code files across ALL Tower services
"""
import os
import sys
import asyncio
from pathlib import Path
import hashlib
import time

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams

sys.path.insert(0, '/opt/tower-echo-brain/src')
from services.embedding_service import create_embedding_service

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "code"
VECTOR_SIZE = 1536

# Find ALL Tower services
TOWER_PATHS = [
    "/opt/tower-*",
    "/home/patrick/tower-*",
    "/home/patrick/Documents/tower-*",
    "/home/patrick/Archive/tower-*"
]

SKIP_DIRS = {"__pycache__", "node_modules", ".git", "venv", ".venv"}
CODE_EXTENSIONS = {".py", ".js", ".ts", ".jsx", ".tsx", ".vue", ".sql", ".json", ".yaml", ".yml", ".sh", ".md"}

async def index_file(filepath: Path, embedding_service, qdrant_client) -> int:
    """Index a single file with chunking"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()[:50000]

        if not content.strip():
            return 0

        # Simple chunking
        chunks = []
        lines = content.splitlines()
        for i in range(0, len(lines), 100):
            chunk = "\n".join(lines[i:i+100])[:5000]
            if chunk.strip():
                chunks.append(chunk)

        if not chunks:
            return 0

        # Embed
        texts = [f"File: {filepath}\n{chunk[:1000]}" for chunk in chunks]
        embeddings = await embedding_service.embed_batch(texts, use_cache=True)

        # Create points
        points = []
        file_hash = hashlib.md5(str(filepath).encode()).hexdigest()

        for i, (chunk, vec) in enumerate(zip(chunks, embeddings)):
            point_id = int(hashlib.md5(f"{file_hash}_{i}".encode()).hexdigest()[:16], 16)
            points.append(PointStruct(
                id=point_id,
                vector=vec,
                payload={
                    "filepath": str(filepath),
                    "content": chunk[:2000],
                    "service": str(filepath).split("/")[2] if "tower-" in str(filepath) else "unknown",
                    "chunk": i
                }
            ))

        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
        return len(points)
    except:
        return 0

async def main():
    print("=" * 80)
    print("🚀 INDEXING ALL 90,235 TOWER FILES")
    print("=" * 80)

    # Init
    embedding_service = await create_embedding_service()
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Recreate collection
    try:
        qdrant.delete_collection(COLLECTION_NAME)
    except:
        pass
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
    )

    # Find ALL files
    print("Finding all Tower files...")
    all_files = []

    from glob import glob
    for pattern in TOWER_PATHS:
        for service_dir in glob(pattern):
            service_path = Path(service_dir)
            if service_path.is_dir():
                for ext in CODE_EXTENSIONS:
                    for f in service_path.rglob(f"*{ext}"):
                        if not any(skip in f.parts for skip in SKIP_DIRS):
                            all_files.append(f)

    print(f"Found {len(all_files)} files to index")

    # Index in batches
    start = time.time()
    total_chunks = 0

    for i in range(0, len(all_files), 100):
        batch = all_files[i:i+100]

        for f in batch:
            chunks = await index_file(f, embedding_service, qdrant)
            total_chunks += chunks

        if i % 1000 == 0 and i > 0:
            elapsed = time.time() - start
            rate = i / elapsed
            eta = (len(all_files) - i) / rate
            print(f"Progress: {i}/{len(all_files)} files, {total_chunks} chunks")
            print(f"Rate: {rate:.1f} files/sec, ETA: {eta:.0f} sec")

    # Done
    elapsed = time.time() - start
    info = qdrant.get_collection(COLLECTION_NAME)

    print("\n" + "=" * 80)
    print("✅ COMPLETE")
    print(f"Files: {len(all_files)}")
    print(f"Chunks: {total_chunks}")
    print(f"Vectors: {info.points_count}")
    print(f"Time: {elapsed:.0f} seconds")
    print(f"Rate: {len(all_files)/elapsed:.1f} files/sec")

    await embedding_service.close()

if __name__ == "__main__":
    asyncio.run(main())