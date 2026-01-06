#!/usr/bin/env python3
"""
Index EVERYTHING - all 40,203 code files in Echo Brain
"""
import os
import sys
import asyncio
from pathlib import Path
from typing import List
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

# Index EVERYTHING except these
SKIP_DIRS = {"__pycache__", "node_modules", ".git", "venv", ".venv"}
SKIP_FILES = {".pyc", ".pyo", ".so", ".dylib", ".dll"}

# Include ALL code/text files
CODE_EXTENSIONS = {
    ".py", ".pyx", ".pyi", ".pyw",
    ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    ".java", ".kt", ".scala", ".groovy",
    ".c", ".cpp", ".cc", ".h", ".hpp", ".cs",
    ".go", ".rs", ".swift", ".m", ".mm",
    ".rb", ".php", ".pl", ".lua", ".r",
    ".sh", ".bash", ".zsh", ".fish", ".ps1", ".bat", ".cmd",
    ".sql", ".psql", ".mysql", ".plsql",
    ".json", ".jsonl", ".geojson",
    ".xml", ".html", ".htm", ".xhtml",
    ".css", ".scss", ".sass", ".less",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".config",
    ".md", ".rst", ".txt", ".text", ".log",
    ".dockerfile", ".dockerignore", ".gitignore", ".env",
    ".proto", ".graphql", ".prisma"
}

async def index_file(filepath: Path, embedding_service, qdrant_client: QdrantClient) -> int:
    """Index a single file"""
    try:
        # Read file
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()[:50000]  # Limit to 50k chars per file

        if not content.strip():
            return 0

        # Create chunks (simple splitting for speed)
        chunks = []
        lines = content.splitlines()
        chunk_size = 100  # lines per chunk

        for i in range(0, len(lines), chunk_size):
            chunk_lines = lines[i:i+chunk_size]
            chunk_text = "\n".join(chunk_lines)
            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text[:5000],  # Limit chunk size
                    "start": i,
                    "end": min(i+chunk_size, len(lines))
                })

        if not chunks:
            return 0

        # Generate embeddings
        texts_to_embed = []
        for chunk in chunks:
            text = f"File: {filepath.name}\nPath: {filepath}\n{chunk['text'][:1000]}"
            texts_to_embed.append(text)

        embeddings = await embedding_service.embed_batch(texts_to_embed, use_cache=True)

        # Create points
        points = []
        file_hash = hashlib.md5(str(filepath).encode()).hexdigest()

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = int(hashlib.md5(f"{file_hash}_{i}".encode()).hexdigest()[:16], 16)

            points.append(PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "filepath": str(filepath),
                    "filename": filepath.name,
                    "extension": filepath.suffix,
                    "content": chunk["text"][:2000],
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            ))

        # Insert into Qdrant
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
        return len(points)

    except Exception as e:
        print(f"Error indexing {filepath}: {e}")
        return 0

async def main():
    print("=" * 80)
    print("🚀 INDEXING EVERYTHING IN ECHO BRAIN")
    print("=" * 80)

    # Initialize
    embedding_service = await create_embedding_service()
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Recreate collection
    print("Recreating collection...")
    try:
        qdrant_client.delete_collection(COLLECTION_NAME)
    except:
        pass

    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
    )

    # Find ALL files
    print("Finding all code files...")
    base_path = Path("/opt/tower-echo-brain")
    files_to_index = []

    for ext in CODE_EXTENSIONS:
        for filepath in base_path.rglob(f"*{ext}"):
            # Skip unwanted directories
            if any(skip in filepath.parts for skip in SKIP_DIRS):
                continue
            # Skip unwanted file types
            if filepath.suffix in SKIP_FILES:
                continue
            files_to_index.append(filepath)

    print(f"Found {len(files_to_index)} files to index")

    # Index files
    start_time = time.time()
    total_chunks = 0
    batch_size = 100

    for i in range(0, len(files_to_index), batch_size):
        batch = files_to_index[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(files_to_index)-1)//batch_size + 1}...")

        for filepath in batch:
            chunks = await index_file(filepath, embedding_service, qdrant_client)
            total_chunks += chunks

        # Progress
        if (i + batch_size) % 500 == 0:
            elapsed = time.time() - start_time
            rate = (i + batch_size) / elapsed
            eta = (len(files_to_index) - i - batch_size) / rate
            print(f"  Progress: {i+batch_size}/{len(files_to_index)} files")
            print(f"  Chunks: {total_chunks}")
            print(f"  Rate: {rate:.1f} files/sec")
            print(f"  ETA: {eta:.0f} seconds")

    # Final stats
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("✅ INDEXING COMPLETE")
    print(f"Files: {len(files_to_index)}")
    print(f"Chunks: {total_chunks}")
    print(f"Time: {elapsed:.1f} seconds")
    print(f"Rate: {len(files_to_index)/elapsed:.1f} files/sec")

    # Verify
    collection_info = qdrant_client.get_collection(COLLECTION_NAME)
    print(f"\n📦 Collection has {collection_info.points_count} vectors")

    await embedding_service.close()

if __name__ == "__main__":
    asyncio.run(main())