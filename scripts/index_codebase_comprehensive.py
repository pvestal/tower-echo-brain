#!/usr/bin/env python3
"""
Comprehensive codebase indexer for ALL Tower services
Indexes 1,805+ files across all Tower microservices into Qdrant
"""
import os
import sys
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import hashlib
import ast
import argparse
import time

# Import Qdrant first
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams

# Add Echo Brain src to path
sys.path.insert(0, '/opt/tower-echo-brain/src')
from services.embedding_service import create_embedding_service

# Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "code"
VECTOR_SIZE = 1536
BATCH_SIZE = 50  # Process files in batches

# File extensions to index
EXTENSIONS = [".py", ".ts", ".tsx", ".js", ".jsx", ".vue", ".sh", ".bash", ".sql", ".yaml", ".yml", ".json", ".md"]

# Directories to ignore
IGNORE_DIRS = ["__pycache__", "node_modules", ".git", "venv", ".venv", "dist", "build", ".pytest_cache", "migrations"]

def discover_tower_services() -> List[str]:
    """Dynamically discover all Tower service directories"""
    service_dirs = []

    # Search in multiple locations
    search_paths = [
        Path("/opt"),
        Path("/home/patrick"),
        Path("/home/patrick/Documents"),
        Path("/home/patrick/Archive"),
        Path("/home/patrick/Work"),
    ]

    for base in search_paths:
        if base.exists():
            for service_dir in base.glob("tower-*"):
                if service_dir.is_dir():
                    service_dirs.append(str(service_dir))

    # Remove duplicates and sort
    service_dirs = sorted(list(set(service_dirs)))
    return service_dirs

def get_file_hash(content: str) -> str:
    """Generate hash for file content"""
    return hashlib.md5(content.encode()).hexdigest()

def extract_code_metadata(filepath: Path, content: str) -> Dict[str, Any]:
    """Extract metadata from code files"""
    metadata = {
        "language": filepath.suffix[1:] if filepath.suffix else "unknown",
        "size": len(content),
        "lines": len(content.splitlines()),
        "service": None,
        "module": None,
    }

    # Determine service from path
    for part in filepath.parts:
        if part.startswith("tower-"):
            metadata["service"] = part
            break

    # Extract module/package for Python files
    if filepath.suffix == ".py":
        try:
            tree = ast.parse(content)
            # Count functions and classes
            metadata["functions"] = len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
            metadata["classes"] = len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)])

            # Extract imports
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
            metadata["imports"] = imports[:10]  # Limit to first 10
        except:
            pass

    return metadata

def chunk_code(content: str, filepath: Path, max_chunk_size: int = 1500) -> List[Dict[str, Any]]:
    """Split code into semantic chunks"""
    lines = content.splitlines()
    chunks = []
    current_chunk = []
    current_size = 0

    for i, line in enumerate(lines):
        current_chunk.append(line)
        current_size += len(line) + 1

        # Check if we should create a chunk
        should_chunk = False

        # Natural boundaries
        if filepath.suffix == ".py":
            # Python function/class boundaries
            if line.strip().startswith(("def ", "class ", "async def ")):
                if current_size > 500:  # Minimum chunk size
                    should_chunk = True
        elif filepath.suffix in [".js", ".ts", ".jsx", ".tsx"]:
            # JavaScript function boundaries
            if "function" in line or "=>" in line:
                if current_size > 500:
                    should_chunk = True

        # Size limit reached
        if current_size > max_chunk_size:
            should_chunk = True

        if should_chunk and current_chunk:
            chunk_text = "\n".join(current_chunk[:-1])  # Exclude the trigger line
            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "start_line": i - len(current_chunk) + 2,
                    "end_line": i,
                })
            current_chunk = [line]  # Start new chunk with trigger line
            current_size = len(line) + 1

    # Add remaining lines
    if current_chunk:
        chunk_text = "\n".join(current_chunk)
        if chunk_text.strip():
            chunks.append({
                "text": chunk_text,
                "start_line": len(lines) - len(current_chunk) + 1,
                "end_line": len(lines),
            })

    # If no chunks created, just split by size
    if not chunks:
        for i in range(0, len(lines), 50):
            chunk_lines = lines[i:i+50]
            if chunk_lines:
                chunks.append({
                    "text": "\n".join(chunk_lines),
                    "start_line": i + 1,
                    "end_line": min(i + 50, len(lines)),
                })

    return chunks

async def index_file(
    filepath: Path,
    embedding_service,
    qdrant_client: QdrantClient,
    file_index: int,
    total_files: int
) -> int:
    """Index a single file"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ Error reading {filepath}: {e}")
        return 0

    if not content.strip():
        return 0

    # Extract metadata
    metadata = extract_code_metadata(filepath, content)

    # Chunk the file
    chunks = chunk_code(content, filepath)

    if not chunks:
        return 0

    # Generate embeddings for chunks
    chunk_texts = []
    for chunk in chunks:
        # Create searchable text with context
        chunk_text = f"File: {filepath.name}\n"
        if metadata["service"]:
            chunk_text += f"Service: {metadata['service']}\n"
        chunk_text += f"Language: {metadata['language']}\n"
        chunk_text += f"Lines {chunk['start_line']}-{chunk['end_line']}:\n"
        chunk_text += chunk['text'][:1000]  # Limit chunk size for embedding
        chunk_texts.append(chunk_text)

    # Batch generate embeddings
    try:
        embeddings = await embedding_service.embed_batch(chunk_texts, use_cache=True)
    except Exception as e:
        print(f"❌ Error generating embeddings for {filepath}: {e}")
        return 0

    # Create points for Qdrant
    points = []
    file_id = get_file_hash(str(filepath) + content)

    for i, (chunk, embedding, chunk_text) in enumerate(zip(chunks, embeddings, chunk_texts)):
        # Generate numeric ID from hash
        point_id = int(hashlib.md5(f"{file_id}_{i}".encode()).hexdigest()[:16], 16)

        points.append(PointStruct(
            id=point_id,
            vector=embedding,
            payload={
                "filepath": str(filepath),
                "filename": filepath.name,
                "service": metadata["service"],
                "language": metadata["language"],
                "content": chunk['text'][:2000],  # Store first 2000 chars
                "full_text": chunk_text,
                "start_line": chunk['start_line'],
                "end_line": chunk['end_line'],
                "chunk_index": i,
                "total_chunks": len(chunks),
                "file_size": metadata["size"],
                "file_lines": metadata["lines"],
                "functions": metadata.get("functions", 0),
                "classes": metadata.get("classes", 0),
                "indexed_at": datetime.utcnow().isoformat(),
            }
        ))

    # Upsert to Qdrant
    qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)

    return len(points)

async def ensure_collection(qdrant_client: QdrantClient, recreate: bool = False):
    """Ensure the code collection exists"""
    try:
        collections = [c.name for c in qdrant_client.get_collections().collections]

        if COLLECTION_NAME in collections:
            if recreate:
                print(f"🗑️  Deleting existing '{COLLECTION_NAME}' collection...")
                qdrant_client.delete_collection(COLLECTION_NAME)
                print("✅ Collection deleted")
            else:
                info = qdrant_client.get_collection(COLLECTION_NAME)
                print(f"📦 Using existing '{COLLECTION_NAME}' collection ({info.points_count} vectors)")
                return

        # Create collection
        print(f"📦 Creating '{COLLECTION_NAME}' collection...")
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )
        print("✅ Collection created")

    except Exception as e:
        print(f"❌ Error managing collection: {e}")
        raise

async def main():
    parser = argparse.ArgumentParser(description="Index Tower services codebase")
    parser.add_argument("--recreate", action="store_true", help="Recreate collection (delete existing)")
    parser.add_argument("--service", help="Index only specific service (e.g., tower-echo-brain)")
    parser.add_argument("--limit", type=int, help="Limit number of files to index")
    args = parser.parse_args()

    print("=" * 80)
    print("🚀 TOWER SERVICES COMPREHENSIVE CODEBASE INDEXER")
    print("=" * 80)

    # Initialize services
    print("Initializing services...")
    embedding_service = await create_embedding_service()
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Ensure collection exists
    await ensure_collection(qdrant_client, recreate=args.recreate)

    # Discover Tower services
    print("\n🔍 Discovering Tower services...")
    service_dirs = discover_tower_services()

    if args.service:
        # Filter to specific service
        service_dirs = [d for d in service_dirs if args.service in d]
        if not service_dirs:
            print(f"❌ Service '{args.service}' not found")
            return

    print(f"📁 Found {len(service_dirs)} Tower services:")
    for service_dir in service_dirs:
        print(f"  • {Path(service_dir).name}")

    # Collect all files to index
    print("\n📂 Collecting files to index...")
    files_to_index = []

    for service_dir in service_dirs:
        service_path = Path(service_dir)
        service_name = service_path.name
        file_count = 0

        for ext in EXTENSIONS:
            for filepath in service_path.rglob(f"*{ext}"):
                # Skip ignored directories
                if any(ignore in filepath.parts for ignore in IGNORE_DIRS):
                    continue

                files_to_index.append((filepath, service_name))
                file_count += 1

        if file_count > 0:
            print(f"  {service_name}: {file_count} files")

    # Apply limit if specified
    if args.limit:
        files_to_index = files_to_index[:args.limit]

    print(f"\n📊 Total files to index: {len(files_to_index)}")

    if not files_to_index:
        print("❌ No files found to index")
        return

    # Start indexing
    print("\n🔄 Starting indexing process...")
    print("=" * 80)

    total_chunks = 0
    start_time = time.time()
    errors = 0

    for i, (filepath, service_name) in enumerate(files_to_index, 1):
        # Progress indicator
        progress = (i / len(files_to_index)) * 100
        print(f"[{i}/{len(files_to_index)}] ({progress:.1f}%) {service_name}/{filepath.name}...", end=" ")

        try:
            chunks = await index_file(filepath, embedding_service, qdrant_client, i, len(files_to_index))
            if chunks > 0:
                print(f"✅ {chunks} chunks")
                total_chunks += chunks
            else:
                print("⏭️  Skipped (empty)")
        except Exception as e:
            print(f"❌ Error: {e}")
            errors += 1

        # Show progress every 50 files
        if i % 50 == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            remaining = (len(files_to_index) - i) / rate
            print(f"\n⏱️  Progress: {i}/{len(files_to_index)} files, {total_chunks} chunks")
            print(f"   Time: {elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining")
            print(f"   Rate: {rate:.1f} files/sec\n")

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("✅ INDEXING COMPLETE")
    print("=" * 80)
    print(f"📊 Statistics:")
    print(f"  • Files processed: {len(files_to_index)}")
    print(f"  • Chunks created: {total_chunks}")
    print(f"  • Errors: {errors}")
    print(f"  • Time taken: {elapsed:.1f} seconds")
    print(f"  • Average: {elapsed/len(files_to_index):.2f} sec/file")

    # Verify collection
    info = qdrant_client.get_collection(COLLECTION_NAME)
    print(f"\n📦 Collection '{COLLECTION_NAME}' now has {info.points_count} vectors")

    # Test search
    print("\n🧪 Testing search capability...")
    test_query = "embedding service OpenAI"
    test_embedding = await embedding_service.embed_single(test_query)
    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=test_embedding,
        limit=3
    ).points

    if results:
        print(f"✅ Search works! Found {len(results)} results for '{test_query}':")
        for r in results:
            print(f"  • {r.payload.get('filename')} (score: {r.score:.3f})")
    else:
        print("⚠️  No search results found")

    # Cleanup
    await embedding_service.close()
    print("\n🎉 Codebase indexing complete!")

if __name__ == "__main__":
    asyncio.run(main())