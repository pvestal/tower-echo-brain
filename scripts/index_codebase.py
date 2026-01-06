"""
Index the Echo Brain codebase into Qdrant for semantic code search
"""
import os
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import hashlib
import ast

# Add src to path AFTER qdrant imports to avoid conflicts
import sys
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

sys.path.insert(0, '/opt/tower-echo-brain/src')
from services.embedding_service import create_embedding_service

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
# Dynamically find ALL Tower services
CODE_DIRS = []

# Add all tower services from /opt
for service_dir in Path("/opt").glob("tower-*"):
    if service_dir.is_dir():
        # Add the entire service directory
        CODE_DIRS.append(str(service_dir))

# Add tower services from home directories
for base in [Path("/home/patrick"), Path("/home/patrick/Documents"), Path("/home/patrick/Archive")]:
    if base.exists():
        for service_dir in base.glob("tower-*"):
            if service_dir.is_dir():
                CODE_DIRS.append(str(service_dir))

# Remove duplicates
CODE_DIRS = sorted(list(set(CODE_DIRS)))

EXTENSIONS = [".py", ".ts", ".tsx", ".js", ".jsx", ".vue", ".sh", ".bash", ".sql"]
IGNORE_DIRS = ["__pycache__", "node_modules", ".git", "venv", ".venv", "dist", "build", ".pytest_cache"]

def get_file_hash(content: str) -> str:
    return hashlib.md5(content.encode()).hexdigest()

def extract_python_metadata(content: str, filepath: str) -> Dict[str, Any]:
    """Extract functions, classes, and docstrings from Python files"""
    metadata = {
        "functions": [],
        "classes": [],
        "imports": [],
        "docstring": None
    }

    try:
        tree = ast.parse(content)

        # Get module docstring
        if ast.get_docstring(tree):
            metadata["docstring"] = ast.get_docstring(tree)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                metadata["functions"].append({
                    "name": node.name,
                    "docstring": ast.get_docstring(node),
                    "line": node.lineno
                })
            elif isinstance(node, ast.ClassDef):
                metadata["classes"].append({
                    "name": node.name,
                    "docstring": ast.get_docstring(node),
                    "line": node.lineno
                })
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    metadata["imports"].append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    metadata["imports"].append(node.module)
    except SyntaxError:
        pass

    return metadata

def chunk_code(content: str, filepath: str, chunk_size: int = 1500) -> List[Dict[str, Any]]:
    """Split code into semantic chunks"""
    chunks = []
    lines = content.split('\n')
    current_chunk = []
    current_size = 0
    chunk_start_line = 1

    for i, line in enumerate(lines, 1):
        line_size = len(line) + 1

        # Start new chunk if we hit size limit and we're at a good break point
        if current_size + line_size > chunk_size and current_chunk:
            # Good break points: empty line, function/class def, comment
            if not line.strip() or line.strip().startswith(('def ', 'class ', '#', '//')):
                chunk_text = '\n'.join(current_chunk)
                chunks.append({
                    "content": chunk_text,
                    "start_line": chunk_start_line,
                    "end_line": i - 1,
                    "filepath": filepath
                })
                current_chunk = []
                current_size = 0
                chunk_start_line = i

        current_chunk.append(line)
        current_size += line_size

    # Don't forget the last chunk
    if current_chunk:
        chunk_text = '\n'.join(current_chunk)
        chunks.append({
            "content": chunk_text,
            "start_line": chunk_start_line,
            "end_line": len(lines),
            "filepath": filepath
        })

    return chunks

async def index_file(filepath: Path, embedding_service, qdrant_client) -> int:
    """Index a single file, return number of chunks indexed"""
    try:
        content = filepath.read_text(encoding='utf-8', errors='ignore')
    except Exception as e:
        print(f"  Error reading {filepath}: {e}")
        return 0

    if not content.strip():
        return 0

    file_hash = get_file_hash(content)
    rel_path = str(filepath)

    # Extract metadata for Python files
    metadata = {}
    if filepath.suffix == '.py':
        metadata = extract_python_metadata(content, rel_path)

    # Chunk the code
    chunks = chunk_code(content, rel_path)

    if not chunks:
        return 0

    # Embed chunks
    chunk_texts = [f"File: {rel_path}\n\n{c['content']}" for c in chunks]
    embeddings = await embedding_service.embed_batch(chunk_texts)

    # Create points
    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        point_id = hashlib.md5(f"{rel_path}:{chunk['start_line']}:{chunk['end_line']}".encode()).hexdigest()
        points.append(PointStruct(
            id=point_id,
            vector=embedding,
            payload={
                "filepath": rel_path,
                "filename": filepath.name,
                "extension": filepath.suffix,
                "start_line": chunk["start_line"],
                "end_line": chunk["end_line"],
                "content": chunk["content"][:2000],  # Truncate for storage
                "file_hash": file_hash,
                "indexed_at": datetime.utcnow().isoformat(),
                "functions": metadata.get("functions", []),
                "classes": metadata.get("classes", []),
                "chunk_index": i,
                "total_chunks": len(chunks)
            }
        ))

    # Upsert to Qdrant
    qdrant_client.upsert(collection_name="code", points=points)

    return len(points)

async def main():
    print("=" * 60)
    print("Echo Brain Codebase Indexer")
    print("=" * 60)

    # Initialize services
    embedding_service = await create_embedding_service()
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Collect files
    files_to_index = []
    for code_dir in CODE_DIRS:
        code_path = Path(code_dir)
        if not code_path.exists():
            print(f"Skipping {code_dir} (not found)")
            continue

        for ext in EXTENSIONS:
            for filepath in code_path.rglob(f"*{ext}"):
                # Skip ignored directories
                if any(ignore in filepath.parts for ignore in IGNORE_DIRS):
                    continue
                files_to_index.append(filepath)

    print(f"\nFound {len(files_to_index)} files to index")

    total_chunks = 0
    for i, filepath in enumerate(files_to_index, 1):
        print(f"[{i}/{len(files_to_index)}] Indexing {filepath.name}...", end=" ")
        chunks = await index_file(filepath, embedding_service, qdrant_client)
        print(f"({chunks} chunks)")
        total_chunks += chunks

    print("\n" + "=" * 60)
    print(f"COMPLETE: Indexed {total_chunks} code chunks from {len(files_to_index)} files")

    # Verify
    collection_info = qdrant_client.get_collection("code")
    print(f"Code collection now has {collection_info.points_count} vectors")

if __name__ == "__main__":
    asyncio.run(main())