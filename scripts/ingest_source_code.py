#!/usr/bin/env python3
"""
Ingest Echo Brain source code using AST parsing for semantic chunking.
Each function/method = one chunk, each class = one chunk.
NOT character-count splitting.
"""

import ast
import httpx
import json
from pathlib import Path
from uuid import uuid4
import time

OLLAMA_URL = "http://localhost:11434"
QDRANT_URL = "http://localhost:6333"
COLLECTION = "echo_memory"
EMBED_MODEL = "mxbai-embed-large"
SRC_ROOT = Path("/opt/tower-echo-brain/src")

def get_embedding(text: str) -> list:
    """Get embedding vector from Ollama"""
    resp = httpx.post(f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text}, timeout=30)
    return resp.json()["embedding"]

def extract_chunks_from_file(filepath: Path) -> list:
    """Use AST to extract meaningful chunks from a Python file"""
    chunks = []
    try:
        source = filepath.read_text()
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        # Fall back to raw text chunking for non-parseable files
        source = filepath.read_text(errors='replace')
        if len(source.strip()) > 50:
            chunks.append({
                "text": f"# File: {filepath.relative_to(SRC_ROOT.parent)}\n{source[:2000]}",
                "chunk_type": "raw_file",
                "name": filepath.stem,
                "source_file": str(filepath.relative_to(SRC_ROOT.parent))
            })
        return chunks

    rel_path = str(filepath.relative_to(SRC_ROOT.parent))
    lines = source.splitlines()

    # Module-level docstring
    if (ast.get_docstring(tree)):
        doc = ast.get_docstring(tree)
        chunks.append({
            "text": f"# Module: {rel_path}\n# Docstring: {doc}",
            "chunk_type": "module_doc",
            "name": filepath.stem,
            "source_file": rel_path
        })

    # Extract imports block
    imports = [node for node in ast.iter_child_nodes(tree)
               if isinstance(node, (ast.Import, ast.ImportFrom))]
    if imports:
        import_lines = []
        for imp in imports:
            start = imp.lineno - 1
            end = imp.end_lineno or imp.lineno
            import_lines.extend(lines[start:end])
        import_text = "\n".join(import_lines)

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            # Get full class source
            start = node.lineno - 1
            end = node.end_lineno or (node.lineno + 20)
            class_source = "\n".join(lines[start:end])

            # If class is very long, chunk the docstring + method signatures
            if len(class_source) > 2000:
                doc = ast.get_docstring(node) or ""
                methods = [n.name for n in ast.iter_child_nodes(node)
                          if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                summary = f"# Class: {node.name} in {rel_path}\n"
                summary += f"# Docstring: {doc}\n" if doc else ""
                summary += f"# Methods: {', '.join(methods)}\n"
                summary += class_source[:1500]
                chunks.append({
                    "text": summary,
                    "chunk_type": "class",
                    "name": node.name,
                    "source_file": rel_path
                })

                # Also chunk each method individually
                for method_node in ast.iter_child_nodes(node):
                    if isinstance(method_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        m_start = method_node.lineno - 1
                        m_end = method_node.end_lineno or (method_node.lineno + 10)
                        method_source = "\n".join(lines[m_start:m_end])
                        if len(method_source.strip()) > 30:
                            chunks.append({
                                "text": f"# Method: {node.name}.{method_node.name} in {rel_path}\n{method_source[:2000]}",
                                "chunk_type": "method",
                                "name": f"{node.name}.{method_node.name}",
                                "source_file": rel_path
                            })
            else:
                chunks.append({
                    "text": f"# Class: {node.name} in {rel_path}\n{class_source}",
                    "chunk_type": "class",
                    "name": node.name,
                    "source_file": rel_path
                })

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno - 1
            end = node.end_lineno or (node.lineno + 10)
            func_source = "\n".join(lines[start:end])
            if len(func_source.strip()) > 30:
                chunks.append({
                    "text": f"# Function: {node.name} in {rel_path}\n{func_source[:2000]}",
                    "chunk_type": "function",
                    "name": node.name,
                    "source_file": rel_path
                })

    return chunks


def store_vectors(chunks: list):
    """Embed and store chunks in Qdrant"""
    points = []
    for i, chunk in enumerate(chunks):
        print(f"  Embedding {i+1}/{len(chunks)}: {chunk['chunk_type']} - {chunk['name']}")
        try:
            embedding = get_embedding(chunk["text"])
            points.append({
                "id": str(uuid4()),
                "vector": embedding,
                "payload": {
                    "text": chunk["text"],
                    "source": "echo_brain_code",
                    "source_file": chunk["source_file"],
                    "chunk_type": chunk["chunk_type"],
                    "name": chunk["name"],
                    "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ")
                }
            })
        except Exception as e:
            print(f"  ERROR embedding {chunk['name']}: {e}")
            continue

        # Batch upsert every 50
        if len(points) >= 50:
            resp = httpx.put(f"{QDRANT_URL}/collections/{COLLECTION}/points",
                json={"points": points}, timeout=30)
            print(f"  Stored batch of {len(points)} vectors")
            points = []

    # Final batch
    if points:
        resp = httpx.put(f"{QDRANT_URL}/collections/{COLLECTION}/points",
            json={"points": points}, timeout=30)
        print(f"  Stored final batch of {len(points)} vectors")


def main():
    print("=" * 70)
    print("ECHO BRAIN SOURCE CODE INGESTION (AST-PARSED)")
    print("=" * 70)

    py_files = list(SRC_ROOT.rglob("*.py"))
    py_files = [f for f in py_files if "__pycache__" not in str(f)]
    print(f"Found {len(py_files)} Python files")

    all_chunks = []
    for filepath in sorted(py_files):
        chunks = extract_chunks_from_file(filepath)
        if chunks:
            print(f"  {filepath.relative_to(SRC_ROOT.parent)}: {len(chunks)} chunks")
            all_chunks.extend(chunks)

    print(f"\nTotal chunks to ingest: {len(all_chunks)}")
    print(f"Breakdown:")
    from collections import Counter
    types = Counter(c["chunk_type"] for c in all_chunks)
    for t, count in types.most_common():
        print(f"  {t}: {count}")

    print(f"\nStoring vectors...")
    store_vectors(all_chunks)

    # Verify
    resp = httpx.get(f"{QDRANT_URL}/collections/{COLLECTION}").json()
    print(f"\nFinal vector count: {resp['result']['points_count']}")


if __name__ == "__main__":
    main()