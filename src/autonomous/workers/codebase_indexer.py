"""Codebase Indexer Worker - Indexes Echo Brain's own source code for self-awareness"""

import asyncio
import ast
import hashlib
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional

import httpx
import asyncpg

logger = logging.getLogger(__name__)


class CodebaseIndexer:
    """Indexes Echo Brain's own source code for self-awareness"""

    def __init__(self):
        self.src_root = "/opt/tower-echo-brain"
        self.db_url = os.environ.get("DATABASE_URL",
            "postgresql://echo:echo_secure_password_123@localhost/echo_brain")
        self.qdrant_url = "http://localhost:6333"
        self.ollama_url = "http://localhost:11434"
        self.collection = "echo_memory"
        self.chunk_size = 500  # tokens approximately
        self.chunk_overlap = 100

    async def run_cycle(self):
        """Main worker cycle — called by scheduler"""
        logger.info("🧠 Codebase Indexer starting cycle")

        try:
            conn = await asyncpg.connect(self.db_url)

            # Scan for Python files and key config files
            files_to_index = await self._scan_files()
            logger.info(f"Found {len(files_to_index)} files to scan")

            # Track statistics
            files_indexed = 0
            files_unchanged = 0
            total_chunks = 0

            for file_path in files_to_index:
                try:
                    # Check if file has changed
                    file_hash = await self._hash_file(file_path)
                    existing = await conn.fetchrow(
                        "SELECT file_hash FROM self_codebase_index WHERE file_path = $1",
                        str(file_path)
                    )

                    if existing and existing['file_hash'] == file_hash:
                        files_unchanged += 1
                        continue

                    # Parse and index the file
                    chunks_created = await self._index_file(conn, file_path, file_hash)
                    files_indexed += 1
                    total_chunks += chunks_created

                except Exception as e:
                    logger.error(f"Failed to index {file_path}: {e}")
                    continue

            # Record metrics
            await conn.execute("""
                INSERT INTO self_health_metrics (metric_name, metric_value, metadata)
                VALUES
                    ('codebase_indexer_files_indexed', $1, $2::jsonb),
                    ('codebase_indexer_chunks_created', $3, $4::jsonb)
            """,
                float(files_indexed),
                json.dumps({"cycle_time": datetime.now(timezone.utc).isoformat()}),
                float(total_chunks),
                json.dumps({"files_unchanged": files_unchanged})
            )

            await conn.close()

            logger.info(f"✅ Codebase Indexer completed: {files_indexed} files indexed, "
                       f"{files_unchanged} unchanged, {total_chunks} chunks created")

        except Exception as e:
            logger.error(f"❌ Codebase Indexer cycle failed: {e}", exc_info=True)

            # Try to record the failure
            try:
                conn = await asyncpg.connect(self.db_url)
                await conn.execute("""
                    INSERT INTO self_detected_issues
                    (issue_type, severity, source, title, description, related_worker)
                    VALUES ($1, $2, $3, $4, $5, $6)
                """,
                    "worker_failure", "critical", "codebase_indexer",
                    "Codebase Indexer cycle failed",
                    str(e), "codebase_indexer"
                )
                await conn.close()
            except:
                pass

    async def _scan_files(self) -> List[Path]:
        """Scan for Python files and key config files"""
        files = []

        # Python files in src/
        src_path = Path(self.src_root) / "src"
        if src_path.exists():
            files.extend(src_path.rglob("*.py"))

        # Key config files in project root
        root = Path(self.src_root)
        for pattern in ["*.env", "*.toml", "*.yaml", "*.yml", "requirements*.txt"]:
            files.extend(root.glob(pattern))

        # System service file (if readable)
        service_file = Path("/etc/systemd/system/tower-echo-brain.service")
        if service_file.exists() and os.access(service_file, os.R_OK):
            files.append(service_file)

        return sorted(files)

    async def _hash_file(self, file_path: Path) -> str:
        """Compute SHA256 hash of file contents"""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()

    async def _index_file(self, conn: asyncpg.Connection, file_path: Path, file_hash: str) -> int:
        """Parse and index a single file"""
        # Read file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Skip binary files
            return 0

        lines = content.splitlines()
        line_count = len(lines)

        # Parse Python files for structure
        functions = []
        classes = []
        imports = []

        if file_path.suffix == '.py' and content.strip():
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        func_info = {
                            "name": node.name,
                            "line_start": node.lineno,
                            "line_end": node.end_lineno or node.lineno,
                            "docstring": ast.get_docstring(node) or "",
                            "args": [arg.arg for arg in node.args.args]
                        }
                        functions.append(func_info)
                    elif isinstance(node, ast.ClassDef):
                        methods = [n.name for n in ast.walk(node) if isinstance(n, ast.FunctionDef)]
                        class_info = {
                            "name": node.name,
                            "line_start": node.lineno,
                            "line_end": node.end_lineno or node.lineno,
                            "docstring": ast.get_docstring(node) or "",
                            "methods": methods
                        }
                        classes.append(class_info)
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.append(node.module)
            except SyntaxError as e:
                logger.warning(f"Syntax error parsing {file_path}: {e}")

        # Chunk the content
        chunks = self._chunk_text(content, file_path)

        # Embed and store chunks in Qdrant
        point_ids = []
        for chunk in chunks:
            point_id = await self._embed_and_store(chunk, file_path)
            if point_id:
                point_ids.append(point_id)

        # Store metadata in database
        relative_path = str(file_path.relative_to(self.src_root) if file_path.is_relative_to(self.src_root) else file_path)

        await conn.execute("""
            INSERT INTO self_codebase_index
            (file_path, file_hash, language, line_count, functions, classes, imports, qdrant_point_ids)
            VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb, $7::jsonb, $8::jsonb)
            ON CONFLICT (file_path)
            DO UPDATE SET
                file_hash = EXCLUDED.file_hash,
                line_count = EXCLUDED.line_count,
                functions = EXCLUDED.functions,
                classes = EXCLUDED.classes,
                imports = EXCLUDED.imports,
                qdrant_point_ids = EXCLUDED.qdrant_point_ids,
                last_indexed_at = NOW()
        """,
            relative_path, file_hash, 'python' if file_path.suffix == '.py' else 'config',
            line_count, json.dumps(functions), json.dumps(classes),
            json.dumps(list(set(imports))), json.dumps(point_ids)
        )

        return len(chunks)

    def _chunk_text(self, text: str, file_path: Path) -> List[Dict[str, Any]]:
        """Chunk text into overlapping segments"""
        chunks = []
        lines = text.splitlines()

        # Simple line-based chunking (more predictable than token-based)
        lines_per_chunk = 30
        overlap_lines = 5

        for i in range(0, len(lines), lines_per_chunk - overlap_lines):
            chunk_lines = lines[i:i + lines_per_chunk]
            if not chunk_lines:
                continue

            chunk_text = '\n'.join(chunk_lines)

            chunks.append({
                "text": chunk_text,
                "file_path": str(file_path),
                "line_start": i + 1,
                "line_end": min(i + lines_per_chunk, len(lines))
            })

        return chunks

    async def _embed_and_store(self, chunk: Dict[str, Any], file_path: Path) -> Optional[str]:
        """Embed a text chunk and store in Qdrant"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Get embedding from Ollama
                embed_response = await client.post(
                    f"{self.ollama_url}/api/embed",
                    json={
                        "model": "mxbai-embed-large:latest",
                        "input": chunk["text"]
                    }
                )

                if embed_response.status_code != 200:
                    logger.error(f"Embedding failed: {embed_response.text}")
                    return None

                embedding = embed_response.json()["embeddings"][0]

                # Generate unique point ID
                point_id = str(uuid.uuid4())

                # Store in Qdrant
                relative_path = str(file_path.relative_to(self.src_root) if file_path.is_relative_to(self.src_root) else file_path)

                payload = {
                    "source": "self_codebase",
                    "file_path": relative_path,
                    "chunk_type": "code",
                    "line_start": chunk["line_start"],
                    "line_end": chunk["line_end"],
                    "indexed_at": datetime.now(timezone.utc).isoformat(),
                    "content": chunk["text"][:500]  # Store preview for debugging
                }

                # Extract function/class names in this chunk
                if file_path.suffix == '.py':
                    # Simple regex-based extraction for chunk metadata
                    import re
                    func_pattern = r'^def\s+(\w+)\s*\('
                    class_pattern = r'^class\s+(\w+)'

                    func_names = re.findall(func_pattern, chunk["text"], re.MULTILINE)
                    class_names = re.findall(class_pattern, chunk["text"], re.MULTILINE)

                    if func_names:
                        payload["functions"] = func_names
                    if class_names:
                        payload["classes"] = class_names

                upsert_response = await client.put(
                    f"{self.qdrant_url}/collections/{self.collection}/points",
                    json={
                        "points": [{
                            "id": point_id,
                            "vector": embedding,
                            "payload": payload
                        }]
                    }
                )

                if upsert_response.status_code == 200:
                    return point_id
                else:
                    logger.error(f"Qdrant upsert failed: {upsert_response.text}")
                    return None

        except Exception as e:
            logger.error(f"Failed to embed/store chunk: {e}")
            return None