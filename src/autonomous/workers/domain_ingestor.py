"""
Domain Ingestor Worker - Echo Brain Phase 2b
Ingests domain knowledge from all Tower ecosystem sources.
Handles: code, JSON workflows, markdown, database records, git history, conversations.
Categories: anime:*, tower:*, external:*
"""
import asyncio
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

import asyncpg
import httpx


# ============================================================
# Configuration: What to ingest and where
# ============================================================

INGESTION_SOURCES = {
    "anime:comfyui": {
        "paths": [
            "/opt/tower-anime-production/workflows/comfyui/",
            "/mnt/1TB-storage/workflows/",
            "/mnt/1TB-storage/ComfyUI/user/default/workflows/",
        ],
        "extensions": [".json", ".py"],
    },
    "anime:lora": {
        "paths": [
            "/opt/tower-anime-production/training/",
            "/opt/tower-echo-brain/scripts/lora/",
        ],
        "extensions": [".py", ".yaml", ".json", ".md", ".sh"],
    },
    "anime:storyline": {
        "paths": [
            "/opt/tower-anime-production/projects/",
            "/opt/tower-anime-production/stories/",
            "/opt/tower-anime-production/docs/",
        ],
        "extensions": [".md", ".txt", ".json"],
    },
    "anime:pipeline": {
        "paths": [
            "/opt/tower-anime-production/api/",
            "/opt/tower-anime-production/src/",
        ],
        "extensions": [".py", ".ts", ".vue"],
    },
    "tower:architecture": {
        "paths": [
            "/opt/tower-echo-brain/docs/",
            "/opt/tower-echo-brain/DEPLOYMENT.md",
            "/opt/tower-echo-brain/README.md",
            "/opt/tower-echo-brain/ARCHITECTURE.md",
        ],
        "extensions": [".md", ".yaml", ".yml", ".toml"],
    },
    "tower:services": {
        "paths": ["/etc/systemd/system/"],
        "extensions": [".service"],
        "pattern": "tower-*",
    },
}

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "mxbai-embed-large:latest")  # Must match collection dimension (1024)
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "echo_memory")
DB_URL = os.getenv("DATABASE_URL", "postgresql://echo:echo_secure_password_123@localhost/echo_brain")

MAX_CHUNK_SIZE = 1500    # ~tokens (chars / 4)
CHUNK_OVERLAP = 200


class DomainIngestor:
    """Ingests domain knowledge into Qdrant with categorization, chunking, and dedup."""

    def __init__(self):
        self.stats = {"files_processed": 0, "chunks_created": 0, "vectors_stored": 0,
                      "skipped_unchanged": 0, "errors": 0}

    async def run_cycle(self):
        """Main worker cycle — scan all sources and ingest new/changed content."""
        print(f"[DomainIngestor] Starting cycle at {datetime.now().isoformat()}")
        self.stats = {k: 0 for k in self.stats}

        conn = await asyncpg.connect(DB_URL)
        try:
            for category, config in INGESTION_SOURCES.items():
                await self._ingest_category(conn, category, config)

            await self._ingest_database_records(conn)
            await self._ingest_git_history(conn)
            await self._update_category_stats(conn)

            print(f"[DomainIngestor] Cycle complete: {json.dumps(self.stats)}")
        except Exception as e:
            print(f"[DomainIngestor] ERROR: {e}")
            self.stats["errors"] += 1
        finally:
            await conn.close()

    # --- File ingestion ---

    async def _ingest_category(self, conn, category: str, config: dict):
        extensions = set(config.get("extensions", []))
        pattern = config.get("pattern")

        for path_str in config.get("paths", []):
            path = Path(path_str)
            if path.is_dir():
                files = []
                for ext in extensions:
                    if pattern:
                        files.extend(path.glob(f"{pattern}{ext}"))
                    else:
                        files.extend(path.rglob(f"*{ext}"))
            elif path.is_file():
                files = [path]
            else:
                continue

            for filepath in files:
                if not filepath.is_file():
                    continue
                skip = {"node_modules", "__pycache__", ".git", "venv", ".venv", "dist"}
                if any(p in skip for p in filepath.parts):
                    continue
                await self._ingest_file(conn, filepath, category)

    async def _ingest_file(self, conn, filepath: Path, category: str):
        try:
            content = filepath.read_text(encoding="utf-8", errors="ignore")
            if not content.strip():
                return

            content_hash = hashlib.sha256(content.encode()).hexdigest()

            # Dedup check
            existing = await conn.fetchrow(
                "SELECT id FROM domain_ingestion_log WHERE source_path = $1 AND content_hash = $2",
                str(filepath), content_hash)
            if existing:
                self.stats["skipped_unchanged"] += 1
                return

            # Delete old vectors if file changed
            old = await conn.fetch(
                "SELECT vector_ids FROM domain_ingestion_log WHERE source_path = $1", str(filepath))
            for r in old:
                if r["vector_ids"]:
                    await self._delete_vectors(r["vector_ids"])
            await conn.execute("DELETE FROM domain_ingestion_log WHERE source_path = $1", str(filepath))

            # Chunk and embed
            chunks = self._chunk_content(content, filepath, category)
            vector_ids = []
            for chunk in chunks:
                vid = await self._embed_and_store(chunk, category, str(filepath))
                if vid:
                    vector_ids.append(vid)

            await conn.execute("""
                INSERT INTO domain_ingestion_log
                    (source_type, source_path, category, content_hash, chunk_count, vector_ids,
                     file_size_bytes, last_modified)
                VALUES ('file', $1, $2, $3, $4, $5, $6, $7)
            """, str(filepath), category, content_hash, len(vector_ids), vector_ids,
                filepath.stat().st_size, datetime.fromtimestamp(filepath.stat().st_mtime))

            self.stats["files_processed"] += 1
            self.stats["chunks_created"] += len(chunks)
            self.stats["vectors_stored"] += len(vector_ids)

        except Exception as e:
            print(f"[DomainIngestor] Error {filepath}: {e}")
            self.stats["errors"] += 1

    # --- Database record ingestion ---

    async def _ingest_database_records(self, conn):
        """Ingest generation profiles, characters, models, scenes from anime DB."""
        anime_urls = [
            "postgresql://patrick:tower_echo_brain_secret_key_2025@localhost/anime_production",
            "postgresql://patrick:RP78eIrW7cI2jYvL5akt1yurE@localhost/anime_production",
            "postgresql://patrick:tower_echo_brain_secret_key_2025@localhost/tower_anime",
        ]

        anime_conn = None
        for url in anime_urls:
            try:
                anime_conn = await asyncpg.connect(url)
                break
            except Exception:
                continue

        if not anime_conn:
            print("[DomainIngestor] Cannot connect to anime database, skipping")
            return

        try:
            # Generation profiles
            try:
                profiles = await anime_conn.fetch("""
                    SELECT p.name, p.sampler, p.steps, p.cfg_scale, p.width, p.height,
                           m.model_path as checkpoint, l.model_path as lora, p.lora_strength
                    FROM generation_profiles p
                    LEFT JOIN ai_models m ON p.checkpoint_id = m.id
                    LEFT JOIN ai_models l ON p.lora_id = l.id
                """)
                for p in profiles:
                    text = (f"Generation Profile: {p['name']}\n"
                            f"Checkpoint: {p.get('checkpoint', 'none')}\n"
                            f"LoRA: {p.get('lora', 'none')} (strength: {p.get('lora_strength', 'N/A')})\n"
                            f"Sampler: {p.get('sampler')}, Steps: {p.get('steps')}, CFG: {p.get('cfg_scale')}\n"
                            f"Resolution: {p.get('width', '?')}x{p.get('height', '?')}")
                    await self._store_db_record(conn, text, "anime:generation",
                                                f"db:profiles:{p['name']}", {"profile": p["name"]})
            except Exception as e:
                print(f"[DomainIngestor] Profiles error: {e}")

            # Characters
            try:
                chars = await anime_conn.fetch("""
                    SELECT c.name, c.description, c.traits, p.name as project_name,
                           m.model_path as lora_path
                    FROM characters c
                    LEFT JOIN projects p ON c.project_id = p.id
                    LEFT JOIN ai_models m ON c.lora_id = m.id
                """)
                for c in chars:
                    text = (f"Character: {c['name']}\nProject: {c.get('project_name', '?')}\n"
                            f"Description: {c.get('description', 'N/A')}\n"
                            f"Traits: {c.get('traits', 'N/A')}\nLoRA: {c.get('lora_path', 'none')}")
                    await self._store_db_record(conn, text, "anime:storyline",
                                                f"db:chars:{c['name']}", {"character": c["name"]})
            except Exception as e:
                print(f"[DomainIngestor] Characters error: {e}")

            # AI Models inventory
            try:
                models = await anime_conn.fetch(
                    "SELECT model_name, model_path, model_type, base_model, description FROM ai_models")
                for m in models:
                    text = (f"AI Model: {m['model_name']}\nType: {m.get('model_type', '?')}\n"
                            f"Path: {m.get('model_path', 'N/A')}\nBase: {m.get('base_model', 'N/A')}\n"
                            f"Description: {m.get('description', 'N/A')}")
                    await self._store_db_record(conn, text, "anime:safetensors",
                                                f"db:models:{m['model_name']}", {"model": m["model_name"]})
            except Exception as e:
                print(f"[DomainIngestor] Models error: {e}")

            # Scenes
            try:
                scenes = await anime_conn.fetch("""
                    SELECT s.title, s.description, s.visual_description, p.name as project_name
                    FROM scenes s LEFT JOIN projects p ON s.project_id = p.id
                """)
                for s in scenes:
                    text = (f"Scene: {s['title']}\nProject: {s.get('project_name', '?')}\n"
                            f"Description: {s.get('description', 'N/A')}\n"
                            f"Visual: {s.get('visual_description', 'N/A')}")
                    await self._store_db_record(conn, text, "anime:storyline",
                                                f"db:scenes:{s['title']}", {"scene": s["title"]})
            except Exception as e:
                print(f"[DomainIngestor] Scenes error: {e}")

        finally:
            await anime_conn.close()

    async def _store_db_record(self, conn, text, category, source_path, metadata):
        content_hash = hashlib.sha256(text.encode()).hexdigest()
        existing = await conn.fetchrow(
            "SELECT id FROM domain_ingestion_log WHERE source_path = $1 AND content_hash = $2",
            source_path, content_hash)
        if existing:
            return
        # Delete old version
        old = await conn.fetch(
            "SELECT vector_ids FROM domain_ingestion_log WHERE source_path = $1", source_path)
        for r in old:
            if r["vector_ids"]:
                await self._delete_vectors(r["vector_ids"])
        await conn.execute("DELETE FROM domain_ingestion_log WHERE source_path = $1", source_path)

        vid = await self._embed_and_store({"text": text, "metadata": metadata}, category, source_path)
        if vid:
            await conn.execute("""
                INSERT INTO domain_ingestion_log
                    (source_type, source_path, category, content_hash, chunk_count, vector_ids)
                VALUES ('database', $1, $2, $3, 1, $4)
            """, source_path, category, content_hash, [vid])

    # --- Git history ingestion ---

    async def _ingest_git_history(self, conn):
        repos = [
            ("/opt/tower-echo-brain", "tower:git_history"),
            ("/opt/tower-anime-production", "anime:pipeline"),
        ]
        for repo_path, category in repos:
            if not Path(repo_path).joinpath(".git").exists():
                continue
            try:
                result = subprocess.run(
                    ["git", "-C", repo_path, "log", "--pretty=format:%H|||%ai|||%s|||%b", "-200"],
                    capture_output=True, text=True, timeout=30)
                if result.returncode != 0:
                    continue

                for line in result.stdout.strip().split("\n"):
                    if not line.strip():
                        continue
                    parts = line.split("|||", 3)
                    if len(parts) < 3:
                        continue
                    commit_hash, date, subject = parts[0], parts[1], parts[2]
                    body = parts[3].strip() if len(parts) > 3 else ""
                    text = f"Git commit ({Path(repo_path).name}): {subject}"
                    if body:
                        text += f"\n{body}"
                    text += f"\nDate: {date}, Hash: {commit_hash[:8]}"

                    source_path = f"git:{repo_path}:{commit_hash[:8]}"
                    existing = await conn.fetchrow(
                        "SELECT id FROM domain_ingestion_log WHERE source_path = $1", source_path)
                    if existing:
                        continue

                    content_hash = hashlib.sha256(text.encode()).hexdigest()
                    vid = await self._embed_and_store(
                        {"text": text, "metadata": {"commit": commit_hash[:8]}},
                        category, source_path)
                    if vid:
                        await conn.execute("""
                            INSERT INTO domain_ingestion_log
                                (source_type, source_path, category, content_hash, chunk_count, vector_ids)
                            VALUES ('git', $1, $2, $3, 1, $4)
                        """, source_path, category, content_hash, [vid])
                        self.stats["vectors_stored"] += 1
            except Exception as e:
                print(f"[DomainIngestor] Git error {repo_path}: {e}")

    # --- Chunking ---

    def _chunk_content(self, content: str, filepath: Path, category: str) -> List[dict]:
        ext = filepath.suffix.lower()
        if ext == ".json":
            return self._chunk_json(content, filepath)
        elif ext == ".py":
            return self._chunk_python(content, filepath)
        elif ext in (".md", ".txt"):
            return self._chunk_markdown(content, filepath)
        else:
            return self._chunk_by_size(f"Source: {filepath}\n{content}", filepath)

    def _chunk_json(self, content: str, filepath: Path) -> List[dict]:
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return self._chunk_by_size(content, filepath)

        # Detect ComfyUI workflow
        if isinstance(data, dict) and any(
            isinstance(v, dict) and "class_type" in v for v in data.values()
        ):
            nodes = []
            for nid, node in data.items():
                if isinstance(node, dict) and "class_type" in node:
                    inputs = {k: v for k, v in node.get("inputs", {}).items()
                             if not isinstance(v, list) and v is not None}
                    nodes.append(f"  Node {nid}: {node['class_type']} — {json.dumps(inputs, default=str)[:200]}")
            text = f"ComfyUI Workflow: {filepath.name}\nNodes: {len(nodes)}\n" + "\n".join(nodes[:50])
            if len(text) > MAX_CHUNK_SIZE * 4:
                return self._chunk_by_size(text, filepath)
            return [{"text": text, "metadata": {"filename": filepath.name, "type": "comfyui_workflow"}}]

        pretty = json.dumps(data, indent=2, default=str)
        return self._chunk_by_size(f"JSON: {filepath.name}\n{pretty}", filepath)

    def _chunk_python(self, content: str, filepath: Path) -> List[dict]:
        chunks = []
        pattern = re.compile(r'^(class |def |async def )', re.MULTILINE)
        boundaries = [(m.start(), m.group()) for m in pattern.finditer(content)]

        if not boundaries:
            return self._chunk_by_size(f"Python: {filepath}\n{content}", filepath)

        for i, (start, _) in enumerate(boundaries):
            end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(content)
            block = content[start:end].strip()
            if len(block) < 50:
                continue
            text = f"From {filepath}:\n{block}"
            if len(text) > MAX_CHUNK_SIZE * 4:
                chunks.extend(self._chunk_by_size(text, filepath))
            else:
                chunks.append({"text": text, "metadata": {
                    "filename": filepath.name, "definition": block.split("\n")[0][:100]}})
        return chunks or self._chunk_by_size(f"Python: {filepath}\n{content}", filepath)

    def _chunk_markdown(self, content: str, filepath: Path) -> List[dict]:
        chunks = []
        sections = re.split(r'^(#{1,3}\s+.+)$', content, flags=re.MULTILINE)
        header = f"Document: {filepath.name}"
        text = ""
        for section in sections:
            if re.match(r'^#{1,3}\s+', section):
                if text.strip():
                    chunks.append({"text": f"{header}\n{text.strip()}",
                                  "metadata": {"filename": filepath.name, "section": header}})
                header = section.strip()
                text = ""
            else:
                text += section
        if text.strip():
            chunks.append({"text": f"{header}\n{text.strip()}",
                          "metadata": {"filename": filepath.name, "section": header}})
        return chunks or [{"text": f"{filepath.name}\n{content[:MAX_CHUNK_SIZE * 4]}",
                          "metadata": {"filename": filepath.name}}]

    def _chunk_by_size(self, content: str, filepath) -> List[dict]:
        chunks = []
        limit = MAX_CHUNK_SIZE * 4
        overlap = CHUNK_OVERLAP * 4
        name = filepath.name if isinstance(filepath, Path) else str(filepath)
        for i in range(0, len(content), limit - overlap):
            chunk = content[i:i + limit]
            if chunk.strip():
                chunks.append({"text": chunk, "metadata": {"filename": name, "offset": i}})
        return chunks

    # --- Embedding and storage ---

    async def _embed_and_store(self, chunk: dict, category: str, source_path: str) -> Optional[str]:
        text = chunk["text"]
        metadata = chunk.get("metadata", {})
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(f"{OLLAMA_URL}/api/embeddings",
                                        json={"model": EMBED_MODEL, "prompt": text})
                if resp.status_code != 200:
                    return None
                embedding = resp.json().get("embedding")
                if not embedding:
                    return None

            point_id = str(uuid4())
            payload = {"text": text[:10000], "category": category, "source": source_path,
                       "ingested_at": datetime.now().isoformat(), **metadata}

            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.put(f"{QDRANT_URL}/collections/{COLLECTION}/points",
                    json={"points": [{"id": point_id, "vector": embedding, "payload": payload}]})
                if resp.status_code not in (200, 201):
                    return None
            return point_id
        except Exception as e:
            print(f"[DomainIngestor] Embed error: {e}")
            return None

    async def _delete_vectors(self, vector_ids: list):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                await client.post(f"{QDRANT_URL}/collections/{COLLECTION}/points/delete",
                                 json={"points": vector_ids})
        except Exception:
            pass

    async def _update_category_stats(self, conn):
        stats = await conn.fetch("""
            SELECT category, COUNT(*) as docs, SUM(chunk_count) as vecs,
                   SUM(file_size_bytes) as bytes, MAX(ingested_at) as last
            FROM domain_ingestion_log WHERE status = 'completed' GROUP BY category
        """)
        for r in stats:
            await conn.execute("""
                INSERT INTO domain_category_stats (category, total_documents, total_vectors, total_bytes, last_ingested_at)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (category) DO UPDATE SET
                    total_documents=$2, total_vectors=$3, total_bytes=$4,
                    last_ingested_at=$5, last_refreshed_at=NOW()
            """, r["category"], r["docs"], r["vecs"] or 0, r["bytes"] or 0, r["last"])