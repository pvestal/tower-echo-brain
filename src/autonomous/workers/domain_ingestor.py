"""
Domain Ingestor Worker - Echo Brain Phase 2b
Ingests domain knowledge from all Tower ecosystem sources.
Handles: code, JSON workflows, markdown, database records, git history, conversations.
Categories: anime:*, tower:*, external:*
"""
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Optional
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
    "anime:lora-studio": {
        "paths": [
            "/opt/tower-anime-production/training/lora-studio/src/",
            "/opt/tower-anime-production/training/lora-studio/packages/",
        ],
        "extensions": [".py", ".ts", ".vue", ".json", ".md"],
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
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")  # Must match collection dimension (768)
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
                vid = await self._embed_and_store(chunk, category, str(filepath), "domain_code")
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
        # Get DB password from environment (set by service/Vault)
        db_password = os.getenv("PGPASSWORD", os.getenv("DB_PASSWORD", ""))
        if not db_password:
            print("[DomainIngestor] Warning: No database password in environment")
            return {}

        anime_urls = [
            f"postgresql://patrick:{db_password}@localhost/anime_production",
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
                    "SELECT model_name, model_path, model_type, base_model, status FROM ai_models")
                for m in models:
                    text = (f"AI Model: {m['model_name']}\nType: {m.get('model_type', '?')}\n"
                            f"Path: {m.get('model_path', 'N/A')}\nBase: {m.get('base_model', 'N/A')}\n"
                            f"Status: {m.get('status', 'N/A')}")
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

            # Approved generation results (keyframes with QC scores and generation metadata)
            try:
                approved_gens = await anime_conn.fetch("""
                    SELECT gh.id, gh.character_slug, gh.project_name, gh.prompt,
                           gh.checkpoint_model, gh.cfg_scale, gh.steps, gh.sampler,
                           gh.width, gh.height, gh.quality_score, gh.character_match,
                           gh.clarity, gh.training_value, gh.generation_type,
                           gh.seed, gh.generated_at,
                           a.auto_approved, a.quality_score as approval_score,
                           a.checkpoint_model as approval_checkpoint
                    FROM generation_history gh
                    LEFT JOIN approvals a ON a.generation_history_id = gh.id
                    WHERE gh.status = 'approved'
                    ORDER BY gh.generated_at DESC
                    LIMIT 2000
                """)
                for g in approved_gens:
                    gen_id = g['id']
                    char = g.get('character_slug') or 'unknown'
                    project = g.get('project_name') or 'unknown'
                    q_score = g.get('quality_score')
                    prompt_preview = (g.get('prompt') or '')[:300]

                    text = (f"Approved Generation Result #{gen_id}\n"
                            f"Project: {project}\nCharacter: {char}\n"
                            f"Type: {g.get('generation_type', 'image')}\n"
                            f"Checkpoint: {g.get('checkpoint_model', 'N/A')}\n"
                            f"Settings: {g.get('sampler', '?')} / {g.get('steps', '?')} steps / "
                            f"CFG {g.get('cfg_scale', '?')} / {g.get('width', '?')}x{g.get('height', '?')}\n"
                            f"Quality Score: {q_score if q_score is not None else 'N/A'}\n"
                            f"Character Match: {g.get('character_match') if g.get('character_match') is not None else 'N/A'}\n"
                            f"Clarity: {g.get('clarity') if g.get('clarity') is not None else 'N/A'}\n"
                            f"Training Value: {g.get('training_value') if g.get('training_value') is not None else 'N/A'}\n"
                            f"Auto-Approved: {g.get('auto_approved', False)}\n"
                            f"Seed: {g.get('seed', 'N/A')}\n"
                            f"Generated: {g.get('generated_at', 'N/A')}\n"
                            f"Prompt: {prompt_preview}")

                    metadata = {
                        "generation_id": gen_id,
                        "character": char,
                        "project": project,
                        "generation_type": g.get('generation_type', 'image'),
                        "checkpoint": g.get('checkpoint_model'),
                        "quality_score": q_score,
                        "character_match": g.get('character_match'),
                        "auto_approved": g.get('auto_approved', False),
                    }
                    # Remove None values from metadata
                    metadata = {k: v for k, v in metadata.items() if v is not None}

                    await self._store_db_record(conn, text, "anime:generation",
                                                f"db:approved_gen:{gen_id}", metadata)
                count = len(approved_gens)
                print(f"[DomainIngestor] Processed {count} approved generation results")
            except Exception as e:
                print(f"[DomainIngestor] Approved generations error: {e}")

            # Standalone approvals (images approved via approvals table, no generation_history link)
            try:
                standalone_approvals = await anime_conn.fetch("""
                    SELECT a.id, a.character_slug, a.project_name, a.image_name,
                           a.quality_score, a.auto_approved, a.vision_review,
                           a.checkpoint_model, a.created_at
                    FROM approvals a
                    WHERE a.generation_history_id IS NULL
                    ORDER BY a.created_at DESC
                    LIMIT 2000
                """)
                for a in standalone_approvals:
                    approval_id = a['id']
                    char = a.get('character_slug') or 'unknown'
                    project = a.get('project_name') or 'unknown'
                    q_score = a.get('quality_score')
                    vision = a.get('vision_review')
                    vision_summary = ''
                    if vision and isinstance(vision, dict):
                        vision_summary = f"\nVision Review: {json.dumps(vision, default=str)[:300]}"

                    text = (f"Approved Image #{approval_id}\n"
                            f"Project: {project}\nCharacter: {char}\n"
                            f"Image: {a.get('image_name', 'N/A')}\n"
                            f"Checkpoint: {a.get('checkpoint_model', 'N/A')}\n"
                            f"Quality Score: {q_score if q_score is not None else 'N/A'}\n"
                            f"Auto-Approved: {a.get('auto_approved', False)}\n"
                            f"Approved: {a.get('created_at', 'N/A')}"
                            f"{vision_summary}")

                    metadata = {
                        "approval_id": approval_id,
                        "character": char,
                        "project": project,
                        "checkpoint": a.get('checkpoint_model'),
                        "quality_score": q_score,
                        "auto_approved": a.get('auto_approved', False),
                    }
                    metadata = {k: v for k, v in metadata.items() if v is not None}

                    await self._store_db_record(conn, text, "anime:generation",
                                                f"db:approval:{approval_id}", metadata)
                count = len(standalone_approvals)
                print(f"[DomainIngestor] Processed {count} standalone approvals")
            except Exception as e:
                print(f"[DomainIngestor] Standalone approvals error: {e}")

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

        vid = await self._embed_and_store({"text": text, "metadata": metadata}, category, source_path, "domain_record")
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
                        category, source_path, "domain_git")
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

    async def _embed_and_store(self, chunk: dict, category: str, source_path: str,
                              source_type: str = "domain_code") -> Optional[str]:
        text = chunk["text"]
        metadata = chunk.get("metadata", {})
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(f"{OLLAMA_URL}/api/embeddings",
                                        json={"model": EMBED_MODEL, "prompt": text})
                if resp.status_code != 200:
                    return None
                data = resp.json()
                embedding = data.get("embedding", [])
                if not embedding:
                    return None

            point_id = str(uuid4())
            payload = {"text": text[:10000], "type": source_type, "category": category,
                       "source": source_path, "ingested_at": datetime.now().isoformat(),
                       **metadata}

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

    # --- Backfill: add type field to existing untyped points ---

    async def backfill_type_field(self):
        """One-time backfill: set 'type' on domain ingestor points that lack it.

        Scrolls all points with a 'source' field (domain ingestor signature),
        skips those already typed, and sets 'type' based on source pattern:
          - source starts with 'git:' → domain_git
          - source starts with 'db:'  → domain_record
          - otherwise                 → domain_code
        """
        print("[DomainIngestor] Starting type field backfill...")
        batch_size = 250
        updated = 0
        skipped = 0
        errors = 0
        offset = None

        async with httpx.AsyncClient(timeout=60.0) as client:
            while True:
                scroll_body = {
                    "limit": batch_size,
                    "with_payload": {"include": ["source", "type"]},
                    "with_vector": False,
                    "filter": {
                        "must": [
                            {"is_empty": {"key": "type"}},
                        ],
                        "must_not": [
                            {"is_empty": {"key": "source"}},
                        ],
                    },
                }
                if offset:
                    scroll_body["offset"] = offset

                resp = await client.post(
                    f"{QDRANT_URL}/collections/{COLLECTION}/points/scroll",
                    json=scroll_body,
                )
                if resp.status_code != 200:
                    # Fallback: if is_empty filter not supported, scroll without filter
                    print(f"[DomainIngestor] Filter scroll failed ({resp.status_code}), trying unfiltered...")
                    return await self._backfill_unfiltered(client)

                result = resp.json().get("result", {})
                points = result.get("points", [])
                next_offset = result.get("next_page_offset")

                if not points:
                    break

                # Group by derived type
                type_groups = {"domain_code": [], "domain_record": [], "domain_git": []}
                for pt in points:
                    payload = pt.get("payload") or {}
                    source = payload.get("source", "")
                    if not source:
                        skipped += 1
                        continue
                    if payload.get("type"):
                        skipped += 1
                        continue
                    if source.startswith("git:"):
                        dtype = "domain_git"
                    elif source.startswith("db:"):
                        dtype = "domain_record"
                    else:
                        dtype = "domain_code"
                    type_groups[dtype].append(pt["id"])

                # Batch set_payload per type
                for dtype, ids in type_groups.items():
                    if not ids:
                        continue
                    set_resp = await client.post(
                        f"{QDRANT_URL}/collections/{COLLECTION}/points/payload",
                        json={"payload": {"type": dtype}, "points": ids},
                    )
                    if set_resp.status_code in (200, 201):
                        updated += len(ids)
                    else:
                        errors += len(ids)
                        print(f"[DomainIngestor] set_payload error: {set_resp.status_code}")

                if updated % 5000 < batch_size:
                    print(f"[DomainIngestor] Backfill progress: {updated} updated, {skipped} skipped, {errors} errors")

                if not next_offset:
                    break
                offset = next_offset

        print(f"[DomainIngestor] Backfill complete: {updated} updated, {skipped} skipped, {errors} errors")
        return {"updated": updated, "skipped": skipped, "errors": errors}

    async def _backfill_unfiltered(self, client):
        """Fallback backfill that scrolls all points and filters client-side."""
        batch_size = 250
        updated = 0
        skipped = 0
        errors = 0
        offset = None

        while True:
            scroll_body = {
                "limit": batch_size,
                "with_payload": {"include": ["source", "type"]},
                "with_vector": False,
            }
            if offset:
                scroll_body["offset"] = offset

            resp = await client.post(
                f"{QDRANT_URL}/collections/{COLLECTION}/points/scroll",
                json=scroll_body,
            )
            if resp.status_code != 200:
                print(f"[DomainIngestor] Scroll error: {resp.status_code}")
                break

            result = resp.json().get("result", {})
            points = result.get("points", [])
            next_offset = result.get("next_page_offset")

            if not points:
                break

            type_groups = {"domain_code": [], "domain_record": [], "domain_git": []}
            for pt in points:
                payload = pt.get("payload") or {}
                source = payload.get("source", "")
                if not source or payload.get("type"):
                    skipped += 1
                    continue
                if source.startswith("git:"):
                    dtype = "domain_git"
                elif source.startswith("db:"):
                    dtype = "domain_record"
                else:
                    dtype = "domain_code"
                type_groups[dtype].append(pt["id"])

            for dtype, ids in type_groups.items():
                if not ids:
                    continue
                set_resp = await client.post(
                    f"{QDRANT_URL}/collections/{COLLECTION}/points/payload",
                    json={"payload": {"type": dtype}, "points": ids},
                )
                if set_resp.status_code in (200, 201):
                    updated += len(ids)
                else:
                    errors += len(ids)

            if updated % 10000 < batch_size:
                print(f"[DomainIngestor] Backfill progress: {updated} updated, {skipped} skipped, {errors} errors")

            if not next_offset:
                break
            offset = next_offset

        print(f"[DomainIngestor] Backfill complete: {updated} updated, {skipped} skipped, {errors} errors")
        return {"updated": updated, "skipped": skipped, "errors": errors}


# --- Standalone runner ---
if __name__ == "__main__":
    import asyncio
    import sys

    async def main():
        ingestor = DomainIngestor()
        if len(sys.argv) > 1 and sys.argv[1] == "backfill":
            result = await ingestor.backfill_type_field()
            print(f"Result: {result}")
        else:
            await ingestor.run_cycle()

    asyncio.run(main())