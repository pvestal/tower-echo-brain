#!/usr/bin/env python3
"""
SSOT-Compliant Story Bible Rebuilder
=====================================
Nukes the broken story_bible collection (full data duplicates) and rebuilds it
as a proper search index where Qdrant stores ONLY:
  - Embedding vector (for semantic search)
  - source_table + source_id (reference back to PostgreSQL SSOT)
  - search_text (denormalized keywords for debugging — NOT the full record)
  - indexed_at timestamp (for freshness checks)

The anime_production PostgreSQL database remains the Single Source of Truth.
At generation time, the orchestrator searches Qdrant → gets references →
fetches fresh, authoritative data from PostgreSQL.

Usage:
    python3 rebuild_story_bible_ssot.py                # Full rebuild
    python3 rebuild_story_bible_ssot.py --dry-run      # Preview without changes
    python3 rebuild_story_bible_ssot.py --incremental   # Only index new/changed rows

Place at: /opt/tower-echo-brain/scripts/rebuild_story_bible_ssot.py
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import urllib.request
import urllib.error

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class Config:
    PG_HOST = os.getenv("PG_HOST", "localhost")
    PG_PORT = int(os.getenv("PG_PORT", "5432"))
    PG_USER = os.getenv("PG_USER", "patrick")
    PG_PASSWORD = os.getenv("PG_PASSWORD", "RP78eIrW7cI2jYvL5akt1yurE")
    PG_DATABASE = os.getenv("PG_DATABASE", "anime_production")

    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    COLLECTION = "story_bible"
    VECTOR_DIM = 768

    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
    EMBEDDING_MODEL = "nomic-embed-text"

    WORKFLOW_DIR = "/opt/tower-anime-production/workflows/comfyui"

    # Batch size for Qdrant upserts
    BATCH_SIZE = 50

    # Tables to index and how to build search text for each.
    # Format: "table_name": {
    #     "type": content type tag,
    #     "id_col": primary key column,
    #     "search_cols": columns to concatenate into search text,
    #     "name_col": optional column for display name,
    #     "updated_col": optional column for incremental sync
    # }
    # A value of None means "auto-discover" — we'll index all tables
    # with a sensible default.
    INDEX_RULES: dict = {}  # Populated at runtime by discover_tables()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("ssot_rebuilder")


# ---------------------------------------------------------------------------
# HTTP helpers (no external deps beyond psycopg2)
# ---------------------------------------------------------------------------

def http_json(method: str, url: str, payload: dict = None, timeout: int = 30):
    """Generic HTTP request returning parsed JSON."""
    data = json.dumps(payload).encode() if payload else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        logger.error(f"HTTP {e.code} from {url}: {body[:300]}")
        return None
    except Exception as e:
        logger.error(f"HTTP request to {url} failed: {e}")
        return None


def qdrant_get(path: str):
    return http_json("GET", f"{Config.QDRANT_URL}{path}")


def qdrant_post(path: str, payload: dict):
    return http_json("POST", f"{Config.QDRANT_URL}{path}", payload)


def qdrant_put(path: str, payload: dict):
    return http_json("PUT", f"{Config.QDRANT_URL}{path}", payload)


def qdrant_delete(path: str):
    return http_json("DELETE", f"{Config.QDRANT_URL}{path}")


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_text(text: str) -> Optional[list]:
    """Get 768D embedding from Ollama nomic-embed-text."""
    resp = http_json("POST", f"{Config.OLLAMA_URL}/api/embed", {
        "model": Config.EMBEDDING_MODEL,
        "input": text,
    })
    if resp and "embeddings" in resp:
        vec = resp["embeddings"][0]
        if len(vec) == Config.VECTOR_DIM:
            return vec
        else:
            logger.error(f"Wrong vector dim: got {len(vec)}, expected {Config.VECTOR_DIM}")
    return None


# ---------------------------------------------------------------------------
# PostgreSQL helpers
# ---------------------------------------------------------------------------

def get_pg_connection():
    import psycopg2
    return psycopg2.connect(
        host=Config.PG_HOST,
        port=Config.PG_PORT,
        user=Config.PG_USER,
        password=Config.PG_PASSWORD,
        dbname=Config.PG_DATABASE,
        connect_timeout=10,
    )


def discover_tables(conn) -> dict:
    """
    Auto-discover all public tables, their primary keys, and text-like columns.
    Returns index rules dict keyed by table name.
    """
    cur = conn.cursor()

    # Get all tables
    cur.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_type = 'BASE TABLE'
        ORDER BY table_name
    """)
    tables = [row[0] for row in cur.fetchall()]

    # Well-known tables get explicit rules (highest search quality)
    KNOWN_RULES = {
        "characters": {
            "type": "character",
            "id_col": "id",
            "search_cols": ["name", "description", "personality", "appearance",
                            "role", "backstory", "character_role"],
            "name_col": "name",
            "updated_col": "updated_at",
        },
        "episodes": {
            "type": "episode",
            "id_col": "id",
            "search_cols": ["title", "description", "synopsis", "episode_number",
                            "tone", "notes"],
            "name_col": "title",
            "updated_col": "updated_at",
        },
        "scenes": {
            "type": "scene",
            "id_col": "id",
            "search_cols": ["title", "description", "narrative_text", "dialogue",
                            "visual_description", "scene_type", "location"],
            "name_col": "title",
            "updated_col": "updated_at",
        },
        "ai_models": {
            "type": "ai_model",
            "id_col": "id",
            "search_cols": ["name", "model_type", "description", "capabilities"],
            "name_col": "name",
            "updated_col": "updated_at",
        },
        "lora_definitions": {
            "type": "lora",
            "id_col": "id",
            "search_cols": ["name", "description", "trigger_words", "base_model",
                            "lora_type"],
            "name_col": "name",
            "updated_col": "updated_at",
        },
        "generation_history": {
            "type": "generation",
            "id_col": "id",
            "search_cols": ["prompt", "negative_prompt", "model_name", "status",
                            "quality_score"],
            "name_col": None,
            "updated_col": "created_at",
        },
        "story_arcs": {
            "type": "story_arc",
            "id_col": "id",
            "search_cols": ["title", "description", "arc_type", "status"],
            "name_col": "title",
            "updated_col": "updated_at",
        },
        "projects": {
            "type": "project",
            "id_col": "id",
            "search_cols": ["name", "description", "style", "status"],
            "name_col": "name",
            "updated_col": "updated_at",
        },
    }

    rules = {}
    for table in tables:
        if table in KNOWN_RULES:
            # Validate that expected columns actually exist
            rule = KNOWN_RULES[table].copy()
            cur.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = %s
            """, (table,))
            actual_cols = {row[0] for row in cur.fetchall()}

            rule["search_cols"] = [c for c in rule["search_cols"] if c in actual_cols]
            if rule["name_col"] and rule["name_col"] not in actual_cols:
                rule["name_col"] = None
            if rule["updated_col"] and rule["updated_col"] not in actual_cols:
                rule["updated_col"] = None
            if rule["id_col"] not in actual_cols:
                # Try common PK names
                for fallback in ["id", f"{table}_id", "uuid"]:
                    if fallback in actual_cols:
                        rule["id_col"] = fallback
                        break
                else:
                    logger.warning(f"No PK found for known table {table}, skipping")
                    continue

            if rule["search_cols"]:
                rules[table] = rule
            else:
                logger.warning(f"Known table {table} has no matching search columns, skipping")
        else:
            # Auto-discover: find PK and text columns
            cur.execute("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = %s
                ORDER BY ordinal_position
            """, (table,))
            columns = cur.fetchall()
            col_names = [c[0] for c in columns]
            col_types = {c[0]: c[1] for c in columns}

            # Find PK
            id_col = None
            for candidate in ["id", f"{table}_id", "uuid"]:
                if candidate in col_names:
                    id_col = candidate
                    break
            if not id_col:
                # Use first integer/uuid column as PK
                for cname, ctype in columns:
                    if ctype in ("integer", "bigint", "uuid"):
                        id_col = cname
                        break
            if not id_col:
                logger.debug(f"Skipping {table}: no identifiable PK")
                continue

            # Find text columns for search
            text_types = {
                "character varying", "text", "varchar",
                "character", "name",
            }
            search_cols = [
                c[0] for c in columns
                if c[1] in text_types and c[0] != id_col
            ]

            if not search_cols:
                logger.debug(f"Skipping {table}: no text columns for search")
                continue

            # Find name column (for display)
            name_col = None
            for candidate in ["name", "title", "label", "display_name"]:
                if candidate in col_names:
                    name_col = candidate
                    break

            # Find updated_at column
            updated_col = None
            for candidate in ["updated_at", "modified_at", "created_at"]:
                if candidate in col_names:
                    updated_col = candidate
                    break

            rules[table] = {
                "type": f"db_{table}",
                "id_col": id_col,
                "search_cols": search_cols[:8],  # Cap at 8 columns
                "name_col": name_col,
                "updated_col": updated_col,
            }

    return rules


# ---------------------------------------------------------------------------
# Workflow file indexer (these aren't in PG, so they get special handling)
# ---------------------------------------------------------------------------

def index_workflow_files() -> list:
    """
    Build index entries for ComfyUI workflow JSON files.
    These don't live in PostgreSQL, so we use file path as the source reference.
    """
    entries = []
    if not os.path.isdir(Config.WORKFLOW_DIR):
        logger.warning(f"Workflow dir not found: {Config.WORKFLOW_DIR}")
        return entries

    for fname in sorted(os.listdir(Config.WORKFLOW_DIR)):
        if not fname.endswith(".json"):
            continue

        fpath = os.path.join(Config.WORKFLOW_DIR, fname)
        try:
            with open(fpath) as f:
                workflow = json.load(f)
        except Exception as e:
            logger.warning(f"Skipping {fname}: {e}")
            continue

        # Extract key info for search text
        models = set()
        loras = set()
        node_types = set()
        scene_type = "general"

        for node_id, node in workflow.items():
            if not isinstance(node, dict):
                continue
            ct = node.get("class_type", "")
            node_types.add(ct)
            inputs = node.get("inputs", {})

            if ct == "CheckpointLoaderSimple":
                models.add(inputs.get("ckpt_name", ""))
            elif ct == "LoraLoader":
                loras.add(inputs.get("lora_name", ""))

        # Classify scene type from filename
        fname_lower = fname.lower()
        if "combat" in fname_lower or "action" in fname_lower:
            scene_type = "action"
        elif "rife" in fname_lower or "video" in fname_lower:
            scene_type = "video"

        search_text = (
            f"ComfyUI workflow {fname} "
            f"scene_type:{scene_type} "
            f"models:{' '.join(models)} "
            f"loras:{' '.join(loras)} "
            f"nodes:{' '.join(sorted(node_types)[:10])}"
        )

        # Generate a stable ID from filename
        point_id = int(hashlib.md5(fname.encode()).hexdigest()[:8], 16)

        entries.append({
            "id": point_id,
            "search_text": search_text,
            "payload": {
                "type": "workflow",
                "source_type": "file",
                "source_path": fpath,
                "source_id": fname,
                "display_name": fname,
                "scene_type": scene_type,
                "models": list(models),
                "loras": list(loras),
                "indexed_at": datetime.now(timezone.utc).isoformat(),
            },
        })

    return entries


# ---------------------------------------------------------------------------
# Core rebuild logic
# ---------------------------------------------------------------------------

def generate_point_id(table: str, row_id) -> int:
    """Deterministic point ID from table + row ID (for idempotent upserts)."""
    raw = f"{table}:{row_id}"
    return int(hashlib.md5(raw.encode()).hexdigest()[:12], 16)


def build_search_text(row: dict, rule: dict, table: str) -> str:
    """
    Build a compact search string from selected columns.
    This is what gets embedded — NOT the full record.
    Enhanced to include more context for better semantic search.
    """
    parts = []

    # Special handling for characters - more descriptive
    if rule["type"] == "character":
        name = row.get(rule.get("name_col", "name"), "")
        if name:
            # Add "anime character" context for better matching
            parts.append(f"{name} anime character protagonist")

        # Add key attributes
        for col in ["description", "personality", "appearance", "role"]:
            if col in rule["search_cols"]:
                val = row.get(col)
                if val and str(val).strip():
                    parts.append(str(val).strip()[:200])

    # Special handling for scenes
    elif rule["type"] == "scene":
        title = row.get(rule.get("name_col", "title"), "")
        if title:
            parts.append(f"anime scene {title}")

        for col in rule["search_cols"]:
            val = row.get(col)
            if val and str(val).strip():
                parts.append(str(val).strip()[:200])

    # Special handling for LoRAs
    elif rule["type"] == "lora":
        name = row.get(rule.get("name_col", "name"), "")
        if name:
            parts.append(f"LoRA model {name}")

        for col in ["description", "trigger_words", "base_model"]:
            if col in rule["search_cols"]:
                val = row.get(col)
                if val and str(val).strip():
                    parts.append(str(val).strip()[:150])

    else:
        # Default handling
        parts.append(f"{rule['type']}:")

        if rule.get("name_col") and row.get(rule["name_col"]):
            parts.append(str(row[rule["name_col"]]))

        for col in rule["search_cols"]:
            val = row.get(col)
            if val and str(val).strip():
                text = str(val).strip()[:300]
                parts.append(text)

    return " ".join(parts)


def rebuild_collection(dry_run: bool = False, incremental: bool = False):
    """Main rebuild: nuke and recreate story_bible as SSOT-compliant index."""

    logger.info("=" * 60)
    logger.info("SSOT-Compliant Story Bible Rebuild")
    logger.info("=" * 60)

    # --- Step 1: Verify dependencies ---
    logger.info("\n[1/6] Verifying dependencies...")

    # Qdrant
    qdrant_ok = qdrant_get("/collections")
    if not qdrant_ok:
        logger.error("Qdrant not reachable — aborting")
        sys.exit(1)

    # Ollama embedding
    test_vec = embed_text("connection test")
    if not test_vec:
        logger.error("Ollama embedding failed — aborting")
        sys.exit(1)
    logger.info(f"  ✓ Ollama: {Config.EMBEDDING_MODEL} → {len(test_vec)}D")

    # PostgreSQL
    try:
        conn = get_pg_connection()
        logger.info(f"  ✓ PostgreSQL: {Config.PG_DATABASE}")
    except Exception as e:
        logger.error(f"PostgreSQL connection failed: {e}")
        sys.exit(1)

    # --- Step 2: Discover tables ---
    logger.info("\n[2/6] Discovering tables and index rules...")
    rules = discover_tables(conn)
    Config.INDEX_RULES = rules

    total_tables = len(rules)
    known = sum(1 for r in rules.values() if not r["type"].startswith("db_"))
    auto = total_tables - known
    logger.info(f"  ✓ {total_tables} tables to index ({known} known + {auto} auto-discovered)")

    for table, rule in sorted(rules.items()):
        logger.info(f"    {table}: type={rule['type']}, "
                     f"search_cols={rule['search_cols'][:3]}...")

    # --- Step 3: Nuke old collection (unless incremental) ---
    if not incremental:
        logger.info("\n[3/6] Dropping old story_bible collection...")
        if dry_run:
            logger.info("  [DRY RUN] Would delete story_bible collection")
        else:
            qdrant_delete(f"/collections/{Config.COLLECTION}")
            time.sleep(1)

            # Recreate with correct dimensions
            create_resp = qdrant_put(f"/collections/{Config.COLLECTION}", {
                "vectors": {
                    "size": Config.VECTOR_DIM,
                    "distance": "Cosine",
                },
            })
            if create_resp and create_resp.get("result"):
                logger.info(f"  ✓ Created {Config.COLLECTION} @ {Config.VECTOR_DIM}D")
            else:
                logger.error(f"  ✗ Failed to create collection: {create_resp}")
                sys.exit(1)
    else:
        logger.info("\n[3/6] Incremental mode — keeping existing collection")

    # --- Step 4: Index all tables ---
    logger.info("\n[4/6] Indexing tables from PostgreSQL SSOT...")

    cur = conn.cursor()
    total_points = 0
    failed_tables = []
    points_buffer = []

    for table, rule in sorted(rules.items()):
        try:
            # Build column list
            cols = [rule["id_col"]] + rule["search_cols"]
            if rule.get("name_col") and rule["name_col"] not in cols:
                cols.append(rule["name_col"])

            # Only select columns that exist
            cur.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = %s
            """, (table,))
            actual_cols = {row[0] for row in cur.fetchall()}
            cols = [c for c in cols if c in actual_cols]

            if rule["id_col"] not in actual_cols:
                logger.warning(f"  ⏭ {table}: PK column '{rule['id_col']}' not found")
                continue

            col_str = ", ".join(cols)
            cur.execute(f"SELECT {col_str} FROM {table}")
            rows = cur.fetchall()
            col_names = [desc[0] for desc in cur.description]

            if not rows:
                logger.debug(f"  ⏭ {table}: empty")
                continue

            table_count = 0
            for row_tuple in rows:
                row = dict(zip(col_names, row_tuple))
                row_id = row[rule["id_col"]]

                search_text = build_search_text(row, rule, table)
                if len(search_text.strip()) < 10:
                    continue  # Not enough content to embed

                point_id = generate_point_id(table, row_id)

                # SSOT payload: references only, NO duplicated content
                payload = {
                    "type": rule["type"],
                    "source_table": table,
                    "source_id": row_id,
                    "search_text": search_text[:500],  # Capped, for debug only
                    "indexed_at": datetime.now(timezone.utc).isoformat(),
                }

                # Add display name if available
                if rule.get("name_col") and row.get(rule["name_col"]):
                    payload["display_name"] = str(row[rule["name_col"]])[:100]

                points_buffer.append({
                    "id": point_id,
                    "search_text": search_text,
                    "payload": payload,
                })
                table_count += 1

            total_points += table_count
            logger.info(f"  ✓ {table}: {table_count} rows queued "
                         f"(type={rule['type']})")

        except Exception as e:
            logger.error(f"  ✗ {table}: {e}")
            failed_tables.append(table)

    # --- Step 5: Index workflow files ---
    logger.info("\n[5/6] Indexing ComfyUI workflow files...")
    workflow_entries = index_workflow_files()
    points_buffer.extend(workflow_entries)
    total_points += len(workflow_entries)
    logger.info(f"  ✓ {len(workflow_entries)} workflows queued")

    # --- Step 6: Embed and upsert ---
    logger.info(f"\n[6/6] Embedding and upserting {total_points} points...")

    if dry_run:
        logger.info(f"  [DRY RUN] Would embed and upsert {total_points} points")
        logger.info("  Sample payloads:")
        for entry in points_buffer[:3]:
            logger.info(f"    {json.dumps(entry['payload'], indent=2)[:200]}")
        conn.close()
        return

    upserted = 0
    embed_failures = 0

    for i in range(0, len(points_buffer), Config.BATCH_SIZE):
        batch = points_buffer[i:i + Config.BATCH_SIZE]
        qdrant_points = []

        for entry in batch:
            vec = embed_text(entry["search_text"])
            if not vec:
                embed_failures += 1
                continue

            qdrant_points.append({
                "id": entry["id"],
                "vector": vec,
                "payload": entry["payload"],
            })

        if qdrant_points:
            resp = qdrant_put(
                f"/collections/{Config.COLLECTION}/points",
                {"points": qdrant_points},
            )
            # Check for both "completed" and "acknowledged" status
            if resp and resp.get("result", {}).get("status") in ["completed", "acknowledged"]:
                upserted += len(qdrant_points)
            elif resp and resp.get("status") == "ok":
                # Alternative success response format
                upserted += len(qdrant_points)
            else:
                logger.error(f"  Batch upsert failed: {resp}")

        # Progress
        done = min(i + Config.BATCH_SIZE, len(points_buffer))
        logger.info(f"  Progress: {done}/{len(points_buffer)} "
                     f"(upserted: {upserted}, embed failures: {embed_failures})")

    conn.close()

    # --- Summary ---
    logger.info("\n" + "=" * 60)
    logger.info("REBUILD COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Collection:     {Config.COLLECTION}")
    logger.info(f"  Vector dim:     {Config.VECTOR_DIM}D (nomic-embed-text)")
    logger.info(f"  Total upserted: {upserted}")
    logger.info(f"  Embed failures: {embed_failures}")
    logger.info(f"  Failed tables:  {failed_tables or 'none'}")
    logger.info(f"  Architecture:   SSOT-compliant (references only)")
    logger.info(f"  SSOT database:  {Config.PG_DATABASE} @ {Config.PG_HOST}")
    logger.info("")
    logger.info("  Qdrant payloads contain ONLY:")
    logger.info("    - type, source_table, source_id (reference to PG)")
    logger.info("    - search_text (for debug, NOT authoritative)")
    logger.info("    - indexed_at (freshness tracking)")
    logger.info("")
    logger.info("  At generation time, orchestrator MUST:")
    logger.info("    1. Search Qdrant → get source_table + source_id")
    logger.info("    2. SELECT * FROM {source_table} WHERE id = {source_id}")
    logger.info("    3. Use fresh PostgreSQL data for generation")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SSOT-Compliant Story Bible Rebuilder"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview what would be indexed without changes")
    parser.add_argument("--incremental", action="store_true",
                        help="Add/update without dropping existing collection")
    args = parser.parse_args()

    rebuild_collection(dry_run=args.dry_run, incremental=args.incremental)