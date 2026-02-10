#!/usr/bin/env python3
"""
SSOT-Compliant Intelligent Generation Orchestrator
====================================================
Searches Qdrant story_bible for semantic matches, then fetches fresh,
authoritative data from the anime_production PostgreSQL SSOT before
building ComfyUI generation requests.

Flow:
    User prompt → Content Analysis → Qdrant Search (references only)
    → PostgreSQL Fetch (fresh SSOT data) → Resource Selection
    → Prompt Engineering → ComfyUI Submission → Quality Check

Usage:
    # As library (from other scripts / orchestrator services)
    from ssot_generation_orchestrator import SSOTOrchestrator
    orch = SSOTOrchestrator()
    plan = orch.plan_generation("Generate Kai fighting goblins")
    result = orch.execute(plan)

    # CLI test mode
    python3 ssot_generation_orchestrator.py "Generate Kai fighting goblins"
    python3 ssot_generation_orchestrator.py --plan-only "Mei romantic Tokyo scene"
    python3 ssot_generation_orchestrator.py --dry-run "cyberpunk battle"

Place at: /opt/tower-echo-brain/scripts/ssot_generation_orchestrator.py
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
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

    COMFYUI_URL = os.getenv("COMFYUI_URL", "http://localhost:8188")
    WORKFLOW_DIR = "/opt/tower-anime-production/workflows/comfyui"
    CHECKPOINT_DIR = "/mnt/1TB-storage/models/checkpoints"
    LORA_DIR = "/mnt/1TB-storage/models/loras"

    GEN_TIMEOUT = 300
    GEN_POLL_INTERVAL = 5


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("ssot_orchestrator")


# ---------------------------------------------------------------------------
# HTTP / DB helpers
# ---------------------------------------------------------------------------

def http_json(method: str, url: str, payload: dict = None, timeout: int = 30):
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
        logger.error(f"Request to {url} failed: {e}")
        return None


def embed_text(text: str) -> Optional[list]:
    resp = http_json("POST", f"{Config.OLLAMA_URL}/api/embed", {
        "model": Config.EMBEDDING_MODEL,
        "input": text,
    })
    if resp and "embeddings" in resp:
        vec = resp["embeddings"][0]
        return vec if len(vec) == Config.VECTOR_DIM else None
    return None


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


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ContentAnalysis:
    """Parsed understanding of what the user wants to generate."""
    characters: list = field(default_factory=list)
    scene_type: str = "general"        # action, romantic, dialogue, establishing, etc.
    style: str = "anime"               # anime, cyberpunk, photorealistic, etc.
    location: str = ""
    mood: str = ""
    raw_prompt: str = ""
    keywords: list = field(default_factory=list)


@dataclass
class SSOTReference:
    """A reference from Qdrant pointing back to the PostgreSQL SSOT."""
    source_table: str
    source_id: int
    content_type: str
    display_name: str = ""
    search_score: float = 0.0


@dataclass
class FreshRecord:
    """Full record fetched from the PostgreSQL SSOT."""
    table: str
    id: int
    data: dict = field(default_factory=dict)
    content_type: str = ""


@dataclass
class ResourceSelection:
    """Chosen generation resources based on content + data analysis."""
    workflow_file: str = ""
    checkpoint: str = ""
    loras: list = field(default_factory=list)   # [{"name": str, "strength": float}]
    positive_prompt: str = ""
    negative_prompt: str = ""
    width: int = 512
    height: int = 768
    steps: int = 20
    cfg_scale: float = 7.0
    reasoning: list = field(default_factory=list)  # Why each choice was made


@dataclass
class GenerationPlan:
    """Complete plan ready for execution."""
    analysis: ContentAnalysis = None
    references: list = field(default_factory=list)     # SSOTReference list
    fresh_data: list = field(default_factory=list)      # FreshRecord list
    resources: ResourceSelection = None
    warnings: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Content Analyzer
# ---------------------------------------------------------------------------

class ContentAnalyzer:
    """Parse user requests into structured generation requirements."""

    SCENE_KEYWORDS = {
        "action": ["fight", "battle", "combat", "attack", "slash", "punch",
                    "explosion", "chase", "duel", "clash"],
        "romantic": ["romantic", "love", "kiss", "embrace", "tender", "intimate",
                     "date", "confession", "blush"],
        "dialogue": ["talking", "conversation", "discuss", "argue", "meeting",
                     "negotiate", "confront"],
        "establishing": ["city", "landscape", "skyline", "overview", "panorama",
                         "environment", "setting"],
    }

    STYLE_KEYWORDS = {
        "cyberpunk": ["cyberpunk", "neon", "cyber", "futuristic", "dystopian",
                      "hologram", "augmented", "cybernetic"],
        "photorealistic": ["photorealistic", "realistic", "photo", "real",
                           "lifelike"],
        "anime": ["anime", "manga", "cel-shaded", "2d", "hand-drawn"],
    }

    def analyze(self, prompt: str) -> ContentAnalysis:
        result = ContentAnalysis(raw_prompt=prompt)
        lower = prompt.lower()

        # Detect scene type
        for scene_type, keywords in self.SCENE_KEYWORDS.items():
            if any(kw in lower for kw in keywords):
                result.scene_type = scene_type
                break

        # Detect style
        for style, keywords in self.STYLE_KEYWORDS.items():
            if any(kw in lower for kw in keywords):
                result.style = style
                break

        # Extract character names (capitalized words not in common stop words)
        stop = {"generate", "create", "make", "show", "draw", "render",
                "scene", "with", "from", "the", "and", "tokyo", "city"}
        words = prompt.split()
        result.characters = [
            w.strip(".,!?;:'\"") for w in words
            if w[0:1].isupper() and w.lower() not in stop and len(w) > 2
        ]

        # Extract keywords for Qdrant search
        result.keywords = [
            w.lower().strip(".,!?;:'\"") for w in words
            if len(w) > 3 and w.lower() not in stop
        ]

        return result


# ---------------------------------------------------------------------------
# SSOT Data Fetcher — the critical piece
# ---------------------------------------------------------------------------

class SSOTFetcher:
    """
    Searches Qdrant for references, then fetches fresh data from PostgreSQL.
    This is the SSOT-compliant replacement for reading stale Qdrant payloads.
    """

    def search_references(self, query: str, limit: int = 10,
                          type_filter: str = None) -> list:
        """Search Qdrant story_bible for SSOT references (not data)."""
        vec = embed_text(query)
        if not vec:
            logger.error("Failed to embed search query")
            return []

        search_payload = {
            "vector": vec,
            "limit": limit,
            "with_payload": True,
        }

        # Optional type filter
        if type_filter:
            search_payload["filter"] = {
                "must": [{"key": "type", "match": {"value": type_filter}}]
            }

        resp = http_json(
            "POST",
            f"{Config.QDRANT_URL}/collections/{Config.COLLECTION}/points/search",
            search_payload,
        )

        if not resp or "result" not in resp:
            return []

        refs = []
        for hit in resp["result"]:
            payload = hit.get("payload", {})
            refs.append(SSOTReference(
                source_table=payload.get("source_table", ""),
                source_id=payload.get("source_id", 0),
                content_type=payload.get("type", ""),
                display_name=payload.get("display_name", ""),
                search_score=hit.get("score", 0),
            ))

        return refs

    def fetch_fresh(self, references: list) -> list:
        """
        Fetch full, authoritative records from PostgreSQL SSOT.
        This is the KEY SSOT step — Qdrant gives us references,
        PostgreSQL gives us the actual data.
        """
        if not references:
            return []

        try:
            conn = get_pg_connection()
            cur = conn.cursor()
        except Exception as e:
            logger.error(f"Cannot connect to SSOT database: {e}")
            return []

        fresh_records = []

        # Group by table for efficient queries
        by_table = {}
        for ref in references:
            if ref.source_table and ref.source_id:
                by_table.setdefault(ref.source_table, []).append(ref)

        for table, refs in by_table.items():
            try:
                # Verify table exists
                cur.execute("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = %s
                    ORDER BY ordinal_position
                """, (table,))
                columns = [row[0] for row in cur.fetchall()]

                if not columns:
                    logger.warning(f"SSOT table {table} not found")
                    continue

                # Determine PK column
                id_col = "id"
                if id_col not in columns:
                    for fallback in [f"{table}_id", "uuid"]:
                        if fallback in columns:
                            id_col = fallback
                            break

                # Fetch all matching rows in one query
                ids = [ref.source_id for ref in refs]
                placeholders = ",".join(["%s"] * len(ids))
                cur.execute(
                    f"SELECT * FROM {table} WHERE {id_col} IN ({placeholders})",
                    ids,
                )
                col_names = [desc[0] for desc in cur.description]
                rows = cur.fetchall()

                for row in rows:
                    row_dict = {}
                    for col, val in zip(col_names, row):
                        # Convert non-serializable types
                        if hasattr(val, "isoformat"):
                            row_dict[col] = val.isoformat()
                        elif isinstance(val, (dict, list)):
                            row_dict[col] = val
                        elif val is not None:
                            row_dict[col] = str(val) if not isinstance(val, (int, float, bool)) else val
                        else:
                            row_dict[col] = None

                    # Find content type from the reference
                    row_id = row_dict.get(id_col)
                    ref_match = next(
                        (r for r in refs if str(r.source_id) == str(row_id)),
                        refs[0],
                    )

                    fresh_records.append(FreshRecord(
                        table=table,
                        id=row_id,
                        data=row_dict,
                        content_type=ref_match.content_type,
                    ))

            except Exception as e:
                logger.error(f"SSOT fetch from {table} failed: {e}")

        conn.close()
        logger.info(f"Fetched {len(fresh_records)} fresh records from SSOT")
        return fresh_records


# ---------------------------------------------------------------------------
# Resource Selector
# ---------------------------------------------------------------------------

class ResourceSelector:
    """Choose optimal models, LoRAs, and workflows based on fresh SSOT data."""

    # Model recommendations by style
    STYLE_MODELS = {
        "cyberpunk": [
            "cyberrealistic_v9.safetensors",
            "realistic_vision_v51.safetensors",
        ],
        "photorealistic": [
            "realistic_vision_v51.safetensors",
            "chilloutmix_NiPrunedFp32Fix.safetensors",
            "chilloutmix.safetensors",
        ],
        "anime": [
            "AOM3A1B.safetensors",
            "anything-v3-full.safetensors",
        ],
    }

    # Scene type → workflow preference
    SCENE_WORKFLOWS = {
        "action": [
            "anime_30sec_rife_workflow_with_lora.json",
            "ACTION_combat_workflow.json",
        ],
        "romantic": [
            "anime_30sec_rife_workflow_with_lora.json",
            "anime_30sec_working_workflow.json",
        ],
        "dialogue": [
            "anime_30sec_working_workflow.json",
            "anime_30sec_fixed_workflow.json",
        ],
        "general": [
            "anime_30sec_rife_workflow_with_lora.json",
            "anime_30sec_working_workflow.json",
        ],
    }

    # LoRA matching: character name patterns → LoRA files
    CHARACTER_LORAS = {
        "kai": [
            ("kai_cyberpunk_slayer.safetensors", 0.8),
            ("kai_nakamura_optimized_v1.safetensors", 0.7),
        ],
        "mei": [
            ("mei_working_v1.safetensors", 0.8),
        ],
    }

    STYLE_LORAS = {
        "cyberpunk": [
            ("cyberpunk_style_proper.safetensors", 0.6),
        ],
    }

    def select(self, analysis: ContentAnalysis,
               fresh_data: list) -> ResourceSelection:
        """Build resource selection using fresh SSOT data."""
        sel = ResourceSelection()

        # --- Workflow ---
        candidates = self.SCENE_WORKFLOWS.get(
            analysis.scene_type,
            self.SCENE_WORKFLOWS["general"],
        )
        for wf in candidates:
            path = os.path.join(Config.WORKFLOW_DIR, wf)
            if os.path.isfile(path):
                sel.workflow_file = wf
                sel.reasoning.append(
                    f"Workflow: {wf} (best for {analysis.scene_type} scenes)"
                )
                break

        if not sel.workflow_file:
            # Fallback to any available workflow
            for f in os.listdir(Config.WORKFLOW_DIR):
                if f.endswith(".json"):
                    sel.workflow_file = f
                    sel.reasoning.append(f"Workflow: {f} (fallback)")
                    break

        # --- Checkpoint ---
        available = set()
        if os.path.isdir(Config.CHECKPOINT_DIR):
            available = set(os.listdir(Config.CHECKPOINT_DIR))

        candidates = self.STYLE_MODELS.get(analysis.style, self.STYLE_MODELS["anime"])
        for model in candidates:
            if model in available:
                sel.checkpoint = model
                sel.reasoning.append(
                    f"Model: {model} (optimized for {analysis.style} style)"
                )
                break

        if not sel.checkpoint and available:
            sel.checkpoint = sorted(available)[0]
            sel.reasoning.append(f"Model: {sel.checkpoint} (fallback)")

        # --- LoRAs from Database (FIXED) ---
        # Check what LoRA files actually exist
        lora_paths = []
        for base_dir in [Config.LORA_DIR, "/opt/ComfyUI/models/loras"]:
            if os.path.isdir(base_dir):
                for f in os.listdir(base_dir):
                    if f.endswith(".safetensors"):
                        lora_paths.append(f)

        # Get LoRAs directly from fresh character data (from database)
        used_loras = set()  # Prevent duplicates
        for record in fresh_data:
            if record.content_type == "character" and record.table == "characters":
                char_data = record.data
                lora_path = char_data.get("lora_path")
                lora_trigger = char_data.get("lora_trigger")
                char_name = char_data.get("name", "unknown")

                if lora_path and lora_path in lora_paths and lora_path not in used_loras:
                    # Use database lora_path, not hard-coded mappings
                    strength = 0.85  # Standard strength
                    sel.loras.append({
                        "name": lora_path,
                        "strength": strength,
                        "trigger": lora_trigger,  # Include trigger word
                    })
                    used_loras.add(lora_path)
                    sel.reasoning.append(
                        f"LoRA: {lora_path} @ {strength} (character: {char_name}, "
                        f"trigger: {lora_trigger}, from database)"
                    )
                elif lora_path and lora_path in used_loras:
                    # Already added this LoRA
                    sel.reasoning.append(f"LoRA {lora_path} already added (character: {char_name})")
                elif lora_path:
                    # LoRA specified in DB but file not found
                    sel.reasoning.append(
                        f"WARNING: {char_name} has LoRA '{lora_path}' in database "
                        f"but file not found in {[Config.LORA_DIR, '/opt/ComfyUI/models/loras']}"
                    )
                else:
                    # No LoRA specified for this character
                    sel.reasoning.append(f"No LoRA specified for character {char_name}")

        # Style LoRAs (keep this as fallback for non-character LoRAs)
        style_candidates = self.STYLE_LORAS.get(analysis.style, [])
        for lora_file, strength in style_candidates:
            if lora_file in lora_paths:
                already_added = any(l["name"] == lora_file for l in sel.loras)
                if not already_added:
                    sel.loras.append({"name": lora_file, "strength": strength})
                    sel.reasoning.append(
                        f"LoRA: {lora_file} @ {strength} (style: {analysis.style})"
                    )

        # --- Prompt Engineering (using FRESH SSOT data) ---
        prompt_parts = ["masterpiece, best quality, high resolution, detailed"]
        neg_parts = [
            "lowres, bad anatomy, bad hands, text, error, missing fingers, "
            "extra digit, fewer digits, cropped, worst quality, low quality, "
            "jpeg artifacts, signature, watermark"
        ]

        # Add LoRA trigger words first (CRITICAL for LoRA activation)
        trigger_words = []
        for lora in sel.loras:
            if lora.get("trigger"):
                trigger_words.append(lora["trigger"])
        if trigger_words:
            prompt_parts.extend(trigger_words)
            sel.reasoning.append(f"Added LoRA triggers: {', '.join(trigger_words)}")

        # Build character description from FRESH PostgreSQL data
        for record in fresh_data:
            if record.content_type == "character":
                d = record.data
                # Pull appearance/description fields fresh from SSOT
                char_desc_parts = []
                for col in ["appearance", "description", "personality",
                            "visual_description", "hair_color", "eye_color",
                            "outfit", "distinguishing_features"]:
                    val = d.get(col)
                    if val and str(val).strip() and str(val).lower() != "none":
                        char_desc_parts.append(str(val).strip()[:150])

                if char_desc_parts:
                    char_name = d.get("name", "character")
                    char_prompt = ", ".join(char_desc_parts[:4])
                    prompt_parts.append(char_prompt)
                    sel.reasoning.append(
                        f"Prompt includes FRESH data for {char_name} "
                        f"from SSOT table '{record.table}'"
                    )

            elif record.content_type == "scene":
                d = record.data
                for col in ["visual_description", "description",
                            "narrative_text", "location"]:
                    val = d.get(col)
                    if val and str(val).strip()[:100] and str(val).lower() != "none":
                        prompt_parts.append(str(val).strip()[:200])
                        break

        # Style-specific prompt additions
        if analysis.style == "cyberpunk":
            prompt_parts.extend([
                "cyberpunk aesthetic", "neon lights", "dark atmosphere",
                "futuristic technology",
            ])
        elif analysis.style == "photorealistic":
            prompt_parts.extend([
                "photorealistic", "8k uhd", "studio lighting",
            ])

        # Scene-specific additions
        if analysis.scene_type == "action":
            prompt_parts.extend(["dynamic pose", "motion blur", "intense action"])
            neg_parts.append("static pose, standing still, calm expression")
        elif analysis.scene_type == "romantic":
            prompt_parts.extend(["warm lighting", "soft focus", "emotional"])
            neg_parts.append("cold, harsh, violent, aggressive")

        # Add location if mentioned
        if analysis.location:
            prompt_parts.append(analysis.location)

        sel.positive_prompt = ", ".join(prompt_parts)
        sel.negative_prompt = ", ".join(neg_parts)

        return sel


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

class SSOTOrchestrator:
    """
    The complete SSOT-compliant generation orchestrator.
    Search → Reference → Fetch Fresh → Select → Build → Execute
    """

    def __init__(self):
        self.analyzer = ContentAnalyzer()
        self.fetcher = SSOTFetcher()
        self.selector = ResourceSelector()

    def plan_generation(self, prompt: str) -> GenerationPlan:
        """
        Build a complete generation plan using SSOT architecture.

        1. Analyze the user prompt
        2. Query PostgreSQL directly for characters (no Qdrant for character lookup)
        3. Search Qdrant for scene/style references only
        4. Fetch FRESH data from PostgreSQL SSOT
        5. Select optimal resources
        """
        plan = GenerationPlan()

        # Step 1: Analyze
        plan.analysis = self.analyzer.analyze(prompt)
        logger.info(
            f"Parsed: characters={plan.analysis.characters}, "
            f"scene={plan.analysis.scene_type}, style={plan.analysis.style}"
        )

        # Step 2: Query PostgreSQL directly for characters (FIXED)
        # This bypasses the Qdrant semantic search bug that returns wrong characters
        try:
            conn = get_pg_connection()
            cur = conn.cursor()

            for char_name in plan.analysis.characters:
                # Query characters table directly by name
                cur.execute("""
                    SELECT id, name, lora_path, lora_trigger, description,
                           appearance_data, personality, project_id
                    FROM characters
                    WHERE name ILIKE %s OR name ILIKE %s
                    ORDER BY
                        CASE WHEN name ILIKE %s THEN 1 ELSE 2 END,
                        name
                    LIMIT 3
                """, (f"%{char_name}%", char_name, char_name))

                rows = cur.fetchall()
                col_names = [desc[0] for desc in cur.description]

                for row in rows:
                    row_dict = dict(zip(col_names, row))
                    # Convert non-serializable types
                    for key, val in row_dict.items():
                        if hasattr(val, "isoformat"):
                            row_dict[key] = val.isoformat()
                        elif isinstance(val, (dict, list)):
                            row_dict[key] = val
                        elif val is not None:
                            row_dict[key] = str(val) if not isinstance(val, (int, float, bool)) else val
                        else:
                            row_dict[key] = None

                    # Add as both reference and fresh data
                    ref = SSOTReference(
                        source_table="characters",
                        source_id=row_dict["id"],
                        content_type="character",
                        display_name=row_dict.get("name", ""),
                        search_score=1.0  # Perfect match from direct query
                    )
                    plan.references.append(ref)

                    fresh = FreshRecord(
                        table="characters",
                        id=row_dict["id"],
                        data=row_dict,
                        content_type="character"
                    )
                    plan.fresh_data.append(fresh)

                    logger.info(f"Direct character match: {row_dict['name']} with LoRA: {row_dict.get('lora_path', 'none')}")

            conn.close()
        except Exception as e:
            logger.error(f"Direct character query failed: {e}")

        # Step 3: Search Qdrant for scene/style references only (not characters)
        scene_query = " ".join(plan.analysis.keywords[:5])
        if scene_query:
            scene_refs = self.fetcher.search_references(
                scene_query, limit=3, type_filter="scene",
            )
            plan.references.extend(scene_refs)

        # Search for style/model info (but NOT characters)
        if plan.analysis.style:
            model_refs = self.fetcher.search_references(
                f"{plan.analysis.style} style visual",
                limit=2, type_filter="scene",  # Use scene for style context
            )
            plan.references.extend(model_refs)

        logger.info(f"Found {len(plan.references)} total references")
        for ref in plan.references:
            logger.info(
                f"  → {ref.content_type}: {ref.source_table}.{ref.source_id} "
                f"({ref.display_name}) score={ref.search_score:.3f}"
            )

        # Step 4: Fetch additional fresh data from remaining references (scenes, etc)
        remaining_refs = [ref for ref in plan.references if ref.source_table != "characters"]
        additional_fresh = self.fetcher.fetch_fresh(remaining_refs)
        plan.fresh_data.extend(additional_fresh)

        logger.info(f"Total fresh records: {len(plan.fresh_data)}")
        for record in plan.fresh_data:
            name = record.data.get("name", record.data.get("title", f"id={record.id}"))
            logger.info(f"  → {record.table}.{record.id}: {name}")

        # Step 5: Select resources using FRESH data
        plan.resources = self.selector.select(plan.analysis, plan.fresh_data)

        # Warnings
        if not any(rec.table == "characters" for rec in plan.fresh_data):
            plan.warnings.append(
                "No characters found in database - generation will use "
                "generic prompts without character-specific details"
            )
        if not plan.resources.loras:
            plan.warnings.append(
                "No character-specific LoRAs selected — output may not "
                "match expected character appearance"
            )
        if not plan.resources.workflow_file:
            plan.warnings.append("No workflow file found!")

        return plan

    def execute(self, plan: GenerationPlan) -> dict:
        """Execute a generation plan by submitting to ComfyUI."""
        if not plan.resources.workflow_file:
            return {"error": "No workflow file in plan"}

        workflow_path = os.path.join(
            Config.WORKFLOW_DIR, plan.resources.workflow_file
        )
        try:
            with open(workflow_path) as f:
                workflow = json.load(f)
        except Exception as e:
            return {"error": f"Failed to load workflow: {e}"}

        # Apply plan to workflow nodes
        for node_id, node in workflow.items():
            if not isinstance(node, dict):
                continue

            ct = node.get("class_type", "")
            inputs = node.get("inputs", {})
            title = node.get("_meta", {}).get("title", "").lower()

            # Set checkpoint
            if ct == "CheckpointLoaderSimple" and plan.resources.checkpoint:
                inputs["ckpt_name"] = plan.resources.checkpoint

            # Set prompts
            if ct == "CLIPTextEncode":
                if "positive" in title or "prompt" in title:
                    inputs["text"] = plan.resources.positive_prompt
                elif "negative" in title:
                    inputs["text"] = plan.resources.negative_prompt

            # Set LoRAs (FIXED - handle multiple LoRAs)
            if ct == "LoraLoader" and plan.resources.loras:
                # Find which LoRA this node should use
                # For now, use first available LoRA, but in future could match by node title/id
                lora = plan.resources.loras[0]
                inputs["lora_name"] = lora["name"]
                inputs["strength_model"] = lora["strength"]
                inputs["strength_clip"] = lora["strength"]
                logger.info(f"Applied LoRA {lora['name']} to node {node_id}")

            # Set image dimensions (FIXED - AnimateDiff batch size)
            if ct == "EmptyLatentImage":
                inputs["width"] = plan.resources.width
                inputs["height"] = plan.resources.height
                # CRITICAL FIX: AnimateDiff requires batch_size >= 16 for temporal coherence
                # Never override to 1 for video generation
                if "batch_size" in inputs:
                    current_batch = inputs.get("batch_size", 24)
                    if current_batch < 16:
                        inputs["batch_size"] = 24  # Good for 1-second @ 24fps
                        logger.info(f"Corrected batch_size from {current_batch} to 24 for AnimateDiff")
                else:
                    inputs["batch_size"] = 24

        # Submit to ComfyUI
        try:
            data = json.dumps({"prompt": workflow}).encode()
            req = urllib.request.Request(
                f"{Config.COMFYUI_URL}/prompt",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                result = json.loads(resp.read().decode())
        except Exception as e:
            return {"error": f"ComfyUI submission failed: {e}"}

        prompt_id = result.get("prompt_id")
        if not prompt_id:
            return {
                "error": "ComfyUI rejected workflow",
                "details": result.get("error", result.get("node_errors", "")),
            }

        logger.info(f"Submitted to ComfyUI: prompt_id={prompt_id}")

        # Poll for completion
        start = time.time()
        while time.time() - start < Config.GEN_TIMEOUT:
            history = http_json(
                "GET", f"{Config.COMFYUI_URL}/history/{prompt_id}"
            )
            if history and prompt_id in history:
                status = history[prompt_id].get("status", {})
                if status.get("completed") or status.get("status_str") == "success":
                    elapsed = round(time.time() - start, 1)
                    outputs = history[prompt_id].get("outputs", {})
                    images = []
                    for node_out in outputs.values():
                        if isinstance(node_out, dict):
                            images.extend(node_out.get("images", []))
                    return {
                        "status": "completed",
                        "prompt_id": prompt_id,
                        "elapsed_seconds": elapsed,
                        "images": images,
                        "ssot_sources": [
                            f"{r.table}.{r.id}" for r in plan.fresh_data
                        ],
                    }
                if status.get("status_str") == "error":
                    return {
                        "error": "Generation failed",
                        "details": status,
                    }
            time.sleep(Config.GEN_POLL_INTERVAL)

        return {"error": f"Timed out after {Config.GEN_TIMEOUT}s"}


# ---------------------------------------------------------------------------
# CLI / Test Mode
# ---------------------------------------------------------------------------

def print_plan(plan: GenerationPlan):
    """Pretty-print a generation plan for review."""
    print("\n" + "=" * 60)
    print("  SSOT GENERATION PLAN")
    print("=" * 60)

    a = plan.analysis
    print(f"\n  📝 Content Analysis:")
    print(f"     Characters: {a.characters or '(none detected)'}")
    print(f"     Scene type: {a.scene_type}")
    print(f"     Style:      {a.style}")
    print(f"     Raw prompt: {a.raw_prompt[:80]}...")

    print(f"\n  🔍 Qdrant References ({len(plan.references)}):")
    for ref in plan.references:
        print(f"     → [{ref.content_type}] {ref.source_table}.{ref.source_id} "
              f"'{ref.display_name}' (score: {ref.search_score:.3f})")

    print(f"\n  📊 Fresh SSOT Data ({len(plan.fresh_data)} records):")
    for rec in plan.fresh_data:
        name = rec.data.get("name", rec.data.get("title", f"id={rec.id}"))
        cols = len(rec.data)
        print(f"     → [{rec.content_type}] {rec.table}.{rec.id}: "
              f"{name} ({cols} columns, FRESH from PostgreSQL)")

    r = plan.resources
    print(f"\n  ⚙️  Resource Selection:")
    print(f"     Workflow:   {r.workflow_file or 'NONE'}")
    print(f"     Checkpoint: {r.checkpoint or 'NONE'}")
    print(f"     LoRAs:      {r.loras or 'none'}")
    print(f"     Dimensions: {r.width}x{r.height}")
    print(f"     Steps:      {r.steps}")

    print(f"\n  💬 Positive Prompt:")
    print(f"     {r.positive_prompt[:200]}...")

    print(f"\n  🚫 Negative Prompt:")
    print(f"     {r.negative_prompt[:150]}...")

    print(f"\n  🧠 Reasoning:")
    for reason in r.reasoning:
        print(f"     • {reason}")

    if plan.warnings:
        print(f"\n  ⚠️  Warnings:")
        for w in plan.warnings:
            print(f"     • {w}")

    # SSOT compliance check
    print(f"\n  ✅ SSOT Compliance:")
    print(f"     Data source:   PostgreSQL anime_production (AUTHORITATIVE)")
    print(f"     Qdrant role:   Search index only (references, not data)")
    print(f"     Fresh records: {len(plan.fresh_data)} fetched at generation time")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SSOT-Compliant Generation Orchestrator"
    )
    parser.add_argument("prompt", nargs="?",
                        default="Generate Kai fighting cyberpunk goblins in Tokyo")
    parser.add_argument("--plan-only", action="store_true",
                        help="Show plan without submitting to ComfyUI")
    parser.add_argument("--dry-run", action="store_true",
                        help="Alias for --plan-only")

    args = parser.parse_args()

    orch = SSOTOrchestrator()
    plan = orch.plan_generation(args.prompt)
    print_plan(plan)

    if args.plan_only or args.dry_run:
        print("\n  [--plan-only] Skipping ComfyUI submission\n")
    else:
        print("\n  Submitting to ComfyUI...")
        result = orch.execute(plan)
        print(f"\n  Result: {json.dumps(result, indent=2)}")