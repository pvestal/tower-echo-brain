"""CLIP-based generation quality scorer for anime production.

Provides three scoring dimensions:
- semantic_score: CLIP similarity of output image vs reference character images (0-100)
- variety_score: inverse of max similarity to recent shots in same scene (0-100)
- text_alignment: CLIP similarity of output image embedding vs prompt text embedding (0-100)

Uses ViT-B-32 (512-dim embeddings), stored in Qdrant collection `generation_clip`.
IMPORTANT: This is SEPARATE from echo_memory (768-dim nomic-embed-text). Never mix.

Runs on AMD RX 9070 XT alongside Ollama. ~400MB VRAM, ~50ms/image after warmup.
"""

import logging
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Lazy-loaded CLIP model singleton (same pattern as anime-studio variety_check.py)
_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None
_clip_device = "cpu"

# Qdrant collection for CLIP embeddings — 512D, cosine distance
COLLECTION_NAME = "generation_clip"
VECTOR_DIM = 512


def _load_clip():
    """Lazy-load CLIP ViT-B-32 model. ~400MB VRAM on first call."""
    global _clip_model, _clip_preprocess, _clip_tokenizer, _clip_device
    if _clip_model is not None:
        return

    import open_clip
    import torch

    device = "cpu"
    try:
        if torch.cuda.is_available():
            device = "cuda"
    except Exception:
        pass

    _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device,
    )
    _clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
    _clip_model.eval()
    _clip_device = device
    logger.info(f"clip_scorer: CLIP ViT-B-32 loaded on {device}")


def embed_image(image_path: str | Path) -> np.ndarray | None:
    """Compute CLIP image embedding. Returns 512-dim normalized vector."""
    try:
        _load_clip()
        import torch
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        img_tensor = _clip_preprocess(img).unsqueeze(0).to(_clip_device)

        with torch.no_grad():
            emb = _clip_model.encode_image(img_tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)

        return emb.cpu().numpy().flatten()
    except Exception as e:
        logger.warning(f"clip_scorer: failed to embed image {image_path}: {e}")
        return None


def embed_text(text: str) -> np.ndarray | None:
    """Compute CLIP text embedding. Returns 512-dim normalized vector."""
    try:
        _load_clip()
        import torch

        tokens = _clip_tokenizer([text]).to(_clip_device)

        with torch.no_grad():
            emb = _clip_model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)

        return emb.cpu().numpy().flatten()
    except Exception as e:
        logger.warning(f"clip_scorer: failed to embed text: {e}")
        return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalized vectors."""
    return float(np.dot(a, b))


def _get_qdrant():
    """Get Qdrant client singleton."""
    from qdrant_client import QdrantClient
    return QdrantClient(url="http://localhost:6333")


def ensure_collection():
    """Create the generation_clip collection if it doesn't exist."""
    from qdrant_client.models import Distance, VectorParams
    client = _get_qdrant()
    try:
        info = client.get_collection(COLLECTION_NAME)
        logger.info(f"clip_scorer: {COLLECTION_NAME} exists ({info.points_count} points)")
    except Exception:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        logger.info(f"clip_scorer: created {COLLECTION_NAME} collection ({VECTOR_DIM}D, cosine)")


async def evaluate_generation(
    image_path: str,
    prompt_text: str,
    shot_id: str,
    scene_id: str,
    project_id: int,
    character_slugs: list[str] | None = None,
    video_engine: str = "",
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score a generated image/frame against multiple quality dimensions.

    Returns:
        {
            "semantic_score": 0-100 (vs reference images),
            "variety_score": 0-100 (inverse similarity to recent shots),
            "text_alignment": 0-100 (image vs prompt text),
            "mhp_bucket": 0-100 (weighted composite),
            "too_similar_to": shot_id or None,
            "suggestion": str or None,
            "embedding_stored": bool,
        }
    """
    result = {
        "semantic_score": 0,
        "variety_score": 100,
        "text_alignment": 0,
        "mhp_bucket": 0,
        "too_similar_to": None,
        "suggestion": None,
        "embedding_stored": False,
    }

    # 1. Embed the output image
    img_emb = embed_image(image_path)
    if img_emb is None:
        result["suggestion"] = "failed to embed output image"
        return result

    # 2. Text alignment: image embedding vs prompt text embedding
    if prompt_text:
        text_emb = embed_text(prompt_text[:77])  # CLIP tokenizer max ~77 tokens
        if text_emb is not None:
            result["text_alignment"] = round(cosine_similarity(img_emb, text_emb) * 100, 1)

    # 3. Variety score: compare against recent shots in same scene via Qdrant
    variety_score = 100.0
    too_similar_to = None
    try:
        client = _get_qdrant()
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        search_filter = Filter(must=[
            FieldCondition(key="scene_id", match=MatchValue(value=scene_id)),
        ])

        hits = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=img_emb.tolist(),
            query_filter=search_filter,
            limit=5,
        )

        if hits:
            max_sim = max(h.score for h in hits)
            variety_score = max(0, round((1.0 - max_sim) * 100, 1))
            if max_sim > 0.85:
                best_hit = max(hits, key=lambda h: h.score)
                too_similar_to = best_hit.payload.get("shot_id")
                result["too_similar_to"] = too_similar_to
                result["suggestion"] = (
                    f"similarity {max_sim:.3f} to shot {too_similar_to}; "
                    "try different pose or camera angle"
                )

    except Exception as e:
        logger.warning(f"clip_scorer: variety check failed: {e}")

    result["variety_score"] = round(variety_score, 1)

    # 4. Semantic score: compare against reference images for the character(s)
    # Look for existing character reference embeddings in Qdrant
    semantic_score = 50.0  # Default when no references exist
    if character_slugs:
        try:
            client = _get_qdrant()
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            ref_filter = Filter(must=[
                FieldCondition(key="type", match=MatchValue(value="character_reference")),
                FieldCondition(key="character_slug", match=MatchValue(value=character_slugs[0])),
            ])

            ref_hits = client.search(
                collection_name=COLLECTION_NAME,
                query_vector=img_emb.tolist(),
                query_filter=ref_filter,
                limit=5,
            )

            if ref_hits:
                avg_sim = sum(h.score for h in ref_hits) / len(ref_hits)
                semantic_score = round(avg_sim * 100, 1)
        except Exception as e:
            logger.debug(f"clip_scorer: reference lookup failed: {e}")

    result["semantic_score"] = round(semantic_score, 1)

    # 5. Composite MHP bucket (weighted average)
    result["mhp_bucket"] = round(
        result["semantic_score"] * 0.3
        + result["variety_score"] * 0.3
        + result["text_alignment"] * 0.4,
        1,
    )

    # 6. Store embedding in Qdrant
    try:
        from qdrant_client.models import PointStruct
        from datetime import datetime, timezone

        point_id = str(uuid.uuid4())
        point = PointStruct(
            id=point_id,
            vector=img_emb.tolist(),
            payload={
                "type": "generation_output",
                "shot_id": shot_id,
                "scene_id": scene_id,
                "project_id": project_id,
                "character_slugs": character_slugs or [],
                "video_engine": video_engine,
                "prompt_text": prompt_text[:500],
                "semantic_score": result["semantic_score"],
                "variety_score": result["variety_score"],
                "text_alignment": result["text_alignment"],
                "mhp_bucket": result["mhp_bucket"],
                "parameters": parameters or {},
                "image_path": image_path,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        client = _get_qdrant()
        client.upsert(collection_name=COLLECTION_NAME, points=[point])
        result["embedding_stored"] = True
        logger.info(
            f"clip_scorer: scored shot {shot_id[:8]} — "
            f"semantic={result['semantic_score']}, variety={result['variety_score']}, "
            f"text={result['text_alignment']}, mhp={result['mhp_bucket']}"
        )
    except Exception as e:
        logger.warning(f"clip_scorer: failed to store embedding: {e}")

    return result


async def store_character_reference(
    image_path: str,
    character_slug: str,
    project_id: int,
) -> bool:
    """Store a character reference image embedding for semantic scoring.

    Call this with LoRA training images to build the reference set.
    """
    emb = embed_image(image_path)
    if emb is None:
        return False

    try:
        from qdrant_client.models import PointStruct
        from datetime import datetime, timezone

        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=emb.tolist(),
            payload={
                "type": "character_reference",
                "character_slug": character_slug,
                "project_id": project_id,
                "image_path": image_path,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        client = _get_qdrant()
        client.upsert(collection_name=COLLECTION_NAME, points=[point])
        logger.info(f"clip_scorer: stored reference for {character_slug} from {image_path}")
        return True
    except Exception as e:
        logger.warning(f"clip_scorer: failed to store reference: {e}")
        return False


async def backfill_from_shots(limit: int = 200) -> dict[str, int]:
    """Backfill CLIP embeddings from existing completed shots in anime_production DB.

    Connects to anime_production directly and embeds last_frame_path for each shot.
    """
    import asyncpg

    stats = {"processed": 0, "stored": 0, "skipped": 0, "failed": 0}

    try:
        conn = await asyncpg.connect(
            host="localhost",
            user="patrick",
            password="RP78eIrW7cI2jYvL5akt1yurE",
            database="anime_production",
        )
    except Exception as e:
        logger.error(f"clip_scorer backfill: DB connection failed: {e}")
        return stats

    try:
        rows = await conn.fetch("""
            SELECT sh.id::text as shot_id, sh.scene_id::text as scene_id,
                   s.project_id, sh.last_frame_path, sh.video_engine,
                   sh.generation_prompt, sh.characters_present
            FROM shots sh
            JOIN scenes s ON sh.scene_id = s.id
            WHERE sh.status = 'completed' AND sh.last_frame_path IS NOT NULL
            ORDER BY sh.created_at DESC
            LIMIT $1
        """, limit)

        ensure_collection()

        for row in rows:
            stats["processed"] += 1
            frame_path = row["last_frame_path"]

            if not frame_path or not Path(frame_path).exists():
                stats["skipped"] += 1
                continue

            try:
                result = await evaluate_generation(
                    image_path=frame_path,
                    prompt_text=row["generation_prompt"] or "",
                    shot_id=row["shot_id"],
                    scene_id=row["scene_id"],
                    project_id=row["project_id"],
                    character_slugs=row["characters_present"],
                    video_engine=row["video_engine"] or "",
                )
                if result.get("embedding_stored"):
                    stats["stored"] += 1
                else:
                    stats["failed"] += 1
            except Exception as e:
                logger.warning(f"clip_scorer backfill: shot {row['shot_id'][:8]} failed: {e}")
                stats["failed"] += 1

        logger.info(f"clip_scorer backfill: {stats}")
    finally:
        await conn.close()

    return stats
