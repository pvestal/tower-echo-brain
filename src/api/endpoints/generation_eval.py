"""Generation evaluation endpoint — CLIP-based quality scoring for anime production.

POST /api/echo/generation-eval/evaluate
  Scores a generated frame/image against prompt text, reference images, and recent shots.

POST /api/echo/generation-eval/store-reference
  Stores a character reference image for semantic scoring.

POST /api/echo/generation-eval/backfill
  Backfills CLIP embeddings from existing completed shots.

GET /api/echo/generation-eval/health
  Health check for CLIP scorer service.
"""

import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/generation-eval", tags=["generation-eval"])


class EvaluateRequest(BaseModel):
    image_path: str
    prompt_text: str = ""
    shot_id: str = ""
    scene_id: str = ""
    project_id: int = 0
    character_slugs: list[str] = []
    video_engine: str = ""
    parameters: dict[str, Any] = {}


class StoreReferenceRequest(BaseModel):
    image_path: str
    character_slug: str
    project_id: int = 0


class BackfillRequest(BaseModel):
    limit: int = 200


@router.post("/evaluate")
async def evaluate_generation(request: EvaluateRequest) -> dict[str, Any]:
    """Score a generated image using CLIP embeddings.

    Returns semantic_score, variety_score, text_alignment, mhp_bucket,
    and optionally flags too-similar shots with suggestions.
    """
    from src.services.clip_scorer import evaluate_generation as _evaluate

    try:
        result = await _evaluate(
            image_path=request.image_path,
            prompt_text=request.prompt_text,
            shot_id=request.shot_id,
            scene_id=request.scene_id,
            project_id=request.project_id,
            character_slugs=request.character_slugs or None,
            video_engine=request.video_engine,
            parameters=request.parameters,
        )
        return result
    except Exception as e:
        logger.error(f"evaluate_generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/store-reference")
async def store_reference(request: StoreReferenceRequest) -> dict[str, Any]:
    """Store a character reference image embedding for semantic scoring."""
    from src.services.clip_scorer import store_character_reference

    try:
        ok = await store_character_reference(
            image_path=request.image_path,
            character_slug=request.character_slug,
            project_id=request.project_id,
        )
        return {"stored": ok, "character_slug": request.character_slug}
    except Exception as e:
        logger.error(f"store_reference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/backfill")
async def backfill(request: BackfillRequest) -> dict[str, Any]:
    """Backfill CLIP embeddings from existing completed shots."""
    from src.services.clip_scorer import backfill_from_shots

    try:
        stats = await backfill_from_shots(limit=request.limit)
        return stats
    except Exception as e:
        logger.error(f"backfill failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def eval_health() -> dict[str, Any]:
    """Health check for CLIP scorer."""
    from src.services.clip_scorer import (
        _clip_model, _clip_device, COLLECTION_NAME, VECTOR_DIM,
    )

    status = {
        "service": "clip_scorer",
        "model_loaded": _clip_model is not None,
        "device": _clip_device,
        "collection": COLLECTION_NAME,
        "vector_dim": VECTOR_DIM,
    }

    # Check Qdrant collection
    try:
        from src.services.clip_scorer import _get_qdrant
        client = _get_qdrant()
        info = client.get_collection(COLLECTION_NAME)
        status["collection_points"] = info.points_count
        status["status"] = "healthy"
    except Exception as e:
        status["collection_points"] = 0
        status["collection_error"] = str(e)
        status["status"] = "degraded"

    return status
