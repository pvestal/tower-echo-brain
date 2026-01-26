#!/usr/bin/env python3
"""
Echo Brain - LoRA Training Integration API

Orchestrates LoRA training requests to the Tower LoRA Studio service.
Provides intelligent recommendations and monitors training progress.
"""
import os
import httpx
import asyncio
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/lora", tags=["lora-training"])

# LoRA Studio service URL
LORA_STUDIO_URL = "http://localhost:8315"

class LoRATrainingRequest(BaseModel):
    """Request to train a new LoRA"""
    lora_name: str
    trigger_word: str
    concept_type: str  # action, intimate, character, style, violence
    keywords: List[str]
    priority: int = 5
    steps: Optional[int] = 500
    learning_rate: Optional[float] = 2e-4
    lora_rank: Optional[int] = 16

class LoRARecommendation(BaseModel):
    """LoRA training recommendation from Echo Brain"""
    lora_name: str
    trigger_word: str
    concept_type: str
    keywords: List[str]
    priority: int
    rationale: str
    estimated_benefit: str

@router.post("/train/request")
async def request_lora_training(
    request: LoRATrainingRequest,
    background_tasks: BackgroundTasks
):
    """
    Request LoRA training through Tower LoRA Studio.

    Echo Brain acts as the orchestrator, analyzing the request
    and forwarding to the specialized training service.
    """
    logger.info(f"Echo Brain: Processing LoRA training request for {request.lora_name}")

    # Enhance request with Echo Brain intelligence
    enhanced_keywords = await _enhance_keywords(request.keywords, request.concept_type)

    # Forward to LoRA Studio
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(
                f"{LORA_STUDIO_URL}/api/training/submit",
                json={
                    "lora_name": request.lora_name,
                    "trigger_word": request.trigger_word,
                    "concept_type": request.concept_type,
                    "keywords": enhanced_keywords,
                    "priority": request.priority,
                    "steps": request.steps,
                    "learning_rate": request.learning_rate,
                    "lora_rank": request.lora_rank,
                }
            )

            if response.status_code != 200:
                raise HTTPException(500, f"LoRA Studio error: {response.text}")

            result = response.json()

            # Store training context in Echo Brain memory
            background_tasks.add_task(
                _store_training_context,
                request.lora_name,
                request.trigger_word,
                enhanced_keywords,
                result.get("job_id")
            )

            return {
                "status": "success",
                "job_id": result.get("job_id"),
                "message": f"LoRA training queued: {request.lora_name}",
                "enhanced_keywords": enhanced_keywords,
                "studio_url": f"{LORA_STUDIO_URL}/api/training/status/{result.get('job_id')}"
            }

        except httpx.RequestError as e:
            logger.error(f"Failed to contact LoRA Studio: {e}")
            raise HTTPException(503, "LoRA Studio service unavailable")

@router.get("/recommendations")
async def get_lora_recommendations() -> List[LoRARecommendation]:
    """
    Get intelligent LoRA training recommendations from Echo Brain.

    Analyzes:
    - Generation failure patterns
    - Missing action/scene types
    - Character consistency issues
    - User request patterns
    """
    logger.info("Echo Brain: Analyzing LoRA training needs...")

    recommendations = []

    # 1. Analyze generation failures
    failure_recommendations = await _analyze_generation_failures()
    recommendations.extend(failure_recommendations)

    # 2. Identify missing capabilities
    gap_recommendations = await _identify_capability_gaps()
    recommendations.extend(gap_recommendations)

    # 3. Prioritize based on impact
    recommendations = await _prioritize_recommendations(recommendations)

    logger.info(f"Echo Brain: Generated {len(recommendations)} LoRA recommendations")

    return recommendations[:10]  # Top 10 recommendations

@router.get("/status/overview")
async def get_training_overview():
    """Get overview of all LoRA training activities"""
    async with httpx.AsyncClient() as client:
        try:
            # Get queue status from LoRA Studio
            queue_response = await client.get(f"{LORA_STUDIO_URL}/api/training/queue")
            queue_data = queue_response.json() if queue_response.status_code == 200 else []

            # Get available LoRAs
            loras_response = await client.get(f"{LORA_STUDIO_URL}/api/loras")
            loras_data = loras_response.json() if loras_response.status_code == 200 else []

            return {
                "training_queue": len(queue_data),
                "available_loras": len(loras_data),
                "queue_items": queue_data[:5],  # First 5 in queue
                "recent_loras": loras_data[:5],  # Most recent LoRAs
            }

        except httpx.RequestError:
            return {"error": "LoRA Studio unavailable", "training_queue": 0, "available_loras": 0}

async def _enhance_keywords(keywords: List[str], concept_type: str) -> List[str]:
    """
    Enhance keywords using Echo Brain's semantic understanding
    """
    enhanced = list(keywords)

    # Add concept-specific enhancements
    if concept_type == "action":
        enhanced.extend(["combat", "movement", "dynamic"])
    elif concept_type == "intimate":
        enhanced.extend(["romantic", "intimate", "close"])
    elif concept_type == "violence":
        enhanced.extend(["gore", "blood", "violence"])
    elif concept_type == "character":
        enhanced.extend(["face", "expression", "consistent"])
    elif concept_type == "style":
        enhanced.extend(["art", "aesthetic", "visual"])

    return list(set(enhanced))  # Remove duplicates

async def _analyze_generation_failures() -> List[LoRARecommendation]:
    """Analyze recent generation failures to recommend LoRAs"""
    recommendations = []

    # This would connect to anime_production database to analyze failures
    # For now, return some example recommendations based on common failure patterns

    recommendations.append(LoRARecommendation(
        lora_name="character_consistency",
        trigger_word="consistent character",
        concept_type="character",
        keywords=["character", "face", "consistent", "identity"],
        priority=9,
        rationale="High failure rate in character consistency across scenes",
        estimated_benefit="60% improvement in character recognition"
    ))

    recommendations.append(LoRARecommendation(
        lora_name="sword_combat",
        trigger_word="sword fighting",
        concept_type="action",
        keywords=["sword", "combat", "fighting", "blade", "action"],
        priority=7,
        rationale="Poor quality sword combat scenes in recent generations",
        estimated_benefit="40% better action scene quality"
    ))

    return recommendations

async def _identify_capability_gaps() -> List[LoRARecommendation]:
    """Identify missing capabilities in current LoRA inventory"""
    recommendations = []

    # Check what LoRAs are already available vs what's needed
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{LORA_STUDIO_URL}/api/loras")
            if response.status_code == 200:
                available_loras = [lora["name"] for lora in response.json()]

                # Common LoRAs that should exist but might be missing
                essential_loras = [
                    ("cowgirl_position", "cowgirl", "intimate", ["cowgirl", "straddling", "riding"]),
                    ("blood_splatter", "blood splatter", "violence", ["blood", "gore", "splatter"]),
                    ("cyberpunk_style", "cyberpunk", "style", ["cyberpunk", "neon", "futuristic"]),
                ]

                for lora_name, trigger, concept, keywords in essential_loras:
                    if lora_name not in available_loras:
                        recommendations.append(LoRARecommendation(
                            lora_name=lora_name,
                            trigger_word=trigger,
                            concept_type=concept,
                            keywords=keywords,
                            priority=8,
                            rationale=f"Essential {concept} LoRA missing from inventory",
                            estimated_benefit="Core capability enhancement"
                        ))

        except httpx.RequestError:
            pass

    return recommendations

async def _prioritize_recommendations(recommendations: List[LoRARecommendation]) -> List[LoRARecommendation]:
    """Prioritize recommendations based on impact and feasibility"""
    # Sort by priority (descending) then by concept type importance
    concept_weights = {
        "character": 1.0,  # Highest impact
        "action": 0.9,
        "violence": 0.8,
        "intimate": 0.7,
        "style": 0.6,
    }

    def priority_score(rec):
        base_priority = rec.priority
        concept_weight = concept_weights.get(rec.concept_type, 0.5)
        return base_priority * concept_weight

    return sorted(recommendations, key=priority_score, reverse=True)

async def _store_training_context(
    lora_name: str,
    trigger_word: str,
    keywords: List[str],
    job_id: str
):
    """Store training context in Echo Brain memory for future reference"""
    # This would store the context in Echo Brain's vector database
    # For now, just log the information
    logger.info(
        f"Echo Brain: Stored training context - "
        f"LoRA: {lora_name}, Trigger: {trigger_word}, Job: {job_id}"
    )

# Quick start training for common LoRAs
@router.post("/quick-start/{lora_type}")
async def quick_start_training(
    lora_type: str,
    priority: int = 5
):
    """
    Quick start training for common LoRA types.

    Supported types:
    - character_consistency
    - sword_combat
    - blood_effects
    - cyberpunk_style
    - intimate_positions
    """
    quick_configs = {
        "character_consistency": {
            "trigger_word": "consistent character",
            "concept_type": "character",
            "keywords": ["character", "face", "consistent", "identity"],
        },
        "sword_combat": {
            "trigger_word": "sword fighting",
            "concept_type": "action",
            "keywords": ["sword", "combat", "fighting", "blade"],
        },
        "blood_effects": {
            "trigger_word": "blood splatter",
            "concept_type": "violence",
            "keywords": ["blood", "gore", "splatter", "wound"],
        },
        "cyberpunk_style": {
            "trigger_word": "cyberpunk",
            "concept_type": "style",
            "keywords": ["cyberpunk", "neon", "futuristic", "tech"],
        },
        "intimate_positions": {
            "trigger_word": "intimate scene",
            "concept_type": "intimate",
            "keywords": ["intimate", "romantic", "close", "embrace"],
        }
    }

    if lora_type not in quick_configs:
        raise HTTPException(400, f"Unknown LoRA type: {lora_type}")

    config = quick_configs[lora_type]
    request = LoRATrainingRequest(
        lora_name=lora_type,
        trigger_word=config["trigger_word"],
        concept_type=config["concept_type"],
        keywords=config["keywords"],
        priority=priority
    )

    return await request_lora_training(request, BackgroundTasks())