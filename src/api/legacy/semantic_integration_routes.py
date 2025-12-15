#!/usr/bin/env python3
"""
Semantic Integration Routes for Echo Brain
Exposes intelligent creative orchestration with semantic memory capabilities
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import traceback

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

# Import our semantic memory client
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.services.semantic_memory_client import SemanticMemoryClient, SemanticEnhancedOrchestrator

logger = logging.getLogger(__name__)

# Create router with prefix
router = APIRouter(prefix="/api/echo", tags=["semantic_integration"])

# Request/Response models
class SemanticPromptEnhanceRequest(BaseModel):
    """Request for semantic prompt enhancement"""
    prompt: str = Field(..., description="Original prompt to enhance")
    generation_type: str = Field("anime_character", description="Type of generation")
    character_name: Optional[str] = Field(None, description="Character name if applicable")

class SemanticPromptEnhanceResponse(BaseModel):
    """Response for semantic prompt enhancement"""
    original_prompt: str
    enhanced_prompt: str
    similar_prompts: List[Dict[str, Any]]
    character_consistency: Dict[str, Any]
    style_recommendations: Dict[str, Any]
    semantic_enhancement: Dict[str, Any]
    processing_time_ms: float

class CharacterConsistencyRequest(BaseModel):
    """Request for character consistency check"""
    character_description: str = Field(..., description="Description of character")
    character_name: Optional[str] = Field(None, description="Character name if known")

class CharacterConsistencyResponse(BaseModel):
    """Response for character consistency check"""
    character_description: str
    character_name: Optional[str]
    similar_characters: List[Dict[str, Any]]
    consistency_score: float
    recommendations: List[str]
    processing_time_ms: float

class CreativeDNAUpdateRequest(BaseModel):
    """Request for creative DNA update"""
    generation_result: Dict[str, Any] = Field(..., description="Generation result data")
    quality_scores: Dict[str, float] = Field(..., description="Quality scores")
    user_feedback: Optional[str] = Field(None, description="User feedback")

class CreativeDNAUpdateResponse(BaseModel):
    """Response for creative DNA update"""
    generation_id: Optional[str]
    quality_scores: Dict[str, float]
    learning_stored: bool
    dna_evolution: Dict[str, Any]
    processing_time_ms: float

class SemanticQualityValidationRequest(BaseModel):
    """Request for semantic quality validation"""
    generated_content: Dict[str, Any] = Field(..., description="Generated content data")
    original_prompt: str = Field(..., description="Original prompt")

class SemanticQualityValidationResponse(BaseModel):
    """Response for semantic quality validation"""
    original_prompt: str
    similar_successful_generations: List[Dict[str, Any]]
    semantic_consistency: Dict[str, Any]
    style_adherence: Dict[str, Any]
    quality_prediction: Dict[str, Any]
    processing_time_ms: float

# Global orchestrator instance
semantic_orchestrator: Optional[SemanticEnhancedOrchestrator] = None

async def get_semantic_orchestrator() -> SemanticEnhancedOrchestrator:
    """Get semantic orchestrator instance"""
    global semantic_orchestrator
    if semantic_orchestrator is None:
        semantic_orchestrator = SemanticEnhancedOrchestrator()
    return semantic_orchestrator

@router.get("/semantic/health")
async def semantic_health_check():
    """Check semantic integration health"""
    try:
        async with SemanticMemoryClient() as client:
            health = await client.health_check()
            return {
                "status": "healthy",
                "semantic_memory": health,
                "integration": "active",
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"Semantic health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@router.post("/semantic/enhance-prompt", response_model=SemanticPromptEnhanceResponse)
async def enhance_prompt_with_semantic_memory(request: SemanticPromptEnhanceRequest):
    """
    Enhance prompt generation using semantic memory

    This endpoint:
    1. Searches for similar past prompts
    2. Extracts successful patterns
    3. Checks character consistency
    4. Provides style recommendations
    5. Generates semantically enhanced prompt
    """
    start_time = datetime.now()

    try:
        orchestrator = await get_semantic_orchestrator()

        # Enhance prompt with semantic memory
        enhancement = await orchestrator.enhance_prompt_with_semantic_memory(
            original_prompt=request.prompt,
            generation_type=request.generation_type
        )

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return SemanticPromptEnhanceResponse(
            original_prompt=enhancement["original_prompt"],
            enhanced_prompt=enhancement["enhanced_prompt"],
            similar_prompts=enhancement.get("similar_prompts", []),
            character_consistency=enhancement.get("character_consistency", {}),
            style_recommendations=enhancement.get("style_recommendations", {}),
            semantic_enhancement=enhancement.get("semantic_enhancement", {}),
            processing_time_ms=processing_time
        )

    except Exception as e:
        logger.error(f"Prompt enhancement failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Prompt enhancement failed: {e}")

@router.post("/semantic/check-character-consistency", response_model=CharacterConsistencyResponse)
async def check_character_consistency(request: CharacterConsistencyRequest):
    """
    Check character consistency against stored character memories

    This endpoint:
    1. Searches for similar characters in semantic memory
    2. Calculates consistency score
    3. Provides recommendations for maintaining consistency
    4. Identifies potential conflicts or opportunities
    """
    start_time = datetime.now()

    try:
        orchestrator = await get_semantic_orchestrator()

        # Check character consistency
        consistency_report = await orchestrator.check_character_consistency(
            character_description=request.character_description,
            character_name=request.character_name
        )

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return CharacterConsistencyResponse(
            character_description=consistency_report["character_description"],
            character_name=consistency_report.get("character_name"),
            similar_characters=consistency_report.get("similar_characters", []),
            consistency_score=consistency_report.get("consistency_score", 0.5),
            recommendations=consistency_report.get("recommendations", []),
            processing_time_ms=processing_time
        )

    except Exception as e:
        logger.error(f"Character consistency check failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Character consistency check failed: {e}")

@router.post("/semantic/update-creative-dna", response_model=CreativeDNAUpdateResponse)
async def update_creative_dna(request: CreativeDNAUpdateRequest):
    """
    Update creative DNA based on generation results

    This endpoint:
    1. Analyzes generation success/failure
    2. Stores learning data in semantic memory
    3. Updates creative patterns and preferences
    4. Evolves generation strategies
    """
    start_time = datetime.now()

    try:
        orchestrator = await get_semantic_orchestrator()

        # Update creative DNA
        dna_update = await orchestrator.update_creative_dna(
            generation_result=request.generation_result,
            quality_scores=request.quality_scores,
            user_feedback=request.user_feedback
        )

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return CreativeDNAUpdateResponse(
            generation_id=dna_update.get("generation_id"),
            quality_scores=request.quality_scores,
            learning_stored=dna_update.get("learning_stored", False),
            dna_evolution=dna_update.get("dna_evolution", {}),
            processing_time_ms=processing_time
        )

    except Exception as e:
        logger.error(f"Creative DNA update failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Creative DNA update failed: {e}")

@router.post("/semantic/validate-quality", response_model=SemanticQualityValidationResponse)
async def validate_generation_quality(request: SemanticQualityValidationRequest):
    """
    Validate generation quality using semantic analysis

    This endpoint:
    1. Compares generation to similar successful examples
    2. Validates semantic consistency
    3. Checks style adherence
    4. Predicts quality based on patterns
    """
    start_time = datetime.now()

    try:
        orchestrator = await get_semantic_orchestrator()

        # Validate quality
        validation = await orchestrator.get_semantic_quality_validation(
            generated_content=request.generated_content,
            original_prompt=request.original_prompt
        )

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return SemanticQualityValidationResponse(
            original_prompt=validation["original_prompt"],
            similar_successful_generations=validation.get("similar_successful_generations", []),
            semantic_consistency=validation.get("semantic_consistency", {}),
            style_adherence=validation.get("style_adherence", {}),
            quality_prediction=validation.get("quality_prediction", {}),
            processing_time_ms=processing_time
        )

    except Exception as e:
        logger.error(f"Quality validation failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Quality validation failed: {e}")

@router.post("/semantic/search")
async def semantic_search(
    query: str,
    content_type: Optional[str] = None,
    max_results: int = 10,
    similarity_threshold: float = 0.7
):
    """
    Perform semantic search across stored content

    Direct access to semantic search capabilities
    """
    try:
        async with SemanticMemoryClient() as client:
            results = await client.search_similar_content(
                query=query,
                content_type=content_type,
                max_results=max_results,
                similarity_threshold=similarity_threshold
            )
            return results

    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Semantic search failed: {e}")

class StoreGenerationRequest(BaseModel):
    """Request for storing generation data"""
    prompt: str = Field(..., description="Generation prompt")
    generation_params: Dict[str, Any] = Field(..., description="Generation parameters")
    quality_scores: Dict[str, float] = Field(..., description="Quality scores")
    job_id: Optional[str] = Field(None, description="Job ID")
    character_name: Optional[str] = Field(None, description="Character name")
    user_feedback: Optional[str] = Field(None, description="User feedback")

@router.post("/semantic/store-generation")
async def store_generation_for_learning(
    background_tasks: BackgroundTasks,
    request: StoreGenerationRequest
):
    """
    Store generation data for semantic learning

    Stores generation in background for future semantic search and learning
    """
    try:
        # Add to background tasks for async processing
        background_tasks.add_task(
            _store_generation_background,
            prompt=request.prompt,
            generation_params=request.generation_params,
            quality_scores=request.quality_scores,
            job_id=request.job_id,
            character_name=request.character_name,
            user_feedback=request.user_feedback
        )

        return {
            "status": "queued",
            "message": "Generation data queued for semantic storage",
            "job_id": request.job_id,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Generation storage queueing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Storage queueing failed: {e}")

async def _store_generation_background(
    prompt: str,
    generation_params: Dict[str, Any],
    quality_scores: Dict[str, float],
    job_id: Optional[str] = None,
    character_name: Optional[str] = None,
    user_feedback: Optional[str] = None
):
    """Background task for storing generation data"""
    try:
        orchestrator = await get_semantic_orchestrator()

        generation_result = {
            "prompt": prompt,
            "generation_params": generation_params,
            "job_id": job_id,
            "character_name": character_name
        }

        await orchestrator.update_creative_dna(
            generation_result=generation_result,
            quality_scores=quality_scores,
            user_feedback=user_feedback
        )

        logger.info(f"Successfully stored generation data for job {job_id}")

    except Exception as e:
        logger.error(f"Background generation storage failed: {e}")
        logger.error(traceback.format_exc())

@router.get("/semantic/stats")
async def get_semantic_stats():
    """
    Get semantic memory system statistics

    Returns information about stored content, learning progress, etc.
    """
    try:
        async with SemanticMemoryClient() as client:
            # Get health info which includes some stats
            health = await client.health_check()

            # Search for some basic stats
            all_prompts = await client.search_similar_content(
                query="anime",
                content_type="prompt",
                max_results=100,
                similarity_threshold=0.0  # Get all
            )

            all_generations = await client.search_similar_content(
                query="generation",
                content_type="generation_result",
                max_results=100,
                similarity_threshold=0.0  # Get all
            )

            stats = {
                "service_health": health.get("status", "unknown"),
                "stored_prompts": len(all_prompts.get("results", [])),
                "stored_generations": len(all_generations.get("results", [])),
                "embedding_dimension": health.get("embedding_test", {}).get("embedding_size", 0),
                "models_loaded": health.get("models", {}),
                "timestamp": datetime.now().isoformat()
            }

            return stats

    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}")
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Integration with existing Echo Brain workflow
class WorkflowEnhanceRequest(BaseModel):
    """Request for workflow enhancement"""
    original_prompt: str = Field(..., description="Original user prompt")
    generation_type: str = Field("anime_character", description="Generation type")
    character_name: Optional[str] = Field(None, description="Character name")

@router.post("/semantic/workflow/enhance")
async def enhance_generation_workflow(request: WorkflowEnhanceRequest):
    """
    Complete semantic enhancement workflow

    This is the main integration point that combines all semantic capabilities:
    1. Enhance prompt with semantic memory
    2. Check character consistency
    3. Provide quality predictions
    4. Return complete enhancement package
    """
    start_time = datetime.now()

    try:
        orchestrator = await get_semantic_orchestrator()

        # Step 1: Enhance prompt
        enhancement = await orchestrator.enhance_prompt_with_semantic_memory(
            original_prompt=request.original_prompt,
            generation_type=request.generation_type
        )

        # Step 2: Check character consistency if character mentioned
        character_consistency = {}
        if request.character_name or "character" in request.original_prompt.lower():
            character_desc = f"{request.character_name} {request.original_prompt}" if request.character_name else request.original_prompt
            character_consistency = await orchestrator.check_character_consistency(
                character_description=character_desc,
                character_name=request.character_name
            )

        # Step 3: Quality prediction based on similar content
        similar_content = enhancement.get("similar_prompts", [])
        quality_prediction = _predict_quality_from_similar_content(similar_content)

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        workflow_result = {
            "original_prompt": request.original_prompt,
            "enhanced_prompt": enhancement.get("enhanced_prompt", request.original_prompt),
            "semantic_enhancement": enhancement.get("semantic_enhancement", {}),
            "style_recommendations": enhancement.get("style_recommendations", {}),
            "character_consistency": character_consistency,
            "quality_prediction": quality_prediction,
            "similar_examples": similar_content,
            "processing_time_ms": processing_time,
            "timestamp": datetime.now().isoformat()
        }

        return workflow_result

    except Exception as e:
        logger.error(f"Semantic workflow enhancement failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Workflow enhancement failed: {e}")

def _predict_quality_from_similar_content(similar_content: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Helper function to predict quality from similar content"""
    if not similar_content:
        return {"predicted_quality": 5.0, "confidence": 0.1, "message": "No similar content found"}

    # Extract quality scores from metadata
    qualities = []
    for content in similar_content:
        metadata = content.get("metadata", {})
        if "average_quality" in metadata:
            qualities.append(metadata["average_quality"])

    if qualities:
        avg_quality = sum(qualities) / len(qualities)
        confidence = min(len(qualities) / 5.0, 1.0)  # Higher confidence with more samples

        return {
            "predicted_quality": round(avg_quality, 1),
            "confidence": round(confidence, 2),
            "based_on_samples": len(qualities),
            "message": f"Prediction based on {len(qualities)} similar generations"
        }
    else:
        return {
            "predicted_quality": 6.0,
            "confidence": 0.3,
            "based_on_samples": 0,
            "message": "No quality data in similar content"
        }