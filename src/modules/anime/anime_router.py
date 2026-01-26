#!/usr/bin/env python3
"""
Echo Brain Anime Module - AI Director for Tower Anime Production
Provides intelligent scene planning, prompt refinement, and learning from feedback
"""

from fastapi import APIRouter, HTTPException, Depends
import os
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import asyncio
import asyncpg
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/echo/anime", tags=["anime"])

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'echo_brain',
    'user': 'patrick',
    'password': os.getenv("TOWER_DB_PASSWORD", "RP78eIrW7cI2jYvL5akt1yurE")
}

# ============= Pydantic Models =============

class ScenePlanRequest(BaseModel):
    session_id: str
    scene_description: str
    style_references: List[str] = Field(default_factory=list)
    characters_in_scene: List[str] = Field(default_factory=list)
    duration_seconds: int = 10

class ShotPlan(BaseModel):
    shot_number: int
    description: str
    suggested_camera_angle: str  # "wide", "medium", "close-up", "over-shoulder"
    character_emotions: Dict[str, str]  # character -> emotion
    suggested_poses: List[Dict[str, Any]]
    duration_seconds: float
    background_context: Optional[str] = None

class ScenePlanResponse(BaseModel):
    shot_list: List[ShotPlan]
    overall_mood: str
    lighting_suggestions: str
    style_keywords: List[str]
    narrative_arc: str

class PromptRefineRequest(BaseModel):
    session_id: str
    raw_prompt: str
    context_tags: List[str] = Field(default_factory=list)
    character_name: Optional[str] = None
    current_emotion: Optional[str] = None
    camera_angle: Optional[str] = None

class PromptRefineResponse(BaseModel):
    enhanced_prompt: str
    style_keywords: List[str]
    negative_prompt: str
    cinematic_terms: List[str]
    character_consistency_hints: List[str]

class FeedbackLearnRequest(BaseModel):
    session_id: str
    prompt_used: str
    quality_scores: Dict[str, float]  # "ssim", "optical_flow", "character_consistency"
    generation_id: str
    context_tags: List[str] = Field(default_factory=list)
    user_feedback: Optional[str] = None

class FeedbackLearnResponse(BaseModel):
    learned_elements: List[str]
    updated_confidence_scores: Dict[str, float]
    recommendations_for_next: List[str]

# ============= Helper Functions =============

async def get_db_connection():
    """Get database connection to echo_brain"""
    return await asyncpg.connect(**DB_CONFIG)

async def get_session_context(session_id: str) -> Dict:
    """Load session context - simplified for now without creative_sessions"""
    conn = await get_db_connection()
    try:
        # For now, just get a default project context
        result = await conn.fetchrow("""
            SELECT
                p.id as project_id,
                p.name as project_name,
                p.description as project_description
            FROM anime.projects p
            LIMIT 1
        """)

        if result:
            context = dict(result)
            context['genre'] = 'anime'  # Default genre
            return context
        return {}
    finally:
        await conn.close()

async def get_character_info(character_name: str) -> Dict:
    """Get character information via FDW"""
    conn = await get_db_connection()
    try:
        result = await conn.fetchrow("""
            SELECT
                id,
                name,
                description,
                lora_trigger,
                lora_path
            FROM anime.characters
            WHERE LOWER(name) = LOWER($1)
        """, character_name)

        if result:
            return dict(result)
        return {}
    finally:
        await conn.close()

async def store_learning_feedback(feedback_data: Dict):
    """Store feedback in echo_brain for learning"""
    conn = await get_db_connection()
    try:
        await conn.execute("""
            INSERT INTO anime_learning_feedback (
                session_id,
                prompt_used,
                quality_scores,
                learned_patterns,
                created_at
            ) VALUES ($1, $2, $3, $4, $5)
        """,
        feedback_data['session_id'],
        feedback_data['prompt_used'],
        json.dumps(feedback_data['quality_scores']),
        feedback_data.get('learned_patterns', []),
        datetime.now()
        )
    finally:
        await conn.close()

# ============= API Endpoints =============

@router.post("/scene/plan", response_model=ScenePlanResponse)
async def plan_scene(request: ScenePlanRequest):
    """
    AI-assisted scene planning: breaks down scene description into shots
    Uses narrative context from anime_production via FDW
    """
    try:
        # Load session context from anime_production
        context = await get_session_context(request.session_id)

        # Get character information
        character_data = {}
        for char_name in request.characters_in_scene:
            char_info = await get_character_info(char_name)
            if char_info:
                character_data[char_name] = char_info

        # Generate scene breakdown based on description and context
        # This is a simplified version - in production, this would use the model router
        shot_list = []

        # Example shot planning logic
        if "action" in request.scene_description.lower():
            # Action scene - more dynamic shots
            shot_list = [
                ShotPlan(
                    shot_number=1,
                    description="Establishing wide shot of the scene",
                    suggested_camera_angle="wide",
                    character_emotions={char: "determined" for char in request.characters_in_scene},
                    suggested_poses=[{"type": "standing", "intensity": "neutral"}],
                    duration_seconds=2.0,
                    background_context="Set the scene location"
                ),
                ShotPlan(
                    shot_number=2,
                    description="Medium shot focusing on main character",
                    suggested_camera_angle="medium",
                    character_emotions={request.characters_in_scene[0]: "intense" if request.characters_in_scene else ""},
                    suggested_poses=[{"type": "action_pose", "intensity": "high"}],
                    duration_seconds=3.0,
                    background_context="Focus on character reaction"
                ),
                ShotPlan(
                    shot_number=3,
                    description="Close-up for emotional impact",
                    suggested_camera_angle="close-up",
                    character_emotions={char: "focused" for char in request.characters_in_scene},
                    suggested_poses=[{"type": "face_closeup", "intensity": "high"}],
                    duration_seconds=2.0,
                    background_context="Emotional climax"
                )
            ]
        else:
            # Default scene - standard coverage
            shot_list = [
                ShotPlan(
                    shot_number=1,
                    description="Wide establishing shot",
                    suggested_camera_angle="wide",
                    character_emotions={char: "neutral" for char in request.characters_in_scene},
                    suggested_poses=[{"type": "standing", "intensity": "neutral"}],
                    duration_seconds=3.0,
                    background_context="Establish location and mood"
                ),
                ShotPlan(
                    shot_number=2,
                    description="Medium shot for dialogue",
                    suggested_camera_angle="medium",
                    character_emotions={char: "engaged" for char in request.characters_in_scene},
                    suggested_poses=[{"type": "conversational", "intensity": "medium"}],
                    duration_seconds=4.0,
                    background_context="Main interaction"
                )
            ]

        # Determine mood and style based on context
        genre = context.get('genre', 'anime')
        overall_mood = "dramatic" if "action" in request.scene_description.lower() else "conversational"

        return ScenePlanResponse(
            shot_list=shot_list,
            overall_mood=overall_mood,
            lighting_suggestions="High contrast for drama" if overall_mood == "dramatic" else "Soft, natural lighting",
            style_keywords=request.style_references + [genre, "cinematic"],
            narrative_arc="Building tension" if overall_mood == "dramatic" else "Character development"
        )

    except Exception as e:
        logger.error(f"Scene planning failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/prompt/refine", response_model=PromptRefineResponse)
async def refine_prompt(request: PromptRefineRequest):
    """
    Enhance raw user prompts with cinematic terms and style consistency
    Uses character data and successful patterns from past generations
    """
    try:
        # Get character data if specified
        character_style = {}
        if request.character_name:
            char_info = await get_character_info(request.character_name)
            if char_info:
                character_style = {
                    'lora_trigger': char_info.get('lora_trigger', ''),
                    'description': char_info.get('description', '')
                }

        # Base enhancement
        enhanced_parts = [request.raw_prompt]

        # Add character-specific triggers
        if character_style.get('lora_trigger'):
            enhanced_parts.append(character_style['lora_trigger'])

        # Add emotion modifiers
        if request.current_emotion:
            emotion_terms = {
                'happy': 'joyful expression, bright eyes, slight smile',
                'sad': 'melancholic expression, downcast eyes, subtle frown',
                'angry': 'intense expression, furrowed brow, determined look',
                'surprised': 'wide eyes, open mouth, shocked expression'
            }
            if request.current_emotion in emotion_terms:
                enhanced_parts.append(emotion_terms[request.current_emotion])

        # Add camera angle terms
        if request.camera_angle:
            camera_terms = {
                'wide': 'wide shot, full body visible, environmental context',
                'medium': 'medium shot, waist up, clear character details',
                'close-up': 'close-up shot, detailed face, emotional focus',
                'over-shoulder': 'over the shoulder shot, depth, perspective'
            }
            if request.camera_angle in camera_terms:
                enhanced_parts.append(camera_terms[request.camera_angle])

        # Build enhanced prompt
        enhanced_prompt = ', '.join(enhanced_parts)

        # Add quality and style keywords
        style_keywords = ['masterpiece', 'best quality', 'detailed', 'anime style'] + request.context_tags

        # Negative prompt to avoid common issues
        negative_prompt = 'worst quality, low quality, blurry, bad anatomy, bad hands, missing fingers, extra digit, fewer digits'

        # Cinematic terms
        cinematic_terms = ['cinematic lighting', 'professional photography', 'depth of field']

        # Character consistency hints
        consistency_hints = []
        if character_style:
            consistency_hints.append(f"Maintain {request.character_name}'s canonical appearance")
            if character_style.get('lora_trigger'):
                consistency_hints.append(f"Use trigger: {character_style['lora_trigger']}")

        return PromptRefineResponse(
            enhanced_prompt=enhanced_prompt,
            style_keywords=style_keywords,
            negative_prompt=negative_prompt,
            cinematic_terms=cinematic_terms,
            character_consistency_hints=consistency_hints
        )

    except Exception as e:
        logger.error(f"Prompt refinement failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/feedback/learn", response_model=FeedbackLearnResponse)
async def learn_from_feedback(request: FeedbackLearnRequest):
    """
    Process generation quality feedback to improve future suggestions
    Stores patterns and updates confidence scores
    """
    try:
        # Analyze quality scores
        avg_quality = sum(request.quality_scores.values()) / len(request.quality_scores) if request.quality_scores else 0

        learned_elements = []
        recommendations = []

        # Determine what worked or didn't
        if avg_quality > 0.7:
            learned_elements.append(f"High quality prompt pattern: {request.prompt_used[:100]}")
            recommendations.append("Continue using similar prompt structure")
        elif avg_quality < 0.4:
            learned_elements.append(f"Low quality prompt pattern detected")
            recommendations.append("Adjust prompt parameters and retry")

            # Specific recommendations based on scores
            if request.quality_scores.get('ssim', 1.0) < 0.5:
                recommendations.append("Focus on character consistency - add more specific descriptors")
            if request.quality_scores.get('optical_flow', 1.0) < 0.5:
                recommendations.append("Improve motion smoothness - simplify action descriptions")

        # Store feedback for future learning
        await store_learning_feedback({
            'session_id': request.session_id,
            'prompt_used': request.prompt_used,
            'quality_scores': request.quality_scores,
            'learned_patterns': learned_elements
        })

        # Calculate confidence scores
        confidence_scores = {
            'prompt_quality': avg_quality,
            'character_consistency': request.quality_scores.get('character_consistency', 0.5),
            'overall_confidence': avg_quality * 0.8 + 0.2  # Baseline confidence
        }

        return FeedbackLearnResponse(
            learned_elements=learned_elements,
            updated_confidence_scores=confidence_scores,
            recommendations_for_next=recommendations
        )

    except Exception as e:
        logger.error(f"Feedback learning failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Check if anime module is operational"""
    try:
        # Test database connection
        conn = await get_db_connection()
        await conn.fetchval("SELECT 1")
        await conn.close()

        # Test FDW access
        conn = await get_db_connection()
        project_count = await conn.fetchval("SELECT COUNT(*) FROM anime.projects")
        await conn.close()

        return {
            "status": "healthy",
            "module": "anime",
            "database": "connected",
            "fdw_access": "working",
            "project_count": project_count
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }