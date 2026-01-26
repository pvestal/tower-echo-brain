#!/usr/bin/env python3
"""
Echo Brain Anime Production API
Provides endpoints for LTX video generation through Echo Brain
"""

import logging
from typing import Dict, Optional, List
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
import asyncio

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/api/echo/anime",
    tags=["anime"]
)

# Import anime manager
try:
    from ..managers.anime_production_manager import anime_manager
    ANIME_ENABLED = anime_manager.enabled
except Exception as e:
    logger.error(f"Failed to import anime manager: {e}")
    anime_manager = None
    ANIME_ENABLED = False

# ============= Request/Response Models =============

class AnimeSceneRequest(BaseModel):
    """Request to generate an anime scene"""
    prompt: str
    character: Optional[str] = None
    scene_type: Optional[str] = None
    episode_id: Optional[int] = 1
    quality: Optional[str] = "standard"

class AnimeSceneResponse(BaseModel):
    """Response for anime scene generation"""
    prompt_id: Optional[str] = None
    episode_id: int
    scene_number: int
    scene_type: Optional[str] = None
    status: str
    echo_brain: Optional[Dict] = None
    error: Optional[str] = None

class CharacterSheetRequest(BaseModel):
    """Request to generate character reference sheet"""
    character_name: str
    angles: Optional[int] = 5

class AnimeStatusResponse(BaseModel):
    """Status of anime production system"""
    enabled: bool
    stats: Dict
    recent: List[Dict]
    loras: List[Dict]

# ============= API Endpoints =============

@router.post("/generate", response_model=AnimeSceneResponse)
async def generate_anime_scene(request: AnimeSceneRequest):
    """
    Generate an anime scene using LTX video model

    Examples:
    - "Mei and Hiroshi kissing under moonlight"
    - "Epic fight scene with martial arts"
    - "Character transformation with magical effects"
    """

    if not ANIME_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="Anime production system is not available"
        )

    try:
        result = await anime_manager.generate_scene(
            prompt=request.prompt,
            character=request.character,
            scene_type=request.scene_type,
            episode_id=request.episode_id
        )

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return AnimeSceneResponse(**result)

    except Exception as e:
        logger.error(f"Scene generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/character-sheet")
async def generate_character_sheet(request: CharacterSheetRequest):
    """
    Generate a character reference sheet with multiple angles
    """

    if not ANIME_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="Anime production system is not available"
        )

    try:
        result = await anime_manager.generate_character_sheet(
            request.character_name
        )

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return result

    except Exception as e:
        logger.error(f"Character sheet generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status", response_model=AnimeStatusResponse)
async def get_anime_status():
    """
    Get status of anime production system
    """

    if not ANIME_ENABLED:
        return AnimeStatusResponse(
            enabled=False,
            stats={"status": "disabled"},
            recent=[],
            loras=[]
        )

    try:
        stats = anime_manager.get_production_stats()
        recent = anime_manager.get_recent_generations(5)
        loras = anime_manager.get_available_loras()

        return AnimeStatusResponse(
            enabled=True,
            stats=stats,
            recent=recent,
            loras=loras
        )

    except Exception as e:
        logger.error(f"Failed to get status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/loras")
async def get_available_loras():
    """
    Get list of available LoRAs for video generation
    """

    if not ANIME_ENABLED:
        return []

    try:
        return anime_manager.get_available_loras()

    except Exception as e:
        logger.error(f"Failed to get LoRAs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recent/{limit}")
async def get_recent_generations(limit: int = 10):
    """
    Get recent video generations
    """

    if not ANIME_ENABLED:
        return []

    try:
        return anime_manager.get_recent_generations(limit)

    except Exception as e:
        logger.error(f"Failed to get recent generations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/interpret")
async def interpret_scene_type(prompt: str):
    """
    Interpret natural language to determine scene type

    Example: "romantic kiss" -> "intimate_scene"
    """

    if not ANIME_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="Anime production system is not available"
        )

    try:
        scene_type = anime_manager.interpret_scene_type(prompt)

        return {
            "prompt": prompt,
            "interpreted_type": scene_type,
            "mappings": anime_manager.scene_mappings
        }

    except Exception as e:
        logger.error(f"Failed to interpret scene: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============= Echo Brain Commands =============

async def handle_anime_command(query: str) -> Dict:
    """
    Handle anime-related commands from Echo Brain chat interface

    Commands:
    - "generate anime scene: <description>"
    - "create character sheet for <name>"
    - "show anime status"
    - "list anime loras"
    """

    query_lower = query.lower()

    if "generate anime scene" in query_lower or "create anime" in query_lower:
        # Extract prompt after colon or "of"
        if ":" in query:
            prompt = query.split(":", 1)[1].strip()
        elif " of " in query_lower:
            prompt = query.split(" of ", 1)[1].strip()
        else:
            prompt = query

        result = await anime_manager.generate_scene(prompt)

        if "error" in result:
            return {
                "response": f"Failed to generate scene: {result['error']}",
                "type": "error"
            }

        return {
            "response": f"🎬 Generating anime scene #{result['scene_number']}\n"
                       f"Type: {result['scene_type']}\n"
                       f"Prompt ID: {result['prompt_id']}\n"
                       f"Status: {result['status']}",
            "type": "anime_generation",
            "data": result
        }

    elif "character sheet" in query_lower:
        # Extract character name
        import re
        match = re.search(r'for\s+(\w+)', query, re.IGNORECASE)
        if match:
            character = match.group(1)
            result = await anime_manager.generate_character_sheet(character)

            return {
                "response": f"📋 Generating character sheet for {character}\n"
                           f"References: {len(result.get('references', []))} angles",
                "type": "character_sheet",
                "data": result
            }

    elif "anime status" in query_lower:
        stats = anime_manager.get_production_stats()

        return {
            "response": f"🎬 Anime Production Status\n"
                       f"Total Scenes: {stats.get('total_scenes', 0)}\n"
                       f"LoRAs Available: {stats.get('total_loras', 0)}\n"
                       f"Model: {stats.get('ltx_model', 'Unknown')}\n"
                       f"Status: {stats.get('status', 'Unknown')}",
            "type": "status",
            "data": stats
        }

    elif "anime loras" in query_lower or "list loras" in query_lower:
        loras = anime_manager.get_available_loras()

        lora_list = "\n".join([
            f"- {l['name']} ({l['type']}): {l['trigger_word']}"
            for l in loras
        ])

        return {
            "response": f"🎨 Available LoRAs:\n{lora_list}",
            "type": "lora_list",
            "data": loras
        }

    return {
        "response": "Unknown anime command. Try: 'generate anime scene', "
                   "'character sheet for NAME', 'anime status', or 'list anime loras'",
        "type": "help"
    }

# Export for Echo Brain integration
__all__ = ['router', 'handle_anime_command', 'ANIME_ENABLED']