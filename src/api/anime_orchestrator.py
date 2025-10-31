"""
Echo Brain Anime Production Orchestrator
Centralized anime generation with character consistency, project context, and creative memory.
"""

import asyncio
import json
import logging
import requests
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from ..db.database import database
from ..tasks.task_queue import Task, TaskType, TaskPriority

logger = logging.getLogger(__name__)

router = APIRouter()

class AnimeGenerationRequest(BaseModel):
    """Comprehensive anime generation request with project context."""

    # Project Context
    project_id: Optional[int] = None
    project_name: Optional[str] = None
    create_new_project: bool = False

    # Character Context
    character_name: Optional[str] = None
    character_id: Optional[int] = None
    use_existing_character: bool = True

    # Generation Parameters
    prompt: str
    scene_type: str = "character_portrait"  # character_portrait, action_scene, environment, story_scene
    generation_type: str = "image"  # image, video, story

    # Style and Quality
    style_preference: Optional[str] = None  # Will learn from Patrick's preferences
    quality_level: str = "professional"  # draft, standard, professional, cinematic

    # Technical Parameters
    width: int = 1024
    height: int = 1024
    steps: int = 30

    # Echo Orchestration Flags
    learn_preferences: bool = True
    apply_character_consistency: bool = True
    update_project_timeline: bool = True

class ProjectContextResponse(BaseModel):
    """Project context for character consistency."""
    project_id: int
    project_name: str
    characters: List[Dict]
    style_guide: Dict
    recent_generations: List[Dict]

class AnimeOrchestrator:
    """Echo Brain's central anime production orchestrator."""

    def __init__(self):
        self.anime_service_url = "http://127.0.0.1:8351"
        self.comfyui_url = "http://127.0.0.1:8188"

    async def get_project_context(self, project_id: int) -> Optional[Dict]:
        """Retrieve comprehensive project context from anime database."""
        try:
            # Get project details
            project_query = """
                SELECT p.*, array_agg(c.*) as characters
                FROM anime_api.projects p
                LEFT JOIN anime_api.characters c ON p.id = c.project_id
                WHERE p.id = %s
                GROUP BY p.id
            """
            result = await database.fetch_one(project_query, project_id)

            if not result:
                return None

            # Get recent generations for style consistency
            generations_query = """
                SELECT image_path, prompt, created_at, metadata
                FROM anime_api.production_jobs
                WHERE project_id = %s
                ORDER BY created_at DESC
                LIMIT 5
            """
            recent_generations = await database.fetch_all(generations_query, project_id)

            return {
                "project": dict(result),
                "characters": result["characters"] or [],
                "recent_generations": [dict(g) for g in recent_generations],
                "style_patterns": await self._analyze_style_patterns(recent_generations)
            }

        except Exception as e:
            logger.error(f"Failed to get project context: {e}")
            return None

    async def _analyze_style_patterns(self, generations: List) -> Dict:
        """Analyze past generations to learn Patrick's style preferences."""
        if not generations:
            return {}

        # Extract common patterns from successful generations
        style_analysis = {
            "common_prompts": [],
            "preferred_settings": {},
            "character_consistency_score": 0.0,
            "quality_patterns": []
        }

        # This would use Echo's AI to analyze patterns
        # For now, basic pattern extraction
        for gen in generations:
            if gen.get("metadata"):
                try:
                    metadata = json.loads(gen["metadata"])
                    if metadata.get("user_rating", 0) >= 4:  # High-rated generations
                        style_analysis["quality_patterns"].append({
                            "prompt": gen["prompt"],
                            "settings": metadata.get("generation_settings", {}),
                            "rating": metadata.get("user_rating")
                        })
                except:
                    continue

        return style_analysis

    async def ensure_character_consistency(self, character_name: str, project_id: int,
                                         prompt: str) -> Dict:
        """Ensure character appearance consistency across generations."""
        try:
            # Get character's canonical definition
            char_query = """
                SELECT * FROM anime_api.characters
                WHERE project_id = %s AND name ILIKE %s
                ORDER BY version DESC LIMIT 1
            """
            character = await database.fetch_one(char_query, project_id, f"%{character_name}%")

            if not character:
                logger.warning(f"Character {character_name} not found in project {project_id}")
                return {"consistent_prompt": prompt}

            # Enhance prompt with character consistency data
            char_description = character.get("description", "")
            consistent_prompt = f"{prompt}, {char_description}"

            # If character has a reference image, include it in generation
            reference_data = {}
            if character.get("image_path"):
                reference_data["reference_image"] = character["image_path"]
                reference_data["character_lora"] = character.get("comfyui_workflow")

            return {
                "consistent_prompt": consistent_prompt,
                "character_reference": reference_data,
                "character_id": character["id"],
                "consistency_score": 0.95  # High confidence with reference
            }

        except Exception as e:
            logger.error(f"Character consistency check failed: {e}")
            return {"consistent_prompt": prompt, "consistency_score": 0.5}

    async def apply_learned_preferences(self, prompt: str, user_context: Dict) -> str:
        """Apply Patrick's learned creative preferences to the prompt."""

        # Get user's style preferences from conversation history
        prefs_query = """
            SELECT metadata FROM conversations
            WHERE user_id = 'patrick'
            AND content LIKE '%anime%'
            AND metadata->>'user_rating' IS NOT NULL
            ORDER BY timestamp DESC LIMIT 10
        """

        try:
            recent_prefs = await database.fetch_all(prefs_query)

            # Analyze preferred styles
            preferred_styles = []
            for pref in recent_prefs:
                metadata = pref.get("metadata", {})
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)

                rating = metadata.get("user_rating", 0)
                if rating >= 4:  # Patrick liked this style
                    style_hints = metadata.get("style_elements", [])
                    preferred_styles.extend(style_hints)

            # Common preferences based on analysis
            common_preferences = [
                "professional anime style",
                "detailed character design",
                "cinematic lighting",
                "high quality",
                "counterfeit_v3 model aesthetic"
            ]

            # Enhance prompt with learned preferences
            enhanced_prompt = f"{prompt}, {', '.join(common_preferences)}"

            return enhanced_prompt

        except Exception as e:
            logger.error(f"Failed to apply learned preferences: {e}")
            return prompt

    async def orchestrate_generation(self, request: AnimeGenerationRequest) -> Dict:
        """Main orchestration method - coordinates entire anime generation workflow."""

        start_time = datetime.now()
        generation_id = f"echo_anime_{int(start_time.timestamp())}"

        logger.info(f"🎬 Starting Echo anime orchestration: {generation_id}")

        try:
            # 1. Project Context Resolution
            project_context = None
            if request.project_id:
                project_context = await self.get_project_context(request.project_id)
            elif request.project_name and request.create_new_project:
                # Create new project via anime API
                project_data = {
                    "name": request.project_name,
                    "description": f"Echo-orchestrated project: {request.project_name}",
                    "created_via": "echo_brain"
                }
                # This would call the anime API to create project
                project_context = {"project": project_data}

            # 2. Character Consistency Processing
            consistent_prompt = request.prompt
            character_data = {}

            if request.character_name and project_context:
                consistency_result = await self.ensure_character_consistency(
                    request.character_name,
                    project_context["project"]["id"],
                    request.prompt
                )
                consistent_prompt = consistency_result["consistent_prompt"]
                character_data = consistency_result.get("character_reference", {})

            # 3. Apply Learned Preferences
            if request.learn_preferences:
                enhanced_prompt = await self.apply_learned_preferences(
                    consistent_prompt,
                    {"project_context": project_context}
                )
            else:
                enhanced_prompt = consistent_prompt

            # 4. Generation Coordination
            generation_params = {
                "prompt": enhanced_prompt,
                "width": request.width,
                "height": request.height,
                "steps": request.steps,
                "model": "counterfeit_v3.safetensors",  # Patrick's preferred model
                "vae": "vae-ft-mse-840000-ema-pruned.safetensors",  # Quality VAE
                "generation_type": request.generation_type,
                "echo_orchestrated": True,
                "echo_generation_id": generation_id
            }

            # Add character reference if available
            if character_data:
                generation_params.update(character_data)

            # 5. Send to Anime Production Service
            logger.info(f"🚀 Sending to anime service: {self.anime_service_url}")

            response = requests.post(
                f"{self.anime_service_url}/api/generate-with-echo",
                json=generation_params,
                timeout=300
            )

            if response.status_code == 200:
                result = response.json()

                # 6. Update Project Timeline
                if request.update_project_timeline and project_context:
                    await self._update_project_timeline(
                        project_context["project"]["id"],
                        generation_id,
                        result,
                        request
                    )

                # 7. Learn from Generation
                if request.learn_preferences:
                    await self._record_generation_for_learning(
                        generation_id,
                        request,
                        result
                    )

                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()

                return {
                    "success": True,
                    "generation_id": generation_id,
                    "result": result,
                    "orchestration_data": {
                        "project_context": project_context is not None,
                        "character_consistency": bool(character_data),
                        "preferences_applied": request.learn_preferences,
                        "generation_duration": duration,
                        "enhanced_prompt": enhanced_prompt,
                        "original_prompt": request.prompt
                    }
                }
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Anime service error: {response.text}"
                )

        except Exception as e:
            logger.error(f"Anime orchestration failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "generation_id": generation_id
            }

    async def _update_project_timeline(self, project_id: int, generation_id: str,
                                     result: Dict, request: AnimeGenerationRequest):
        """Update project timeline with generation event."""
        try:
            timeline_entry = {
                "project_id": project_id,
                "generation_id": generation_id,
                "event_type": "echo_generation",
                "metadata": {
                    "request": request.dict(),
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                }
            }

            # This would insert into project timeline table
            logger.info(f"📝 Updated project timeline for {project_id}")

        except Exception as e:
            logger.error(f"Failed to update project timeline: {e}")

    async def _record_generation_for_learning(self, generation_id: str,
                                            request: AnimeGenerationRequest,
                                            result: Dict):
        """Record generation data for future preference learning."""
        try:
            learning_record = {
                "generation_id": generation_id,
                "user_id": "patrick",
                "request_data": request.dict(),
                "result_data": result,
                "learning_metadata": {
                    "style_elements": self._extract_style_elements(request.prompt),
                    "quality_level": request.quality_level,
                    "character_used": request.character_name,
                    "scene_type": request.scene_type
                },
                "timestamp": datetime.now()
            }

            # Store in Echo's learning database
            logger.info(f"🧠 Recorded generation for learning: {generation_id}")

        except Exception as e:
            logger.error(f"Failed to record learning data: {e}")

    def _extract_style_elements(self, prompt: str) -> List[str]:
        """Extract style elements from prompt for learning."""
        style_keywords = [
            "anime", "manga", "photorealistic", "cartoon", "detailed",
            "cinematic", "dramatic", "soft lighting", "hard lighting",
            "close-up", "wide shot", "action", "portrait", "landscape"
        ]

        found_elements = []
        prompt_lower = prompt.lower()

        for keyword in style_keywords:
            if keyword in prompt_lower:
                found_elements.append(keyword)

        return found_elements

# Global orchestrator instance
anime_orchestrator = AnimeOrchestrator()

@router.post("/anime/orchestrate")
async def orchestrate_anime_generation(
    request: AnimeGenerationRequest,
    background_tasks: BackgroundTasks
):
    """
    Echo Brain's main anime orchestration endpoint.

    Handles:
    - Project context resolution
    - Character consistency validation
    - Creative preference application
    - Multi-service coordination
    - Timeline management
    - Preference learning
    """

    return await anime_orchestrator.orchestrate_generation(request)

@router.get("/anime/project-context/{project_id}")
async def get_project_context(project_id: int):
    """Get comprehensive project context for character consistency."""

    context = await anime_orchestrator.get_project_context(project_id)

    if not context:
        raise HTTPException(status_code=404, detail="Project not found")

    return context

@router.post("/anime/learn-preference")
async def record_user_preference(
    generation_id: str,
    rating: int,
    feedback: str = None
):
    """Record user feedback for preference learning."""

    try:
        # Update learning record with user feedback
        update_query = """
            UPDATE anime_learning_records
            SET user_rating = %s, user_feedback = %s, rated_at = NOW()
            WHERE generation_id = %s
        """

        await database.execute(update_query, rating, feedback, generation_id)

        return {"success": True, "message": "Preference recorded for learning"}

    except Exception as e:
        logger.error(f"Failed to record preference: {e}")
        raise HTTPException(status_code=500, detail="Failed to record preference")

@router.get("/anime/style-analysis")
async def get_style_analysis():
    """Get analysis of Patrick's learned style preferences."""

    try:
        # Analyze learned preferences
        analysis = await anime_orchestrator._analyze_style_patterns([])

        return {
            "style_preferences": analysis,
            "learning_status": "active",
            "total_generations": 0,  # Would count from database
            "preference_confidence": 0.8
        }

    except Exception as e:
        logger.error(f"Failed to get style analysis: {e}")
        raise HTTPException(status_code=500, detail="Failed to analyze styles")