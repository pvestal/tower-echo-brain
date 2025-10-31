#!/usr/bin/env python3
"""
Echo Brain Anime Production Director API
Advanced anime orchestration endpoints serving as the central coordinator for anime workflows.

This module implements Echo as the intelligent Production Director that:
- Analyzes user intent and determines optimal generation paths
- Manages project context and character consistency across sessions
- Applies learned style preferences and quality standards
- Coordinates multi-service workflows seamlessly
- Provides git-like version control for creative decisions
"""

import asyncio
import json
import logging
import os
import requests
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from pydantic import BaseModel, Field
import sys

sys.path.append('/opt/tower-echo-brain/src')

from db.database import database
from tasks.task_queue import Task, TaskType, TaskPriority
from anime.echo_anime_coordinator import EchoAnimeCoordinator, AnimeRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/echo/anime", tags=["Echo Anime Production Director"])

class RequestType(str, Enum):
    """Types of anime generation requests"""
    NEW_PROJECT = "new_project"
    CONTINUE_EXISTING = "continue_existing"
    CHARACTER_FOCUS = "character_focus"
    SCENE_FOCUS = "scene_focus"
    STORY_DEVELOPMENT = "story_development"
    STYLE_EXPERIMENT = "style_experiment"

class GenerationContext(str, Enum):
    """Context for generation routing"""
    PROJECT_BOUND = "project_bound"
    CHARACTER_BOUND = "character_bound"
    STANDALONE = "standalone"
    CONTINUATION = "continuation"

class IntentClassification(BaseModel):
    """Result of intent analysis"""
    request_type: RequestType
    context: GenerationContext
    confidence: float
    project_binding: Optional[str] = None
    character_binding: Optional[str] = None
    service_route: str
    reasoning: str
    required_context: List[str] = []

class CoordinationRequest(BaseModel):
    """Main coordination request model"""
    prompt: str = Field(..., description="User's generation prompt")
    user_id: str = Field("patrick", description="User identifier")
    platform: str = Field("echo_brain", description="Source platform (echo_brain, telegram, browser)")
    session_id: Optional[str] = Field(None, description="Session ID for context continuity")

    # Context hints (optional)
    project_hint: Optional[str] = Field(None, description="Project name hint")
    character_hint: Optional[str] = Field(None, description="Character name hint")
    style_hint: Optional[str] = Field(None, description="Style preference hint")

    # Generation preferences
    quality_level: str = Field("professional", description="Quality level")
    generation_type: str = Field("image", description="Type: image, video, story")
    apply_learning: bool = Field(True, description="Apply learned preferences")

    # Override flags
    force_new_project: bool = Field(False, description="Force creation of new project")
    ignore_context: bool = Field(False, description="Ignore existing context")

class ProjectSummary(BaseModel):
    """Project summary for listing"""
    project_id: Optional[int]
    project_name: str
    character_count: int
    generation_count: int
    last_activity: datetime
    style_signature: Dict[str, Any]
    timeline_depth: int
    active_sessions: int

class CharacterProfile(BaseModel):
    """Character profile with consistency data"""
    name: str
    project_context: Optional[str]
    visual_description: str
    consistency_score: float
    generation_count: int
    last_used: datetime
    learned_traits: Dict[str, Any]
    reference_images: List[str]

class TimelineEntry(BaseModel):
    """Timeline entry for project version control"""
    entry_id: str
    timestamp: datetime
    action: str
    description: str
    generation_id: Optional[str]
    branch_point: bool
    metadata: Dict[str, Any]

class FeedbackData(BaseModel):
    """Feedback for learning system"""
    generation_id: str
    rating: int = Field(..., ge=1, le=5)
    style_feedback: Optional[str] = None
    character_feedback: Optional[str] = None
    quality_feedback: Optional[str] = None
    preference_updates: Optional[Dict[str, Any]] = None

class AnimeProductionDirector:
    """Echo Brain's intelligent anime production director"""

    def __init__(self):
        self.coordinator = EchoAnimeCoordinator()
        self.intent_classifier = IntentClassifier()
        self.project_manager = ProjectContextManager()
        self.character_manager = CharacterConsistencyManager()
        self.timeline_manager = TimelineManager()
        self.learning_engine = StyleLearningEngine()

    async def analyze_intent(self, request: CoordinationRequest) -> IntentClassification:
        """Analyze user intent and determine optimal generation path"""
        try:
            # Multi-factor intent analysis
            prompt_analysis = await self._analyze_prompt_intent(request.prompt)
            context_analysis = await self._analyze_context_clues(request)
            session_analysis = await self._analyze_session_context(request.session_id, request.user_id)

            # Weighted scoring system
            intent_scores = {
                RequestType.NEW_PROJECT: 0.0,
                RequestType.CONTINUE_EXISTING: 0.0,
                RequestType.CHARACTER_FOCUS: 0.0,
                RequestType.SCENE_FOCUS: 0.0,
                RequestType.STORY_DEVELOPMENT: 0.0,
                RequestType.STYLE_EXPERIMENT: 0.0
            }

            # Prompt-based scoring
            if any(word in request.prompt.lower() for word in ["new", "create", "start", "begin"]):
                intent_scores[RequestType.NEW_PROJECT] += 0.3

            if any(word in request.prompt.lower() for word in ["continue", "more", "next", "another"]):
                intent_scores[RequestType.CONTINUE_EXISTING] += 0.4

            if any(word in request.prompt.lower() for word in ["character", "person", "face", "portrait"]):
                intent_scores[RequestType.CHARACTER_FOCUS] += 0.3

            if any(word in request.prompt.lower() for word in ["scene", "background", "environment", "setting"]):
                intent_scores[RequestType.SCENE_FOCUS] += 0.3

            if any(word in request.prompt.lower() for word in ["story", "sequence", "narrative", "timeline"]):
                intent_scores[RequestType.STORY_DEVELOPMENT] += 0.4

            if any(word in request.prompt.lower() for word in ["style", "artistic", "experiment", "different"]):
                intent_scores[RequestType.STYLE_EXPERIMENT] += 0.2

            # Context-based scoring
            if request.project_hint:
                intent_scores[RequestType.CONTINUE_EXISTING] += 0.4
            if request.character_hint:
                intent_scores[RequestType.CHARACTER_FOCUS] += 0.4
            if request.force_new_project:
                intent_scores[RequestType.NEW_PROJECT] = 1.0

            # Session-based scoring
            if session_analysis.get("has_active_project"):
                intent_scores[RequestType.CONTINUE_EXISTING] += 0.3
            if session_analysis.get("recent_character_work"):
                intent_scores[RequestType.CHARACTER_FOCUS] += 0.2

            # Determine winning intent
            top_intent = max(intent_scores.items(), key=lambda x: x[1])
            confidence = min(top_intent[1], 1.0)

            # Determine context binding
            context = GenerationContext.STANDALONE
            project_binding = None
            character_binding = None

            if session_analysis.get("active_project"):
                context = GenerationContext.PROJECT_BOUND
                project_binding = session_analysis["active_project"]

            if request.character_hint or session_analysis.get("active_character"):
                context = GenerationContext.CHARACTER_BOUND
                character_binding = request.character_hint or session_analysis["active_character"]

            # Determine service routing
            service_route = await self._determine_service_route(top_intent[0], context)

            # Generate reasoning
            reasoning = f"Intent '{top_intent[0].value}' selected with {confidence:.2f} confidence. "
            reasoning += f"Context: {context.value}. "
            if project_binding:
                reasoning += f"Project binding: {project_binding}. "
            if character_binding:
                reasoning += f"Character binding: {character_binding}. "

            return IntentClassification(
                request_type=top_intent[0],
                context=context,
                confidence=confidence,
                project_binding=project_binding,
                character_binding=character_binding,
                service_route=service_route,
                reasoning=reasoning,
                required_context=await self._get_required_context(top_intent[0], context)
            )

        except Exception as e:
            logger.error(f"Intent analysis failed: {e}")
            # Fallback to safe default
            return IntentClassification(
                request_type=RequestType.STANDALONE,
                context=GenerationContext.STANDALONE,
                confidence=0.5,
                service_route="anime_production",
                reasoning=f"Fallback due to analysis error: {str(e)}"
            )

    async def _analyze_prompt_intent(self, prompt: str) -> Dict[str, Any]:
        """Analyze prompt text for intent clues"""
        # This could be enhanced with LLM analysis
        words = prompt.lower().split()

        return {
            "character_references": len([w for w in words if w in ["character", "person", "face"]]),
            "scene_references": len([w for w in words if w in ["scene", "background", "environment"]]),
            "continuation_signals": len([w for w in words if w in ["continue", "more", "next"]]),
            "new_project_signals": len([w for w in words if w in ["new", "create", "start"]]),
            "word_count": len(words),
            "complexity_score": len(words) / 10.0  # Simple complexity metric
        }

    async def _analyze_context_clues(self, request: CoordinationRequest) -> Dict[str, Any]:
        """Analyze request context clues"""
        return {
            "has_project_hint": bool(request.project_hint),
            "has_character_hint": bool(request.character_hint),
            "has_style_hint": bool(request.style_hint),
            "platform": request.platform,
            "force_flags": request.force_new_project or request.ignore_context
        }

    async def _analyze_session_context(self, session_id: Optional[str], user_id: str) -> Dict[str, Any]:
        """Analyze session context for continuity"""
        if not session_id:
            return {}

        try:
            # Query recent session activity
            query = """
                SELECT project_binding_id, active_character, context_state
                FROM anime_echo_sessions
                WHERE session_id = %s AND user_id = %s
                ORDER BY last_activity DESC LIMIT 1
            """
            result = await database.fetch_one(query, session_id, user_id)

            if not result:
                return {}

            return {
                "has_active_project": bool(result["project_binding_id"]),
                "active_project": result["project_binding_id"],
                "active_character": result["active_character"],
                "context_state": result["context_state"] or {}
            }

        except Exception as e:
            logger.error(f"Session context analysis failed: {e}")
            return {}

    async def _determine_service_route(self, intent: RequestType, context: GenerationContext) -> str:
        """Determine which service should handle the request"""
        routing_map = {
            (RequestType.NEW_PROJECT, GenerationContext.STANDALONE): "anime_production",
            (RequestType.CONTINUE_EXISTING, GenerationContext.PROJECT_BOUND): "anime_production",
            (RequestType.CHARACTER_FOCUS, GenerationContext.CHARACTER_BOUND): "character_studio",
            (RequestType.SCENE_FOCUS, GenerationContext.STANDALONE): "comfyui_direct",
            (RequestType.STORY_DEVELOPMENT, GenerationContext.PROJECT_BOUND): "story_orchestrator",
            (RequestType.STYLE_EXPERIMENT, GenerationContext.STANDALONE): "style_lab"
        }

        return routing_map.get((intent, context), "anime_production")

    async def _get_required_context(self, intent: RequestType, context: GenerationContext) -> List[str]:
        """Get list of required context elements"""
        required = []

        if context == GenerationContext.PROJECT_BOUND:
            required.append("project_data")
        if context == GenerationContext.CHARACTER_BOUND:
            required.append("character_profile")
        if intent == RequestType.CONTINUE_EXISTING:
            required.append("generation_history")

        return required

class IntentClassifier:
    """Advanced intent classification for anime requests"""

    async def classify(self, prompt: str, context: Dict) -> IntentClassification:
        """Classify request intent using multiple analysis methods"""
        # Placeholder for advanced classification logic
        pass

class ProjectContextManager:
    """Manages project context and memory"""

    async def get_user_projects(self, user_id: str) -> List[ProjectSummary]:
        """Get user's anime projects with metadata"""
        try:
            query = """
                SELECT
                    pb.anime_project_id,
                    pb.project_name,
                    COUNT(DISTINCT acm.character_name) as character_count,
                    COUNT(DISTINCT asl.generation_id) as generation_count,
                    MAX(pb.last_accessed) as last_activity,
                    COUNT(DISTINCT aes.session_id) as active_sessions
                FROM anime_echo_project_bindings pb
                LEFT JOIN anime_echo_character_memory acm ON pb.id = acm.project_binding_id
                LEFT JOIN anime_echo_style_learning asl ON asl.context_tags->>'project_id' = pb.id::text
                LEFT JOIN anime_echo_sessions aes ON pb.id = aes.project_binding_id
                WHERE pb.user_id = %s AND pb.active = true
                GROUP BY pb.id, pb.anime_project_id, pb.project_name
                ORDER BY last_activity DESC
            """

            results = await database.fetch_all(query, user_id)

            projects = []
            for row in results:
                # Get style signature
                style_signature = await self._get_project_style_signature(row["anime_project_id"])

                # Calculate timeline depth
                timeline_depth = await self._get_timeline_depth(row["anime_project_id"])

                projects.append(ProjectSummary(
                    project_id=row["anime_project_id"],
                    project_name=row["project_name"],
                    character_count=row["character_count"] or 0,
                    generation_count=row["generation_count"] or 0,
                    last_activity=row["last_activity"],
                    style_signature=style_signature,
                    timeline_depth=timeline_depth,
                    active_sessions=row["active_sessions"] or 0
                ))

            return projects

        except Exception as e:
            logger.error(f"Failed to get user projects: {e}")
            return []

    async def _get_project_style_signature(self, project_id: Optional[int]) -> Dict[str, Any]:
        """Get project's learned style signature"""
        if not project_id:
            return {}

        try:
            query = """
                SELECT style_elements, learned_preferences
                FROM anime_echo_style_learning
                WHERE context_tags->>'project_id' = %s
                ORDER BY created_at DESC LIMIT 10
            """

            results = await database.fetch_all(query, str(project_id))

            # Aggregate style patterns
            style_signature = {
                "preferred_styles": [],
                "common_elements": [],
                "quality_preferences": {},
                "generation_patterns": {}
            }

            for row in results:
                if row["style_elements"]:
                    style_signature["common_elements"].extend(row["style_elements"].get("tags", []))
                if row["learned_preferences"]:
                    style_signature["preferred_styles"].append(row["learned_preferences"])

            return style_signature

        except Exception as e:
            logger.error(f"Failed to get style signature: {e}")
            return {}

    async def _get_timeline_depth(self, project_id: Optional[int]) -> int:
        """Get project timeline depth (number of major events)"""
        if not project_id:
            return 0

        try:
            # This would query a timeline events table
            # For now, estimate based on generation count
            query = """
                SELECT COUNT(*) as timeline_events
                FROM anime_echo_style_learning
                WHERE context_tags->>'project_id' = %s
            """

            result = await database.fetch_one(query, str(project_id))
            return result["timeline_events"] if result else 0

        except Exception as e:
            logger.error(f"Failed to get timeline depth: {e}")
            return 0

class CharacterConsistencyManager:
    """Manages character consistency across generations"""

    async def get_character_profiles(self, user_id: str, project_id: Optional[int] = None) -> List[CharacterProfile]:
        """Get character profiles with consistency data"""
        try:
            if project_id:
                query = """
                    SELECT acm.*, pb.project_name
                    FROM anime_echo_character_memory acm
                    JOIN anime_echo_project_bindings pb ON acm.project_binding_id = pb.id
                    WHERE pb.user_id = %s AND pb.anime_project_id = %s
                    ORDER BY acm.last_used DESC
                """
                params = [user_id, project_id]
            else:
                query = """
                    SELECT acm.*, pb.project_name
                    FROM anime_echo_character_memory acm
                    JOIN anime_echo_project_bindings pb ON acm.project_binding_id = pb.id
                    WHERE pb.user_id = %s
                    ORDER BY acm.last_used DESC
                """
                params = [user_id]

            results = await database.fetch_all(query, *params)

            characters = []
            for row in results:
                # Get reference images
                reference_images = await self._get_character_references(row["character_name"])

                characters.append(CharacterProfile(
                    name=row["character_name"],
                    project_context=row.get("project_name"),
                    visual_description=row["learned_traits"].get("visual_description", ""),
                    consistency_score=row["consistency_score"],
                    generation_count=row["generation_count"],
                    last_used=row["last_used"],
                    learned_traits=row["learned_traits"] or {},
                    reference_images=reference_images
                ))

            return characters

        except Exception as e:
            logger.error(f"Failed to get character profiles: {e}")
            return []

    async def _get_character_references(self, character_name: str) -> List[str]:
        """Get reference images for character"""
        # This would query generation results for character images
        # For now, return empty list
        return []

class TimelineManager:
    """Manages project timeline and version control"""

    async def get_project_timeline(self, project_id: int, user_id: str) -> List[TimelineEntry]:
        """Get project timeline with git-like version control"""
        try:
            # This would query a comprehensive timeline table
            # For now, create timeline from generation history
            query = """
                SELECT generation_id, created_at, prompt_used, quality_assessment
                FROM anime_echo_style_learning
                WHERE context_tags->>'project_id' = %s
                ORDER BY created_at DESC
                LIMIT 50
            """

            results = await database.fetch_all(query, str(project_id))

            timeline = []
            for i, row in enumerate(results):
                timeline.append(TimelineEntry(
                    entry_id=f"tl_{row['generation_id']}",
                    timestamp=row["created_at"],
                    action="generation",
                    description=f"Generated: {row['prompt_used'][:50]}...",
                    generation_id=row["generation_id"],
                    branch_point=i % 5 == 0,  # Mark every 5th as potential branch point
                    metadata={
                        "quality": row["quality_assessment"] or {},
                        "prompt": row["prompt_used"]
                    }
                ))

            return timeline

        except Exception as e:
            logger.error(f"Failed to get project timeline: {e}")
            return []

class StyleLearningEngine:
    """Handles style learning and feedback processing"""

    async def record_feedback(self, feedback: FeedbackData, user_id: str) -> Dict[str, Any]:
        """Record user feedback for learning"""
        try:
            # Update existing learning record
            update_query = """
                UPDATE anime_echo_style_learning
                SET user_feedback = %s, feedback_weight = %s
                WHERE generation_id = %s
            """

            feedback_data = {
                "rating": feedback.rating,
                "style_feedback": feedback.style_feedback,
                "character_feedback": feedback.character_feedback,
                "quality_feedback": feedback.quality_feedback,
                "preference_updates": feedback.preference_updates
            }

            # Weight calculation based on rating
            weight = feedback.rating / 5.0

            await database.execute(update_query, json.dumps(feedback_data), weight, feedback.generation_id)

            # Update learned preferences
            preference_updates = await self._process_preference_updates(feedback, user_id)

            return {
                "success": True,
                "feedback_recorded": True,
                "preference_updates": preference_updates,
                "learning_weight": weight
            }

        except Exception as e:
            logger.error(f"Failed to record feedback: {e}")
            return {"success": False, "error": str(e)}

    async def _process_preference_updates(self, feedback: FeedbackData, user_id: str) -> Dict[str, Any]:
        """Process feedback into preference updates"""
        updates = {}

        if feedback.rating >= 4:  # Positive feedback
            updates["positive_patterns"] = {
                "style_elements": feedback.style_feedback,
                "quality_aspects": feedback.quality_feedback
            }
        elif feedback.rating <= 2:  # Negative feedback
            updates["avoid_patterns"] = {
                "style_elements": feedback.style_feedback,
                "quality_issues": feedback.quality_feedback
            }

        return updates

# Initialize global instances
director = AnimeProductionDirector()

# API Endpoints

@router.post("/coordinate")
async def coordinate_anime_generation(
    request: CoordinationRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Main coordination endpoint - Echo's intelligent anime orchestration.

    Analyzes user intent, determines project context, applies learned preferences,
    and routes to appropriate generation services with full workflow management.
    """
    try:
        logger.info(f"🎭 Echo anime coordination request: {request.prompt[:50]}...")

        # 1. Analyze intent and determine routing
        intent_analysis = await director.analyze_intent(request)

        logger.info(f"🧠 Intent analysis: {intent_analysis.request_type.value} "
                   f"(confidence: {intent_analysis.confidence:.2f})")

        # 2. Gather required context
        context_data = await director._gather_required_context(intent_analysis, request)

        # 3. Create enhanced anime request
        anime_request = AnimeRequest(
            prompt=request.prompt,
            project_name=intent_analysis.project_binding,
            character_name=intent_analysis.character_binding,
            user_id=request.user_id,
            platform=request.platform,
            session_id=request.session_id,
            apply_learning=request.apply_learning,
            quality_level=request.quality_level,
            generation_type=request.generation_type
        )

        # 4. Coordinate generation through Echo system
        generation_result = await director.coordinator.coordinate_generation(anime_request)

        # 5. Return comprehensive response
        return {
            "success": generation_result.get("success", False),
            "generation_id": generation_result.get("orchestration_id"),
            "intent_analysis": {
                "request_type": intent_analysis.request_type.value,
                "context": intent_analysis.context.value,
                "confidence": intent_analysis.confidence,
                "reasoning": intent_analysis.reasoning,
                "service_route": intent_analysis.service_route
            },
            "coordination_data": {
                "project_binding": intent_analysis.project_binding,
                "character_binding": intent_analysis.character_binding,
                "required_context": intent_analysis.required_context,
                "context_applied": list(context_data.keys()) if context_data else []
            },
            "generation_result": generation_result,
            "workflow_metadata": {
                "platform": request.platform,
                "session_id": request.session_id,
                "learning_applied": request.apply_learning,
                "timestamp": datetime.now().isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Anime coordination failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/projects")
async def get_user_projects(
    user_id: str = "patrick",
    include_inactive: bool = False
) -> Dict[str, Any]:
    """
    Get user's anime projects with intelligent metadata and context.
    Provides project selection intelligence and management capabilities.
    """
    try:
        projects = await director.project_manager.get_user_projects(user_id)

        # Add intelligent project selection recommendations
        recommendations = await director._get_project_recommendations(projects, user_id)

        return {
            "projects": [proj.dict() for proj in projects],
            "total_count": len(projects),
            "active_count": len([p for p in projects if p.active_sessions > 0]),
            "recommendations": recommendations,
            "user_analytics": {
                "total_generations": sum(p.generation_count for p in projects),
                "total_characters": sum(p.character_count for p in projects),
                "most_active_project": projects[0].project_name if projects else None,
                "avg_timeline_depth": sum(p.timeline_depth for p in projects) / len(projects) if projects else 0
            }
        }

    except Exception as e:
        logger.error(f"Failed to get projects: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/character")
async def coordinate_character_generation(
    character_name: str,
    prompt: str,
    project_id: Optional[int] = None,
    consistency_mode: str = "strict",
    evolution_allowed: bool = False,
    user_id: str = "patrick"
) -> Dict[str, Any]:
    """
    Character-focused coordination with consistency management.
    Handles character consistency across scenes, reference management,
    and character evolution tracking.
    """
    try:
        logger.info(f"🎭 Character coordination: {character_name}")

        # Get character profile and consistency data
        characters = await director.character_manager.get_character_profiles(user_id, project_id)
        target_character = next((c for c in characters if c.name.lower() == character_name.lower()), None)

        if not target_character and consistency_mode == "strict":
            raise HTTPException(
                status_code=404,
                detail=f"Character '{character_name}' not found in strict consistency mode"
            )

        # Build character-consistent generation request
        character_request = CoordinationRequest(
            prompt=f"{prompt}, {character_name}",
            user_id=user_id,
            character_hint=character_name,
            project_hint=str(project_id) if project_id else None
        )

        # Enhanced prompt with character consistency
        if target_character:
            consistent_prompt = await director._enhance_prompt_with_character_data(
                prompt, target_character, consistency_mode
            )
            character_request.prompt = consistent_prompt

        # Coordinate generation
        result = await coordinate_anime_generation(character_request, BackgroundTasks())

        # Update character evolution if allowed
        if evolution_allowed and result.get("success"):
            await director._update_character_evolution(
                character_name, result["generation_id"], prompt, user_id
            )

        return {
            **result,
            "character_data": {
                "character_name": character_name,
                "consistency_mode": consistency_mode,
                "found_existing": target_character is not None,
                "consistency_score": target_character.consistency_score if target_character else 0.0,
                "evolution_applied": evolution_allowed,
                "reference_count": len(target_character.reference_images) if target_character else 0
            }
        }

    except Exception as e:
        logger.error(f"Character coordination failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/feedback")
async def record_generation_feedback(
    feedback: FeedbackData,
    user_id: str = "patrick"
) -> Dict[str, Any]:
    """
    Record generation feedback for learning system.
    Updates style preferences, character evolution, and quality standards.
    """
    try:
        logger.info(f"🧠 Recording feedback for generation {feedback.generation_id}")

        # Record feedback through learning engine
        learning_result = await director.learning_engine.record_feedback(feedback, user_id)

        # Update character data if character feedback provided
        character_updates = {}
        if feedback.character_feedback:
            character_updates = await director._process_character_feedback(feedback, user_id)

        # Update style preferences
        style_updates = await director._process_style_feedback(feedback, user_id)

        return {
            "success": learning_result.get("success", False),
            "feedback_processed": True,
            "learning_updates": {
                "style_preferences": style_updates,
                "character_evolution": character_updates,
                "feedback_weight": learning_result.get("learning_weight", 0.0)
            },
            "system_improvements": {
                "preference_confidence": await director._calculate_preference_confidence(user_id),
                "learning_completeness": await director._calculate_learning_completeness(user_id)
            }
        }

    except Exception as e:
        logger.error(f"Feedback recording failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/timeline/{project_id}")
async def get_project_timeline(
    project_id: int,
    user_id: str = "patrick",
    branch_view: bool = False,
    limit: int = 50
) -> Dict[str, Any]:
    """
    Get project timeline with git-like version control.
    Provides story branch management and scene sequence planning.
    """
    try:
        timeline = await director.timeline_manager.get_project_timeline(project_id, user_id)

        # Apply branch view if requested
        if branch_view:
            timeline = await director._create_branch_view(timeline)

        # Get timeline analytics
        analytics = await director._get_timeline_analytics(timeline, project_id)

        return {
            "timeline": [entry.dict() for entry in timeline[:limit]],
            "total_entries": len(timeline),
            "branch_view": branch_view,
            "analytics": analytics,
            "navigation": {
                "branch_points": len([e for e in timeline if e.branch_point]),
                "major_milestones": await director._identify_major_milestones(timeline),
                "suggested_branches": await director._suggest_timeline_branches(timeline, project_id)
            }
        }

    except Exception as e:
        logger.error(f"Timeline retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper methods for the director class would be implemented here
# These are placeholder implementations that would need to be fully developed

async def _gather_required_context(self, intent_analysis: IntentClassification, request: CoordinationRequest) -> Dict[str, Any]:
    """Gather required context based on intent analysis"""
    context = {}

    for requirement in intent_analysis.required_context:
        if requirement == "project_data" and intent_analysis.project_binding:
            context["project"] = await self._get_project_data(intent_analysis.project_binding)
        elif requirement == "character_profile" and intent_analysis.character_binding:
            context["character"] = await self._get_character_data(intent_analysis.character_binding)
        elif requirement == "generation_history":
            context["history"] = await self._get_generation_history(request.user_id)

    return context

# Add this method to the AnimeProductionDirector class
AnimeProductionDirector._gather_required_context = _gather_required_context

logger.info("🎬 Echo Brain Anime Production Director API loaded")