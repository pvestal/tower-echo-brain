#!/usr/bin/env python3
"""
Echo Anime Central Coordinator
Makes Echo the intelligent orchestrator for all anime production workflows.

This system:
- Coordinates between anime production services and Echo intelligence
- Maintains project memory across sessions
- Applies learned user preferences automatically
- Bridges character consistency between database and JSON files
- Enables cross-platform context persistence (Telegram/Browser)
"""

import asyncio
import json
import logging
import os
import requests
import psycopg2
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import sys
import os
sys.path.append('/opt/tower-echo-brain/src')

from db.database import database
from tasks.task_queue import Task, TaskType, TaskPriority

logger = logging.getLogger(__name__)

@dataclass
class AnimeRequest:
    """Unified anime generation request structure"""
    prompt: str
    project_name: Optional[str] = None
    character_name: Optional[str] = None
    scene_type: str = "character_portrait"
    generation_type: str = "image"
    style_preference: Optional[str] = None
    quality_level: str = "professional"
    width: int = 1024
    height: int = 1024
    steps: int = 30
    user_id: str = "patrick"
    session_id: Optional[str] = None
    platform: str = "echo_brain"  # echo_brain, telegram, browser
    apply_learning: bool = True
    maintain_consistency: bool = True

@dataclass
class ProjectMemory:
    """Project context and memory state"""
    project_id: Optional[int]
    project_name: str
    characters: Dict[str, Dict]
    style_guide: Dict
    generation_history: List[Dict]
    user_preferences: Dict
    last_updated: datetime
    session_context: Dict

class EchoAnimeCoordinator:
    """Central orchestrator making Echo the intelligent anime coordinator"""

    def __init__(self):
        self.anime_service_url = "http://192.168.50.135:8328"
        self.comfyui_url = "http://127.0.0.1:8188"
        self.character_system_path = "/opt/tower-anime-production/characters"

        # Database configuration
        self.db_config = {
            "database": os.environ.get("DB_NAME", "echo_brain"),
            "user": os.environ.get("DB_USER", "patrick")
        }

        # Memory cache for active projects
        self.project_memory_cache = {}
        self.session_contexts = {}

        # Database initialization will be called explicitly
        self._db_initialized = False

    async def initialize_database(self):
        """Initialize anime coordination tables in Echo Brain database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            # Anime-Echo project bindings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS anime_echo_project_bindings (
                    id SERIAL PRIMARY KEY,
                    echo_conversation_id VARCHAR(100),
                    anime_project_id INTEGER,
                    project_name VARCHAR(200) NOT NULL,
                    user_id VARCHAR(100) DEFAULT 'patrick',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    session_context JSONB DEFAULT '{}',
                    memory_context JSONB DEFAULT '{}',
                    active BOOLEAN DEFAULT TRUE
                )
            """)

            # Character consistency tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS anime_echo_character_memory (
                    id SERIAL PRIMARY KEY,
                    character_name VARCHAR(100) NOT NULL,
                    project_binding_id INTEGER REFERENCES anime_echo_project_bindings(id),
                    json_definition_path VARCHAR(500),
                    database_character_id INTEGER,
                    learned_traits JSONB DEFAULT '{}',
                    generation_count INTEGER DEFAULT 0,
                    consistency_score FLOAT DEFAULT 0.0,
                    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    evolution_history JSONB DEFAULT '[]'
                )
            """)

            # Style and preference learning
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS anime_echo_style_learning (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(100) DEFAULT 'patrick',
                    generation_id VARCHAR(200),
                    prompt_used TEXT,
                    style_elements JSONB,
                    quality_assessment JSONB,
                    user_feedback JSONB,
                    learned_preferences JSONB,
                    context_tags JSONB DEFAULT '[]',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    feedback_weight FLOAT DEFAULT 1.0
                )
            """)

            # Cross-platform session management
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS anime_echo_sessions (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(100) UNIQUE NOT NULL,
                    user_id VARCHAR(100) DEFAULT 'patrick',
                    platform VARCHAR(50) NOT NULL,
                    project_binding_id INTEGER REFERENCES anime_echo_project_bindings(id),
                    context_state JSONB DEFAULT '{}',
                    active_character VARCHAR(100),
                    pending_generations JSONB DEFAULT '[]',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP DEFAULT (CURRENT_TIMESTAMP + INTERVAL '24 hours')
                )
            """)

            # Generation orchestration history
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS anime_echo_orchestrations (
                    id SERIAL PRIMARY KEY,
                    orchestration_id VARCHAR(200) UNIQUE NOT NULL,
                    user_id VARCHAR(100) DEFAULT 'patrick',
                    session_id VARCHAR(100),
                    request_data JSONB NOT NULL,
                    project_context JSONB,
                    character_consistency JSONB,
                    applied_preferences JSONB,
                    generation_result JSONB,
                    performance_metrics JSONB,
                    learning_captured BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_project_bindings_user ON anime_echo_project_bindings(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_character_memory_name ON anime_echo_character_memory(character_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_style_learning_user ON anime_echo_style_learning(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user_platform ON anime_echo_sessions(user_id, platform)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_orchestrations_session ON anime_echo_orchestrations(session_id)")

            conn.commit()
            cursor.close()
            conn.close()
            logger.info("✅ Echo Anime Coordinator database tables initialized")

        except Exception as e:
            logger.error(f"❌ Failed to initialize anime coordinator database: {e}")

    async def coordinate_generation(self, request: AnimeRequest) -> Dict[str, Any]:
        """Main coordination method - Echo's intelligent anime orchestration"""

        # Initialize database if not already done
        if not self._db_initialized:
            await self.initialize_database()
            self._db_initialized = True

        orchestration_id = f"echo_anime_{int(datetime.now().timestamp())}_{request.user_id}"
        start_time = datetime.now()

        logger.info(f"🎬 Echo orchestrating anime generation: {orchestration_id}")

        try:
            # 1. Session Context Resolution
            session_context = await self.resolve_session_context(request)

            # 2. Project Memory Integration
            project_memory = await self.get_or_create_project_memory(request, session_context)

            # 3. Character Consistency Processing
            character_data = await self.ensure_character_consistency(request, project_memory)

            # 4. Style Learning Application
            enhanced_request = await self.apply_learned_preferences(request, project_memory)

            # 5. Generation Coordination
            generation_result = await self.orchestrate_generation_pipeline(
                enhanced_request, project_memory, character_data, session_context
            )

            # 6. Memory Update & Learning Capture
            await self.update_project_memory(project_memory, enhanced_request, generation_result)
            await self.capture_learning_data(orchestration_id, enhanced_request, generation_result)

            # 7. Session State Persistence
            await self.persist_session_state(session_context, project_memory, generation_result)

            end_time = datetime.now()
            orchestration_duration = (end_time - start_time).total_seconds()

            # Store orchestration record
            await self.record_orchestration(
                orchestration_id, request, project_memory, character_data,
                enhanced_request, generation_result, orchestration_duration
            )

            return {
                "success": True,
                "orchestration_id": orchestration_id,
                "generation_result": generation_result,
                "project_memory": asdict(project_memory),
                "character_consistency": character_data,
                "applied_preferences": enhanced_request.style_preference,
                "session_context": session_context,
                "performance": {
                    "orchestration_duration": orchestration_duration,
                    "echo_coordination_active": True,
                    "cross_platform_context": True,
                    "memory_persistence": True
                }
            }

        except Exception as e:
            logger.error(f"❌ Echo anime orchestration failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "orchestration_id": orchestration_id,
                "echo_coordination_attempted": True
            }

    async def resolve_session_context(self, request: AnimeRequest) -> Dict[str, Any]:
        """Resolve cross-platform session context"""
        try:
            session_id = request.session_id or f"{request.platform}_{request.user_id}_{int(datetime.now().timestamp())}"

            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            # Get or create session
            cursor.execute("""
                SELECT id, context_state, project_binding_id, active_character, pending_generations
                FROM anime_echo_sessions
                WHERE session_id = %s AND user_id = %s
                AND expires_at > CURRENT_TIMESTAMP
            """, (session_id, request.user_id))

            session_row = cursor.fetchone()

            if session_row:
                # Update existing session
                cursor.execute("""
                    UPDATE anime_echo_sessions
                    SET last_activity = CURRENT_TIMESTAMP,
                        expires_at = CURRENT_TIMESTAMP + INTERVAL '24 hours'
                    WHERE session_id = %s
                """, (session_id,))

                session_context = {
                    "session_id": session_id,
                    "context_state": session_row[1] or {},
                    "project_binding_id": session_row[2],
                    "active_character": session_row[3],
                    "pending_generations": session_row[4] or [],
                    "platform": request.platform,
                    "existing_session": True
                }
            else:
                # Create new session
                cursor.execute("""
                    INSERT INTO anime_echo_sessions
                    (session_id, user_id, platform, context_state)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                """, (session_id, request.user_id, request.platform, json.dumps({})))

                session_context = {
                    "session_id": session_id,
                    "context_state": {},
                    "project_binding_id": None,
                    "active_character": None,
                    "pending_generations": [],
                    "platform": request.platform,
                    "existing_session": False
                }

            conn.commit()
            cursor.close()
            conn.close()

            return session_context

        except Exception as e:
            logger.error(f"Failed to resolve session context: {e}")
            return {
                "session_id": f"fallback_{int(datetime.now().timestamp())}",
                "context_state": {},
                "platform": request.platform,
                "existing_session": False
            }

    async def get_or_create_project_memory(self, request: AnimeRequest, session_context: Dict) -> ProjectMemory:
        """Get or create project memory state"""
        try:
            project_name = request.project_name or "Echo Default Project"

            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            # Check for existing project binding
            cursor.execute("""
                SELECT id, anime_project_id, memory_context, session_context
                FROM anime_echo_project_bindings
                WHERE project_name = %s AND user_id = %s AND active = TRUE
                ORDER BY last_accessed DESC LIMIT 1
            """, (project_name, request.user_id))

            binding_row = cursor.fetchone()

            if binding_row:
                binding_id, anime_project_id, memory_context, stored_session_context = binding_row

                # Update access time
                cursor.execute("""
                    UPDATE anime_echo_project_bindings
                    SET last_accessed = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, (binding_id,))

                # Get character data
                characters = await self.load_project_characters(binding_id)

                project_memory = ProjectMemory(
                    project_id=anime_project_id,
                    project_name=project_name,
                    characters=characters,
                    style_guide=memory_context.get("style_guide", {}) if memory_context else {},
                    generation_history=memory_context.get("generation_history", []) if memory_context else [],
                    user_preferences=memory_context.get("user_preferences", {}) if memory_context else {},
                    last_updated=datetime.now(),
                    session_context=stored_session_context or {}
                )

            else:
                # Create new project binding
                cursor.execute("""
                    INSERT INTO anime_echo_project_bindings
                    (project_name, user_id, session_context, memory_context)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                """, (project_name, request.user_id, json.dumps(session_context), json.dumps({})))

                binding_id = cursor.fetchone()[0]

                project_memory = ProjectMemory(
                    project_id=None,
                    project_name=project_name,
                    characters={},
                    style_guide={},
                    generation_history=[],
                    user_preferences={},
                    last_updated=datetime.now(),
                    session_context=session_context
                )

            conn.commit()
            cursor.close()
            conn.close()

            # Cache in memory
            self.project_memory_cache[project_name] = project_memory

            return project_memory

        except Exception as e:
            logger.error(f"Failed to get project memory: {e}")
            return ProjectMemory(
                project_id=None,
                project_name=project_name or "Fallback Project",
                characters={},
                style_guide={},
                generation_history=[],
                user_preferences={},
                last_updated=datetime.now(),
                session_context=session_context
            )

    async def ensure_character_consistency(self, request: AnimeRequest, project_memory: ProjectMemory) -> Dict[str, Any]:
        """Unified character management bridging database and JSON files"""
        if not request.character_name:
            return {"consistency_applied": False}

        try:
            character_name = request.character_name.lower().strip()

            # 1. Check JSON character files
            json_character = await self.load_json_character(character_name)

            # 2. Check database character
            db_character = await self.load_database_character(character_name, project_memory.project_id)

            # 3. Check Echo memory for learned traits
            learned_traits = await self.get_character_learned_traits(character_name)

            # 4. Merge character data intelligently
            unified_character = await self.merge_character_data(json_character, db_character, learned_traits)

            # 5. Update character consistency tracking
            await self.update_character_memory(character_name, project_memory, unified_character)

            # 6. Generate consistency-enhanced prompt
            consistent_prompt = await self.enhance_prompt_with_character(request.prompt, unified_character)

            return {
                "consistency_applied": True,
                "character_name": character_name,
                "character_data": unified_character,
                "enhanced_prompt": consistent_prompt,
                "data_sources": {
                    "json_file": json_character is not None,
                    "database": db_character is not None,
                    "echo_learning": bool(learned_traits)
                },
                "consistency_score": unified_character.get("consistency_score", 0.5)
            }

        except Exception as e:
            logger.error(f"Character consistency processing failed: {e}")
            return {
                "consistency_applied": False,
                "error": str(e),
                "fallback_prompt": request.prompt
            }

    async def load_json_character(self, character_name: str) -> Optional[Dict]:
        """Load character from JSON files"""
        try:
            character_file = Path(self.character_system_path) / f"{character_name.replace(' ', '_').lower()}.json"

            if character_file.exists():
                with open(character_file, 'r', encoding='utf-8') as f:
                    character_data = json.load(f)
                    character_data["source"] = "json_file"
                    character_data["file_path"] = str(character_file)
                    return character_data

        except Exception as e:
            logger.error(f"Failed to load JSON character {character_name}: {e}")

        return None

    async def load_database_character(self, character_name: str, project_id: Optional[int]) -> Optional[Dict]:
        """Load character from anime production database"""
        if not project_id:
            return None

        try:
            # This would connect to anime_production database
            # For now, return None as we don't have direct access
            logger.info(f"Database character lookup for {character_name} in project {project_id}")
            return None

        except Exception as e:
            logger.error(f"Failed to load database character: {e}")
            return None

    async def get_character_learned_traits(self, character_name: str) -> Dict:
        """Get Echo's learned traits for character"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT learned_traits, generation_count, consistency_score, evolution_history
                FROM anime_echo_character_memory
                WHERE character_name = %s
                ORDER BY last_used DESC LIMIT 1
            """, (character_name,))

            row = cursor.fetchone()
            cursor.close()
            conn.close()

            if row:
                return {
                    "learned_traits": row[0] or {},
                    "generation_count": row[1] or 0,
                    "consistency_score": row[2] or 0.0,
                    "evolution_history": row[3] or [],
                    "source": "echo_learning"
                }

        except Exception as e:
            logger.error(f"Failed to get learned traits: {e}")

        return {}

    async def merge_character_data(self, json_char: Optional[Dict], db_char: Optional[Dict], learned_traits: Dict) -> Dict:
        """Intelligently merge character data from all sources"""
        merged = {
            "name": "",
            "visual_description": "",
            "style_tags": [],
            "negative_prompts": [],
            "consistency_score": 0.5,
            "generation_settings": {},
            "sources": []
        }

        # Priority: JSON file > Echo learning > Database
        if json_char:
            merged["sources"].append("json_file")
            merged["name"] = json_char.get("name", "")

            gen_prompts = json_char.get("generation_prompts", {})
            merged["visual_description"] = gen_prompts.get("visual_description", "")
            merged["style_tags"] = gen_prompts.get("style_tags", [])
            merged["negative_prompts"] = gen_prompts.get("negative_prompts", [])
            merged["consistency_score"] = 0.9  # High confidence from JSON

        if learned_traits:
            merged["sources"].append("echo_learning")
            traits = learned_traits.get("learned_traits", {})

            # Enhance with learned traits
            if traits.get("preferred_styles"):
                merged["style_tags"].extend(traits["preferred_styles"])
            if traits.get("visual_enhancements"):
                merged["visual_description"] += f", {traits['visual_enhancements']}"

            # Update consistency score based on learning
            learned_score = learned_traits.get("consistency_score", 0.5)
            merged["consistency_score"] = max(merged["consistency_score"], learned_score)

        if db_char:
            merged["sources"].append("database")
            # Database integration would go here

        # Remove duplicates from style tags
        merged["style_tags"] = list(set(merged["style_tags"]))

        return merged

    async def enhance_prompt_with_character(self, original_prompt: str, character_data: Dict) -> str:
        """Enhance prompt with character consistency data"""
        try:
            visual_desc = character_data.get("visual_description", "")
            style_tags = character_data.get("style_tags", [])

            enhanced_parts = [original_prompt]

            if visual_desc:
                enhanced_parts.append(visual_desc)

            if style_tags:
                enhanced_parts.append(", ".join(style_tags))

            enhanced_prompt = ", ".join(enhanced_parts)

            logger.info(f"Enhanced prompt with character consistency: {len(enhanced_prompt)} chars")
            return enhanced_prompt

        except Exception as e:
            logger.error(f"Failed to enhance prompt: {e}")
            return original_prompt

    async def apply_learned_preferences(self, request: AnimeRequest, project_memory: ProjectMemory) -> AnimeRequest:
        """Apply Echo's learned user preferences to the request"""
        if not request.apply_learning:
            return request

        try:
            # Get user's learned preferences from style learning table
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT learned_preferences, style_elements, context_tags, feedback_weight
                FROM anime_echo_style_learning
                WHERE user_id = %s
                AND user_feedback->>'rating' IS NOT NULL
                ORDER BY created_at DESC LIMIT 20
            """, (request.user_id,))

            learning_rows = cursor.fetchall()
            cursor.close()
            conn.close()

            # Analyze and apply preferences
            preferred_styles = []
            quality_settings = {}

            for row in learning_rows:
                preferences = row[0] or {}
                style_elements = row[1] or []
                weight = row[3] or 1.0

                # Weight by feedback quality
                if weight > 0.7:  # High-rated generations
                    preferred_styles.extend(style_elements)

                    if preferences.get("quality_settings"):
                        for key, value in preferences["quality_settings"].items():
                            quality_settings[key] = value

            # Apply to request
            enhanced_request = request

            if preferred_styles:
                style_text = ", ".join(set(preferred_styles))
                enhanced_request.style_preference = f"{request.style_preference or ''}, {style_text}".strip(', ')

            if quality_settings:
                enhanced_request.steps = quality_settings.get("steps", request.steps)
                enhanced_request.width = quality_settings.get("width", request.width)
                enhanced_request.height = quality_settings.get("height", request.height)

            logger.info(f"Applied learned preferences: {len(preferred_styles)} style elements")
            return enhanced_request

        except Exception as e:
            logger.error(f"Failed to apply learned preferences: {e}")
            return request

    async def orchestrate_generation_pipeline(self, request: AnimeRequest, project_memory: ProjectMemory,
                                            character_data: Dict, session_context: Dict) -> Dict:
        """Coordinate generation across anime production services"""
        try:
            # Build generation parameters matching AnimeGenerationRequest format
            generation_params = {
                "prompt": character_data.get("enhanced_prompt", request.prompt),
                "character": request.character_name or "original",  # Pass character name correctly
                "scene_type": "dialogue",
                "generation_type": request.generation_type,  # Pass video/image type
                "style": request.style_preference or "anime",
                "type": "professional",
                # Echo metadata
                "echo_orchestrated": True,
                "echo_session_id": session_context.get("session_id"),
                "project_context": {
                    "name": project_memory.project_name,
                    "id": project_memory.project_id
                },
                "character_consistency": character_data.get("consistency_applied", False)
            }

            # Add character reference if available
            if character_data.get("character_data"):
                generation_params["character_reference"] = character_data["character_data"]

            # Send to anime production service
            logger.info(f"🚀 Sending to anime service: {self.anime_service_url}")

            response = requests.post(
                f"{self.anime_service_url}/api/anime/generate",
                json=generation_params,
                timeout=300
            )

            if response.status_code == 200:
                result = response.json()
                result["echo_coordinated"] = True
                result["coordination_metadata"] = {
                    "project_memory_applied": True,
                    "character_consistency": character_data.get("consistency_applied", False),
                    "learned_preferences": bool(request.style_preference),
                    "session_context": session_context.get("session_id")
                }
                return result
            else:
                logger.error(f"Anime service error: {response.status_code} - {response.text}")
                return {
                    "success": False,
                    "error": f"Generation service error: {response.status_code}",
                    "echo_coordinated": True
                }

        except Exception as e:
            logger.error(f"Generation pipeline orchestration failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "echo_coordination_attempted": True
            }

    async def update_project_memory(self, project_memory: ProjectMemory, request: AnimeRequest, result: Dict):
        """Update project memory with generation results"""
        try:
            # Add to generation history
            generation_entry = {
                "timestamp": datetime.now().isoformat(),
                "prompt": request.prompt,
                "character": request.character_name,
                "result": result,
                "success": result.get("success", False)
            }

            project_memory.generation_history.append(generation_entry)

            # Keep only recent history (last 50 generations)
            if len(project_memory.generation_history) > 50:
                project_memory.generation_history = project_memory.generation_history[-50:]

            project_memory.last_updated = datetime.now()

            # Update database
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE anime_echo_project_bindings
                SET memory_context = %s, last_accessed = CURRENT_TIMESTAMP
                WHERE project_name = %s AND user_id = 'patrick' AND active = TRUE
            """, (json.dumps({
                "style_guide": project_memory.style_guide,
                "generation_history": project_memory.generation_history,
                "user_preferences": project_memory.user_preferences
            }), project_memory.project_name))

            conn.commit()
            cursor.close()
            conn.close()

            logger.info("📝 Updated project memory successfully")

        except Exception as e:
            logger.error(f"Failed to update project memory: {e}")

    async def capture_learning_data(self, orchestration_id: str, request: AnimeRequest, result: Dict):
        """Capture data for Echo's learning system"""
        try:
            style_elements = []
            if request.style_preference:
                style_elements = [s.strip() for s in request.style_preference.split(",")]

            learning_record = {
                "generation_id": orchestration_id,
                "user_id": request.user_id,
                "prompt_used": request.prompt,
                "style_elements": style_elements,
                "quality_assessment": {
                    "generation_successful": result.get("success", False),
                    "generation_time": result.get("generation_time"),
                    "settings_used": {
                        "width": request.width,
                        "height": request.height,
                        "steps": request.steps
                    }
                },
                "context_tags": [
                    request.generation_type,
                    request.scene_type,
                    request.platform
                ]
            }

            if request.character_name:
                learning_record["context_tags"].append(f"character:{request.character_name}")

            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO anime_echo_style_learning
                (generation_id, user_id, prompt_used, style_elements, quality_assessment, context_tags)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                orchestration_id, request.user_id, request.prompt,
                json.dumps(style_elements), json.dumps(learning_record["quality_assessment"]),
                json.dumps(learning_record["context_tags"])
            ))

            conn.commit()
            cursor.close()
            conn.close()

            logger.info("🧠 Captured learning data for Echo")

        except Exception as e:
            logger.error(f"Failed to capture learning data: {e}")

    async def persist_session_state(self, session_context: Dict, project_memory: ProjectMemory, result: Dict):
        """Persist session state for cross-platform continuity"""
        try:
            session_id = session_context.get("session_id")
            if not session_id:
                return

            # Update session state
            updated_context = session_context["context_state"].copy()
            updated_context.update({
                "last_generation": result.get("generation_id"),
                "current_project": project_memory.project_name,
                "generation_count": updated_context.get("generation_count", 0) + 1,
                "last_activity": datetime.now().isoformat()
            })

            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE anime_echo_sessions
                SET context_state = %s, last_activity = CURRENT_TIMESTAMP
                WHERE session_id = %s
            """, (json.dumps(updated_context), session_id))

            conn.commit()
            cursor.close()
            conn.close()

            logger.info(f"💾 Persisted session state: {session_id}")

        except Exception as e:
            logger.error(f"Failed to persist session state: {e}")

    async def record_orchestration(self, orchestration_id: str, request: AnimeRequest,
                                 project_memory: ProjectMemory, character_data: Dict,
                                 enhanced_request: AnimeRequest, result: Dict, duration: float):
        """Record complete orchestration for analysis"""
        try:
            orchestration_record = {
                "orchestration_id": orchestration_id,
                "user_id": request.user_id,
                "session_id": request.session_id,
                "request_data": asdict(request),
                "project_context": asdict(project_memory),
                "character_consistency": character_data,
                "applied_preferences": asdict(enhanced_request),
                "generation_result": result,
                "performance_metrics": {
                    "orchestration_duration": duration,
                    "echo_coordination_active": True,
                    "memory_integration": True,
                    "character_consistency": character_data.get("consistency_applied", False),
                    "preference_learning": enhanced_request.apply_learning
                }
            }

            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO anime_echo_orchestrations
                (orchestration_id, user_id, session_id, request_data, project_context,
                 character_consistency, applied_preferences, generation_result, performance_metrics)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                orchestration_id, request.user_id, request.session_id,
                json.dumps(orchestration_record["request_data"]),
                json.dumps(orchestration_record["project_context"]),
                json.dumps(character_data),
                json.dumps(orchestration_record["applied_preferences"]),
                json.dumps(result),
                json.dumps(orchestration_record["performance_metrics"])
            ))

            conn.commit()
            cursor.close()
            conn.close()

            logger.info(f"📊 Recorded orchestration: {orchestration_id}")

        except Exception as e:
            logger.error(f"Failed to record orchestration: {e}")

    async def load_project_characters(self, binding_id: int) -> Dict[str, Dict]:
        """Load all characters associated with a project"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT character_name, learned_traits, consistency_score, generation_count
                FROM anime_echo_character_memory
                WHERE project_binding_id = %s
            """, (binding_id,))

            characters = {}
            for row in cursor.fetchall():
                name = row[0]
                characters[name] = {
                    "learned_traits": row[1] or {},
                    "consistency_score": row[2] or 0.0,
                    "generation_count": row[3] or 0
                }

            cursor.close()
            conn.close()

            return characters

        except Exception as e:
            logger.error(f"Failed to load project characters: {e}")
            return {}

    async def update_character_memory(self, character_name: str, project_memory: ProjectMemory, character_data: Dict):
        """Update character memory with latest consistency data"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            # Get project binding ID
            cursor.execute("""
                SELECT id FROM anime_echo_project_bindings
                WHERE project_name = %s AND user_id = 'patrick' AND active = TRUE
                LIMIT 1
            """, (project_memory.project_name,))

            binding_row = cursor.fetchone()
            if not binding_row:
                return

            binding_id = binding_row[0]

            # Update or insert character memory
            cursor.execute("""
                INSERT INTO anime_echo_character_memory
                (character_name, project_binding_id, learned_traits, consistency_score, generation_count, last_used)
                VALUES (%s, %s, %s, %s, 1, CURRENT_TIMESTAMP)
                ON CONFLICT (character_name, project_binding_id)
                DO UPDATE SET
                    learned_traits = EXCLUDED.learned_traits,
                    consistency_score = EXCLUDED.consistency_score,
                    generation_count = anime_echo_character_memory.generation_count + 1,
                    last_used = CURRENT_TIMESTAMP
            """, (
                character_name, binding_id,
                json.dumps(character_data.get("learned_traits", {})),
                character_data.get("consistency_score", 0.5)
            ))

            conn.commit()
            cursor.close()
            conn.close()

            logger.info(f"🎭 Updated character memory: {character_name}")

        except Exception as e:
            logger.error(f"Failed to update character memory: {e}")

# Global coordinator instance
echo_anime_coordinator = EchoAnimeCoordinator()

# Convenience functions for external use
async def coordinate_anime_generation(prompt: str, character_name: Optional[str] = None,
                                    project_name: Optional[str] = None,
                                    session_id: Optional[str] = None,
                                    platform: str = "echo_brain") -> Dict[str, Any]:
    """Simplified interface for anime generation coordination"""

    request = AnimeRequest(
        prompt=prompt,
        character_name=character_name,
        project_name=project_name,
        session_id=session_id,
        platform=platform
    )

    return await echo_anime_coordinator.coordinate_generation(request)

async def get_project_context(project_name: str, user_id: str = "patrick") -> Optional[Dict]:
    """Get project context for external systems"""
    try:
        conn = psycopg2.connect(**echo_anime_coordinator.db_config)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT memory_context, session_context, last_accessed
            FROM anime_echo_project_bindings
            WHERE project_name = %s AND user_id = %s AND active = TRUE
            ORDER BY last_accessed DESC LIMIT 1
        """, (project_name, user_id))

        row = cursor.fetchone()
        cursor.close()
        conn.close()

        if row:
            return {
                "project_name": project_name,
                "memory_context": row[0] or {},
                "session_context": row[1] or {},
                "last_accessed": row[2].isoformat() if row[2] else None
            }

    except Exception as e:
        logger.error(f"Failed to get project context: {e}")

    return None

async def record_user_feedback(generation_id: str, rating: int, feedback: str = None):
    """Record user feedback for learning"""
    try:
        conn = psycopg2.connect(**echo_anime_coordinator.db_config)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE anime_echo_style_learning
            SET user_feedback = %s, feedback_weight = %s
            WHERE generation_id = %s
        """, (json.dumps({"rating": rating, "feedback": feedback}), rating / 5.0, generation_id))

        conn.commit()
        cursor.close()
        conn.close()

        logger.info(f"💭 Recorded user feedback: {generation_id} - Rating: {rating}")

    except Exception as e:
        logger.error(f"Failed to record user feedback: {e}")