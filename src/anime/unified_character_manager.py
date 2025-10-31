#!/usr/bin/env python3
"""
Unified Character Management System
Bridges character definitions between JSON files, database records, and Echo learning.

This system provides a single interface for character consistency across:
- JSON character files (/opt/tower-anime-production/characters/*.json)
- Anime production database character records
- Echo's learned character traits and evolution data
"""

import asyncio
import json
import logging
import os
import psycopg2
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import sys
import os
sys.path.append('/opt/tower-echo-brain/src')

from db.database import database

logger = logging.getLogger(__name__)

@dataclass
class CharacterDefinition:
    """Unified character definition structure"""
    name: str
    visual_description: str
    style_tags: List[str]
    negative_prompts: List[str]
    generation_settings: Dict[str, Any]
    consistency_score: float
    sources: List[str]  # json_file, database, echo_learning

    # Evolution tracking
    generation_count: int = 0
    evolution_history: List[Dict] = None
    learned_traits: Dict[str, Any] = None

    # Reference data
    json_file_path: Optional[str] = None
    database_id: Optional[int] = None
    echo_memory_id: Optional[int] = None

    def __post_init__(self):
        if self.evolution_history is None:
            self.evolution_history = []
        if self.learned_traits is None:
            self.learned_traits = {}

class UnifiedCharacterManager:
    """Manages character definitions across all data sources"""

    def __init__(self):
        self.json_characters_path = "/opt/tower-anime-production/characters"
        self.db_config = {
            "database": os.environ.get("DB_NAME", "echo_brain"),
            "user": os.environ.get("DB_USER", "patrick")
        }
        self.character_cache = {}
        self.last_cache_update = None

    async def get_character(self, character_name: str, project_id: Optional[int] = None) -> Optional[CharacterDefinition]:
        """Get unified character definition from all sources"""
        try:
            character_name_normalized = character_name.lower().strip()

            # Check cache first
            cache_key = f"{character_name_normalized}_{project_id}"
            if cache_key in self.character_cache:
                cache_entry = self.character_cache[cache_key]
                # Return cached if less than 5 minutes old
                if (datetime.now() - cache_entry["timestamp"]).total_seconds() < 300:
                    return cache_entry["character"]

            # Load from all sources
            json_char = await self.load_from_json(character_name_normalized)
            db_char = await self.load_from_database(character_name_normalized, project_id)
            echo_char = await self.load_from_echo_memory(character_name_normalized)

            # Merge into unified definition
            unified_char = await self.merge_character_sources(
                character_name, json_char, db_char, echo_char
            )

            # Cache the result
            self.character_cache[cache_key] = {
                "character": unified_char,
                "timestamp": datetime.now()
            }

            return unified_char

        except Exception as e:
            logger.error(f"Failed to get character {character_name}: {e}")
            return None

    async def load_from_json(self, character_name: str) -> Optional[Dict]:
        """Load character definition from JSON files"""
        try:
            # Try different filename formats
            possible_names = [
                character_name.replace(" ", "_").lower(),
                character_name.replace(" ", "").lower(),
                character_name.lower()
            ]

            json_path = Path(self.json_characters_path)

            for name_variant in possible_names:
                json_file = json_path / f"{name_variant}.json"
                if json_file.exists():
                    with open(json_file, 'r', encoding='utf-8') as f:
                        char_data = json.load(f)
                        char_data["source"] = "json_file"
                        char_data["file_path"] = str(json_file)
                        char_data["last_modified"] = datetime.fromtimestamp(
                            json_file.stat().st_mtime
                        ).isoformat()

                        logger.info(f"✅ Loaded character from JSON: {json_file}")
                        return char_data

            logger.info(f"📄 No JSON file found for character: {character_name}")
            return None

        except Exception as e:
            logger.error(f"Failed to load character from JSON: {e}")
            return None

    async def load_from_database(self, character_name: str, project_id: Optional[int]) -> Optional[Dict]:
        """Load character definition from anime production database"""
        try:
            # For now, we'll simulate database loading since we don't have direct access
            # In a real implementation, this would connect to the anime_production database

            # Check if we have any database reference in Echo's memory
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT database_character_id, learned_traits
                FROM anime_echo_character_memory
                WHERE character_name = %s
                AND database_character_id IS NOT NULL
                ORDER BY last_used DESC LIMIT 1
            """, (character_name,))

            row = cursor.fetchone()
            cursor.close()
            conn.close()

            if row:
                db_id, learned_traits = row
                return {
                    "source": "database",
                    "database_id": db_id,
                    "name": character_name,
                    "learned_traits": learned_traits or {},
                    "note": "Database character reference found in Echo memory"
                }

            logger.info(f"🗄️ No database record found for character: {character_name}")
            return None

        except Exception as e:
            logger.error(f"Failed to load character from database: {e}")
            return None

    async def load_from_echo_memory(self, character_name: str) -> Optional[Dict]:
        """Load character traits learned by Echo"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT id, learned_traits, generation_count, consistency_score,
                       evolution_history, last_used
                FROM anime_echo_character_memory
                WHERE character_name = %s
                ORDER BY last_used DESC LIMIT 1
            """, (character_name,))

            row = cursor.fetchone()
            cursor.close()
            conn.close()

            if row:
                echo_id, learned_traits, gen_count, consistency, evolution, last_used = row

                echo_data = {
                    "source": "echo_learning",
                    "echo_memory_id": echo_id,
                    "learned_traits": learned_traits or {},
                    "generation_count": gen_count or 0,
                    "consistency_score": consistency or 0.0,
                    "evolution_history": evolution or [],
                    "last_used": last_used.isoformat() if last_used else None
                }

                logger.info(f"🧠 Loaded Echo learning data for character: {character_name}")
                return echo_data

            logger.info(f"🤖 No Echo learning data for character: {character_name}")
            return None

        except Exception as e:
            logger.error(f"Failed to load character from Echo memory: {e}")
            return None

    async def merge_character_sources(self, character_name: str, json_char: Optional[Dict],
                                    db_char: Optional[Dict], echo_char: Optional[Dict]) -> CharacterDefinition:
        """Merge character data from all sources into unified definition"""

        # Initialize with defaults
        name = character_name
        visual_description = ""
        style_tags = []
        negative_prompts = []
        generation_settings = {}
        consistency_score = 0.5
        sources = []
        generation_count = 0
        evolution_history = []
        learned_traits = {}

        # JSON file data (highest priority for base definition)
        if json_char:
            sources.append("json_file")
            name = json_char.get("name", character_name)

            gen_prompts = json_char.get("generation_prompts", {})
            visual_description = gen_prompts.get("visual_description", "")
            style_tags = gen_prompts.get("style_tags", [])
            negative_prompts = gen_prompts.get("negative_prompts", [])

            generation_settings = json_char.get("generation_settings", {})
            consistency_score = 0.9  # High confidence from JSON definition

        # Echo learning data (enhances base definition)
        if echo_char:
            sources.append("echo_learning")

            learned_traits = echo_char.get("learned_traits", {})
            generation_count = echo_char.get("generation_count", 0)
            evolution_history = echo_char.get("evolution_history", [])

            # Update consistency score with Echo's learning
            echo_consistency = echo_char.get("consistency_score", 0.5)
            if echo_consistency > consistency_score:
                consistency_score = echo_consistency

            # Enhance visual description with learned traits
            if learned_traits.get("visual_enhancements"):
                if visual_description:
                    visual_description += f", {learned_traits['visual_enhancements']}"
                else:
                    visual_description = learned_traits["visual_enhancements"]

            # Add learned style preferences
            if learned_traits.get("preferred_styles"):
                style_tags.extend(learned_traits["preferred_styles"])

        # Database data (provides additional context)
        if db_char:
            sources.append("database")
            # Database integration would enhance here

        # Remove duplicates and clean up
        style_tags = list(set(style_tags))  # Remove duplicates
        negative_prompts = list(set(negative_prompts))

        # Create unified character definition
        unified_char = CharacterDefinition(
            name=name,
            visual_description=visual_description,
            style_tags=style_tags,
            negative_prompts=negative_prompts,
            generation_settings=generation_settings,
            consistency_score=consistency_score,
            sources=sources,
            generation_count=generation_count,
            evolution_history=evolution_history,
            learned_traits=learned_traits,
            json_file_path=json_char.get("file_path") if json_char else None,
            database_id=db_char.get("database_id") if db_char else None,
            echo_memory_id=echo_char.get("echo_memory_id") if echo_char else None
        )

        logger.info(f"🎭 Merged character from {len(sources)} sources: {name} (score: {consistency_score:.2f})")
        return unified_char

    async def save_character_evolution(self, character_name: str, generation_result: Dict,
                                     prompt_used: str, feedback: Optional[Dict] = None):
        """Save character evolution data to Echo memory"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            # Get existing character memory or create new
            cursor.execute("""
                SELECT id, learned_traits, generation_count, evolution_history
                FROM anime_echo_character_memory
                WHERE character_name = %s
                ORDER BY last_used DESC LIMIT 1
            """, (character_name,))

            row = cursor.fetchone()

            if row:
                memory_id, current_traits, gen_count, evolution = row
                learned_traits = current_traits or {}
                generation_count = (gen_count or 0) + 1
                evolution_history = evolution or []
            else:
                memory_id = None
                learned_traits = {}
                generation_count = 1
                evolution_history = []

            # Add evolution entry
            evolution_entry = {
                "timestamp": datetime.now().isoformat(),
                "prompt_used": prompt_used,
                "generation_result": generation_result,
                "feedback": feedback,
                "generation_number": generation_count
            }

            evolution_history.append(evolution_entry)

            # Keep only recent evolution (last 20 entries)
            if len(evolution_history) > 20:
                evolution_history = evolution_history[-20:]

            # Update learned traits based on feedback
            if feedback and feedback.get("rating", 0) >= 4:  # Good feedback
                if "visual_enhancements" not in learned_traits:
                    learned_traits["visual_enhancements"] = ""

                # Extract positive elements from prompt
                prompt_elements = [elem.strip() for elem in prompt_used.split(",")]
                if "preferred_styles" not in learned_traits:
                    learned_traits["preferred_styles"] = []

                # Add successful prompt elements to preferences
                for element in prompt_elements[-3:]:  # Last 3 elements likely style-related
                    if element and element not in learned_traits["preferred_styles"]:
                        learned_traits["preferred_styles"].append(element)

            # Calculate new consistency score
            consistency_score = min(0.95, 0.5 + (generation_count * 0.05))

            if memory_id:
                # Update existing record
                cursor.execute("""
                    UPDATE anime_echo_character_memory
                    SET learned_traits = %s, generation_count = %s, consistency_score = %s,
                        evolution_history = %s, last_used = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, (json.dumps(learned_traits), generation_count, consistency_score,
                          json.dumps(evolution_history), memory_id))
            else:
                # Create new record
                cursor.execute("""
                    INSERT INTO anime_echo_character_memory
                    (character_name, learned_traits, generation_count, consistency_score,
                     evolution_history, last_used)
                    VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                """, (character_name, json.dumps(learned_traits), generation_count,
                      consistency_score, json.dumps(evolution_history)))

            conn.commit()
            cursor.close()
            conn.close()

            logger.info(f"📈 Saved character evolution: {character_name} (gen #{generation_count})")

        except Exception as e:
            logger.error(f"Failed to save character evolution: {e}")

    async def create_character_from_learning(self, character_name: str, visual_traits: Dict,
                                           project_binding_id: Optional[int] = None) -> CharacterDefinition:
        """Create new character definition from Echo's learning"""
        try:
            # Create JSON structure from learned traits
            character_json = {
                "name": character_name,
                "gender": visual_traits.get("gender", "unknown"),
                "age": visual_traits.get("age", "unknown"),
                "generation_prompts": {
                    "visual_description": visual_traits.get("visual_description", ""),
                    "style_tags": visual_traits.get("style_tags", []),
                    "negative_prompts": visual_traits.get("negative_prompts", [])
                },
                "created_by": "echo_learning",
                "created_at": datetime.now().isoformat()
            }

            # Save to JSON file
            json_filename = character_name.replace(" ", "_").lower()
            json_path = Path(self.json_characters_path) / f"{json_filename}.json"

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(character_json, f, indent=2, ensure_ascii=False)

            # Create Echo memory record
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO anime_echo_character_memory
                (character_name, project_binding_id, json_definition_path, learned_traits,
                 generation_count, consistency_score, last_used)
                VALUES (%s, %s, %s, %s, 0, 0.8, CURRENT_TIMESTAMP)
                RETURNING id
            """, (character_name, project_binding_id, str(json_path), json.dumps(visual_traits)))

            memory_id = cursor.fetchone()[0]
            conn.commit()
            cursor.close()
            conn.close()

            # Create unified character definition
            unified_char = CharacterDefinition(
                name=character_name,
                visual_description=visual_traits.get("visual_description", ""),
                style_tags=visual_traits.get("style_tags", []),
                negative_prompts=visual_traits.get("negative_prompts", []),
                generation_settings={},
                consistency_score=0.8,
                sources=["echo_learning", "json_file"],
                generation_count=0,
                evolution_history=[],
                learned_traits=visual_traits,
                json_file_path=str(json_path),
                echo_memory_id=memory_id
            )

            # Clear cache to force reload
            self.character_cache.clear()

            logger.info(f"🎨 Created new character from learning: {character_name}")
            return unified_char

        except Exception as e:
            logger.error(f"Failed to create character from learning: {e}")
            raise

    async def sync_character_to_json(self, character: CharacterDefinition) -> bool:
        """Sync unified character definition back to JSON file"""
        try:
            if not character.json_file_path:
                # Create new JSON file
                json_filename = character.name.replace(" ", "_").lower()
                character.json_file_path = str(Path(self.json_characters_path) / f"{json_filename}.json")

            # Build JSON structure
            character_json = {
                "name": character.name,
                "generation_prompts": {
                    "visual_description": character.visual_description,
                    "style_tags": character.style_tags,
                    "negative_prompts": character.negative_prompts
                },
                "generation_settings": character.generation_settings,
                "echo_metadata": {
                    "consistency_score": character.consistency_score,
                    "generation_count": character.generation_count,
                    "learned_traits": character.learned_traits,
                    "last_sync": datetime.now().isoformat()
                }
            }

            # Write to file
            with open(character.json_file_path, 'w', encoding='utf-8') as f:
                json.dump(character_json, f, indent=2, ensure_ascii=False)

            logger.info(f"💾 Synced character to JSON: {character.json_file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to sync character to JSON: {e}")
            return False

    async def list_all_characters(self) -> List[Dict[str, Any]]:
        """List all characters from all sources"""
        try:
            characters = {}

            # Load from JSON files
            json_path = Path(self.json_characters_path)
            if json_path.exists():
                for json_file in json_path.glob("*.json"):
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            char_data = json.load(f)
                            name = char_data.get("name", json_file.stem)
                            characters[name.lower()] = {
                                "name": name,
                                "sources": ["json_file"],
                                "json_file": str(json_file)
                            }
                    except Exception as e:
                        logger.error(f"Error reading {json_file}: {e}")

            # Add Echo learning data
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT character_name, generation_count, consistency_score, last_used
                FROM anime_echo_character_memory
                ORDER BY last_used DESC
            """)

            for row in cursor.fetchall():
                name, gen_count, consistency, last_used = row
                name_key = name.lower()

                if name_key in characters:
                    characters[name_key]["sources"].append("echo_learning")
                else:
                    characters[name_key] = {
                        "name": name,
                        "sources": ["echo_learning"]
                    }

                characters[name_key].update({
                    "generation_count": gen_count,
                    "consistency_score": consistency,
                    "last_used": last_used.isoformat() if last_used else None
                })

            cursor.close()
            conn.close()

            return list(characters.values())

        except Exception as e:
            logger.error(f"Failed to list characters: {e}")
            return []

    async def analyze_character_consistency(self, character_name: str) -> Dict[str, Any]:
        """Analyze character consistency across all sources"""
        try:
            character = await self.get_character(character_name)
            if not character:
                return {"error": f"Character '{character_name}' not found"}

            analysis = {
                "character_name": character.name,
                "consistency_score": character.consistency_score,
                "data_sources": character.sources,
                "generation_count": character.generation_count,
                "analysis": {
                    "has_json_definition": "json_file" in character.sources,
                    "has_database_record": "database" in character.sources,
                    "has_echo_learning": "echo_learning" in character.sources,
                    "visual_description_length": len(character.visual_description),
                    "style_tags_count": len(character.style_tags),
                    "negative_prompts_count": len(character.negative_prompts),
                    "learned_traits_count": len(character.learned_traits),
                    "evolution_entries": len(character.evolution_history)
                },
                "recommendations": []
            }

            # Generate recommendations
            if not character.visual_description:
                analysis["recommendations"].append("Add visual description for better consistency")

            if len(character.style_tags) < 3:
                analysis["recommendations"].append("Add more style tags for refined generation")

            if character.generation_count < 5:
                analysis["recommendations"].append("Generate more images to improve learning")

            if character.consistency_score < 0.7:
                analysis["recommendations"].append("Provide feedback on generations to improve consistency")

            return analysis

        except Exception as e:
            logger.error(f"Failed to analyze character consistency: {e}")
            return {"error": str(e)}

# Global unified character manager
unified_character_manager = UnifiedCharacterManager()

# Convenience functions
async def get_unified_character(character_name: str, project_id: Optional[int] = None) -> Optional[CharacterDefinition]:
    """Get unified character definition"""
    return await unified_character_manager.get_character(character_name, project_id)

async def save_character_feedback(character_name: str, generation_result: Dict,
                                prompt_used: str, rating: int, feedback: str = None):
    """Save user feedback for character learning"""
    feedback_data = {"rating": rating, "feedback": feedback}
    await unified_character_manager.save_character_evolution(
        character_name, generation_result, prompt_used, feedback_data
    )

async def list_available_characters() -> List[Dict[str, Any]]:
    """List all available characters"""
    return await unified_character_manager.list_all_characters()