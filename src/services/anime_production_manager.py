#!/usr/bin/env python3
"""
Echo Brain Anime Production Manager
Integrates with Tower LTX video generation pipeline
"""

import logging
import os
import asyncio
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import psycopg2
from pathlib import Path
import sys

# Add production directory to path
sys.path.append('/opt/tower-anime-production/production')

logger = logging.getLogger(__name__)

class EchoBrainAnimeManager:
    """
    Manages anime video production through Echo Brain
    Bridges Echo Brain's AI capabilities with Tower's LTX pipeline
    """

    def __init__(self):
        """Initialize anime production manager"""
        self.db_config = {
            "host": "localhost",
            "database": "anime_production",
            "user": "patrick",
            "password": os.getenv("TOWER_DB_PASSWORD", "RP78eIrW7cI2jYvL5akt1yurE")
        }

        # Import Tower's production manager
        try:
            from anime_production_manager import AnimeProductionManager
            self.tower_manager = AnimeProductionManager()
            self.enabled = True
            logger.info("✅ Tower Anime Production Manager loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load Tower Anime Manager: {e}")
            self.tower_manager = None
            self.enabled = False

        # Scene type mappings for Echo Brain context
        self.scene_mappings = {
            "romantic": "intimate_scene",
            "kiss": "intimate_scene",
            "fight": "action_fight",
            "action": "action_fight",
            "talking": "dialogue",
            "conversation": "dialogue",
            "transform": "transformation",
            "magical": "transformation",
            "adult": "nsfw",
            "intimate": "nsfw"
        }

        # Available LoRAs cache
        self._loras_cache = None
        self._cache_time = None

    def get_available_loras(self) -> List[Dict]:
        """Get list of available LoRAs from database"""

        # Cache for 5 minutes
        if self._loras_cache and self._cache_time:
            if (datetime.now() - self._cache_time).seconds < 300:
                return self._loras_cache

        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT name, type, trigger_word, strength_default
                FROM lora_models
                ORDER BY type, name
            """)

            loras = []
            for name, lora_type, trigger, strength in cursor.fetchall():
                loras.append({
                    "name": name,
                    "type": lora_type,
                    "trigger_word": trigger,
                    "strength": strength
                })

            conn.close()

            self._loras_cache = loras
            self._cache_time = datetime.now()

            return loras

        except Exception as e:
            logger.error(f"Failed to get LoRAs: {e}")
            return []

    def interpret_scene_type(self, description: str) -> str:
        """
        Interpret natural language scene description to scene type
        """
        description_lower = description.lower()

        for keyword, scene_type in self.scene_mappings.items():
            if keyword in description_lower:
                return scene_type

        # Default to dialogue if unclear
        return "dialogue"

    async def generate_scene(self,
                           prompt: str,
                           character: Optional[str] = None,
                           scene_type: Optional[str] = None,
                           episode_id: int = 1) -> Dict:
        """
        Generate a scene from natural language prompt
        Echo Brain interprets the request and calls Tower pipeline
        """

        if not self.enabled or not self.tower_manager:
            return {
                "error": "Anime production system not available",
                "status": "disabled"
            }

        try:
            # Auto-detect scene type if not provided
            if not scene_type:
                scene_type = self.interpret_scene_type(prompt)
                logger.info(f"Detected scene type: {scene_type}")

            # Extract character if not provided
            if not character:
                # Simple extraction - look for common names
                common_names = ["Mei", "Hiroshi", "Sakura", "Yuki"]
                for name in common_names:
                    if name.lower() in prompt.lower():
                        character = name
                        break

                if not character:
                    character = "character"  # Generic fallback

            # Get next scene number
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT COALESCE(MAX(scene_number), 0) + 1
                FROM scene_generations
                WHERE episode_id = %s
            """, (episode_id,))

            scene_number = cursor.fetchone()[0]
            conn.close()

            # Generate through Tower pipeline
            result = await self.tower_manager.produce_episode_scene(
                episode_id=episode_id,
                scene_number=scene_number,
                scene_type=scene_type,
                character_names=[character] if character else [],
                description=prompt
            )

            # Add Echo Brain metadata
            result["echo_brain"] = {
                "interpreted_type": scene_type,
                "detected_character": character,
                "original_prompt": prompt,
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f"✅ Scene generation started: {result}")
            return result

        except Exception as e:
            logger.error(f"Scene generation failed: {e}")
            return {
                "error": str(e),
                "status": "failed"
            }

    async def generate_character_sheet(self, character_name: str) -> Dict:
        """
        Generate character reference sheet
        """

        if not self.enabled or not self.tower_manager:
            return {
                "error": "Anime production system not available",
                "status": "disabled"
            }

        try:
            result = await self.tower_manager.generate_character_reference_sheet(
                character_name
            )

            # Add Echo Brain context
            result["echo_brain"] = {
                "character": character_name,
                "timestamp": datetime.now().isoformat(),
                "reference_count": len(result.get("references", []))
            }

            return result

        except Exception as e:
            logger.error(f"Character sheet generation failed: {e}")
            return {
                "error": str(e),
                "status": "failed"
            }

    def get_recent_generations(self, limit: int = 10) -> List[Dict]:
        """
        Get recent video generations for context
        """

        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    episode_id,
                    scene_number,
                    scene_type,
                    description,
                    prompt_id,
                    status,
                    generated_at
                FROM scene_generations
                ORDER BY generated_at DESC
                LIMIT %s
            """, (limit,))

            results = []
            for row in cursor.fetchall():
                results.append({
                    "episode_id": row[0],
                    "scene_number": row[1],
                    "scene_type": row[2],
                    "description": row[3],
                    "prompt_id": row[4],
                    "status": row[5],
                    "generated_at": row[6].isoformat() if row[6] else None
                })

            conn.close()
            return results

        except Exception as e:
            logger.error(f"Failed to get recent generations: {e}")
            return []

    def get_production_stats(self) -> Dict:
        """
        Get production statistics for Echo Brain context
        """

        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            stats = {}

            # Total scenes
            cursor.execute("SELECT COUNT(*) FROM scene_generations")
            stats["total_scenes"] = cursor.fetchone()[0]

            # Scenes by type
            cursor.execute("""
                SELECT scene_type, COUNT(*)
                FROM scene_generations
                GROUP BY scene_type
            """)
            stats["by_type"] = dict(cursor.fetchall())

            # Available LoRAs
            cursor.execute("SELECT COUNT(*) FROM lora_models")
            stats["total_loras"] = cursor.fetchone()[0]

            # LoRAs by type
            cursor.execute("""
                SELECT type, COUNT(*)
                FROM lora_models
                GROUP BY type
            """)
            stats["loras_by_type"] = dict(cursor.fetchall())

            conn.close()

            stats["status"] = "operational"
            stats["ltx_model"] = "ltxv-2b-fp8"
            stats["vram_usage"] = "4GB"

            return stats

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}


# Create singleton instance
anime_manager = EchoBrainAnimeManager()


async def generate_anime_scene(prompt: str, **kwargs) -> Dict:
    """
    Echo Brain API endpoint for anime scene generation
    """
    return await anime_manager.generate_scene(prompt, **kwargs)


async def get_anime_status() -> Dict:
    """
    Echo Brain API endpoint for production status
    """
    return {
        "enabled": anime_manager.enabled,
        "stats": anime_manager.get_production_stats(),
        "recent": anime_manager.get_recent_generations(5),
        "loras": anime_manager.get_available_loras()
    }