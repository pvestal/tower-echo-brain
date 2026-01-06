"""Narration Agent - Anime scene narration and creative writing"""
import logging
import psycopg2
from psycopg2.extras import DictCursor
from typing import Dict, List, Optional
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

# Uses ANIME database - completely separate from Echo Brain
ANIME_DB_CONFIG = {
    'host': 'localhost',
    'database': 'tower_anime',
    'user': 'patrick',
    'password': 'tower_echo_brain_secret_key_2025'
}

class NarrationAgent(BaseAgent):
    """Agent for anime scene narration and creative content"""
    
    def __init__(self):
        super().__init__(
            name="NarrationAgent",
            model="gemma2:9b"  # Good balance for creative writing
        )
        self.system_prompt = """You are a creative anime scene narrator. You excel at:
1. Writing vivid, atmospheric scene descriptions
2. Capturing character emotions and motivations
3. Using cinematic language and pacing
4. Matching tone to genre (action, drama, romance, noir, etc.)
5. Suggesting visual composition and camera work

Output your response as:
## Narration
[The scene narration - 2-4 paragraphs]

## Mood
[Single word or short phrase describing the mood]

## Visual Notes
[Brief camera/composition suggestions for animation]

## ComfyUI Prompt
[A detailed prompt for image generation, if applicable]"""

    async def process(self, task: str, context: Dict = None) -> Dict:
        """Generate narration for a scene"""
        context = context or {}
        
        # Get character info if specified
        character_info = None
        if context.get("character"):
            character_info = self._get_character_info(context["character"])
        
        # Get project style if specified
        project_info = None
        if context.get("project_id"):
            project_info = self._get_project_info(context["project_id"])
        
        # Build prompt
        prompt = self._build_prompt(task, character_info, project_info, context)
        
        # Generate narration
        logger.info(f"NarrationAgent processing: {task[:50]}...")
        response = await self.call_model(prompt, self.system_prompt)
        
        # Extract components
        narration = self._extract_section(response, "Narration")
        mood = self._extract_section(response, "Mood")
        visual_notes = self._extract_section(response, "Visual Notes")
        comfyui_prompt = self._extract_section(response, "ComfyUI Prompt")
        
        result = {
            "scene": task,
            "narration": narration or response,  # Fallback to full response
            "mood": mood,
            "visual_notes": visual_notes,
            "comfyui_prompt": comfyui_prompt,
            "character": context.get("character"),
            "project_id": context.get("project_id"),
            "model": self.model
        }
        
        self.add_to_history(task, {"mood": mood, "has_comfyui": bool(comfyui_prompt)})
        return result
    
    def _get_character_info(self, character_name: str) -> Optional[Dict]:
        """Get character details from anime database"""
        try:
            with psycopg2.connect(**ANIME_DB_CONFIG) as conn:
                with conn.cursor(cursor_factory=DictCursor) as cur:
                    cur.execute("""
                        SELECT name, description, personality, appearance
                        FROM character_profiles
                        WHERE name ILIKE %s
                        LIMIT 1
                    """, (f'%{character_name}%',))
                    row = cur.fetchone()
                    if row:
                        logger.info(f"Found character: {row['name']}")
                        return dict(row)
                    return None
        except Exception as e:
            logger.warning(f"Character lookup failed: {e}")
            return None
    
    def _get_project_info(self, project_id: int) -> Optional[Dict]:
        """Get project style preferences"""
        try:
            with psycopg2.connect(**ANIME_DB_CONFIG) as conn:
                with conn.cursor(cursor_factory=DictCursor) as cur:
                    cur.execute("""
                        SELECT name, style, description
                        FROM animation_projects
                        WHERE id = %s
                    """, (project_id,))
                    row = cur.fetchone()
                    if row:
                        logger.info(f"Found project: {row['name']}")
                        return dict(row)
                    return None
        except Exception as e:
            logger.warning(f"Project lookup failed: {e}")
            return None
    
    def _build_prompt(self, scene: str, character: Dict, project: Dict, context: Dict) -> str:
        """Build narration prompt with context"""
        parts = []
        
        if project:
            parts.append(f"Project: {project.get('name')}")
            parts.append(f"Style: {project.get('style', 'cinematic anime')}")
            if project.get('description'):
                parts.append(f"Setting: {project['description'][:200]}")
            parts.append("")
        
        if character:
            parts.append(f"Character: {character.get('name')}")
            if character.get('personality'):
                parts.append(f"Personality: {character['personality'][:150]}")
            if character.get('appearance'):
                parts.append(f"Appearance: {character['appearance'][:150]}")
            parts.append("")
        
        if context.get("mood"):
            parts.append(f"Desired Mood: {context['mood']}")
        
        if context.get("genre"):
            parts.append(f"Genre: {context['genre']}")
        
        if context.get("previous_scene"):
            parts.append(f"Previous Scene: {context['previous_scene'][:200]}")
        
        parts.append(f"\nScene to narrate: {scene}")
        
        return "\n".join(parts)
    
    def _extract_section(self, text: str, section: str) -> str:
        """Extract a section from formatted response"""
        marker = f"## {section}"
        if marker in text:
            start = text.find(marker) + len(marker)
            next_section = text.find("##", start)
            if next_section > 0:
                return text[start:next_section].strip()
            return text[start:].strip()
        return ""

# Singleton
narration_agent = NarrationAgent()
