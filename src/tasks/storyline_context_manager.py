"""
Storyline Context Management System for Echo Brain
Helps users develop and maintain comprehensive project storylines with interactive context gathering
"""

import logging
import psycopg2
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class StorylineContextManager:
    """Manages storyline development and context gathering for anime/creative projects"""

    def __init__(self):
        self.db_host = "localhost"
        self.db_user = "patrick"
        self.db_password = None  # Use peer authentication
        self.db_name = "anime_production"

    def get_project_storyline(self, project_id: int) -> Dict[str, Any]:
        """Get comprehensive storyline context for a project"""
        try:
            conn = psycopg2.connect(
                host=self.db_host,
                database=self.db_name,
                user=self.db_user
            )

            # Get project details
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, name, description, created_at
                FROM anime_api.projects
                WHERE id = %s
            """, (project_id,))

            project = cursor.fetchone()
            if not project:
                return {"error": f"Project {project_id} not found"}

            # Get characters
            cursor.execute("""
                SELECT id, name, description, image_path
                FROM anime_api.characters
                WHERE project_id = %s
                ORDER BY id
            """, (project_id,))
            characters = cursor.fetchall()

            # Get scenes (using existing scenes table)
            cursor.execute("""
                SELECT id, scene_number, description, status
                FROM anime_api.scenes
                WHERE project_id = %s
                ORDER BY scene_number
            """, (project_id,))
            scenes = cursor.fetchall()

            # Get existing storyline context
            cursor.execute("""
                SELECT context_key, context_value
                FROM storyline_context
                WHERE project_id = %s
            """, (project_id,))
            storyline_data = cursor.fetchall()

            cursor.close()
            conn.close()

            return {
                "project": {
                    "id": project[0],
                    "name": project[1],
                    "description": project[2],
                    "created_at": project[3]
                },
                "characters": [
                    {
                        "id": char[0],
                        "name": char[1],
                        "description": char[2],
                        "image_path": char[3]
                    } for char in characters
                ],
                "scenes": [
                    {
                        "id": scene[0],
                        "scene_number": scene[1],
                        "description": scene[2],
                        "status": scene[3]
                    } for scene in scenes
                ],
                "storyline_context": {
                    data[0]: data[1] for data in storyline_data
                }
            }

        except Exception as e:
            logger.error(f"Error getting project storyline: {e}")
            return {"error": str(e)}

    def analyze_storyline_gaps(self, project_id: int) -> Dict[str, List[str]]:
        """Analyze what's missing from the project storyline"""
        storyline = self.get_project_storyline(project_id)

        if "error" in storyline:
            return storyline

        gaps = {
            "character_development": [],
            "plot_structure": [],
            "world_building": [],
            "technical_details": []
        }

        # Check character development gaps
        for char in storyline["characters"]:
            if not char["description"] or len(char["description"]) < 50:
                gaps["character_development"].append(
                    f"Character '{char['name']}' needs more detailed description"
                )

        # Check plot structure gaps
        if not storyline["scenes"]:
            gaps["plot_structure"].append("No scenes defined - need scene breakdown and structure")

        if len(storyline["scenes"]) < 5:
            gaps["plot_structure"].append("Very few scenes planned - consider expanding story structure")

        # Check world building gaps
        project_desc = storyline["project"]["description"]
        if not project_desc or len(project_desc) < 100:
            gaps["world_building"].append("Project needs more detailed world/setting description")

        # Check technical details
        if not any(char["image_path"] for char in storyline["characters"]):
            gaps["technical_details"].append("No character visual references - need character designs")

        return gaps

    def generate_context_questions(self, project_id: int) -> List[Dict[str, str]]:
        """Generate targeted questions to help user build storyline context"""
        gaps = self.analyze_storyline_gaps(project_id)
        storyline = self.get_project_storyline(project_id)

        if "error" in gaps or "error" in storyline:
            return [{"category": "error", "question": "Unable to analyze project", "reason": "Database error"}]

        questions = []

        # Character development questions
        for gap in gaps["character_development"]:
            char_name = gap.split("'")[1]  # Extract character name
            questions.extend([
                {
                    "category": "character_development",
                    "question": f"What is {char_name}'s primary motivation and goal?",
                    "context_type": "character_motivation",
                    "character": char_name
                },
                {
                    "category": "character_development",
                    "question": f"What is {char_name}'s biggest internal conflict or struggle?",
                    "context_type": "character_conflict",
                    "character": char_name
                },
                {
                    "category": "character_development",
                    "question": f"How does {char_name} change throughout the story?",
                    "context_type": "character_arc",
                    "character": char_name
                }
            ])

        # Plot structure questions
        if gaps["plot_structure"]:
            questions.extend([
                {
                    "category": "plot_structure",
                    "question": "What is the main conflict or problem that drives the entire story?",
                    "context_type": "central_conflict"
                },
                {
                    "category": "plot_structure",
                    "question": "How does the story begin? What's the inciting incident?",
                    "context_type": "story_opening"
                },
                {
                    "category": "plot_structure",
                    "question": "What are the major turning points or climaxes in the story?",
                    "context_type": "plot_points"
                },
                {
                    "category": "plot_structure",
                    "question": "How does the story resolve? What's the conclusion?",
                    "context_type": "story_resolution"
                }
            ])

        # World building questions
        if gaps["world_building"]:
            questions.extend([
                {
                    "category": "world_building",
                    "question": "Where and when does the story take place? Describe the setting.",
                    "context_type": "setting_details"
                },
                {
                    "category": "world_building",
                    "question": "What are the rules and limitations of this world?",
                    "context_type": "world_rules"
                },
                {
                    "category": "world_building",
                    "question": "What's the tone and mood of the story? (dark, hopeful, comedic, etc.)",
                    "context_type": "story_tone"
                }
            ])

        return questions[:10]  # Limit to most important questions

    def save_user_context(self, project_id: int, context_type: str, content: str,
                         character: str = None) -> bool:
        """Save user-provided context to the storyline database"""
        try:
            conn = psycopg2.connect(
                host=self.db_host,
                database=self.db_name,
                user=self.db_user
            )

            cursor = conn.cursor()

            # Create storyline_context table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS storyline_context (
                    id SERIAL PRIMARY KEY,
                    project_id INTEGER REFERENCES anime_api.projects(id),
                    context_type VARCHAR(100) NOT NULL,
                    content TEXT NOT NULL,
                    character_name VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Insert or update context
            cursor.execute("""
                INSERT INTO storyline_context (project_id, context_type, content, character_name)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (project_id, context_type, COALESCE(character_name, ''))
                DO UPDATE SET content = EXCLUDED.content, updated_at = CURRENT_TIMESTAMP
            """, (project_id, context_type, content, character))

            conn.commit()
            cursor.close()
            conn.close()

            logger.info(f"Saved context: {context_type} for project {project_id}")
            return True

        except Exception as e:
            logger.error(f"Error saving context: {e}")
            return False

    def get_storyline_context(self, project_id: int) -> Dict[str, Any]:
        """Get all saved storyline context for a project"""
        try:
            conn = psycopg2.connect(
                host=self.db_host,
                database=self.db_name,
                user=self.db_user
            )

            cursor = conn.cursor()
            cursor.execute("""
                SELECT context_type, content, character_name, created_at
                FROM storyline_context
                WHERE project_id = %s
                ORDER BY created_at DESC
            """, (project_id,))

            contexts = cursor.fetchall()
            cursor.close()
            conn.close()

            context_dict = {}
            for ctx in contexts:
                key = f"{ctx[0]}" + (f"_{ctx[2]}" if ctx[2] else "")
                context_dict[key] = {
                    "content": ctx[1],
                    "character": ctx[2],
                    "created_at": ctx[3]
                }

            return context_dict

        except Exception as e:
            logger.error(f"Error getting storyline context: {e}")
            return {}

    def generate_storyline_summary(self, project_id: int) -> str:
        """Generate a comprehensive storyline summary from all context"""
        storyline = self.get_project_storyline(project_id)
        context = self.get_storyline_context(project_id)

        if "error" in storyline:
            return f"Error: {storyline['error']}"

        project = storyline["project"]
        summary_parts = [
            f"# {project['name']} - Storyline Summary",
            "",
            f"**Project Description:** {project['description']}",
            ""
        ]

        # Add characters
        if storyline["characters"]:
            summary_parts.extend([
                "## Characters",
                ""
            ])
            for char in storyline["characters"]:
                summary_parts.append(f"**{char['name']}:** {char['description']}")

                # Add character-specific context
                for key, ctx in context.items():
                    if ctx["character"] == char["name"]:
                        summary_parts.append(f"  - {key.replace('_', ' ').title()}: {ctx['content']}")
                summary_parts.append("")

        # Add plot structure
        plot_contexts = {k: v for k, v in context.items() if "conflict" in k or "plot" in k or "story" in k}
        if plot_contexts:
            summary_parts.extend([
                "## Plot Structure",
                ""
            ])
            for key, ctx in plot_contexts.items():
                summary_parts.append(f"**{key.replace('_', ' ').title()}:** {ctx['content']}")
                summary_parts.append("")

        # Add world building
        world_contexts = {k: v for k, v in context.items() if "setting" in k or "world" in k or "tone" in k}
        if world_contexts:
            summary_parts.extend([
                "## World Building",
                ""
            ])
            for key, ctx in world_contexts.items():
                summary_parts.append(f"**{key.replace('_', ' ').title()}:** {ctx['content']}")
                summary_parts.append("")

        return "\n".join(summary_parts)