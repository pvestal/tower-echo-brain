#!/usr/bin/env python3
"""
Dynamic Project Loader - Automatically loads projects and characters from database
No hardcoding, works with any new projects added to the system
"""

import asyncpg
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    'host': os.getenv('TOWER_DB_HOST', '192.168.50.135'),
    'database': 'anime_production',
    'user': 'patrick',
    'password': os.getenv('TOWER_DB_PASSWORD', os.getenv('DB_PASSWORD', ''))
}

@dataclass
class ProjectConfig:
    """Dynamic project configuration"""
    id: int
    name: str
    style: str
    status: str
    description: Optional[str] = None
    settings: Optional[Dict] = None

@dataclass
class CharacterConfig:
    """Dynamic character configuration"""
    id: int
    name: str
    project_id: int
    gender: Optional[str] = None
    description: Optional[str] = None
    lora_path: Optional[str] = None
    visual_features: Optional[Dict] = None

class DynamicProjectLoader:
    """Loads projects and characters dynamically from database"""

    def __init__(self):
        self.projects = {}
        self.characters = {}
        self.model_assignments = {}

    async def load_from_database(self):
        """Load all projects and characters from database"""
        try:
            conn = await asyncpg.connect(**DB_CONFIG)

            # Load projects
            projects_data = await conn.fetch("""
                SELECT id, name, style, status, description
                FROM projects
                WHERE status = 'active'
                ORDER BY id
            """)

            for proj in projects_data:
                self.projects[proj['id']] = ProjectConfig(
                    id=proj['id'],
                    name=proj['name'],
                    style=proj['style'] or 'anime',
                    status=proj['status'],
                    description=proj['description']
                )
                logger.info(f"Loaded project: {proj['name']} (ID: {proj['id']})")

            # Load characters
            characters_data = await conn.fetch("""
                SELECT c.id, c.name, c.project_id, c.gender, c.description,
                       l.model_path as lora_path
                FROM characters c
                LEFT JOIN lora_models l ON c.lora_id = l.id
                ORDER BY c.project_id, c.id
            """)

            for char in characters_data:
                if char['project_id'] not in self.characters:
                    self.characters[char['project_id']] = {}

                self.characters[char['project_id']][char['id']] = CharacterConfig(
                    id=char['id'],
                    name=char['name'],
                    project_id=char['project_id'],
                    gender=char['gender'],
                    description=char['description'],
                    lora_path=char['lora_path']
                )
                logger.info(f"  - Character: {char['name']} (Project: {char['project_id']})")

            await conn.close()

            logger.info(f"\nLoaded {len(self.projects)} projects with {sum(len(chars) for chars in self.characters.values())} characters")
            return True

        except Exception as e:
            logger.error(f"Failed to load from database: {e}")
            # Fallback to API if database is unavailable
            return await self.load_from_api()

    async def load_from_api(self):
        """Fallback: Load projects from JSON config or Echo Brain API"""
        import json
        import aiohttp
        from pathlib import Path

        # First try JSON config file
        config_file = Path("/opt/tower-echo-brain/projects_config.json")
        if config_file.exists():
            try:
                with open(config_file) as f:
                    data = json.load(f)

                for proj in data['projects']:
                    self.projects[proj['id']] = ProjectConfig(
                        id=proj['id'],
                        name=proj['name'],
                        style=proj['style'],
                        status=proj['status'],
                        description=proj.get('description')
                    )

                    # Load characters for this project
                    self.characters[proj['id']] = {}
                    for char in proj.get('characters', []):
                        self.characters[proj['id']][char['id']] = CharacterConfig(
                            id=char['id'],
                            name=char['name'],
                            project_id=proj['id'],
                            gender=char.get('gender'),
                            description=char.get('description')
                        )

                logger.info(f"Loaded from JSON: {len(self.projects)} projects")
                return True
            except Exception as e:
                logger.error(f"Failed to load from JSON: {e}")

        # Try API as last resort
        try:
            async with aiohttp.ClientSession() as session:
                # Try production router endpoint
                async with session.get('http://localhost:8309/anime/projects') as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for proj in data.get('projects', []):
                            self.projects[proj['id']] = ProjectConfig(
                                id=proj['id'],
                                name=proj['name'],
                                style='anime',
                                status=proj.get('status', 'active')
                            )

                # Try to get characters
                async with session.get('http://localhost:8309/anime/characters') as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for char in data.get('characters', []):
                            project_id = char.get('project_id', 1)  # Default to first project
                            if project_id not in self.characters:
                                self.characters[project_id] = {}

                            self.characters[project_id][char['id']] = CharacterConfig(
                                id=char['id'],
                                name=char['name'],
                                project_id=project_id,
                                description=char.get('description')
                            )

            logger.info(f"Loaded from API: {len(self.projects)} projects")
            return True

        except Exception as e:
            logger.error(f"Failed to load from API: {e}")
            return False

    def get_optimal_model_for_style(self, style: str) -> str:
        """Select optimal model based on project style"""
        # Import SSOT models for selection logic
        import yaml
        from pathlib import Path

        models_config_path = Path("/opt/tower-echo-brain/models_config.yaml")
        if models_config_path.exists():
            with open(models_config_path) as f:
                models = yaml.safe_load(f)['models']
        else:
            # Fallback models
            models = {
                'primary': {'file': 'epicrealism_v5.safetensors'},
                'fallback': {'file': 'realistic_vision_v51.safetensors'}
            }

        style_lower = style.lower()

        if 'photorealistic' in style_lower or 'realistic' in style_lower:
            return models['primary']['file']  # epiCRealism
        elif 'anime' in style_lower or 'cartoon' in style_lower:
            return models['fallback']['file']  # Realistic Vision
        elif 'cyberpunk' in style_lower:
            return models['fallback']['file']
        else:
            return models['primary']['file']  # Default to best quality

    def get_project_config(self, project_id: int) -> Optional[Dict]:
        """Get complete configuration for a project"""
        if project_id not in self.projects:
            return None

        project = self.projects[project_id]

        # Build dynamic configuration
        config = {
            'id': project.id,
            'name': project.name,
            'style': project.style,
            'model': self.get_optimal_model_for_style(project.style),
            'resolution': {'width': 512, 'height': 768},
            'characters': {}
        }

        # Add character configurations
        if project_id in self.characters:
            for char_id, char in self.characters[project_id].items():
                config['characters'][char.name] = {
                    'id': char.id,
                    'gender': char.gender,
                    'description': char.description,
                    'lora_path': char.lora_path
                }

        return config

    def get_all_projects(self) -> List[Dict]:
        """Get all active projects with their configurations"""
        return [self.get_project_config(pid) for pid in self.projects.keys()]

    async def initialize(self):
        """Initialize loader with all projects"""
        success = await self.load_from_database()
        if not success:
            logger.warning("Using fallback data sources")
        return self

# Singleton instance
loader = DynamicProjectLoader()

async def get_dynamic_projects():
    """Helper function to get all projects dynamically"""
    if not loader.projects:
        await loader.initialize()
    return loader.get_all_projects()

async def get_project_by_name(name: str):
    """Get project configuration by name"""
    if not loader.projects:
        await loader.initialize()

    for project in loader.projects.values():
        if project.name.lower() == name.lower():
            return loader.get_project_config(project.id)
    return None

# Example usage
if __name__ == "__main__":
    import asyncio

    async def main():
        await loader.initialize()

        # Show all loaded projects
        projects = loader.get_all_projects()
        for proj in projects:
            print(f"\nProject: {proj['name']}")
            print(f"  Model: {proj['model']}")
            print(f"  Characters: {list(proj['characters'].keys())}")

    asyncio.run(main())