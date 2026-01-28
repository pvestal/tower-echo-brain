#!/usr/bin/env python3
"""
Single Source of Truth - Project Configuration
Centralized configuration for models and characters
"""

from typing import Dict, List, Any
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ModelConfig:
    """Model configuration with optimal settings"""
    file: str
    name: str
    type: str
    sharpness_score: int
    settings: Dict[str, Any]
    capabilities: List[str]

@dataclass
class CharacterConfig:
    """Character configuration with model assignment"""
    name: str
    project: str
    gender: str
    personality: str
    model_override: str = None  # Use project default if None

# SSOT Model Definitions
MODELS = {
    "primary": ModelConfig(
        file="epicrealism_v5.safetensors",
        name="epiCRealism v5",
        type="photorealistic",
        sharpness_score=10,
        settings={
            "cfg": 5.0,
            "sampler": "dpmpp_sde",
            "scheduler": "karras",
            "steps": 30
        },
        capabilities=["sharp_detail", "skin_texture", "photographic", "both_genders"]
    ),
    "backup": ModelConfig(
        file="cyberrealistic_v9.safetensors",
        name="CyberRealistic v9",
        type="photorealistic",
        sharpness_score=10,
        settings={
            "cfg": 5.0,
            "sampler": "dpmpp_sde",
            "scheduler": "karras",
            "steps": 30
        },
        capabilities=["crisp_detail", "natural_lighting", "photographic", "both_genders"]
    ),
    "fallback": ModelConfig(
        file="realistic_vision_v51.safetensors",
        name="Realistic Vision v5.1",
        type="photorealistic",
        sharpness_score=7,
        settings={
            "cfg": 6.0,
            "sampler": "dpmpp_2m",
            "scheduler": "karras",
            "steps": 25
        },
        capabilities=["versatile", "consistent", "good_quality", "both_genders"]
    ),
    "asian_specialized": ModelConfig(
        file="chilloutmix.safetensors",
        name="ChilloutMix",
        type="photorealistic",
        sharpness_score=5,
        settings={
            "cfg": 7.0,
            "sampler": "euler",
            "scheduler": "normal",
            "steps": 25
        },
        capabilities=["asian_faces", "beauty_style", "soft_render", "both_genders"]
    )
}

# SSOT Project Configurations
PROJECTS = {
    "tokyo_debt_desire": {
        "name": "Tokyo Debt Desire",
        "style": "photorealistic",
        "content_rating": "nsfw",
        "default_model": "primary",  # Use epiCRealism by default
        "fallback_model": "backup",   # Use CyberRealistic if primary fails
        "resolution": {"width": 512, "height": 768},
        "negative_prompt": "anime, cartoon, blurry, soft focus, smooth skin, airbrushed",
        "style_prompt": "RAW photo, photorealistic, detailed skin texture, sharp focus, 85mm lens, DSLR quality"
    },
    "cyberpunk_goblin_slayer": {
        "name": "Cyberpunk Goblin Slayer: Neon Shadows",
        "style": "cyberpunk_anime",
        "content_rating": "mature",
        "default_model": "fallback",  # Anime style works better with RV
        "fallback_model": "backup",
        "resolution": {"width": 512, "height": 768},
        "negative_prompt": "realistic, photograph, blurry",
        "style_prompt": "anime style, cyberpunk aesthetic, neon lighting, detailed"
    },
    "super_mario_galaxy": {
        "name": "Super Mario Galaxy Anime Adventure",
        "style": "anime",
        "content_rating": "family",
        "default_model": "fallback",  # Anime style
        "fallback_model": "asian_specialized",
        "resolution": {"width": 512, "height": 768},
        "negative_prompt": "realistic, photograph, blurry, scary, violent",
        "style_prompt": "anime style, vibrant colors, nintendo character, expressive, cheerful"
    }
}

# SSOT Character Definitions
CHARACTERS = {
    "tokyo_debt_desire": {
        "Yuki_Tanaka": CharacterConfig(
            name="Yuki Tanaka",
            project="tokyo_debt_desire",
            gender="male",
            personality="nervous_anxious",
            model_override=None  # Use project default
        ),
        "Mei_Kobayashi": CharacterConfig(
            name="Mei Kobayashi",
            project="tokyo_debt_desire",
            gender="female",
            personality="gentle_caring",
            model_override=None  # Young housewife, soft features
        ),
        "Rina_Suzuki": CharacterConfig(
            name="Rina Suzuki",
            project="tokyo_debt_desire",
            gender="female",
            personality="confident_assertive",
            model_override=None  # Business woman, sharp features
        ),
        "Takeshi_Sato": CharacterConfig(
            name="Takeshi Sato",
            project="tokyo_debt_desire",
            gender="male",
            personality="intimidating_cold",
            model_override=None
        )
    },
    "cyberpunk_goblin_slayer": {
        "Kai": CharacterConfig(
            name="Kai",
            project="cyberpunk_goblin_slayer",
            gender="male",
            personality="young_fighter",
            model_override=None
        ),
        "Goblin_Slayer": CharacterConfig(
            name="Goblin Slayer",
            project="cyberpunk_goblin_slayer",
            gender="male",
            personality="armored_warrior",
            model_override=None
        ),
        "Ryuu": CharacterConfig(
            name="Ryuu",
            project="cyberpunk_goblin_slayer",
            gender="female",
            personality="elf_archer",
            model_override=None
        )
    }
}

class ProjectConfigManager:
    """Manages project configurations and model assignments"""

    def __init__(self):
        self.models = MODELS
        self.projects = PROJECTS
        self.characters = CHARACTERS

    def get_model_for_character(self, project: str, character: str) -> ModelConfig:
        """Get the optimal model for a character"""
        char_config = self.characters.get(project, {}).get(character)
        if not char_config:
            raise ValueError(f"Character {character} not found in project {project}")

        # Check for character-specific override
        if char_config.model_override:
            return self.models[char_config.model_override]

        # Use project default
        project_config = self.projects.get(project)
        if not project_config:
            raise ValueError(f"Project {project} not found")

        model_key = project_config["default_model"]
        return self.models[model_key]

    def get_workflow_config(self, project: str, character: str) -> Dict[str, Any]:
        """Get complete workflow configuration for a character"""
        model = self.get_model_for_character(project, character)
        project_config = self.projects[project]

        return {
            "model_file": model.file,
            "model_settings": model.settings,
            "resolution": project_config["resolution"],
            "negative_prompt": project_config["negative_prompt"],
            "style_prompt": project_config["style_prompt"]
        }

    def get_all_characters(self, project: str) -> List[str]:
        """Get all characters for a project"""
        return list(self.characters.get(project, {}).keys())

    def update_model_assignment(self, project: str, character: str, model_key: str):
        """Update model assignment for a character"""
        if project in self.characters and character in self.characters[project]:
            self.characters[project][character].model_override = model_key

# Singleton instance
config_manager = ProjectConfigManager()