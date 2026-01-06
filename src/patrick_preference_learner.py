#!/usr/bin/env python3
"""
Patrick's Preference Learning System
Echo learns what Patrick likes and generates accordingly
"""

import json
import asyncio
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class PatrickPreferenceLearner:
    """Learn and apply Patrick's content preferences"""

    def __init__(self):
        self.pref_file = Path("/opt/tower-echo-brain/data/patrick_preferences.json")
        self.pref_file.parent.mkdir(exist_ok=True)
        self.preferences = self._load_preferences()

    def _load_preferences(self) -> Dict:
        """Load Patrick's preferences"""
        if self.pref_file.exists():
            return json.loads(self.pref_file.read_text())
        else:
            # Initialize with what we know about Patrick
            return {
                "technical_style": {
                    "preferred": ["modern", "technical", "systematic"],
                    "quality": ["accurate", "efficient", "reliable"],
                    "avoid": ["outdated", "inefficient", "unreliable"]
                },
                "projects": {
                    "echo_brain_optimization": {
                        "theme": "AI system optimization, performance tuning",
                        "mood": "technical, analytical, solution-focused",
                        "setting": "local development environment"
                    },
                    "tower_infrastructure": {
                        "theme": "system architecture, automation, monitoring",
                        "mood": "systematic, efficient, robust",
                        "setting": "home server environment"
                    }
                },
                "character_traits": {
                    "likes": ["detailed eyes", "dynamic poses", "expressive faces"],
                    "dislikes": ["static poses", "bland expressions"]
                },
                "generation_stats": {
                    "total_generated": 0,
                    "favorites": [],
                    "last_generated": None
                }
            }

    def save_preferences(self):
        """Save preferences to disk"""
        self.pref_file.write_text(json.dumps(self.preferences, indent=2))

    def learn_from_generation(self, character: str, project: str, prompt: str, success: bool):
        """Learn from each generation"""

        # Update stats
        self.preferences["generation_stats"]["total_generated"] += 1
        self.preferences["generation_stats"]["last_generated"] = {
            "character": character,
            "project": project,
            "timestamp": datetime.now().isoformat(),
            "success": success
        }

        # If successful, remember what worked
        if success and character:
            if character not in self.preferences["generation_stats"].get("successful_characters", []):
                self.preferences["generation_stats"].setdefault("successful_characters", []).append(character)

        self.save_preferences()

    def enhance_prompt_with_preferences(self, base_prompt: str, project: str = None) -> str:
        """Enhance prompts with Patrick's preferences"""

        # Add preferred quality tags
        quality_tags = self.preferences["art_style"]["quality"]
        enhanced = f"{base_prompt}, {', '.join(quality_tags)}"

        # Add project-specific styling if known
        if project and project in self.preferences["projects"]:
            proj_data = self.preferences["projects"][project]
            enhanced += f", {proj_data['mood']}"

        # Add character traits Patrick likes
        for trait in self.preferences["character_traits"]["likes"][:2]:
            if trait not in enhanced:
                enhanced += f", {trait}"

        return enhanced

    def get_recommendation(self) -> Dict:
        """Recommend what to generate next based on patterns"""

        stats = self.preferences["generation_stats"]

        # Suggest characters that haven't been generated recently
        all_chars = ["riku", "yuki", "sakura", "kai_nakamura", "ryuu", "hiroshi"]
        successful = stats.get("successful_characters", [])

        # Find characters not generated yet
        not_generated = [c for c in all_chars if c not in successful]

        if not_generated:
            return {
                "recommendation": f"Generate {not_generated[0]} next",
                "reason": "Character not yet generated with new system"
            }
        else:
            return {
                "recommendation": "Generate variations of existing characters",
                "reason": "All characters have base images"
            }

# Singleton
preference_learner = PatrickPreferenceLearner()