#!/usr/bin/env python3
"""
Echo Anime Coordination System
Complete orchestration layer making Echo the central coordinator for anime production.
"""

from .echo_anime_coordinator import (
    echo_anime_coordinator,
    AnimeRequest,
    ProjectMemory,
    coordinate_anime_generation,
    get_project_context,
    record_user_feedback
)

from .unified_character_manager import (
    unified_character_manager,
    CharacterDefinition,
    get_unified_character,
    save_character_feedback,
    list_available_characters
)

from .style_learning_engine import (
    style_learning_engine,
    StylePreference,
    LearnedPreferences,
    analyze_and_enhance_prompt,
    record_style_feedback,
    get_user_style_analytics
)

from .session_context_manager import (
    session_context_manager,
    SessionContext,
    Platform,
    SessionState,
    get_or_create_session,
    migrate_to_platform,
    get_session_continuity_data
)

__all__ = [
    # Main coordinators
    "echo_anime_coordinator",
    "unified_character_manager",
    "style_learning_engine",
    "session_context_manager",

    # Data structures
    "AnimeRequest",
    "ProjectMemory",
    "CharacterDefinition",
    "StylePreference",
    "LearnedPreferences",
    "SessionContext",
    "Platform",
    "SessionState",

    # Convenience functions
    "coordinate_anime_generation",
    "get_project_context",
    "record_user_feedback",
    "get_unified_character",
    "save_character_feedback",
    "list_available_characters",
    "analyze_and_enhance_prompt",
    "record_style_feedback",
    "get_user_style_analytics",
    "get_or_create_session",
    "migrate_to_platform",
    "get_session_continuity_data"
]