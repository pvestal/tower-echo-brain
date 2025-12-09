"""
Anime Story Orchestrator - Echo Brain Integration
Wrapper for AnimeQualityOrchestrator to provide story orchestration capabilities
"""

import sys
sys.path.append('/opt/tower-anime-production/quality')

from anime_quality_orchestrator import AnimeQualityOrchestrator as AnimeStoryOrchestrator

# Export for Echo Brain
__all__ = ['AnimeStoryOrchestrator']
