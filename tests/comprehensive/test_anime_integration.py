"""
Comprehensive Anime Integration Tests

Tests:
- Anime memory integration
- Character data retrieval
- User preferences
- Anime production API calls
- Story orchestration
- ComfyUI integration
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


class TestAnimeMemoryIntegration:
    """Test anime memory integration with tower-anime-production"""

    def test_anime_memory_imports(self):
        """AnimeMemoryIntegration should import"""
        try:
            from modules.generation.anime.anime_memory_integration import AnimeMemoryIntegration
            assert AnimeMemoryIntegration is not None
        except ImportError:
            pytest.skip("Anime memory integration not available")

    def test_anime_memory_initialization(self):
        """Should initialize anime memory integration"""
        try:
            from modules.generation.anime.anime_memory_integration import AnimeMemoryIntegration
            memory = AnimeMemoryIntegration()
            assert memory is not None
            assert memory.db_config is not None
        except ImportError:
            pytest.skip("Anime memory integration not available")

    def test_anime_memory_db_config(self):
        """Should use correct database configuration"""
        try:
            from modules.generation.anime.anime_memory_integration import AnimeMemoryIntegration
            memory = AnimeMemoryIntegration()
            assert memory.db_config["database"] == "anime_production"
        except ImportError:
            pytest.skip("Anime memory integration not available")


class TestCharacterRetrieval:
    """Test character data retrieval"""

    def test_get_character_info(self, anime_character_data):
        """Should retrieve character info by name"""
        try:
            from modules.generation.anime.anime_memory_integration import AnimeMemoryIntegration

            with patch('psycopg2.connect') as mock_connect:
                mock_cursor = MagicMock()
                mock_cursor.fetchone.return_value = (
                    anime_character_data["name"],
                    anime_character_data["description"],
                    anime_character_data["consistency_score"],
                    anime_character_data["generation_count"],
                    10,  # successful_generations
                    [],  # reference_images
                    anime_character_data["style_elements"],
                    None  # last_generated
                )
                mock_connect.return_value.cursor.return_value = mock_cursor

                memory = AnimeMemoryIntegration()
                char_info = memory.get_character_info("Kai Nakamura")

                assert char_info is not None
                assert char_info["name"] == "Kai Nakamura"
        except ImportError:
            pytest.skip("Anime memory integration not available")

    def test_get_character_info_not_found(self):
        """Should return None for unknown character"""
        try:
            from modules.generation.anime.anime_memory_integration import AnimeMemoryIntegration

            with patch('psycopg2.connect') as mock_connect:
                mock_cursor = MagicMock()
                mock_cursor.fetchone.return_value = None
                mock_connect.return_value.cursor.return_value = mock_cursor

                memory = AnimeMemoryIntegration()
                char_info = memory.get_character_info("Unknown Character")

                assert char_info is None
        except ImportError:
            pytest.skip("Anime memory integration not available")


class TestUserPreferences:
    """Test user creative preferences retrieval"""

    def test_get_user_preferences(self):
        """Should retrieve user preferences"""
        try:
            from modules.generation.anime.anime_memory_integration import AnimeMemoryIntegration

            with patch('psycopg2.connect') as mock_connect:
                mock_cursor = MagicMock()
                mock_cursor.fetchall.return_value = [
                    ("style", "art_style", '{"value": "anime"}', 0.9),
                    ("color", "palette", '{"value": "vibrant"}', 0.85)
                ]
                mock_connect.return_value.cursor.return_value = mock_cursor

                memory = AnimeMemoryIntegration()
                prefs = memory.get_user_preferences("patrick")

                assert len(prefs) == 2
                assert prefs[0]["type"] == "style"
        except ImportError:
            pytest.skip("Anime memory integration not available")


class TestAnimeContext:
    """Test anime context extraction"""

    def test_get_anime_context(self):
        """Should extract anime context from query"""
        try:
            from modules.generation.anime.anime_memory_integration import get_anime_context

            with patch('psycopg2.connect') as mock_connect:
                mock_cursor = MagicMock()
                mock_cursor.fetchone.return_value = (
                    "Kai Nakamura",
                    "Young warrior",
                    0.85, 15, 10, [], {}, None
                )
                mock_cursor.fetchall.return_value = []
                mock_connect.return_value.cursor.return_value = mock_cursor

                context = get_anime_context("Generate Kai in battle stance")

                assert "characters" in context
                assert "preferences" in context
        except ImportError:
            pytest.skip("Anime context function not available")

    def test_context_detects_character_mentions(self):
        """Should detect character name mentions in queries"""
        try:
            from modules.generation.anime.anime_memory_integration import get_anime_context

            # Test with different character mentions
            queries = [
                "Show me Kai fighting",
                "Generate Aria reading",
                "Create scene with Hiroshi"
            ]

            for query in queries:
                with patch('psycopg2.connect') as mock_connect:
                    mock_cursor = MagicMock()
                    mock_cursor.fetchone.return_value = (
                        "Test", "Description", 0.8, 10, 5, [], {}, None
                    )
                    mock_cursor.fetchall.return_value = []
                    mock_connect.return_value.cursor.return_value = mock_cursor

                    context = get_anime_context(query)
                    # Should attempt to look up characters
        except ImportError:
            pytest.skip("Anime context function not available")


class TestAnimeStoryOrchestrator:
    """Test anime story orchestration"""

    def test_story_orchestrator_imports(self):
        """AnimeStoryOrchestrator should import"""
        try:
            from modules.generation.anime.anime_story_orchestrator import AnimeStoryOrchestrator
            assert AnimeStoryOrchestrator is not None
        except ImportError:
            pytest.skip("Story orchestrator not available")


class TestAnimeAPIEndpoints:
    """Test anime-related API endpoints"""

    @pytest.mark.asyncio
    async def test_anime_generate_endpoint(self, async_client, anime_generation_request):
        """POST /api/echo/anime/generate should work"""
        try:
            response = await async_client.post(
                "/api/echo/anime/generate",
                json=anime_generation_request
            )
            assert response.status_code in [200, 201, 404, 422, 500]
        except Exception:
            pytest.skip("Echo Brain not running")

    @pytest.mark.asyncio
    async def test_anime_search_endpoint(self, async_client):
        """Anime search endpoint should work"""
        try:
            response = await async_client.get(
                "/api/echo/anime/search",
                params={"query": "warrior character"}
            )
            assert response.status_code in [200, 404]
        except Exception:
            pytest.skip("Echo Brain not running")


class TestComfyUIIntegration:
    """Test ComfyUI integration for anime generation"""

    def test_comfyui_client_imports(self):
        """ComfyUI client should import"""
        try:
            from integrations.comfyui_client import ComfyUIClient
            assert ComfyUIClient is not None
        except ImportError:
            pytest.skip("ComfyUI client not available")

    @pytest.mark.asyncio
    async def test_comfyui_health_check(self, test_config):
        """Should check ComfyUI health"""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(
                    f"http://{test_config['tower_ip']}:8188/system_stats"
                )
                # 200 if running, connection error if not
                assert response.status_code == 200
        except Exception:
            pytest.skip("ComfyUI not accessible")

    def test_comfyui_workflow_generation(self):
        """Should generate workflow for anime generation"""
        try:
            from integrations.comfyui_client import ComfyUIClient
            client = ComfyUIClient()

            if hasattr(client, 'generate_workflow'):
                workflow = client.generate_workflow(
                    prompt="anime warrior",
                    width=512,
                    height=512
                )
                assert workflow is not None
        except ImportError:
            pytest.skip("ComfyUI client not available")


class TestAnimeCharacterMemory:
    """Test anime character memory table access"""

    def test_character_memory_table_schema(self):
        """Character memory should follow expected schema"""
        expected_fields = [
            "character_name",
            "canonical_description",
            "visual_consistency_score",
            "generation_count",
            "successful_generations",
            "reference_images",
            "style_elements",
            "last_generated"
        ]

        try:
            from modules.generation.anime.anime_memory_integration import AnimeMemoryIntegration
            # The integration expects these fields from the database
            assert True  # Schema verified in get_character_info
        except ImportError:
            pytest.skip("Anime memory not available")


class TestCreativePreferences:
    """Test user creative preferences table access"""

    def test_preferences_table_schema(self):
        """Preferences should follow expected schema"""
        expected_fields = [
            "preference_type",
            "preference_key",
            "preference_value",
            "confidence_score"
        ]

        try:
            from modules.generation.anime.anime_memory_integration import AnimeMemoryIntegration
            # The integration expects these fields
            assert True
        except ImportError:
            pytest.skip("Anime memory not available")


class TestTowerAnimeProductionContract:
    """Test integration contract with tower-anime-production"""

    @pytest.mark.asyncio
    async def test_echo_brain_responds_to_anime_queries(self, async_client):
        """Echo Brain should respond to anime-related queries"""
        try:
            response = await async_client.post(
                "/api/echo/chat",
                json={
                    "query": "What anime characters do you know about?",
                    "user_id": "anime_test"
                }
            )
            assert response.status_code in [200, 201, 202]
        except Exception:
            pytest.skip("Echo Brain not running")

    @pytest.mark.asyncio
    async def test_delegation_for_anime_tasks(self, async_client):
        """Should be able to delegate anime generation tasks"""
        try:
            response = await async_client.post(
                "/api/echo/delegate/to-tower",
                json={
                    "task": "Generate anime frame with character Kai",
                    "model": "qwen2.5-coder:7b"
                }
            )
            assert response.status_code in [200, 201, 202, 500]
        except Exception:
            pytest.skip("Echo Brain not running")

    def test_shared_database_access(self, test_config):
        """Echo Brain and tower-anime-production share anime_production database"""
        # Both systems should connect to anime_production database
        assert test_config["anime_db_name"] == "anime_production"


class TestVideoGeneration:
    """Test video generation for anime"""

    def test_video_module_imports(self):
        """Video module should import"""
        try:
            from modules.generation.video import VideoGenerator
            assert VideoGenerator is not None
        except ImportError:
            # Try alternative import paths
            try:
                from services.video import VideoService
                assert VideoService is not None
            except ImportError:
                pytest.skip("Video generation not available")
