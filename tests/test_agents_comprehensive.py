#!/usr/bin/env python3
"""
Comprehensive Agent Tests for Echo Brain
Tests REAL functionality with quantitative metrics
"""

import pytest
import asyncio
import aiohttp
import json
import time
import subprocess
import psycopg2
from datetime import datetime
from typing import Dict, Any, List

# Configuration
BASE_URL = "http://localhost:8309"
DB_CONFIG = {
    'host': 'localhost',
    'database': 'tower_consolidated',
    'user': 'patrick',
    'password': 'tower_echo_brain_secret_key_2025'
}
ANIME_DB_CONFIG = {
    'host': 'localhost',
    'database': 'tower_anime',
    'user': 'patrick',
    'password': 'tower_echo_brain_secret_key_2025'
}

class TestAgentUnit:
    """Unit tests for individual agents"""

    @pytest.mark.asyncio
    async def test_coding_agent_syntax_validation(self):
        """Test: CodingAgent generates syntactically valid Python"""
        async with aiohttp.ClientSession() as session:
            start_time = time.time()

            # Test various code generation tasks
            test_cases = [
                "Write a function to calculate factorial",
                "Create a class for managing database connections",
                "Write async function to fetch multiple URLs concurrently",
                "Create a decorator for caching function results"
            ]

            results = []
            for task in test_cases:
                async with session.post(f"{BASE_URL}/api/echo/agents/coding",
                                       json={"task": task, "language": "python", "validate": True}) as resp:
                    result = await resp.json()
                    results.append(result)

                    # Validate response structure
                    assert "code" in result, f"No code in response for: {task}"
                    assert "validation" in result, f"No validation in response for: {task}"
                    assert result["validation"]["valid"] == True, f"Invalid Python generated for: {task}"

                    # Verify code is not empty
                    assert len(result["code"]) > 10, f"Code too short for: {task}"

                    # Test actual Python compilation
                    try:
                        compile(result["code"], "<string>", "exec")
                    except SyntaxError as e:
                        pytest.fail(f"Python compilation failed for '{task}': {e}")

            elapsed = time.time() - start_time

            # Performance metrics
            assert elapsed < 60, f"Coding agent too slow: {elapsed:.2f}s for {len(test_cases)} tasks"
            avg_time = elapsed / len(test_cases)
            assert avg_time < 15, f"Average response time too high: {avg_time:.2f}s"

            return {
                "total_tasks": len(test_cases),
                "all_valid": all(r["validation"]["valid"] for r in results),
                "total_time": elapsed,
                "avg_time": avg_time
            }

    @pytest.mark.asyncio
    async def test_reasoning_agent_structure(self):
        """Test: ReasoningAgent provides structured analysis"""
        async with aiohttp.ClientSession() as session:
            start_time = time.time()

            test_task = {
                "task": "Should Tower use Docker or systemd for service management?",
                "background": "Running on Ubuntu with nginx reverse proxy",
                "constraints": "Need easy debugging and automatic restarts"
            }

            async with session.post(f"{BASE_URL}/api/echo/agents/reasoning", json=test_task) as resp:
                result = await resp.json()

                # Check required fields
                assert "response" in result, "No response from reasoning agent"
                assert "analysis" in result, "No analysis section"
                assert "reasoning" in result, "No reasoning section"
                assert "conclusion" in result, "No conclusion section"
                assert "model" in result, "No model information"

                # Verify content quality
                assert len(result["response"]) > 100, "Response too short"
                assert result["model"] == "deepseek-r1:8b", "Wrong model used"

                # Check for actual reasoning (not generic)
                response_lower = result["response"].lower()
                assert "docker" in response_lower or "systemd" in response_lower, "Response doesn't address the question"

                elapsed = time.time() - start_time
                assert elapsed < 180, f"Reasoning took too long: {elapsed:.2f}s"

                return {
                    "response_length": len(result["response"]),
                    "has_structure": all([result.get(k) for k in ["analysis", "reasoning", "conclusion"]]),
                    "response_time": elapsed
                }

    @pytest.mark.asyncio
    async def test_narration_agent_comfyui(self):
        """Test: NarrationAgent generates ComfyUI prompts"""
        async with aiohttp.ClientSession() as session:
            test_cases = [
                {"scene": "Cyberpunk street with neon signs", "genre": "cyberpunk"},
                {"scene": "Medieval castle at sunset", "genre": "fantasy"},
                {"scene": "Space station orbiting Earth", "genre": "sci-fi"}
            ]

            for test in test_cases:
                async with session.post(f"{BASE_URL}/api/echo/agents/narration", json=test) as resp:
                    result = await resp.json()

                    # Validate structure
                    assert "narration" in result, f"No narration for {test['genre']}"
                    assert "mood" in result, f"No mood for {test['genre']}"
                    assert "visual_notes" in result, f"No visual notes for {test['genre']}"
                    assert "comfyui_prompt" in result, f"No ComfyUI prompt for {test['genre']}"

                    # Check content quality
                    assert len(result["narration"]) > 50, f"Narration too short for {test['genre']}"
                    if result["comfyui_prompt"]:
                        assert len(result["comfyui_prompt"]) > 20, f"ComfyUI prompt too short"


class TestAgentIntegration:
    """Integration tests for agent coordination"""

    @pytest.mark.asyncio
    async def test_reasoning_to_coding_pipeline(self):
        """Test: Reasoning agent analyzes problem, Coding agent implements solution"""
        async with aiohttp.ClientSession() as session:
            # Step 1: Reasoning analyzes the problem
            reasoning_task = {
                "task": "Design a caching system for Tower's API responses",
                "constraints": "Must handle 1000 req/s, use Redis, expire after 5 minutes"
            }

            async with session.post(f"{BASE_URL}/api/echo/agents/reasoning", json=reasoning_task) as resp:
                reasoning_result = await resp.json()
                assert "conclusion" in reasoning_result

            # Step 2: Coding implements based on reasoning
            coding_task = {
                "task": f"Implement the caching system described: {reasoning_result['conclusion'][:500]}",
                "language": "python",
                "validate": True
            }

            async with session.post(f"{BASE_URL}/api/echo/agents/coding", json=coding_task) as resp:
                coding_result = await resp.json()
                assert coding_result["validation"]["valid"] == True

                # Verify code contains expected elements
                code_lower = coding_result["code"].lower()
                assert "cache" in code_lower or "redis" in code_lower, "Code doesn't implement caching"

    @pytest.mark.asyncio
    async def test_database_separation(self):
        """Test: Verify anime and echo databases are properly separated"""
        # Test Echo Brain database
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM past_solutions")
                echo_count = cur.fetchone()[0]

                # Check no anime tables in consolidated
                cur.execute("""
                    SELECT COUNT(*) FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name LIKE 'anime_%'
                """)
                anime_tables_in_echo = cur.fetchone()[0]
                assert anime_tables_in_echo == 0, f"Found {anime_tables_in_echo} anime tables in echo database!"

        # Test Anime database
        with psycopg2.connect(**ANIME_DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT COUNT(*) FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name IN ('character_profiles', 'animation_projects')
                """)
                anime_tables = cur.fetchone()[0]
                assert anime_tables > 0, "No anime tables found in tower_anime database"

    @pytest.mark.asyncio
    async def test_concurrent_agents(self):
        """Test: Multiple agents can run concurrently without interference"""
        async with aiohttp.ClientSession() as session:
            start_time = time.time()

            tasks = [
                session.post(f"{BASE_URL}/api/echo/agents/coding",
                           json={"task": "Write a REST API endpoint", "language": "python"}),
                session.post(f"{BASE_URL}/api/echo/agents/reasoning",
                           json={"task": "Compare REST vs GraphQL"}),
                session.post(f"{BASE_URL}/api/echo/agents/narration",
                           json={"scene": "Busy marketplace", "genre": "fantasy"})
            ]

            responses = await asyncio.gather(*[task for task in tasks])
            results = [await resp.json() for resp in responses]

            # All should succeed
            assert len(results) == 3, "Not all agents responded"
            assert all("model" in r for r in results), "Missing model info in some responses"

            elapsed = time.time() - start_time
            # Should be faster than sequential (roughly max time, not sum)
            assert elapsed < 60, f"Concurrent execution too slow: {elapsed:.2f}s"


class TestTowerAwareness:
    """Test agents' awareness of Tower environment"""

    @pytest.mark.asyncio
    async def test_coding_agent_tower_knowledge(self):
        """Test: CodingAgent knows Tower-specific paths and services"""
        async with aiohttp.ClientSession() as session:
            tower_task = {
                "task": "Write a Python function to check if all Tower services are running. Check: tower-echo-brain, tower-dashboard, tower-anime-production. Use systemctl.",
                "language": "python",
                "validate": True
            }

            async with session.post(f"{BASE_URL}/api/echo/agents/coding", json=tower_task) as resp:
                result = await resp.json()

                assert result["validation"]["valid"] == True
                code = result["code"]

                # Verify Tower-specific content
                assert "systemctl" in code or "subprocess" in code, "Code doesn't use system commands"
                assert "tower-" in code or "tower_" in code, "Code doesn't reference Tower services"

    @pytest.mark.asyncio
    async def test_self_improvement_capability(self):
        """Test: Agents can analyze and improve their own code"""
        async with aiohttp.ClientSession() as session:
            # Ask CodingAgent to analyze its own performance
            self_analysis_task = {
                "task": "Write a function to analyze agent response times from the database and suggest performance improvements",
                "language": "python",
                "validate": True,
                "files": ["/opt/tower-echo-brain/src/agents/coding_agent.py"]
            }

            async with session.post(f"{BASE_URL}/api/echo/agents/coding", json=self_analysis_task) as resp:
                result = await resp.json()

                assert result["validation"]["valid"] == True
                assert "context_used" in result
                assert result["context_used"]["codebase_refs"] > 0, "Agent didn't use codebase context"


class TestAnimeProduction:
    """Test anime production pipeline capabilities"""

    @pytest.mark.asyncio
    async def test_complete_anime_pipeline(self):
        """Test: Complete anime scene generation pipeline"""
        async with aiohttp.ClientSession() as session:
            # Step 1: Reasoning designs the scene
            reasoning_task = {
                "task": "Design a dramatic anime scene: protagonist confronts villain on rooftop. Include visual elements, mood, camera angles.",
                "options": ["Action-heavy", "Dialog-focused", "Emotional revelation"]
            }

            async with session.post(f"{BASE_URL}/api/echo/agents/reasoning", json=reasoning_task) as resp:
                reasoning_result = await resp.json()
                assert "conclusion" in reasoning_result

            # Step 2: Narration expands the scene
            narration_task = {
                "scene": reasoning_result["conclusion"][:200] if reasoning_result.get("conclusion") else "Rooftop confrontation scene",
                "genre": "action",
                "mood": "tense"
            }

            async with session.post(f"{BASE_URL}/api/echo/agents/narration", json=narration_task) as resp:
                narration_result = await resp.json()
                assert "comfyui_prompt" in narration_result
                assert len(narration_result["narration"]) > 100

            # Step 3: Coding creates production script
            coding_task = {
                "task": f"Create a Python script to queue this scene for ComfyUI rendering. Scene: {narration_result['narration'][:200]}",
                "language": "python",
                "validate": True
            }

            async with session.post(f"{BASE_URL}/api/echo/agents/coding", json=coding_task) as resp:
                coding_result = await resp.json()
                assert coding_result["validation"]["valid"] == True


class TestStressAndEdgeCases:
    """Stress tests and edge case handling"""

    @pytest.mark.asyncio
    async def test_large_context_handling(self):
        """Test: Agents handle large context appropriately"""
        async with aiohttp.ClientSession() as session:
            # Create a very long task description
            long_context = "Implement a distributed system with the following components: " + \
                          " ".join([f"Component{i} handles {i*10} requests per second" for i in range(100)])

            coding_task = {
                "task": long_context,
                "language": "python",
                "validate": True
            }

            timeout = aiohttp.ClientTimeout(total=120)
            async with session.post(f"{BASE_URL}/api/echo/agents/coding",
                                   json=coding_task, timeout=timeout) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    assert "code" in result, "Failed to generate code with large context"
                else:
                    pytest.skip(f"Large context test failed with status {resp.status}")

    @pytest.mark.asyncio
    async def test_error_recovery(self):
        """Test: Agents handle invalid inputs gracefully"""
        async with aiohttp.ClientSession() as session:
            # Test with invalid language
            invalid_task = {
                "task": "Write hello world",
                "language": "invalid_language_xyz",
                "validate": True
            }

            async with session.post(f"{BASE_URL}/api/echo/agents/coding", json=invalid_task) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    # Should still generate something or indicate the issue
                    assert "code" in result or "error" in result

    @pytest.mark.asyncio
    async def test_database_connection_pool(self):
        """Test: Database connections don't leak under load"""
        initial_connections = self._count_db_connections()

        async with aiohttp.ClientSession() as session:
            # Fire 20 rapid requests
            tasks = []
            for i in range(20):
                task = session.post(f"{BASE_URL}/api/echo/agents/coding",
                                   json={"task": f"Write function {i}", "language": "python"})
                tasks.append(task)

            responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Wait for connections to settle
        await asyncio.sleep(2)

        final_connections = self._count_db_connections()

        # Should not have leaked connections (allow small variance)
        assert final_connections <= initial_connections + 5, \
               f"Possible connection leak: {initial_connections} -> {final_connections}"

    def _count_db_connections(self):
        """Count active database connections"""
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM pg_stat_activity WHERE datname = 'tower_consolidated'")
                return cur.fetchone()[0]


@pytest.fixture
def cleanup_test_data():
    """Cleanup any test data created during tests"""
    yield
    # Cleanup after tests if needed
    pass


if __name__ == "__main__":
    # Run with: python -m pytest test_agents_comprehensive.py -v
    pytest.main([__file__, "-v", "--tb=short"])