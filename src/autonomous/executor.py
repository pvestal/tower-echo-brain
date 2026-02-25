"""
Task Executor for Echo Brain Autonomous Operations

Routes autonomous tasks to the appropriate Ollama agent via the AgentRegistry,
executes them, and returns structured results.
"""

import logging
import os
import time
from typing import Optional

import asyncpg
import httpx

from .models import TaskResult

logger = logging.getLogger(__name__)

# Map task_type values to agent registry intent strings
_TASK_TYPE_TO_INTENT = {
    "testing": "code_query",
    "analysis": "general_knowledge",
    "monitoring": "general_knowledge",
    "coding": "code_query",
    "system": "system_admin",
    "creative": "anime_production",
    "reasoning": "self_introspection",
}

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")


class Executor:
    """
    Executes autonomous tasks by routing them to Ollama agents.

    Uses the AgentRegistry for intent-based model selection and
    sends task descriptions to the selected model for execution.
    """

    def __init__(self):
        self._pool: Optional[asyncpg.Pool] = None
        self._registry = None
        self.db_config = {
            "host": "localhost",
            "port": 5432,
            "database": "echo_brain",
            "user": "patrick",
            "password": os.environ.get(
                "ECHO_BRAIN_DB_PASSWORD", os.getenv("TOWER_DB_PASSWORD", "")
            ),
        }

    async def _get_pool(self) -> asyncpg.Pool:
        if self._pool is None or self._pool._closed:
            self._pool = await asyncpg.create_pool(**self.db_config, min_size=1, max_size=3)
        return self._pool

    async def initialize_agents(self):
        """Load the agent registry so we can route tasks to models."""
        from src.core.agent_registry import get_agent_registry

        self._registry = get_agent_registry()
        agent_count = len(self._registry._agents) if self._registry._loaded else 0
        logger.info(f"Executor initialized with {agent_count} agents from registry")

    async def execute(self, task_id: int) -> TaskResult:
        """
        Execute an autonomous task by its ID.

        Fetches the task from DB, selects the right agent/model,
        sends the task description to Ollama, and returns the result.
        """
        start = time.time()
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT id, name, task_type, goal_id, metadata::text as metadata_text,
                           safety_level, priority
                    FROM autonomous_tasks WHERE id = $1
                    """,
                    task_id,
                )

            if not row:
                return TaskResult(
                    task_id=task_id,
                    success=False,
                    error=f"Task {task_id} not found in database",
                )

            task_name = row["name"]
            task_type = row["task_type"]
            metadata_raw = row["metadata_text"]
            metadata = {}
            if metadata_raw:
                import json
                try:
                    metadata = json.loads(metadata_raw)
                except Exception:
                    pass

            description = metadata.get("description", task_name)

            # Select agent via registry
            intent = _TASK_TYPE_TO_INTENT.get(task_type, "general_knowledge")
            agent = self._registry.select(intent) if self._registry else None

            if agent:
                model = await self._registry.resolve_model(agent)
                system_prompt = agent.system_prompt
                agent_name = agent.name
            else:
                model = "mistral:7b"
                system_prompt = "You are a helpful AI assistant."
                agent_name = "default"

            logger.info(
                f"Executing task {task_id} ({task_name}) via agent={agent_name} model={model}"
            )

            # Call Ollama
            prompt = (
                f"You are executing an autonomous task for the Echo Brain system.\n"
                f"Task: {task_name}\n"
                f"Type: {task_type}\n"
                f"Description: {description}\n\n"
                f"Execute this task and provide a concise result summary."
            )

            result_text = await self._call_ollama(model, system_prompt, prompt)
            elapsed = time.time() - start

            # Mark task completed in DB
            async with pool.acquire() as conn:
                await conn.execute(
                    "UPDATE autonomous_tasks SET status = 'completed' WHERE id = $1",
                    task_id,
                )

            return TaskResult(
                task_id=task_id,
                success=True,
                result=result_text,
                agent_used=agent_name,
                execution_time=round(elapsed, 2),
                metadata={"model": model, "task_type": task_type},
            )

        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"Task {task_id} execution failed: {e}")
            return TaskResult(
                task_id=task_id,
                success=False,
                error=str(e),
                execution_time=round(elapsed, 2),
            )

    async def _call_ollama(self, model: str, system_prompt: str, prompt: str) -> str:
        """Send a prompt to Ollama and return the response text."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "")

    async def cleanup(self):
        """Close database pool."""
        if self._pool and not self._pool._closed:
            await self._pool.close()
        logger.info("Executor cleaned up")
