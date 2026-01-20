"""
Task Executor for Echo Brain Autonomous Operations

The Executor class handles the routing and execution of autonomous tasks,
integrating with specialized agents and providing structured results.
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass
import asyncpg
from contextlib import asynccontextmanager

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import existing agents
from src.agents.coding_agent import CodingAgent
from src.agents.reasoning_agent import ReasoningAgent
from src.agents.narration_agent import NarrationAgent

# Import context provider
from src.core.unified_context import get_context_provider

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """Structured result from task execution"""
    task_id: int
    success: bool
    result: Optional[str] = None
    error: Optional[str] = None
    agent_used: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class Executor:
    """
    Executes autonomous tasks by routing them to appropriate agents.

    Provides task execution capabilities with proper error handling,
    context integration, and structured result reporting.
    """

    def __init__(self):
        """Initialize the Executor with database configuration and agents."""
        self.db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'tower_consolidated',
            'user': 'patrick',
            'password': os.environ.get('ECHO_BRAIN_DB_PASSWORD', 'RP78eIrW7cI2jYvL5akt1yurE')
        }
        self._pool = None

        # Initialize agents
        self.coding_agent = None
        self.reasoning_agent = None
        self.narration_agent = None

        # Context provider
        self.context_provider = get_context_provider()

        # Task type routing map
        self.agent_routing = {
            'coding': 'coding_agent',
            'programming': 'coding_agent',
            'code_generation': 'coding_agent',
            'debugging': 'coding_agent',
            'script_creation': 'coding_agent',

            'reasoning': 'reasoning_agent',
            'analysis': 'reasoning_agent',
            'decision_making': 'reasoning_agent',
            'problem_solving': 'reasoning_agent',
            'research': 'reasoning_agent',

            'narration': 'narration_agent',
            'creative_writing': 'narration_agent',
            'scene_description': 'narration_agent',
            'anime_content': 'narration_agent',
            'storytelling': 'narration_agent'
        }

    @asynccontextmanager
    async def get_connection(self):
        """Get a database connection from the pool."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(**self.db_config, min_size=2, max_size=10)

        async with self._pool.acquire() as connection:
            yield connection

    async def initialize_agents(self):
        """Initialize all agents lazily."""
        try:
            if self.coding_agent is None:
                self.coding_agent = CodingAgent()
                logger.info("Initialized CodingAgent")

            if self.reasoning_agent is None:
                self.reasoning_agent = ReasoningAgent()
                logger.info("Initialized ReasoningAgent")

            if self.narration_agent is None:
                self.narration_agent = NarrationAgent()
                logger.info("Initialized NarrationAgent")

        except Exception as e:
            logger.error(f"Failed to initialize agents: {e}")
            raise

    async def execute(self, task_id: int) -> TaskResult:
        """
        Execute a task by routing it to the appropriate agent.

        Args:
            task_id: Database ID of the task to execute

        Returns:
            TaskResult with execution outcome and details
        """
        start_time = datetime.now()

        try:
            # Get task details from database
            async with self.get_connection() as conn:
                task = await conn.fetchrow("""
                    SELECT id, name, task_type, goal_id, result, metadata
                    FROM autonomous_tasks
                    WHERE id = $1
                """, task_id)

                if not task:
                    return TaskResult(
                        task_id=task_id,
                        success=False,
                        error=f"Task {task_id} not found in database"
                    )

            # Update task status to in_progress
            await self._update_task_status(task_id, 'in_progress', start_time)

            # Prepare context for the task
            context = await self._prepare_context(task)

            # Route to appropriate agent
            agent = await self._route_to_agent(task['task_type'])

            if not agent:
                return TaskResult(
                    task_id=task_id,
                    success=False,
                    error=f"No agent available for task type: {task['task_type']}",
                    agent_used="none"
                )

            # Execute the task
            logger.info(f"Executing task {task_id} ({task['task_type']}) with {agent.__class__.__name__}")

            # Build the prompt with context
            prompt = self._build_task_prompt(task, context)

            # Execute with the agent
            result = await agent.process_request(
                user_input=prompt,
                context=context.get('context_summary', '')
            )

            execution_time = (datetime.now() - start_time).total_seconds()

            # Update task in database
            await self._update_task_completion(task_id, result, None, execution_time)

            return TaskResult(
                task_id=task_id,
                success=True,
                result=result,
                agent_used=agent.__class__.__name__,
                execution_time=execution_time,
                metadata={
                    'task_type': task['task_type'],
                    'goal_id': task['goal_id'],
                    'context_items': {
                        'memories': len(context.get('memories', [])),
                        'facts': len(context.get('facts', [])),
                        'recent_conversations': len(context.get('recent_conversations', []))
                    }
                }
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Task execution failed: {str(e)}"
            logger.error(error_msg)

            # Update task with error
            await self._update_task_completion(task_id, None, error_msg, execution_time)

            return TaskResult(
                task_id=task_id,
                success=False,
                error=error_msg,
                execution_time=execution_time
            )

    async def _route_to_agent(self, task_type: str) -> Optional[Any]:
        """
        Route task to appropriate agent based on task type.

        Args:
            task_type: The type of task to route

        Returns:
            Agent instance or None if no suitable agent found
        """
        await self.initialize_agents()

        # Normalize task type for routing
        task_type_lower = task_type.lower()

        # Check direct mapping first
        if task_type_lower in self.agent_routing:
            agent_name = self.agent_routing[task_type_lower]
            return getattr(self, agent_name, None)

        # Fallback logic for partial matches
        if any(keyword in task_type_lower for keyword in ['code', 'program', 'script', 'debug']):
            return self.coding_agent
        elif any(keyword in task_type_lower for keyword in ['reason', 'analys', 'decide', 'research']):
            return self.reasoning_agent
        elif any(keyword in task_type_lower for keyword in ['narrat', 'story', 'creative', 'anime']):
            return self.narration_agent

        # Default to reasoning agent for unknown types
        logger.warning(f"Unknown task type '{task_type}', defaulting to ReasoningAgent")
        return self.reasoning_agent

    async def _prepare_context(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare context for task execution.

        Args:
            task: Task dictionary from database

        Returns:
            Context dictionary with relevant information
        """
        try:
            # Build context query from task name and type
            context_query = f"{task['name']} {task['task_type']}"

            # Get context from unified context provider
            context = await self.context_provider.get_context(
                query=context_query,
                limit=10
            )

            # Add task-specific metadata
            if task.get('metadata'):
                context['task_metadata'] = task['metadata']

            return context

        except Exception as e:
            logger.error(f"Failed to prepare context for task {task['id']}: {e}")
            return {
                'query': f"{task['name']} {task['task_type']}",
                'memories': [],
                'facts': [],
                'recent_conversations': [],
                'error': str(e)
            }

    def _build_task_prompt(self, task: Dict[str, Any], context: Dict[str, Any]) -> str:
        """
        Build the prompt for the agent based on task and context.

        Args:
            task: Task dictionary
            context: Context dictionary

        Returns:
            Formatted prompt string
        """
        prompt_parts = [
            f"Task: {task['name']}",
            f"Task Type: {task['task_type']}"
        ]

        # Add context summary if available
        if context.get('context_summary'):
            prompt_parts.append(f"Context: {context['context_summary']}")

        # Add relevant memories
        if context.get('memories'):
            memories_text = "\n".join([
                f"- {memory['text'][:200]}..." if len(memory['text']) > 200 else f"- {memory['text']}"
                for memory in context['memories'][:3]  # Top 3 memories
            ])
            prompt_parts.append(f"Relevant Information:\n{memories_text}")

        # Add task metadata if available
        if task.get('metadata') and isinstance(task['metadata'], dict):
            for key, value in task['metadata'].items():
                if isinstance(value, str) and value:
                    prompt_parts.append(f"{key.replace('_', ' ').title()}: {value}")

        return "\n\n".join(prompt_parts)

    async def _update_task_status(self, task_id: int, status: str, timestamp: Optional[datetime] = None) -> None:
        """Update task status in database."""
        try:
            async with self.get_connection() as conn:
                if status == 'in_progress':
                    await conn.execute("""
                        UPDATE autonomous_tasks
                        SET status = $1, started_at = $2
                        WHERE id = $3
                    """, status, timestamp or datetime.now(), task_id)
                else:
                    await conn.execute("""
                        UPDATE autonomous_tasks
                        SET status = $1
                        WHERE id = $2
                    """, status, task_id)

        except Exception as e:
            logger.error(f"Failed to update task {task_id} status to {status}: {e}")

    async def _update_task_completion(self, task_id: int, result: Optional[str],
                                    error: Optional[str], execution_time: float) -> None:
        """Update task completion in database."""
        try:
            async with self.get_connection() as conn:
                status = 'completed' if result and not error else 'failed'

                await conn.execute("""
                    UPDATE autonomous_tasks
                    SET status = $1, result = $2, error = $3, completed_at = $4
                    WHERE id = $5
                """, status, result, error, datetime.now(), task_id)

                logger.info(f"Task {task_id} marked as {status} (execution time: {execution_time:.2f}s)")

        except Exception as e:
            logger.error(f"Failed to update task {task_id} completion: {e}")

    async def get_agent_capabilities(self) -> Dict[str, List[str]]:
        """
        Get the capabilities of all available agents.

        Returns:
            Dictionary mapping agent names to their supported task types
        """
        await self.initialize_agents()

        capabilities = {}

        for task_type, agent_name in self.agent_routing.items():
            agent = getattr(self, agent_name, None)
            if agent:
                agent_class_name = agent.__class__.__name__
                if agent_class_name not in capabilities:
                    capabilities[agent_class_name] = []
                capabilities[agent_class_name].append(task_type)

        return capabilities

    async def cleanup(self):
        """Cleanup resources."""
        if self._pool:
            await self._pool.close()

        # Context provider cleanup
        if self.context_provider:
            await self.context_provider.cleanup()

        logger.info("Executor cleaned up")