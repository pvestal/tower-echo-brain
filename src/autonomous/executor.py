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

# Import Tower LLM delegation for heavy reasoning tasks
from src.core.tower_llm_executor import TowerLLMExecutor

# Import context provider
from src.core.context import get_optimized_omniscient_context

logger = logging.getLogger(__name__)


# Import TaskResult from models module
from .models import TaskResult


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
            'database': 'echo_brain',
            'user': 'patrick',
            'password': os.environ.get('ECHO_BRAIN_DB_PASSWORD', 'RP78eIrW7cI2jYvL5akt1yurE')
        }
        self._pool = None

        # Initialize agents
        self.coding_agent = None
        self.reasoning_agent = None
        self.narration_agent = None

        # Initialize Tower LLM executor for delegation
        self.tower_executor = TowerLLMExecutor()

        # Model preferences for different task types
        self.model_preferences = {
            'heavy_reasoning': 'deepseek-r1:8b',  # DeepSeek for complex reasoning
            'code_generation': 'deepseek-coder-v2:16b',  # DeepSeek Coder for programming
            'quick_analysis': 'qwen2.5-coder:7b',  # Qwen for quick tasks
            'documentation': 'qwen2.5:3b'  # Small model for simple docs
        }

        # Context provider
        self.context_provider = get_optimized_omniscient_context()

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
        Enforces safety levels before execution.

        Args:
            task_id: Database ID of the task to execute

        Returns:
            TaskResult with execution outcome and details
        """
        start_time = datetime.now()

        try:
            # Get task details from database including safety level and status
            async with self.get_connection() as conn:
                task = await conn.fetchrow("""
                    SELECT id, name, task_type, goal_id, result, metadata, safety_level, status
                    FROM autonomous_tasks
                    WHERE id = $1
                """, task_id)

                if not task:
                    return TaskResult(
                        task_id=task_id,
                        success=False,
                        error=f"Task {task_id} not found in database"
                    )

            # Extract safety level and status from task
            metadata = task.get('metadata')
            if isinstance(metadata, str):
                # If metadata is a JSON string, parse it
                import json
                try:
                    metadata = json.loads(metadata) if metadata else {}
                except:
                    metadata = {}
            elif metadata is None:
                metadata = {}

            # Use the actual safety_level column from database
            safety_level = (task.get('safety_level') or 'review').lower()
            task_status = task.get('status')

            # Skip safety checks for already approved tasks
            if task_status == 'approved':
                logger.info(f"Task {task_id} is pre-approved with safety level '{safety_level}', proceeding with execution")
                # Continue to actual execution below
            else:
                # Apply safety level restrictions for non-approved tasks
                safety_level = safety_level.lower()  # Ensure lowercase for comparison

                # FORBIDDEN - Reject immediately
                if safety_level == 'forbidden':
                    error_msg = f"Task {task_id} has FORBIDDEN safety level and cannot be executed"
                    logger.warning(error_msg)

                    # Log the attempt in notifications
                    await self._create_notification(
                        notification_type='forbidden_attempt',
                        title=f"Forbidden Task Attempted: {task['name']}",
                        message=f"Task {task_id} with FORBIDDEN safety level was attempted but blocked",
                        task_id=task_id
                    )

                    await self._update_task_status(task_id, 'rejected')

                    return TaskResult(
                        task_id=task_id,
                        success=False,
                        error=error_msg,
                        metadata={'safety_level': 'forbidden', 'blocked': True}
                    )

                # REVIEW - Mark for approval, do not execute
                elif safety_level == 'review':
                    logger.info(f"Task {task_id} requires human approval (REVIEW safety level)")

                    # Mark as needing approval
                    await self._update_task_status(task_id, 'needs_approval')

                    # Create notification for approval
                    await self._create_notification(
                        notification_type='approval_required',
                        title=f"Approval Required: {task['name']}",
                        message=f"Task {task_id} requires your approval before execution",
                        task_id=task_id
                    )

                    return TaskResult(
                        task_id=task_id,
                        success=True,  # Successfully queued for approval
                        result="Task queued for human approval",
                        metadata={'safety_level': 'review', 'status': 'needs_approval'}
                    )

            # AUTO or NOTIFY - Proceed with execution
            # Update task status to in_progress
            await self._update_task_status(task_id, 'in_progress', start_time)

            # Prepare context for the task
            context = await self._prepare_context(task)

            # Check if task should be delegated to Tower LLM
            should_delegate, model = await self._should_delegate_to_tower(task, context)

            if should_delegate:
                # Delegate to Tower LLM for execution
                logger.info(f"Delegating task {task_id} to Tower LLM ({model})")

                # Set the appropriate model
                self.tower_executor.model = model

                # Build delegation request
                delegation_result = await self.tower_executor.delegate_task(
                    task=f"{task['name']}: {task.get('description', '')}",
                    context={
                        'task_type': task['task_type'],
                        'goal': task.get('goal_name', ''),
                        'context_summary': context.get('context_summary', ''),
                        'facts': context.get('facts', [])[:5],  # Include top 5 relevant facts
                        'recent_memories': context.get('memories', [])[:3]  # Include 3 recent memories
                    }
                )

                if delegation_result.get('success'):
                    result = f"Tower LLM ({model}) executed: {delegation_result.get('results', [])}"
                    agent_used = f"TowerLLM-{model}"
                else:
                    # Fallback to regular agent if delegation fails
                    logger.warning(f"Delegation failed, falling back to regular agent")
                    agent = await self._route_to_agent(task['task_type'])

                    if not agent:
                        return TaskResult(
                            task_id=task_id,
                            success=False,
                            error=f"No agent available for task type: {task['task_type']}",
                            agent_used="none"
                        )

                    prompt = self._build_task_prompt(task, context)
                    result = await agent.process(
                        task=prompt,
                        include_context=True,
                        context=context
                    )
                    agent_used = agent.__class__.__name__
            else:
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
                result = await agent.process(
                    task=prompt,
                    include_context=True,
                    context=context
                )
                agent_used = agent.__class__.__name__

            execution_time = (datetime.now() - start_time).total_seconds()

            # Update task in database
            await self._update_task_completion(task_id, result, None, execution_time)

            # NOTIFY - Create notification after successful execution
            if safety_level == 'notify':
                await self._create_notification(
                    notification_type='task_executed',
                    title=f"Task Executed: {task['name']}",
                    message=f"Task {task_id} was automatically executed with NOTIFY safety level",
                    task_id=task_id
                )
                logger.info(f"Notification created for NOTIFY task {task_id}")

            return TaskResult(
                task_id=task_id,
                success=True,
                result=result,
                agent_used=agent_used if 'agent_used' in locals() else agent.__class__.__name__,
                execution_time=execution_time,
                metadata={
                    'task_type': task['task_type'],
                    'goal_id': task['goal_id'],
                    'safety_level': safety_level,
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

    async def _should_delegate_to_tower(self, task: Dict[str, Any], context: Dict[str, Any]) -> tuple[bool, str]:
        """
        Determine if a task should be delegated to Tower LLM.

        Returns:
            Tuple of (should_delegate, model_to_use)
        """
        task_type = task.get('task_type', '').lower()
        task_name = task.get('name', '').lower()

        # Keywords indicating heavy reasoning tasks for DeepSeek
        heavy_reasoning_keywords = [
            'analyze', 'research', 'investigate', 'complex', 'reasoning',
            'decision', 'strategy', 'optimize', 'evaluate', 'compare',
            'multi-step', 'deep', 'comprehensive'
        ]

        # Keywords indicating code generation for DeepSeek Coder
        code_keywords = [
            'implement', 'code', 'program', 'script', 'function',
            'class', 'api', 'algorithm', 'refactor', 'debug'
        ]

        # Check for heavy reasoning tasks
        if any(keyword in task_name for keyword in heavy_reasoning_keywords):
            return True, self.model_preferences['heavy_reasoning']  # DeepSeek-R1

        # Check for complex coding tasks
        if task_type in ['coding', 'programming', 'code_generation']:
            # Complex coding goes to DeepSeek Coder
            if 'complex' in task_name or 'implement' in task_name:
                return True, self.model_preferences['code_generation']  # DeepSeek Coder V2
            # Simple coding can go to Qwen
            else:
                return True, self.model_preferences['quick_analysis']  # Qwen 2.5

        # Documentation tasks
        if 'document' in task_name or 'explain' in task_name:
            return True, self.model_preferences['documentation']  # Small Qwen

        # Default: Don't delegate simple tasks
        return False, None

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

    async def _create_notification(self, notification_type: str, title: str,
                                  message: str, task_id: Optional[int] = None) -> None:
        """Create a notification in the database."""
        try:
            async with self.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO autonomous_notifications
                    (notification_type, title, message, task_id, read, created_at)
                    VALUES ($1, $2, $3, $4, false, $5)
                """, notification_type, title, message, task_id, datetime.now())

        except Exception as e:
            logger.error(f"Failed to create notification: {e}")

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
                elif status in ('needs_approval', 'rejected'):
                    await conn.execute("""
                        UPDATE autonomous_tasks
                        SET status = $1, updated_at = $2
                        WHERE id = $3
                    """, status, datetime.now(), task_id)
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