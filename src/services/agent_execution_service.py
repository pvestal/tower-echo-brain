"""
Agent Execution Service - Handles agent task execution
"""
import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

class AgentExecutionService:
    def __init__(self):
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.task_history: Dict[str, Dict[str, Any]] = {}
        logger.info("Agent Execution Service initialized")

    async def execute_task(self, agent_type: str, task: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a task with the specified agent type"""
        task_id = str(uuid.uuid4())
        parameters = parameters or {}

        self.active_tasks[task_id] = {
            "task_id": task_id,
            "agent_type": agent_type,
            "status": "running",
            "task": task,
            "started_at": datetime.now().isoformat()
        }

        try:
            # Route to appropriate agent
            if agent_type == "coding":
                result = await self._execute_coding_task(task_id, task, parameters)
            elif agent_type == "reasoning":
                result = await self._execute_reasoning_task(task_id, task, parameters)
            elif agent_type == "narration":
                result = await self._execute_narration_task(task_id, task, parameters)
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")

            # Update task status
            result["status"] = "completed"
            result["completed_at"] = datetime.now().isoformat()
            self.task_history[task_id] = result
            self.active_tasks.pop(task_id, None)

            return result

        except Exception as e:
            error_result = {
                "task_id": task_id,
                "agent_type": agent_type,
                "status": "failed",
                "error": str(e),
                "completed_at": datetime.now().isoformat()
            }
            self.task_history[task_id] = error_result
            self.active_tasks.pop(task_id, None)
            logger.error(f"Task {task_id} failed: {e}")
            return error_result

    async def _execute_coding_task(self, task_id: str, task: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a coding task"""
        # In production, this would call the actual agent
        # For now, simulate the work
        await asyncio.sleep(0.2)

        code = parameters.get("code", f"# Solution for: {task}\ndef solution():\n    pass")

        return {
            "task_id": task_id,
            "agent_type": "coding",
            "task": task,
            "result": f"Code generated for: {task}",
            "code": code,
            "language": parameters.get("language", "python"),
            "execution_time": 0.2
        }

    async def _execute_reasoning_task(self, task_id: str, task: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a reasoning task"""
        await asyncio.sleep(0.15)

        return {
            "task_id": task_id,
            "agent_type": "reasoning",
            "task": task,
            "result": f"Analysis: {task}",
            "reasoning_steps": [
                f"1. Analyzed: {task}",
                "2. Considered implications",
                "3. Formulated conclusion"
            ],
            "confidence": 0.85,
            "execution_time": 0.15
        }

    async def _execute_narration_task(self, task_id: str, task: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a narration task"""
        await asyncio.sleep(0.25)

        style = parameters.get("style", "descriptive")

        return {
            "task_id": task_id,
            "agent_type": "narration",
            "task": task,
            "result": f"Narrative for: {task}",
            "narration": f"In the context of {task}, the narrative unfolds...",
            "style": style,
            "execution_time": 0.25
        }

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task"""
        if task_id in self.active_tasks:
            return self.active_tasks[task_id]
        elif task_id in self.task_history:
            return self.task_history[task_id]
        return None

    def get_all_tasks(self) -> Dict[str, Any]:
        """Get all tasks"""
        return {
            "active": list(self.active_tasks.values()),
            "completed": list(self.task_history.values())[-10:] if self.task_history else [],
            "total_completed": len(self.task_history),
            "total_active": len(self.active_tasks)
        }

# Global instance
agent_execution_service = AgentExecutionService()