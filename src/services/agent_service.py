"""
Unified Agent Service
Combines agents/, autonomous/, and collaboration functionality
"""

import asyncio
from typing import Dict, Any, List, Optional
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class AgentType(Enum):
    CODING = "coding"
    REASONING = "reasoning"
    NARRATION = "narration"
    AUTONOMOUS = "autonomous"
    MOLTBOOK = "moltbook"

class AgentStatus(Enum):
    IDLE = "idle"
    WORKING = "working"
    ERROR = "error"
    OFFLINE = "offline"

class UnifiedAgentService:
    """
    Single agent system replacing multiple agent implementations.
    Provides a unified interface for all agent operations.
    """

    def __init__(self):
        self.agents: Dict[str, Any] = {}
        self.agent_tasks: Dict[str, asyncio.Task] = {}
        self.agent_status: Dict[str, AgentStatus] = {}
        self._initialized = False

    async def initialize(self):
        """Initialize all agent types"""
        if self._initialized:
            return

        # Initialize each agent type with basic capabilities
        for agent_type in AgentType:
            self.agents[agent_type.value] = {
                "type": agent_type.value,
                "created_at": datetime.now().isoformat(),
                "capabilities": self._get_agent_capabilities(agent_type)
            }
            self.agent_status[agent_type.value] = AgentStatus.IDLE

        self._initialized = True
        logger.info(f"Initialized {len(self.agents)} agent types")

    def _get_agent_capabilities(self, agent_type: AgentType) -> List[str]:
        """Get capabilities for each agent type"""
        capabilities_map = {
            AgentType.CODING: ["code_generation", "code_review", "debugging", "refactoring"],
            AgentType.REASONING: ["analysis", "problem_solving", "decision_making", "planning"],
            AgentType.NARRATION: ["storytelling", "dialogue", "description", "summarization"],
            AgentType.AUTONOMOUS: ["self_directed", "goal_seeking", "continuous_learning"],
            AgentType.MOLTBOOK: ["social_interaction", "personality_simulation", "agent_networking"]
        }
        return capabilities_map.get(agent_type, ["general"])

    async def execute_agent_task(self, agent_type: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task with the specified agent"""
        if not self._initialized:
            await self.initialize()

        if agent_type not in self.agents:
            return {
                "status": "error",
                "message": f"Unknown agent type: {agent_type}"
            }

        # Set agent status to working
        self.agent_status[agent_type] = AgentStatus.WORKING

        try:
            # Simulate agent processing
            result = await self._process_task(agent_type, task)

            # Set agent status back to idle
            self.agent_status[agent_type] = AgentStatus.IDLE

            return {
                "status": "success",
                "agent_type": agent_type,
                "task_id": task.get("id", "unknown"),
                "result": result,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Agent {agent_type} task failed: {e}")
            self.agent_status[agent_type] = AgentStatus.ERROR

            return {
                "status": "error",
                "agent_type": agent_type,
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def _process_task(self, agent_type: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task based on agent type"""
        # This is where specific agent logic would go
        # For now, return a mock response

        if agent_type == "coding":
            return {
                "code": "def hello():\n    return 'Hello from coding agent'",
                "language": "python"
            }
        elif agent_type == "reasoning":
            return {
                "analysis": "Task analyzed successfully",
                "confidence": 0.95
            }
        elif agent_type == "narration":
            return {
                "narrative": "Once upon a time in Echo Brain...",
                "style": "descriptive"
            }
        elif agent_type == "autonomous":
            return {
                "action": "Completed autonomous task",
                "iterations": 3
            }
        elif agent_type == "moltbook":
            return {
                "interaction": "Agent network synchronized",
                "connections": 5
            }
        else:
            return {"message": f"Processed by {agent_type}"}

    def get_agent_status(self, agent_type: Optional[str] = None) -> Dict[str, Any]:
        """Get status of specific agent or all agents"""
        if not self._initialized:
            return {
                "status": "not_initialized",
                "message": "Agent service not initialized"
            }

        if agent_type:
            if agent_type in self.agents:
                return {
                    "agent_type": agent_type,
                    "status": self.agent_status.get(agent_type, AgentStatus.OFFLINE).value,
                    "info": self.agents[agent_type]
                }
            else:
                return {
                    "status": "error",
                    "message": f"Agent {agent_type} not found"
                }

        # Return all agent statuses
        return {
            "total_agents": len(self.agents),
            "active_tasks": len([s for s in self.agent_status.values() if s == AgentStatus.WORKING]),
            "agents": {
                agent: {
                    "status": self.agent_status.get(agent, AgentStatus.OFFLINE).value,
                    "capabilities": self.agents[agent]["capabilities"]
                }
                for agent in self.agents
            }
        }

    async def collaborate_agents(self, agent_types: List[str], task: Dict[str, Any]) -> Dict[str, Any]:
        """Enable collaboration between multiple agents"""
        if not self._initialized:
            await self.initialize()

        results = {}
        tasks = []

        # Create tasks for each agent
        for agent_type in agent_types:
            if agent_type in self.agents:
                task_coro = self.execute_agent_task(agent_type, task)
                tasks.append(task_coro)

        # Execute all agent tasks concurrently
        if tasks:
            agent_results = await asyncio.gather(*tasks)
            for i, agent_type in enumerate(agent_types):
                results[agent_type] = agent_results[i]

        return {
            "collaboration_id": f"collab_{datetime.now().timestamp()}",
            "participating_agents": agent_types,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }

# Global instance for easy import
agent_service = UnifiedAgentService()