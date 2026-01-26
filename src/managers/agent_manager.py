#!/usr/bin/env python3
"""
Echo Brain Internal Agent Manager
Manages Echo Brain's core AI agents (coding, reasoning, narration)
Separate from Tower Agent Manager which handles toolbelt/workflow agents
"""

import logging
import httpx
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class AgentManager:
    """
    Manages Echo Brain's internal AI agents
    These are tightly integrated with Echo Brain's memory and conversation system
    """

    def __init__(self):
        """Initialize internal agent manager"""
        self.internal_agents = {}
        self.agent_status = {}
        self.toolbelt_service_url = "http://localhost:8301"  # Optional integration
        self._initialize_internal_agents()

    def _initialize_internal_agents(self):
        """Initialize Echo Brain's core internal agents"""
        try:
            # Import internal agents (these have deep Echo Brain integration)
            from ..agents.coding_agent import coding_agent
            from ..agents.reasoning_agent import reasoning_agent
            from ..agents.narration_agent import narration_agent

            # Register internal agents
            self.internal_agents = {
                'coding': coding_agent,
                'reasoning': reasoning_agent,
                'narration': narration_agent
            }

            # Initialize status tracking
            for name in self.internal_agents:
                self.agent_status[name] = {
                    'type': 'internal',
                    'status': 'active',
                    'last_used': None,
                    'total_requests': 0,
                    'description': f'Echo Brain internal {name} agent'
                }

            logger.info(f"✅ Internal Agent Manager initialized with {len(self.internal_agents)} agents")

        except Exception as e:
            logger.error(f"Failed to initialize internal agents: {e}")

    async def execute_task(self, task: str, agent_type: str = "general", context: Optional[Dict] = None) -> Dict:
        """
        Execute a task using the appropriate agent
        This is the main entry point for the API
        """
        try:
            # Map general request to appropriate internal agent
            if agent_type == "general" or agent_type not in self.internal_agents:
                # Analyze task to determine best agent
                if any(word in task.lower() for word in ["code", "function", "class", "debug", "fix"]):
                    agent_type = "coding"
                elif any(word in task.lower() for word in ["analyze", "explain", "why", "reason"]):
                    agent_type = "reasoning"
                elif any(word in task.lower() for word in ["story", "narrate", "describe"]):
                    agent_type = "narration"
                else:
                    agent_type = "reasoning"  # Default to reasoning

            # Get the agent
            if agent_type in self.internal_agents:
                agent = self.internal_agents[agent_type]

                # Update status
                self.agent_status[agent_type]['last_used'] = datetime.now()
                self.agent_status[agent_type]['total_requests'] += 1

                # Execute with the agent
                if hasattr(agent, 'process'):
                    result = await agent.process(task)
                    return {
                        "output": result.get("response", result),
                        "agent": agent_type,
                        "task_id": str(datetime.now().timestamp()),
                        "status": "completed"
                    }
                else:
                    return {
                        "output": f"Agent {agent_type} does not have a process method",
                        "agent": agent_type,
                        "task_id": str(datetime.now().timestamp()),
                        "status": "error"
                    }
            else:
                # Try external toolbelt agent
                result = await self.execute_toolbelt_agent(agent_type, task)
                if result:
                    return {
                        "output": result.get("result", "No output"),
                        "agent": f"toolbelt_{agent_type}",
                        "task_id": str(datetime.now().timestamp()),
                        "status": "completed"
                    }
                else:
                    return {
                        "output": f"No agent available for type: {agent_type}",
                        "agent": "none",
                        "task_id": str(datetime.now().timestamp()),
                        "status": "error"
                    }
        except Exception as e:
            logger.error(f"Error executing task: {e}")
            return {
                "output": str(e),
                "agent": agent_type,
                "task_id": str(datetime.now().timestamp()),
                "status": "error"
            }

    def get_internal_agents(self) -> List[str]:
        """Get list of internal agent names"""
        return list(self.internal_agents.keys())

    async def process_with_internal_agent(self, agent_name: str, task: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Process task with internal agent (has access to Echo Brain memory/context)"""
        if agent_name not in self.internal_agents:
            raise ValueError(f"Internal agent '{agent_name}' not found")

        agent = self.internal_agents[agent_name]

        # Update usage tracking
        self.agent_status[agent_name]['last_used'] = datetime.now()
        self.agent_status[agent_name]['total_requests'] += 1

        try:
            result = await agent.process(task, context or {})
            logger.info(f"Internal agent '{agent_name}' completed task successfully")
            return result
        except Exception as e:
            logger.error(f"Internal agent '{agent_name}' failed: {e}")
            raise

    async def get_toolbelt_agents(self) -> Optional[List[str]]:
        """Get list of available toolbelt agents from Tower Agent Manager"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.toolbelt_service_url}/api/agents")
                if response.status_code == 200:
                    data = response.json()
                    return data.get('agents', [])
        except Exception as e:
            logger.debug(f"Could not connect to toolbelt service: {e}")
        return None

    async def execute_toolbelt_agent(self, agent_type: str, message: str) -> Optional[Dict]:
        """Execute a toolbelt agent via Tower Agent Manager"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.toolbelt_service_url}/api/agents/execute",
                    json={"agent_type": agent_type, "message": message}
                )
                if response.status_code == 200:
                    return response.json()
        except Exception as e:
            logger.error(f"Toolbelt agent execution failed: {e}")
        return None

    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of internal agents"""
        return self.agent_status.copy()

    async def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get status of both internal and toolbelt agents"""
        status = {
            'internal_agents': self.agent_status.copy(),
            'toolbelt_agents': None
        }

        # Try to get toolbelt status
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.toolbelt_service_url}/api/agents/status")
                if response.status_code == 200:
                    status['toolbelt_agents'] = response.json()
        except Exception as e:
            logger.debug(f"Could not get toolbelt status: {e}")

        return status

    def health_check(self) -> Dict[str, Any]:
        """Health check for internal agents"""
        health = {
            'status': 'healthy',
            'internal_agents': len(self.internal_agents),
            'agent_details': {}
        }

        for name in self.internal_agents:
            health['agent_details'][name] = {
                'status': self.agent_status[name]['status'],
                'total_requests': self.agent_status[name]['total_requests']
            }

        return health