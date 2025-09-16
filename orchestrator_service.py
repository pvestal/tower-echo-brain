#!/usr/bin/env python3
"""
Tower Orchestrator Service - Production Version
Handles agent delegation and task orchestration for Echo Brain
"""

import asyncio
import json
import logging
import sys
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import aiohttp

# Add the source directory to path for imports
sys.path.insert(0, '/home/patrick/Documents/Tower/services/ai/echo-brain-unified/orchestrator')

try:
    from orchestrator import Orchestrator, Task, TaskType, AgentCapability
except ImportError as e:
    print(f"Failed to import orchestrator: {e}")
    print("Creating minimal orchestrator class...")
    
    class TaskType(Enum):
        DESIGN = "design"
        SECURITY = "security"
        DEVELOPMENT = "development"
        REVIEW = "review"
        DEPLOYMENT = "deployment"
        MONITORING = "monitoring"
        ANALYSIS = "analysis"

    @dataclass
    class Task:
        task_id: str
        type: TaskType
        description: str
        requirements: Dict[str, Any]
        priority: int  # 1-10, higher is more urgent
        assigned_agent: Optional[str] = None
        status: str = "pending"  # pending, assigned, in_progress, completed, failed
        result: Optional[Any] = None
        created_at: datetime = field(default_factory=datetime.now)
        completed_at: Optional[datetime] = None

    @dataclass
    class AgentCapability:
        agent_id: str
        agent_type: str
        capabilities: List[str]
        trust_level: float
        current_load: int
        max_load: int
        websocket_url: Optional[str] = None
        api_url: Optional[str] = None
        last_heartbeat: datetime = field(default_factory=datetime.now)
        status: str = "online"

    # Minimal orchestrator implementation
    class Orchestrator:
        def __init__(self, port: int = 8400):
            self.port = port
            self.agents: Dict[str, AgentCapability] = {}
            self.task_queue: List[Task] = []
            self.active_tasks: Dict[str, Task] = {}
            self.completed_tasks: List[Task] = []
            self.running = False
            
        async def start(self):
            self.running = True
            await self._register_default_agents()
            asyncio.create_task(self.task_processing_loop())
            asyncio.create_task(self.start_api_server())
            
        async def _register_default_agents(self):
            # VisionGuardian
            self.register_agent(AgentCapability(
                agent_id="vision_guardian_001",
                agent_type="guardian",
                capabilities=["vision", "alignment", "governance"],
                trust_level=0.95,
                current_load=0,
                max_load=10,
                status="online"
            ))
            
            # ArchitectAgent
            self.register_agent(AgentCapability(
                agent_id="architect_001",
                agent_type="architect",
                capabilities=["architecture", "design", "planning"],
                trust_level=0.8,
                current_load=0,
                max_load=5,
                status="online"
            ))
            
            # SecurityAgent
            self.register_agent(AgentCapability(
                agent_id="security_001",
                agent_type="security",
                capabilities=["security", "audit", "vulnerability"],
                trust_level=0.9,
                current_load=0,
                max_load=5,
                status="online"
            ))
            
            # Weights and Balances AI
            self.register_agent(AgentCapability(
                agent_id="weights_balances_001",
                agent_type="governance",
                capabilities=["oversight", "democracy", "balance"],
                trust_level=0.95,
                current_load=0,
                max_load=10,
                status="online"
            ))
            
            print(f"Registered {len(self.agents)} default agents")
        
        def register_agent(self, agent: AgentCapability):
            self.agents[agent.agent_id] = agent
            print(f"Registered agent: {agent.agent_id} with capabilities: {agent.capabilities}")
        
        async def submit_task(self, task: Task) -> str:
            self.task_queue.append(task)
            self.task_queue.sort(key=lambda t: t.priority, reverse=True)
            print(f"Task {task.task_id} submitted: {task.description[:50]}...")
            asyncio.create_task(self.process_task(task))
            return task.task_id
        
        async def task_processing_loop(self):
            while self.running:
                try:
                    if self.task_queue:
                        for t in self.task_queue:
                            if t.status == "pending":
                                await self.process_task(t)
                                break
                    await asyncio.sleep(1)
                except Exception as e:
                    print(f"Error in task processing loop: {e}")
                    await asyncio.sleep(5)
        
        async def process_task(self, task: Task):
            try:
                agent = self.find_suitable_agent(task)
                if not agent:
                    print(f"No suitable agent for task {task.task_id}")
                    task.status = "failed"
                    return
                
                task.assigned_agent = agent.agent_id
                task.status = "assigned"
                self.active_tasks[task.task_id] = task
                agent.current_load += 1
                
                # Simulate execution
                await asyncio.sleep(2)
                result = {
                    "status": "completed",
                    "agent": agent.agent_id,
                    "task_type": task.type.value,
                    "simulated": True,
                    "result": f"Simulated execution of {task.description[:50]}...",
                    "timestamp": datetime.now().isoformat()
                }
                
                task.result = result
                task.status = "completed"
                task.completed_at = datetime.now()
                
                del self.active_tasks[task.task_id]
                self.completed_tasks.append(task)
                agent.current_load -= 1
                
                print(f"Task {task.task_id} completed")
                
            except Exception as e:
                print(f"Error processing task {task.task_id}: {e}")
                task.status = "failed"
        
        def find_suitable_agent(self, task: Task) -> Optional[AgentCapability]:
            routing_rules = {
                TaskType.DESIGN: ["architecture", "design", "planning"],
                TaskType.SECURITY: ["security", "audit", "vulnerability"],
                TaskType.DEVELOPMENT: ["development", "coding", "implementation"],
                TaskType.REVIEW: ["review", "quality", "testing"],
                TaskType.DEPLOYMENT: ["deployment", "devops", "infrastructure"],
                TaskType.MONITORING: ["monitoring", "observability", "metrics"],
                TaskType.ANALYSIS: ["analysis", "data", "ml"]
            }
            
            required_capabilities = routing_rules.get(task.type, [])
            suitable_agents = []
            
            for agent in self.agents.values():
                if (agent.status == "online" and 
                    agent.current_load < agent.max_load and
                    any(cap in agent.capabilities for cap in required_capabilities)):
                    suitable_agents.append(agent)
            
            if not suitable_agents:
                return None
            
            suitable_agents.sort(key=lambda a: (a.trust_level, -a.current_load), reverse=True)
            return suitable_agents[0]
        
        async def start_api_server(self):
            from aiohttp import web
            
            app = web.Application()
            app.router.add_get('/health', self.handle_health)
            app.router.add_post('/submit_task', self.handle_submit_task)
            app.router.add_get('/status', self.handle_status)
            app.router.add_get('/agents', self.handle_list_agents)
            app.router.add_get('/tasks', self.handle_list_tasks)
            
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, 'localhost', self.port)
            await site.start()
            
            print(f"Orchestrator API server started on port {self.port}")
        
        async def handle_health(self, request):
            from aiohttp import web
            active_agents = len([a for a in self.agents.values() if a.status == "online"])
            return web.json_response({
                'status': 'healthy' if self.running else 'starting',
                'agents_online': active_agents,
                'agents_total': len(self.agents),
                'tasks_pending': len(self.task_queue),
                'tasks_active': len(self.active_tasks),
                'port': self.port
            })
        
        async def handle_submit_task(self, request):
            from aiohttp import web
            data = await request.json()
            
            task = Task(
                task_id=f"task_{datetime.now().timestamp()}",
                type=TaskType(data.get("type", "analysis")),
                description=data.get("description", ""),
                requirements=data.get("requirements", {}),
                priority=data.get("priority", 5)
            )
            
            task_id = await self.submit_task(task)
            return web.json_response({"task_id": task_id, "status": "submitted"})
        
        async def handle_status(self, request):
            from aiohttp import web
            return web.json_response({
                "status": "running",
                "agents": len(self.agents),
                "active_tasks": len(self.active_tasks),
                "queued_tasks": len([t for t in self.task_queue if t.status == "pending"]),
                "completed_tasks": len(self.completed_tasks)
            })
        
        async def handle_list_agents(self, request):
            from aiohttp import web
            agents_data = []
            for agent in self.agents.values():
                agents_data.append({
                    "agent_id": agent.agent_id,
                    "type": agent.agent_type,
                    "capabilities": agent.capabilities,
                    "trust_level": agent.trust_level,
                    "status": agent.status,
                    "current_load": agent.current_load,
                    "max_load": agent.max_load
                })
            return web.json_response(agents_data)
        
        async def handle_list_tasks(self, request):
            from aiohttp import web
            tasks_data = []
            
            # Active tasks
            for task in self.active_tasks.values():
                tasks_data.append({
                    "task_id": task.task_id,
                    "type": task.type.value,
                    "status": task.status,
                    "assigned_agent": task.assigned_agent,
                    "priority": task.priority
                })
            
            # Queued tasks
            for task in self.task_queue:
                if task.status == "pending":
                    tasks_data.append({
                        "task_id": task.task_id,
                        "type": task.type.value,
                        "status": task.status,
                        "priority": task.priority
                    })
            
            return web.json_response(tasks_data)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/tower-echo-brain/logs/orchestrator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def main():
    """Main function to run Tower Orchestrator Service"""
    logger.info("🚀 Starting Tower Orchestrator Service")
    
    orchestrator = Orchestrator(port=8400)
    
    try:
        await orchestrator.start()
        logger.info("✅ Tower Orchestrator Service started on port 8400")
        
        # Keep running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down Tower Orchestrator Service")
        await orchestrator.stop()
    except Exception as e:
        logger.error(f"❌ Orchestrator service error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())