"""
Autonomous Event Loop for Echo Brain
Continuously monitors, decides, and executes tasks autonomously
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

from .capability_registry import CapabilityRegistry, CapabilityType
from .code_executor import SandboxedCodeExecutor

logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    """Priority levels for autonomous tasks"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5

class TaskStatus(Enum):
    """Status of autonomous tasks"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class AutonomousTask:
    """Represents an autonomous task"""
    id: str
    name: str
    description: str
    priority: TaskPriority
    capability_required: str
    parameters: Dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    retries: int = 0
    max_retries: int = 3

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now()

class AutonomousEventLoop:
    """Main autonomous event loop for Echo Brain"""

    def __init__(self, capability_registry: CapabilityRegistry):
        self.capability_registry = capability_registry
        self.code_executor = SandboxedCodeExecutor()
        self.task_queue: List[AutonomousTask] = []
        self.running_tasks: Dict[str, AutonomousTask] = {}
        self.completed_tasks: List[AutonomousTask] = []
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.is_running = False
        self.loop_interval = 5  # seconds
        self.max_concurrent_tasks = 3

    async def start(self):
        """Start the autonomous event loop"""

        if self.is_running:
            logger.warning("Autonomous loop already running")
            return

        self.is_running = True
        logger.info("Starting autonomous event loop")

        # Start background tasks
        asyncio.create_task(self._main_loop())
        asyncio.create_task(self._monitor_loop())
        asyncio.create_task(self._cleanup_loop())

    async def stop(self):
        """Stop the autonomous event loop"""

        logger.info("Stopping autonomous event loop")
        self.is_running = False

        # Cancel running tasks
        for task in self.running_tasks.values():
            task.status = TaskStatus.CANCELLED

    async def _main_loop(self):
        """Main processing loop"""

        while self.is_running:
            try:
                # Process pending tasks
                await self._process_pending_tasks()

                # Check for new events
                await self._check_events()

                # Self-improvement check
                await self._self_improvement_check()

                await asyncio.sleep(self.loop_interval)

            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(self.loop_interval)

    async def _process_pending_tasks(self):
        """Process tasks in the queue"""

        # Sort tasks by priority
        self.task_queue.sort(key=lambda x: (x.priority.value, x.created_at))

        # Process tasks up to max concurrent
        while (len(self.running_tasks) < self.max_concurrent_tasks and self.task_queue):
            task = self.task_queue.pop(0)
            asyncio.create_task(self._execute_task(task))

    async def _execute_task(self, task: AutonomousTask):
        """Execute a single autonomous task"""

        logger.info(f"Executing task: {task.name}")
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        self.running_tasks[task.id] = task

        try:
            # Execute via capability registry
            result = await self.capability_registry.execute_capability(
                task.capability_required,
                **task.parameters
            )

            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()

            logger.info(f"Task completed: {task.name}")

            # Trigger completion events
            await self._trigger_event("task_completed", task)

        except Exception as e:
            logger.error(f"Task failed: {task.name} - {e}")
            task.status = TaskStatus.FAILED
            task.result = {"error": str(e)}

            # Retry if possible
            if task.retries < task.max_retries:
                task.retries += 1
                task.status = TaskStatus.PENDING
                self.task_queue.append(task)
                logger.info(f"Retrying task: {task.name} (attempt {task.retries})")
            else:
                await self._trigger_event("task_failed", task)

        finally:
            del self.running_tasks[task.id]
            self.completed_tasks.append(task)

    async def _monitor_loop(self):
        """Monitor system state and generate tasks"""

        while self.is_running:
            try:
                # Monitor system resources
                await self._monitor_resources()

                # Monitor service health
                await self._monitor_services()

                # Monitor for anomalies
                await self._detect_anomalies()

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                await asyncio.sleep(30)

    async def _monitor_resources(self):
        """Monitor system resources and create tasks if needed"""

        import psutil

        # Check CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 80:
            await self.add_task(
                name="Optimize CPU usage",
                description="High CPU detected, optimizing processes",
                priority=TaskPriority.HIGH,
                capability="analysis",
                parameters={"metric": "cpu", "threshold": 80}
            )

        # Check memory usage
        memory = psutil.virtual_memory()
        if memory.percent > 85:
            await self.add_task(
                name="Free memory",
                description="High memory usage detected",
                priority=TaskPriority.HIGH,
                capability="code_execution",
                parameters={
                    "code": "import gc; gc.collect()",
                    "language": "python"
                }
            )

        # Check disk space
        disk = psutil.disk_usage('/')
        if disk.percent > 90:
            await self.add_task(
                name="Clean disk space",
                description="Low disk space detected",
                priority=TaskPriority.CRITICAL,
                capability="file_system",
                parameters={"action": "cleanup", "threshold": 90}
            )

    async def _monitor_services(self):
        """Monitor Tower services and create tasks if needed"""

        services = [
            ("tower-echo-brain", 8309),
            ("tower-anime-production", 8305),
            ("tower-dashboard", 8080),
            ("comfyui", 8188)
        ]

        for service_name, port in services:
            # Check if service is responding
            import aiohttp
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"http://localhost:{port}/health", timeout=5) as response:
                        if response.status != 200:
                            await self.add_task(
                                name=f"Restart {service_name}",
                                description=f"Service {service_name} not healthy",
                                priority=TaskPriority.HIGH,
                                capability="code_execution",
                                parameters={
                                    "code": f"sudo systemctl restart {service_name}",
                                    "language": "bash"
                                }
                            )
            except:
                pass  # Service might not have health endpoint

    async def _detect_anomalies(self):
        """Detect anomalies in system behavior"""

        # Check for unusual patterns in logs
        log_path = Path("/var/log/tower-echo-brain.log")
        if log_path.exists():
            # Read last 100 lines
            lines = log_path.read_text().splitlines()[-100:]
            error_count = sum(1 for line in lines if "ERROR" in line)

            if error_count > 10:
                await self.add_task(
                    name="Analyze error patterns",
                    description="High error rate detected in logs",
                    priority=TaskPriority.NORMAL,
                    capability="analysis",
                    parameters={"log_file": str(log_path), "pattern": "ERROR"}
                )

    async def _self_improvement_check(self):
        """Check for self-improvement opportunities"""

        # Analyze capability performance
        stats = self.capability_registry.get_statistics()

        # Find underperforming capabilities
        for capability_name, capability in self.capability_registry.capabilities.items():
            if capability.usage_count > 10 and capability.success_rate < 0.5:
                await self.add_task(
                    name=f"Improve {capability_name}",
                    description=f"Low success rate for {capability_name}",
                    priority=TaskPriority.LOW,
                    capability="self_modification",
                    parameters={
                        "target": capability_name,
                        "metric": "success_rate",
                        "current_value": capability.success_rate
                    }
                )

    async def _cleanup_loop(self):
        """Clean up old tasks and data"""

        while self.is_running:
            try:
                # Remove old completed tasks (keep last 100)
                if len(self.completed_tasks) > 100:
                    self.completed_tasks = self.completed_tasks[-100:]

                # Clean up Docker images
                self.code_executor.cleanup_old_images()

                await asyncio.sleep(3600)  # Cleanup every hour

            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(3600)

    async def _check_events(self):
        """Check for external events and triggers"""

        # Check for user requests
        request_file = Path("/tmp/echo_brain_requests.json")
        if request_file.exists():
            try:
                requests = json.loads(request_file.read_text())
                for req in requests:
                    await self.add_task(
                        name=req.get("name", "User request"),
                        description=req.get("description", ""),
                        priority=TaskPriority(req.get("priority", 3)),
                        capability=req.get("capability", "analysis"),
                        parameters=req.get("parameters", {})
                    )
                request_file.unlink()  # Remove after processing
            except Exception as e:
                logger.error(f"Error processing requests: {e}")

    async def add_task(
        self,
        name: str,
        description: str,
        priority: TaskPriority,
        capability: str,
        parameters: Dict[str, Any]
    ) -> str:
        """Add a new autonomous task"""

        import uuid
        task_id = str(uuid.uuid4())[:8]

        task = AutonomousTask(
            id=task_id,
            name=name,
            description=description,
            priority=priority,
            capability_required=capability,
            parameters=parameters
        )

        self.task_queue.append(task)
        logger.info(f"Added task: {name} (priority: {priority.name})")

        # Trigger event
        await self._trigger_event("task_added", task)

        return task_id

    def register_event_handler(self, event: str, handler: Callable):
        """Register an event handler"""

        if event not in self.event_handlers:
            self.event_handlers[event] = []
        self.event_handlers[event].append(handler)

    async def _trigger_event(self, event: str, data: Any):
        """Trigger an event"""

        if event in self.event_handlers:
            for handler in self.event_handlers[event]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(data)
                    else:
                        handler(data)
                except Exception as e:
                    logger.error(f"Error in event handler for {event}: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get current status of autonomous loop"""

        return {
            "is_running": self.is_running,
            "pending_tasks": len(self.task_queue),
            "running_tasks": len(self.running_tasks),
            "completed_tasks": len(self.completed_tasks),
            "tasks": {
                "pending": [
                    {"name": t.name, "priority": t.priority.name}
                    for t in self.task_queue[:5]
                ],
                "running": [
                    {"name": t.name, "started": t.started_at.isoformat()}
                    for t in self.running_tasks.values()
                ],
                "recent_completed": [
                    {
                        "name": t.name,
                        "status": t.status.value,
                        "duration": (t.completed_at - t.started_at).total_seconds() if t.completed_at and t.started_at else None
                    }
                    for t in self.completed_tasks[-5:]
                ]
            },
            "capability_stats": self.capability_registry.get_statistics()
        }