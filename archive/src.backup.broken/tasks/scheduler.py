"""Behavior scheduler module for Echo Brain autonomous behaviors"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class BehaviorScheduler:
    """Schedule and manage autonomous behaviors"""

    def __init__(self, task_queue=None):
        self.scheduled_behaviors = {}
        self.running_behaviors = set()
        self.task_queue = task_queue

    async def schedule_behavior(self, behavior_name: str, interval_minutes: int = 60):
        """Schedule a behavior to run at regular intervals"""
        self.scheduled_behaviors[behavior_name] = {
            "interval": interval_minutes,
            "last_run": None,
            "next_run": datetime.now() + timedelta(minutes=interval_minutes)
        }

    async def should_run_behavior(self, behavior_name: str) -> bool:
        """Check if a behavior should run now"""
        if behavior_name not in self.scheduled_behaviors:
            return False

        schedule = self.scheduled_behaviors[behavior_name]
        return datetime.now() >= schedule["next_run"]

    async def mark_behavior_complete(self, behavior_name: str):
        """Mark a behavior as completed and schedule next run"""
        if behavior_name in self.scheduled_behaviors:
            schedule = self.scheduled_behaviors[behavior_name]
            schedule["last_run"] = datetime.now()
            schedule["next_run"] = datetime.now() + timedelta(minutes=schedule["interval"])

        if behavior_name in self.running_behaviors:
            self.running_behaviors.remove(behavior_name)

    async def get_scheduled_behaviors(self) -> Dict:
        """Get all scheduled behaviors"""
        return self.scheduled_behaviors

    async def setup_schedules(self):
        """Setup default behavior schedules"""
        await self.schedule_behavior("system_monitoring", 60)
        await self.schedule_behavior("service_health_check", 30)
        await self.schedule_behavior("code_quality_check", 1440)  # Daily

    async def process_schedules(self):
        """Process scheduled behaviors and create tasks as needed"""
        try:
            behaviors_to_run = []

            for behavior_name, schedule in self.scheduled_behaviors.items():
                if await self.should_run_behavior(behavior_name):
                    if behavior_name not in self.running_behaviors:
                        behaviors_to_run.append(behavior_name)
                        self.running_behaviors.add(behavior_name)

            # Create tasks for behaviors that should run
            for behavior_name in behaviors_to_run:
                if self.task_queue:
                    from src.tasks.task_queue import Task, TaskType, TaskPriority

                    # Map behavior names to task types
                    task_type_map = {
                        "system_monitoring": TaskType.MONITORING,
                        "service_health_check": TaskType.MONITORING,
                        "code_quality_check": TaskType.CODE_REFACTOR
                    }

                    task_type = task_type_map.get(behavior_name, TaskType.MAINTENANCE)
                    task = Task(
                        type=task_type,
                        priority=TaskPriority.NORMAL,
                        description=f"Scheduled behavior: {behavior_name}",
                        payload={"behavior": behavior_name, "scheduled": True}
                    )
                    await self.task_queue.add_task(task)

            return {
                "processed": len(self.scheduled_behaviors),
                "queued": len(behaviors_to_run),
                "running": len(self.running_behaviors)
            }

        except Exception as e:
            return {"error": f"Schedule processing failed: {str(e)}"}