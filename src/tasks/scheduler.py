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