#!/usr/bin/env python3
"""
Task Manager for Echo Brain - Prevents memory leaks in background tasks
Manages asyncio tasks with proper cleanup
"""

import asyncio
import logging
from typing import Set, Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class TaskManager:
    """Manages background tasks with proper cleanup to prevent memory leaks"""

    def __init__(self):
        self.active_tasks: Set[asyncio.Task] = set()
        self.task_metadata: Dict[asyncio.Task, Dict[str, Any]] = {}
        self._shutdown_event = asyncio.Event()

    def create_task(self, coro, name: str = None, description: str = None) -> asyncio.Task:
        """Create a managed task that will be properly cleaned up"""
        task = asyncio.create_task(coro, name=name)

        # Store task reference for cleanup
        self.active_tasks.add(task)

        # Store metadata for monitoring
        self.task_metadata[task] = {
            "name": name or "unnamed_task",
            "description": description or "No description",
            "created_at": datetime.now(),
            "status": "running"
        }

        # Add cleanup callback
        task.add_done_callback(self._task_done_callback)

        logger.info(f"Created managed task: {name or 'unnamed'}")
        return task

    def _task_done_callback(self, task: asyncio.Task):
        """Callback when task completes to clean up references"""
        self.active_tasks.discard(task)

        # Update metadata
        if task in self.task_metadata:
            self.task_metadata[task]["status"] = "completed"
            if task.exception():
                self.task_metadata[task]["error"] = str(task.exception())
                logger.error(f"Task {self.task_metadata[task]['name']} failed: {task.exception()}")

        # Clean up metadata after some time (keep recent history)
        if len(self.task_metadata) > 100:
            # Remove oldest completed tasks
            completed_tasks = [t for t, meta in self.task_metadata.items()
                             if meta["status"] == "completed"]
            if completed_tasks:
                oldest_task = min(completed_tasks,
                                key=lambda t: self.task_metadata[t]["created_at"])
                del self.task_metadata[oldest_task]

    async def cancel_all_tasks(self, timeout: float = 10.0) -> int:
        """Cancel all active tasks gracefully"""
        if not self.active_tasks:
            return 0

        logger.info(f"Cancelling {len(self.active_tasks)} background tasks...")

        # Cancel all tasks
        for task in self.active_tasks.copy():
            if not task.done():
                task.cancel()

        # Wait for tasks to complete or timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*self.active_tasks, return_exceptions=True),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning("Some tasks did not complete within timeout")

        cancelled_count = len(self.active_tasks)
        self.active_tasks.clear()

        logger.info(f"Cancelled {cancelled_count} background tasks")
        return cancelled_count

    def get_task_stats(self) -> Dict[str, Any]:
        """Get statistics about managed tasks"""
        active_count = len(self.active_tasks)
        total_count = len(self.task_metadata)

        task_info = []
        for task, metadata in self.task_metadata.items():
            duration = (datetime.now() - metadata["created_at"]).total_seconds()
            task_info.append({
                "name": metadata["name"],
                "description": metadata["description"],
                "status": metadata["status"],
                "duration": duration,
                "is_active": task in self.active_tasks
            })

        return {
            "active_tasks": active_count,
            "total_tasks": total_count,
            "task_details": task_info[-20:],  # Last 20 tasks
            "timestamp": datetime.now().isoformat()
        }

    async def health_check(self) -> Dict[str, Any]:
        """Check health of managed tasks"""
        stuck_tasks = []
        healthy_tasks = []

        for task in self.active_tasks:
            metadata = self.task_metadata.get(task, {})
            name = metadata.get("name", "unknown")
            created_at = metadata.get("created_at")

            if created_at:
                duration = (datetime.now() - created_at).total_seconds()

                # Consider tasks stuck if running for more than 1 hour
                if duration > 3600:
                    stuck_tasks.append({
                        "name": name,
                        "duration": duration,
                        "status": "possibly_stuck"
                    })
                else:
                    healthy_tasks.append({
                        "name": name,
                        "duration": duration,
                        "status": "healthy"
                    })

        return {
            "total_active": len(self.active_tasks),
            "healthy_tasks": len(healthy_tasks),
            "stuck_tasks": len(stuck_tasks),
            "stuck_task_details": stuck_tasks,
            "health_status": "healthy" if not stuck_tasks else "warning"
        }

    async def restart_stuck_tasks(self) -> int:
        """Restart tasks that appear to be stuck"""
        health = await self.health_check()
        stuck_tasks = [task for task in self.active_tasks
                      if self.task_metadata.get(task, {}).get("name") in
                      [t["name"] for t in health["stuck_task_details"]]]

        restarted_count = 0
        for task in stuck_tasks:
            if not task.done():
                task.cancel()
                restarted_count += 1
                logger.warning(f"Cancelled stuck task: {self.task_metadata.get(task, {}).get('name')}")

        return restarted_count

# Global task manager instance
task_manager = TaskManager()