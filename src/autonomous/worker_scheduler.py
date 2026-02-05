#!/usr/bin/env python3
"""
Simple Worker Scheduler for Echo Brain Workers
Runs the three phase 2 workers on their schedules.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional
import os

logger = logging.getLogger(__name__)

class WorkerScheduler:
    """Manages and runs autonomous workers on schedule."""

    def __init__(self):
        self._workers: Dict[str, dict] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None

    def register_worker(self, name: str, worker_callable, interval_minutes: int):
        """Register a worker to be run on schedule."""
        self._workers[name] = {
            "callable": worker_callable,
            "interval": timedelta(minutes=interval_minutes),
            "last_run": None,
            "running": False,
            "error_count": 0
        }
        logger.info(f"Registered worker: {name} (every {interval_minutes}min)")

    async def start(self):
        """Start the scheduler loop."""
        if self._running:
            logger.warning("Scheduler already running")
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(f"Worker scheduler started with {len(self._workers)} workers")

    async def stop(self):
        """Stop the scheduler gracefully."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Worker scheduler stopped")

    async def _run_loop(self):
        """Main scheduler loop - checks for due workers every 60 seconds."""
        while self._running:
            try:
                now = datetime.utcnow()
                for name, worker in self._workers.items():
                    if worker["running"]:
                        continue
                    if worker["last_run"] is None or (now - worker["last_run"]) >= worker["interval"]:
                        asyncio.create_task(self._execute_worker(name, worker))
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
            await asyncio.sleep(60)  # Check every minute

    async def _execute_worker(self, name: str, worker: dict):
        """Execute a single worker with error handling."""
        worker["running"] = True
        try:
            logger.info(f"Starting worker: {name}")
            # Call the worker
            result = await worker["callable"]()
            worker["error_count"] = 0
            logger.info(f"Worker {name} completed successfully")
        except Exception as e:
            worker["error_count"] += 1
            logger.error(f"Worker {name} failed (attempt {worker['error_count']}): {e}")
            # Back off after repeated failures
            if worker["error_count"] >= 3:
                logger.warning(f"Worker {name} has failed 3 times, will retry later")
        finally:
            worker["running"] = False
            worker["last_run"] = datetime.utcnow()

    def get_status(self) -> dict:
        """Return current status of all workers."""
        return {
            "running": self._running,
            "workers": {
                name: {
                    "interval_minutes": int(w["interval"].total_seconds() / 60),
                    "last_run": w["last_run"].isoformat() if w["last_run"] else None,
                    "currently_running": w["running"],
                    "error_count": w["error_count"]
                }
                for name, w in self._workers.items()
            }
        }

# Global instance
worker_scheduler = WorkerScheduler()