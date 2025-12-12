#!/usr/bin/env python3
"""
Echo Brain Autonomous Behaviors - Refactored Modular Architecture
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any
from src.tasks.task_queue import TaskQueue

# Import behavior modules
from .service_monitor import ServiceMonitor
from src.behaviors.system_monitor import SystemMonitor
from .code_quality_monitor import CodeQualityMonitor
from .code_refactor_executor import CodeRefactorExecutor
from .multi_language_linter import MultiLanguageLinter
from .scheduler import BehaviorScheduler

logger = logging.getLogger(__name__)

class AutonomousBehaviors:
    """Refactored autonomous behaviors with modular architecture"""

    def __init__(self, task_queue: TaskQueue):
        self.task_queue = task_queue
        self.running = False

        # Initialize behavior modules
        self.service_monitor = ServiceMonitor(task_queue)
        self.system_monitor = SystemMonitor(task_queue)
        self.code_quality_monitor = CodeQualityMonitor(task_queue)
        self.code_refactor_executor = CodeRefactorExecutor()
        self.multi_language_linter = MultiLanguageLinter()
        self.scheduler = BehaviorScheduler(task_queue)

        # Behavior loop intervals (seconds)
        self.intervals = {
            'service_monitoring': 60,      # Check services every minute
            'system_monitoring': 300,      # Check system every 5 minutes
            'code_quality': 86400,         # Check code quality daily
            'scheduled_tasks': 60          # Process schedules every minute
        }

    async def start(self):
        """Start all autonomous behavior loops"""
        if self.running:
            logger.warning("Autonomous behaviors already running")
            return

        self.running = True
        logger.info("🤖 Starting autonomous behaviors with modular architecture...")

        try:
            # Setup scheduled tasks
            await self.scheduler.setup_schedules()

            # Start behavior loops (non-blocking)
            asyncio.create_task(self._service_monitoring_loop())
            asyncio.create_task(self._system_monitoring_loop())
            # TEMPORARILY DISABLED - blocking startup with pylint timeouts
            # asyncio.create_task(self._code_quality_loop())
            asyncio.create_task(self._scheduled_task_loop())

        except Exception as e:
            logger.error(f"❌ Autonomous behaviors startup failed: {e}")
            self.running = False

    async def stop(self):
        """Stop all autonomous behaviors"""
        logger.info("🛑 Stopping autonomous behaviors...")
        self.running = False

    async def _service_monitoring_loop(self):
        """Service monitoring behavior loop"""
        while self.running:
            try:
                await self.service_monitor.monitor_services()
                await asyncio.sleep(self.intervals['service_monitoring'])
            except Exception as e:
                logger.error(f"Service monitoring loop error: {e}")
                await asyncio.sleep(60)  # Error recovery delay

    async def _system_monitoring_loop(self):
        """System resource monitoring behavior loop"""
        while self.running:
            try:
                await self.system_monitor.monitor_resources()
                await self.system_monitor.monitor_processes()
                await asyncio.sleep(self.intervals['system_monitoring'])
            except Exception as e:
                logger.error(f"System monitoring loop error: {e}")
                await asyncio.sleep(300)  # Error recovery delay

    async def _code_quality_loop(self):
        """Code quality monitoring and refactoring behavior loop - Multi-language support"""
        while self.running:
            try:
                # Tower projects to analyze (including frontend code)
                tower_projects = [
                    "/opt/tower-echo-brain",
                    "/opt/tower-anime-production",
                    "/opt/tower-auth",
                    "/opt/tower-kb",
                    "/opt/tower-apple-music",
                    "/opt/tower-dashboard",  # HTML/CSS/JS
                    "/opt/tower-crypto-trader",
                    "/opt/tower-personal-media"
                ]

                for project in tower_projects:
                    try:
                        # Multi-language analysis (Python, JS, TS, HTML, CSS, SQL, etc.)
                        analysis = await self.multi_language_linter.analyze_project(project)

                        logger.info(f"📊 {project}: Score {analysis['average_score']:.1f}/10, "
                                  f"{analysis['total_issues']} issues in {analysis['total_files']} files")

                        # Auto-fix files with low scores
                        for filepath in analysis['fixable_files'][:10]:  # Limit to 10 files per run
                            if await self.multi_language_linter.fix_file(filepath):
                                logger.info(f"✅ Auto-fixed: {filepath}")

                        # Also run Python-specific deep analysis
                        if any('.py' in str(f) for f in analysis.get('languages', {}).get('python', {}).get('files', [])):
                            await self.code_quality_monitor.analyze_code_quality(project)

                    except Exception as e:
                        logger.warning(f"Failed to analyze {project}: {e}")

                await asyncio.sleep(self.intervals['code_quality'])
            except Exception as e:
                logger.error(f"Code quality loop error: {e}")
                await asyncio.sleep(3600)  # Error recovery delay

    async def _scheduled_task_loop(self):
        """Scheduled task processing loop"""
        while self.running:
            try:
                await self.scheduler.process_schedules()
                await asyncio.sleep(self.intervals['scheduled_tasks'])
            except Exception as e:
                logger.error(f"Scheduled task loop error: {e}")
                await asyncio.sleep(60)  # Error recovery delay

    def get_behavior_stats(self) -> Dict[str, Any]:
        """Get comprehensive behavior statistics"""
        return {
            'running': self.running,
            'intervals': self.intervals,
            'service_monitor': self.service_monitor.get_monitor_stats(),
            'system_monitor': self.system_monitor.get_resource_stats(),
            'code_quality': self.code_quality_monitor.get_quality_stats(),
            'code_refactor_tools': self.code_refactor_executor.tools_available,
            'scheduler': self.scheduler.get_schedule_info(),
            'timestamp': datetime.now().isoformat()
        }

    async def trigger_emergency_behavior(self, issue_type: str, details: Dict[str, Any]):
        """Trigger emergency behavior for critical issues"""
        try:
            logger.warning(f"🚨 Emergency behavior triggered: {issue_type}")

            if issue_type == "service_failure":
                await self.service_monitor.monitor_services()
            elif issue_type == "resource_critical":
                await self.system_monitor.monitor_resources()
            elif issue_type == "code_critical":
                await self.code_quality_monitor.analyze_code_quality("/opt/tower-echo-brain/src")
                # Also trigger immediate refactoring if needed
                await self.code_quality_monitor.trigger_emergency_refactoring(details.get('file_path'))

        except Exception as e:
            logger.error(f"Emergency behavior failed: {e}")

    def update_interval(self, behavior: str, new_interval: int):
        """Update behavior loop interval"""
        if behavior in self.intervals:
            self.intervals[behavior] = new_interval
            logger.info(f"Updated {behavior} interval to {new_interval} seconds")
        else:
            logger.warning(f"Unknown behavior: {behavior}")

    def get_module_health(self) -> Dict[str, bool]:
        """Get health status of all behavior modules"""
        return {
            'service_monitor': hasattr(self.service_monitor, 'last_checked'),
            'system_monitor': bool(self.system_monitor.last_stats),
            'code_quality_monitor': len(self.code_quality_monitor.monitored_projects) > 0,
            'scheduler': len(self.scheduler.get_schedule_info()['jobs']) > 0
        }