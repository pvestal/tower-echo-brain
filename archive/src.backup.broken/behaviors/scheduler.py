#!/usr/bin/env python3
"""
Scheduled behavior management for Echo Brain
"""
import asyncio
import logging
import schedule
from datetime import datetime, time
from typing import Dict, List, Callable
from src.tasks.task_queue import TaskQueue, create_scheduled_task, TaskType, TaskPriority

logger = logging.getLogger(__name__)

class BehaviorScheduler:
    """Manages scheduled autonomous behaviors"""

    def __init__(self, task_queue: TaskQueue):
        self.task_queue = task_queue
        self.scheduled_jobs = []

    async def setup_schedules(self):
        """Setup all scheduled behavior loops"""
        try:
            # Clear existing schedules
            schedule.clear()

            # Daily maintenance at 2 AM
            schedule.every().day.at("02:00").do(self._schedule_maintenance)

            # Code quality analysis every Sunday at 3 AM
            schedule.every().sunday.at("03:00").do(self._schedule_code_analysis)

            # System optimization check every 6 hours
            schedule.every(6).hours.do(self._schedule_optimization)

            # Daily digest at 8 AM
            schedule.every().day.at("08:00").do(self._schedule_daily_digest)

            # Security scan every Tuesday at 1 AM
            schedule.every().tuesday.at("01:00").do(self._schedule_security_scan)

            logger.info("✅ Scheduled behaviors configured")

        except Exception as e:
            logger.error(f"❌ Failed to setup schedules: {e}")

    async def process_schedules(self):
        """Process scheduled tasks (call periodically)"""
        try:
            # Run pending scheduled jobs
            schedule.run_pending()

        except Exception as e:
            logger.error(f"❌ Schedule processing failed: {e}")

    def _schedule_maintenance(self):
        """Schedule maintenance tasks"""
        asyncio.create_task(self._create_maintenance_tasks())

    def _schedule_code_analysis(self):
        """Schedule code quality analysis"""
        asyncio.create_task(self._create_code_analysis_task())

    def _schedule_optimization(self):
        """Schedule system optimization"""
        asyncio.create_task(self._create_optimization_task())

    def _schedule_daily_digest(self):
        """Schedule daily digest generation"""
        asyncio.create_task(self._create_digest_task())

    def _schedule_security_scan(self):
        """Schedule security configuration scan"""
        asyncio.create_task(self._create_security_task())

    async def _create_maintenance_tasks(self):
        """Create maintenance tasks"""
        try:
            # Log rotation task
            task = await create_scheduled_task(
                TaskType.MAINTENANCE,
                TaskPriority.LOW,
                "Daily log rotation and cleanup",
                "Perform daily log rotation, temporary file cleanup, and disk space optimization",
                {
                    'maintenance_type': 'daily_cleanup',
                    'tasks': ['log_rotation', 'temp_cleanup', 'disk_optimization'],
                    'scheduled_time': datetime.now().isoformat()
                }
            )
            await self.task_queue.add_task_object(task)

            # Database maintenance task
            db_task = await create_scheduled_task(
                TaskType.MAINTENANCE,
                TaskPriority.LOW,
                "Database maintenance and optimization",
                "Perform database VACUUM, ANALYZE, and index optimization",
                {
                    'maintenance_type': 'database_maintenance',
                    'tasks': ['vacuum', 'analyze', 'index_optimization'],
                    'scheduled_time': datetime.now().isoformat()
                }
            )
            await self.task_queue.add_task_object(db_task)

            logger.info("📅 Scheduled maintenance tasks created")

        except Exception as e:
            logger.error(f"Failed to create maintenance tasks: {e}")

    async def _create_code_analysis_task(self):
        """Create code quality analysis task"""
        try:
            task = await create_scheduled_task(
                TaskType.CODE_REFACTOR,
                TaskPriority.NORMAL,
                "Weekly code quality analysis",
                "Analyze code quality across all Tower projects and create refactoring tasks",
                {
                    'analysis_type': 'weekly_quality_scan',
                    'projects': [
                        '/opt/tower-echo-brain/src',
                        '/opt/tower-anime-production',
                        '/opt/tower-auth',
                        '/opt/tower-kb'
                    ],
                    'scheduled_time': datetime.now().isoformat()
                }
            )
            await self.task_queue.add_task_object(task)

            logger.info("📅 Scheduled code analysis task created")

        except Exception as e:
            logger.error(f"Failed to create code analysis task: {e}")

    async def _create_optimization_task(self):
        """Create system optimization task"""
        try:
            task = await create_scheduled_task(
                TaskType.SYSTEM_OPTIMIZATION,
                TaskPriority.NORMAL,
                "System performance optimization check",
                "Check system resources and optimize performance if needed",
                {
                    'optimization_type': 'scheduled_check',
                    'checks': ['cpu_usage', 'memory_usage', 'disk_usage', 'process_health'],
                    'scheduled_time': datetime.now().isoformat()
                }
            )
            await self.task_queue.add_task_object(task)

            logger.info("📅 Scheduled optimization task created")

        except Exception as e:
            logger.error(f"Failed to create optimization task: {e}")

    async def _create_digest_task(self):
        """Create daily digest task"""
        try:
            task = await create_scheduled_task(
                TaskType.LEARNING,
                TaskPriority.HIGH,
                "Daily system digest and analysis",
                "Generate daily digest of system activities, repairs, and improvements",
                {
                    'digest_type': 'daily_summary',
                    'include': ['repairs', 'optimizations', 'tasks_completed', 'system_health'],
                    'scheduled_time': datetime.now().isoformat()
                }
            )
            await self.task_queue.add_task_object(task)

            logger.info("📅 Scheduled digest task created")

        except Exception as e:
            logger.error(f"Failed to create digest task: {e}")

    async def _create_security_task(self):
        """Create security scan task"""
        try:
            task = await create_scheduled_task(
                TaskType.MONITORING,
                TaskPriority.HIGH,
                "Weekly security configuration scan",
                "Scan security configurations and identify potential vulnerabilities",
                {
                    'scan_type': 'security_configuration',
                    'checks': ['service_configs', 'file_permissions', 'network_security'],
                    'scheduled_time': datetime.now().isoformat()
                }
            )
            await self.task_queue.add_task_object(task)

            logger.info("📅 Scheduled security task created")

        except Exception as e:
            logger.error(f"Failed to create security task: {e}")

    def get_schedule_info(self) -> Dict:
        """Get information about scheduled jobs"""
        jobs_info = []
        for job in schedule.jobs:
            jobs_info.append({
                'function': job.job_func.__name__,
                'interval': str(job.interval),
                'unit': job.unit,
                'at_time': str(job.at_time) if job.at_time else None,
                'next_run': job.next_run.isoformat() if job.next_run else None
            })

        return {
            'total_jobs': len(schedule.jobs),
            'jobs': jobs_info
        }