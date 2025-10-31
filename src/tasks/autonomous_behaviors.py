#!/usr/bin/env python3
"""
Echo Brain Autonomous Behaviors
Self-initiated tasks and proactive problem detection
"""

import asyncio
import logging
import json
import time
import os
import subprocess
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import psutil
import schedule

from src.db.database import database
from .task_queue import (
    TaskQueue, Task, TaskType, TaskPriority, TaskStatus,
    create_monitoring_task, create_optimization_task,

    create_learning_task, create_maintenance_task, create_scheduled_task
)

# Import repair executor for daily digest
from .autonomous_repair_executor import repair_executor
from .code_refactor_executor import code_refactor_executor
from .security_monitoring import SecurityConfigMonitor, SecurityIssue

logger = logging.getLogger(__name__)

class AutonomousBehaviors:
    """Echo's autonomous task generation and self-initiated behaviors"""
    
    def __init__(self, task_queue: TaskQueue):
        self.task_queue = task_queue
        self.running = False

        # Map monitoring names to actual systemd service names
        self.service_name_map = {
            'echo': 'tower-echo-brain.service',
            'telegram_bot': 'echo-telegram-bot.service',
            'auth': 'tower-auth.service',
            'comfyui': 'comfyui.service',
            'knowledge_base': 'tower-kb.service',
            'anime': 'tower-anime-production.service',
            'apple_music': 'tower-apple-music.service',
            'notifications': 'tower-notification-bot.service',
            'vault': 'vault.service'
        }

        # Restart cooldown tracking (prevent aggressive restarts)
        self.last_restart_times: Dict[str, datetime] = {}
        self.RESTART_COOLDOWN_MINUTES = 5  # Don't restart same service within 5 minutes
        self.STARTUP_GRACE_PERIOD = 120  # 2 minutes for services to stabilize

        self.behavior_stats = {
            'started_at': None,
            'tasks_generated': 0,
            'proactive_detections': 0,
            'last_health_check': None,
            'last_optimization': None,
            'system_issues_detected': 0
        }
        
        # Database for persistent state
        self.db = database
        
        # Tower service endpoints for monitoring
        self.tower_services = {
            'echo': 'http://localhost:8309/api/echo/health',
            'anime': 'http://localhost:8328/api/health',
            'knowledge_base': 'http://localhost:8307/api/health',
            # 'comfyui': 'http://localhost:8188/api/health',  # Removed: No health endpoint available
            'auth': 'http://localhost:8088/api/auth/health',
            'apple_music': 'http://localhost:8315/api/music/health',
            # 'vault': 'http://localhost:8200/v1/sys/health',  # Removed: Vault is sealed
            # 'notifications': 'http://localhost:8350/api/notifications/health',  # Removed: Service not running
            # 'telegram_bot': 'http://localhost:8309/api/telegram/health',  # Removed: Service not needed
        }
        
        # Thresholds for proactive detection
        self.thresholds = {
            'cpu_warning': 75.0,
            'cpu_critical': 90.0,
            'memory_warning': 80.0,
            'memory_critical': 95.0,
            'disk_warning': 85.0,
            'disk_critical': 95.0,
            'response_time_warning': 5.0,
            'response_time_critical': 10.0
        }
        
        # Learning patterns
        self.learning_patterns = [
            'error', 'warning', 'failed', 'timeout', 'connection',
            'memory', 'cpu', 'disk', 'permission', 'authentication'
        ]

        # Security monitoring
        self.security_monitor = SecurityConfigMonitor()
        
    async def start(self):
        """Start autonomous behaviors"""
        if self.running:
            logger.warning("🤖 Autonomous behaviors already running")
            return
            
        self.running = True
        self.behavior_stats['started_at'] = datetime.now()
        
        logger.info("🧠 Echo autonomous behaviors starting...")
        
        # Initialize scheduled tasks
        await self._setup_scheduled_tasks()
        
        # Start behavior loops
        asyncio.create_task(self._proactive_monitoring_loop())
        asyncio.create_task(self._system_optimization_loop())
        asyncio.create_task(self._learning_loop())
        asyncio.create_task(self._maintenance_loop())
        asyncio.create_task(self._daily_digest_loop())
        asyncio.create_task(self._scheduled_task_processor())
        # asyncio.create_task(self._code_quality_loop())  # Temporarily disabled - causing startup delays
        asyncio.create_task(self._security_configuration_loop())

        logger.info("✅ Echo autonomous behaviors active")
        
    async def stop(self):
        """Stop autonomous behaviors"""
        if not self.running:
            return
            
        self.running = False
        logger.info("🛑 Echo autonomous behaviors stopping...")
        
    async def _setup_scheduled_tasks(self):
        """Setup recurring scheduled tasks"""
        now = datetime.now()
        
        # Daily system health report (9 AM)
        daily_health = now.replace(hour=9, minute=0, second=0, microsecond=0)
        if daily_health <= now:
            daily_health += timedelta(days=1)
            
        health_task = create_scheduled_task(
            "Daily System Health Report",
            TaskType.ANALYSIS,
            daily_health,
            {
                'report_type': 'daily_health',
                'include_services': True,
                'include_resources': True,
                'include_logs': True
            }
        )
        await self.task_queue.add_task(health_task)
        
        # Weekly cleanup (Sunday 2 AM)
        weekly_cleanup = now.replace(hour=2, minute=0, second=0, microsecond=0)
        days_until_sunday = (6 - now.weekday()) % 7
        if days_until_sunday == 0 and weekly_cleanup <= now:
            days_until_sunday = 7
        weekly_cleanup += timedelta(days=days_until_sunday)
        
        cleanup_task = create_scheduled_task(
            "Weekly System Cleanup",
            TaskType.MAINTENANCE,
            weekly_cleanup,
            {
                'action': 'full_cleanup',
                'targets': ['temp_files', 'old_logs', 'cache', 'zombie_processes']
            }
        )
        await self.task_queue.add_task(cleanup_task)
        
        # Database backup (Daily 3 AM)
        daily_backup = now.replace(hour=3, minute=0, second=0, microsecond=0)
        if daily_backup <= now:
            daily_backup += timedelta(days=1)
            
        backup_task = create_scheduled_task(
            "Daily Database Backup",
            TaskType.BACKUP,
            daily_backup,
            {
                'action': 'backup',
                'target': 'echo_brain_database',
                'retention_days': 7
            }
        )
        await self.task_queue.add_task(backup_task)
        
        logger.info("📅 Scheduled tasks created")
        
    async def _proactive_monitoring_loop(self):
        """Continuous proactive system monitoring"""
        logger.info("👁️ Proactive monitoring loop started")
        
        while self.running:
            try:
                # Monitor Tower services
                await self._monitor_tower_services()
                
                # Monitor system resources
                await self._monitor_system_resources()
                
                # Monitor process health
                await self._monitor_process_health()
                
                # Update stats
                self.behavior_stats['last_health_check'] = datetime.now()
                
                # Wait before next monitoring cycle
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"❌ Error in proactive monitoring: {e}")
                await asyncio.sleep(60)
                

    async def _check_restart_cooldown(self, service_name: str):
        """Check if service can be restarted (persists across Echo restarts)"""
        try:
            import psycopg2
            conn = psycopg2.connect(**self.db.db_config)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT last_restart_time, restart_count FROM restart_cooldowns WHERE service_name = %s",
                (service_name,)
            )
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if result:
                last_restart, restart_count = result
                minutes_since = (datetime.now() - last_restart).total_seconds() / 60
                if restart_count >= 3:
                    required_cooldown = self.RESTART_COOLDOWN_MINUTES * 3
                    logger.warning(f"⚠️ Restart loop detected for {service_name}: {restart_count} restarts")
                    self.behavior_stats['restart_loops_prevented'] += 1
                    return (minutes_since >= required_cooldown, minutes_since)
                return (minutes_since >= self.RESTART_COOLDOWN_MINUTES, minutes_since)
            return (True, 999.0)
        except Exception as e:
            logger.error(f"Error checking cooldown: {e}")
            return (True, 999.0)

    async def _record_restart(self, service_name: str):
        """Record service restart in database"""
        try:
            import psycopg2
            conn = psycopg2.connect(**self.db.db_config)
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO restart_cooldowns (service_name, last_restart_time, restart_count, updated_at)
                   VALUES (%s, NOW(), 1, NOW())
                   ON CONFLICT (service_name) DO UPDATE SET
                   last_restart_time = NOW(), restart_count = restart_cooldowns.restart_count + 1, updated_at = NOW()""",
                (service_name,)
            )
            conn.commit()
            cursor.close()
            conn.close()
            logger.info(f"✅ Recorded restart for {service_name} in database")
        except Exception as e:
            logger.error(f"Error recording restart: {e}")

    async def _log_self_awareness_issue(self, issue_type: str, details: str):
        """Log when Echo detects an issue with itself"""
        try:
            import json
            logger.error(f"🧠 SELF-AWARENESS: {issue_type} - {details}")
            await self.db.execute(
                """INSERT INTO autonomous_tasks (task_id, name, description, priority, status, created_at, result)
                   VALUES ($1, $2, $3, $4, $5, NOW(), $6::jsonb) ON CONFLICT (task_id) DO NOTHING""",
                f"self-awareness-{datetime.now().timestamp()}",
                "Echo Self-Awareness Alert",
                f"{issue_type}: {details}",
                10, "pending",
                json.dumps({"type": issue_type, "details": details, "requires_human_intervention": True})
            )
        except Exception as e:
            logger.error(f"Error logging self-awareness: {e}")

    async def _monitor_tower_services(self):
        """Monitor all Tower services and create tasks for issues"""
        for service_name, service_url in self.tower_services.items():
            try:
                start_time = time.time()
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(service_url, timeout=5) as response:
                        response_time = time.time() - start_time
                        
                        if response.status != 200:
                            # Service unhealthy - create repair task
                            # Map to actual systemd service name
                            actual_service_name = self.service_name_map.get(service_name, service_name)
                            logger.info(f"🔍 Service mapping: {service_name} → {actual_service_name}")
                            
                            # SELF-EXCLUSION: Never restart Echo itself
                            if actual_service_name == 'tower-echo-brain.service':
                                logger.warning(f"🛑 SELF-EXCLUSION: Refusing to restart myself")
                                await self._log_self_awareness_issue("Attempted self-restart", f"HTTP {response.status}")
                                continue
                            
                            # Check cooldown from database (persistent)
                            can_restart, minutes_since = await self._check_restart_cooldown(actual_service_name)
                            if not can_restart:
                                logger.info(f"⏱️ Skipping restart of {actual_service_name} - cooldown active ({minutes_since:.1f}m)")
                                continue
                            
                            # Record this restart in database
                            await self._record_restart(actual_service_name)
                            
                            # Create maintenance task to ACTUALLY RESTART THE SERVICE
                            repair_task = create_maintenance_task(
                                f"Emergency Service Restart: {service_name}",
                                "restart",
                                actual_service_name,
                                TaskPriority.URGENT
                            )
                            repair_task.payload.update({
                                'issue': f'http_error: HTTP {response.status}',
                                'status_code': response.status,
                                'service_name': actual_service_name
                            })
                            await self.task_queue.add_task(repair_task)
                            
                            logger.info(f"🔧 Created repair task for {actual_service_name}")
                            
                            self.behavior_stats['system_issues_detected'] += 1
                            logger.warning(f"⚠️ Service {service_name} unhealthy: HTTP {response.status}")
                            
                        elif response_time > self.thresholds['response_time_warning']:
                            # Slow response - create optimization task
                            priority = TaskPriority.URGENT if response_time > self.thresholds['response_time_critical'] else TaskPriority.HIGH
                            
                            task = create_optimization_task(
                                f"Slow Response: {service_name}",
                                service_name,
                                "response_time",
                                priority
                            )
                            task.payload.update({
                                'response_time': response_time,
                                'threshold_exceeded': 'critical' if response_time > self.thresholds['response_time_critical'] else 'warning'
                            })
                            await self.task_queue.add_task(task)
                            
                            logger.warning(f"🐌 Service {service_name} slow response: {response_time:.2f}s")
                            
            except asyncio.TimeoutError:
                # Service timeout - create urgent task
                task = create_monitoring_task(
                    f"Service Timeout: {service_name}",
                    service_url,
                    "service_health",
                    TaskPriority.URGENT
                )
                task.payload.update({
                    'issue': 'timeout',
                    'service_name': service_name
                })
                await self.task_queue.add_task(task)
                
                self.behavior_stats['system_issues_detected'] += 1
                logger.error(f"⏰ Service {service_name} timeout")
                
            except Exception as e:
                # Connection error - create high priority task
                task = create_monitoring_task(
                    f"Service Connection Error: {service_name}",
                    service_url,
                    "service_health",
                    TaskPriority.HIGH
                )
                task.payload.update({
                    'issue': 'connection_error',
                    'error': str(e),
                    'service_name': service_name
                })
                await self.task_queue.add_task(task)
                
                logger.error(f"🔌 Service {service_name} connection error: {e}")
                
    async def _monitor_system_resources(self):
        """Monitor system resources and create optimization tasks"""
        try:
            # CPU monitoring
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > self.thresholds['cpu_warning']:
                priority = TaskPriority.URGENT if cpu_percent > self.thresholds['cpu_critical'] else TaskPriority.HIGH
                
                task = create_optimization_task(
                    f"High CPU Usage: {cpu_percent:.1f}%",
                    "cpu",
                    "usage",
                    priority
                )
                task.payload.update({
                    'cpu_percent': cpu_percent,
                    'threshold_type': 'critical' if cpu_percent > self.thresholds['cpu_critical'] else 'warning'
                })
                await self.task_queue.add_task(task)
                
                logger.warning(f"🔥 High CPU usage detected: {cpu_percent:.1f}%")
                
            # Memory monitoring
            memory = psutil.virtual_memory()
            if memory.percent > self.thresholds['memory_warning']:
                priority = TaskPriority.URGENT if memory.percent > self.thresholds['memory_critical'] else TaskPriority.HIGH
                
                task = create_optimization_task(
                    f"High Memory Usage: {memory.percent:.1f}%",
                    "memory",
                    "usage",
                    priority
                )
                task.payload.update({
                    'memory_percent': memory.percent,
                    'memory_available': memory.available,
                    'threshold_type': 'critical' if memory.percent > self.thresholds['memory_critical'] else 'warning'
                })
                await self.task_queue.add_task(task)
                
                logger.warning(f"🧠 High memory usage detected: {memory.percent:.1f}%")
                
            # Disk monitoring
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            if disk_percent > self.thresholds['disk_warning']:
                priority = TaskPriority.URGENT if disk_percent > self.thresholds['disk_critical'] else TaskPriority.HIGH
                
                task = create_optimization_task(
                    f"High Disk Usage: {disk_percent:.1f}%",
                    "disk",
                    "usage",
                    priority
                )
                task.payload.update({
                    'disk_percent': disk_percent,
                    'disk_free': disk.free,
                    'threshold_type': 'critical' if disk_percent > self.thresholds['disk_critical'] else 'warning'
                })
                await self.task_queue.add_task(task)
                
                logger.warning(f"💽 High disk usage detected: {disk_percent:.1f}%")
                
        except Exception as e:
            logger.error(f"❌ Error monitoring system resources: {e}")
            
    async def _monitor_process_health(self):
        """Monitor critical processes and detect issues"""
        try:
            # Check for zombie processes
            zombie_count = 0
            high_memory_processes = []
            
            for proc in psutil.process_iter(['pid', 'name', 'status', 'memory_percent', 'cpu_percent']):
                try:
                    if proc.info['status'] == psutil.STATUS_ZOMBIE:
                        zombie_count += 1
                    elif proc.info['memory_percent'] > 10:  # Processes using > 10% memory
                        high_memory_processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
            if zombie_count > 0:
                task = create_maintenance_task(
                    f"Zombie Processes Detected: {zombie_count}",
                    "cleanup",
                    "zombie_processes",
                    TaskPriority.HIGH
                )
                task.payload.update({'zombie_count': zombie_count})
                await self.task_queue.add_task(task)
                
                logger.warning(f"🧟 {zombie_count} zombie processes detected")
                
            if len(high_memory_processes) > 5:
                task = create_optimization_task(
                    f"Multiple High Memory Processes: {len(high_memory_processes)}",
                    "processes",
                    "memory",
                    TaskPriority.NORMAL
                )
                task.payload.update({'high_memory_processes': high_memory_processes})
                await self.task_queue.add_task(task)
                
                logger.info(f"🔍 {len(high_memory_processes)} high memory processes detected")
                
        except Exception as e:
            logger.error(f"❌ Error monitoring process health: {e}")
            
    async def _system_optimization_loop(self):
        """Periodic system optimization tasks"""
        logger.info("⚡ System optimization loop started")
        
        while self.running:
            try:
                # Create periodic optimization tasks
                current_hour = datetime.now().hour
                
                # Memory optimization every 4 hours during low activity
                if current_hour % 4 == 0 and datetime.now().minute < 5:
                    task = create_optimization_task(
                        "Scheduled Memory Optimization",
                        "memory",
                        "periodic_cleanup",
                        TaskPriority.LOW
                    )
                    await self.task_queue.add_task(task)
                    self.behavior_stats['tasks_generated'] += 1
                    
                # Disk optimization daily at 4 AM
                if current_hour == 4 and datetime.now().minute < 5:
                    task = create_optimization_task(
                        "Scheduled Disk Optimization",
                        "disk",
                        "cleanup",
                        TaskPriority.LOW
                    )
                    await self.task_queue.add_task(task)
                    self.behavior_stats['tasks_generated'] += 1
                    
                self.behavior_stats['last_optimization'] = datetime.now()
                
                # Wait 5 minutes before next check
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"❌ Error in optimization loop: {e}")
                await asyncio.sleep(300)
                
    async def _learning_loop(self):
        """Continuous learning from system behavior"""
        logger.info("🎓 Learning loop started")
        
        while self.running:
            try:
                # Analyze system logs for patterns
                for pattern in self.learning_patterns:
                    task = create_learning_task(
                        f"Log Pattern Analysis: {pattern}",
                        "system_logs",
                        pattern,
                        TaskPriority.LOW
                    )
                    await self.task_queue.add_task(task)
                    
                self.behavior_stats['tasks_generated'] += len(self.learning_patterns)
                
                # Learn from task execution patterns
                task = create_learning_task(
                    "Task Execution Pattern Analysis",
                    "task_history",
                    "execution_patterns",
                    TaskPriority.LOW
                )
                await self.task_queue.add_task(task)
                
                # Wait 1 hour before next learning cycle
                await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error(f"❌ Error in learning loop: {e}")
                await asyncio.sleep(3600)
                
    async def _maintenance_loop(self):
        """Periodic maintenance tasks"""
        logger.info("🔧 Maintenance loop started")
        
        while self.running:
            try:
                current_time = datetime.now()
                
                # Daily log rotation check (2 AM)
                if current_time.hour == 2 and current_time.minute < 5:
                    task = create_maintenance_task(
                        "Daily Log Rotation Check",
                        "cleanup",
                        "log_files",
                        TaskPriority.LOW
                    )
                    await self.task_queue.add_task(task)
                    
                # Weekly task queue cleanup (Sunday 1 AM)
                if current_time.weekday() == 6 and current_time.hour == 1 and current_time.minute < 5:
                    task = create_maintenance_task(
                        "Weekly Task Queue Cleanup",
                        "cleanup",
                        "old_tasks",
                        TaskPriority.LOW
                    )
                    await self.task_queue.add_task(task)
                    
                # Wait 5 minutes before next check
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"❌ Error in maintenance loop: {e}")
                await asyncio.sleep(300)
                
                
    async def _daily_digest_loop(self):
        """Send daily repair digest at 9 AM"""
        logger.info("📧 Daily repair digest loop started")
        
        while self.running:
            try:
                now = datetime.now()
                # Calculate next 9 AM
                next_digest = now.replace(hour=9, minute=0, second=0, microsecond=0)
                
                # If it's already past 9 AM today, schedule for tomorrow
                if now >= next_digest:
                    next_digest += timedelta(days=1)
                
                # Calculate seconds until next digest
                wait_seconds = (next_digest - now).total_seconds()
                logger.info(f"📧 Next repair digest scheduled for {next_digest.strftime('%Y-%m-%d %H:%M')}")
                
                # Wait until 9 AM
                await asyncio.sleep(wait_seconds)
                
                # Send the digest
                logger.info("📧 Sending daily repair digest...")
                await repair_executor.send_daily_digest()
                
            except Exception as e:
                logger.error(f"❌ Error in daily digest loop: {e}")
                await asyncio.sleep(3600)  # Wait an hour on error
                
    async def _scheduled_task_processor(self):
        """Process scheduled tasks when their time comes"""
        logger.info("⏰ Scheduled task processor started")
        
        while self.running:
            try:
                # This would query for scheduled tasks due for execution
                # and move them to the active queue
                
                # Wait 1 minute before next check
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"❌ Error in scheduled task processor: {e}")
                await asyncio.sleep(60)
                
    async def generate_emergency_tasks(self, issue_type: str, details: Dict[str, Any]) -> List[Task]:
        """Generate emergency tasks for critical issues"""
        tasks = []
        
        if issue_type == "service_down":
            # Create immediate restart task
            restart_task = create_maintenance_task(
                f"Emergency Service Restart: {details['service']}",
                "restart",
                details['service'],
                TaskPriority.URGENT
            )
            restart_task.payload.update(details)
            tasks.append(restart_task)
            
            # Create diagnostic task
            diagnostic_task = create_monitoring_task(
                f"Service Failure Diagnosis: {details['service']}",
                details.get('service_url', ''),
                "failure_analysis",
                TaskPriority.HIGH
            )
            diagnostic_task.payload.update(details)
            tasks.append(diagnostic_task)
            
        elif issue_type == "resource_exhaustion":
            # Create immediate optimization task
            optimize_task = create_optimization_task(
                f"Emergency Resource Optimization: {details['resource']}",
                details['resource'],
                "emergency_cleanup",
                TaskPriority.URGENT
            )
            optimize_task.payload.update(details)
            tasks.append(optimize_task)
            
        elif issue_type == "security_breach":
            # Create security assessment task
            security_task = create_monitoring_task(
                "Security Breach Assessment",
                "system",
                "security_analysis",
                TaskPriority.URGENT
            )
            security_task.payload.update(details)
            tasks.append(security_task)
            
        # Add all emergency tasks to queue
        for task in tasks:
            await self.task_queue.add_task(task)
            self.behavior_stats['proactive_detections'] += 1
            
        logger.critical(f"🚨 Generated {len(tasks)} emergency tasks for {issue_type}")
        return tasks
        
    async def _code_quality_loop(self):
        """Monitor code quality and create refactoring tasks"""
        logger.info("🔧 Code quality monitoring started")
        
        while self.running:
            try:
                # Check code quality for tower services
                service_paths = [
                    "/opt/tower-echo-brain/src",
                    "/opt/tower-anime-production",
                    "/opt/tower-auth",
                    "/opt/tower-kb"
                ]
                
                for service_path in service_paths:
                    from pathlib import Path
                    if Path(service_path).exists():
                        # Analyze code quality
                        results = await code_refactor_executor.analyze_code_quality(service_path)
                        
                        # Only create task if quality is below threshold
                        pylint_score = results.get("pylint_score")
                        if pylint_score is not None and pylint_score < 7.0:
                            # Code quality below threshold - create refactor task
                            import uuid
                            task = Task(
                                id=str(uuid.uuid4()),
                                name=f"Code Refactor: {Path(service_path).name}",
                                task_type=TaskType.CODE_REFACTOR,
                                priority=TaskPriority.LOW,
                                payload={"project_path": service_path, "results": results}
                            )
                            await self.task_queue.add_task(task)
                            logger.info(f"📝 Created refactor task for {service_path} (score: {pylint_score:.1f}/10)")
                        elif pylint_score is not None:
                            logger.info(f"✅ {Path(service_path).name} code quality OK (score: {pylint_score:.1f}/10)")
                
                # Check every hour (3600 seconds) for more proactive monitoring
                await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error(f"Code quality monitoring error: {e}")
                await asyncio.sleep(3600)  # Retry in 1 hour on error

    async def _security_configuration_loop(self):
        """Monitor security configurations and create alerts"""
        logger.info("🔒 Security configuration monitoring started")

        while self.running:
            try:
                # Run security checks
                issues = await self.security_monitor.run_all_checks()

                # Generate tasks for each critical/high issue
                for issue in issues:
                    if issue.severity in ['CRITICAL', 'HIGH']:
                        # Create security breach task
                        task = create_monitoring_task(
                            f"Security Issue: {issue.title}",
                            "system",
                            f"security_{issue.category}",
                            TaskPriority.URGENT if issue.severity == 'CRITICAL' else TaskPriority.HIGH
                        )
                        task.payload.update({
                            'severity': issue.severity,
                            'category': issue.category,
                            'description': issue.description,
                            'remediation': issue.remediation,
                            'detected_at': issue.detected_at.isoformat(),
                            'details': issue.details
                        })
                        await self.task_queue.add_task(task)

                        # Log the security issue
                        logger.critical(f"🚨 {issue.severity}: {issue.title}")

                # Update stats
                self.behavior_stats['system_issues_detected'] += len([i for i in issues if i.severity in ['CRITICAL', 'HIGH']])

                # Wait 1 hour before next security check
                await asyncio.sleep(3600)

            except Exception as e:
                logger.error(f"❌ Error in security configuration loop: {e}")
                await asyncio.sleep(3600)

    def get_behavior_stats(self) -> Dict[str, Any]:
        """Get autonomous behavior statistics"""
        return {
            'running': self.running,
            'stats': self.behavior_stats,
            'thresholds': self.thresholds,
            'monitored_services': list(self.tower_services.keys()),
            'learning_patterns': self.learning_patterns
        }
