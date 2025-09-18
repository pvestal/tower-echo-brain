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

from .task_queue import (
    TaskQueue, Task, TaskType, TaskPriority, TaskStatus,
    create_monitoring_task, create_optimization_task,
    create_learning_task, create_maintenance_task, create_scheduled_task
)

logger = logging.getLogger(__name__)

class AutonomousBehaviors:
    """Echo's autonomous task generation and self-initiated behaviors"""
    
    def __init__(self, task_queue: TaskQueue):
        self.task_queue = task_queue
        self.running = False
        self.behavior_stats = {
            'started_at': None,
            'tasks_generated': 0,
            'proactive_detections': 0,
            'last_health_check': None,
            'last_optimization': None,
            'system_issues_detected': 0
        }
        
        # Tower service endpoints for monitoring
        self.tower_services = {
            'echo': 'http://localhost:8309',
            'anime': 'http://localhost:8328',
            'knowledge_base': 'http://localhost:8307',
            'comfyui': 'http://localhost:8188',
            'auth': 'http://localhost:8088',
            'apple_music': 'http://localhost:8315',
            'vault': 'http://localhost:8200',
            'notifications': 'http://localhost:8350'
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
        asyncio.create_task(self._scheduled_task_processor())
        
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
                
    async def _monitor_tower_services(self):
        """Monitor all Tower services and create tasks for issues"""
        for service_name, service_url in self.tower_services.items():
            try:
                start_time = time.time()
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{service_url}/api/health", timeout=5) as response:
                        response_time = time.time() - start_time
                        
                        if response.status != 200:
                            # Service unhealthy - create urgent task
                            task = create_monitoring_task(
                                f"Service Health Issue: {service_name}",
                                service_url,
                                "service_health",
                                TaskPriority.URGENT
                            )
                            task.payload.update({
                                'issue': 'http_error',
                                'status_code': response.status,
                                'service_name': service_name
                            })
                            await self.task_queue.add_task(task)
                            
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
        
    def get_behavior_stats(self) -> Dict[str, Any]:
        """Get autonomous behavior statistics"""
        return {
            'running': self.running,
            'stats': self.behavior_stats,
            'thresholds': self.thresholds,
            'monitored_services': list(self.tower_services.keys()),
            'learning_patterns': self.learning_patterns
        }
