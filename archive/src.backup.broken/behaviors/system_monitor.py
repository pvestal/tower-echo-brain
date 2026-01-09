#!/usr/bin/env python3
"""
System resource monitoring behavior for Echo Brain
"""
import asyncio
import logging
import psutil
from datetime import datetime
from typing import Dict
from src.tasks.task_queue import TaskQueue, create_optimization_task, create_maintenance_task

logger = logging.getLogger(__name__)

class SystemMonitor:
    """Monitors system resources and creates optimization tasks"""

    def __init__(self, task_queue: TaskQueue):
        self.task_queue = task_queue
        self.last_stats = {}

    async def monitor_resources(self):
        """Monitor CPU, memory, disk usage"""
        try:
            # Get system stats
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            stats = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / (1024**3),
                'timestamp': datetime.now().isoformat()
            }

            # Check for concerning resource usage
            await self._check_resource_thresholds(stats)

            self.last_stats = stats
            logger.debug(f"📊 System stats: CPU {cpu_percent}%, Memory {memory.percent}%, Disk {disk.percent}%")

        except Exception as e:
            logger.error(f"❌ System monitoring failed: {e}")

    async def monitor_processes(self):
        """Monitor process health and detect issues"""
        try:
            tower_processes = []

            for proc in psutil.process_iter(['pid', 'name', 'memory_percent', 'cpu_percent']):
                try:
                    pinfo = proc.info
                    if 'tower' in pinfo['name'].lower() or 'echo' in pinfo['name'].lower():
                        tower_processes.append(pinfo)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            # Check for high resource usage
            for proc in tower_processes:
                if proc['memory_percent'] > 10.0:  # Over 10% memory
                    await self._handle_high_memory_process(proc)

                if proc['cpu_percent'] > 80.0:  # Over 80% CPU
                    await self._handle_high_cpu_process(proc)

            logger.debug(f"🔍 Monitored {len(tower_processes)} Tower processes")

        except Exception as e:
            logger.error(f"❌ Process monitoring failed: {e}")

    async def _check_resource_thresholds(self, stats: Dict):
        """Check if system resources exceed safe thresholds"""
        # High CPU usage
        if stats['cpu_percent'] > 85:
            await self._create_cpu_optimization_task(stats)

        # Low memory
        if stats['memory_percent'] > 90:
            await self._create_memory_optimization_task(stats)

        # Low disk space
        if stats['disk_percent'] > 90:
            await self._create_disk_cleanup_task(stats)

    async def _create_cpu_optimization_task(self, stats: Dict):
        """Create task for CPU optimization"""
        task = await create_optimization_task(
            f"High CPU usage detected: {stats['cpu_percent']:.1f}%",
            {
                'optimization_type': 'cpu_usage',
                'current_usage': stats['cpu_percent'],
                'threshold': 85,
                'stats': stats
            }
        )
        await self.task_queue.add_task_object(task)

    async def _create_memory_optimization_task(self, stats: Dict):
        """Create task for memory optimization"""
        task = await create_optimization_task(
            f"High memory usage detected: {stats['memory_percent']:.1f}%",
            {
                'optimization_type': 'memory_usage',
                'current_usage': stats['memory_percent'],
                'available_gb': stats['memory_available_gb'],
                'threshold': 90,
                'stats': stats
            }
        )
        await self.task_queue.add_task_object(task)

    async def _create_disk_cleanup_task(self, stats: Dict):
        """Create task for disk cleanup"""
        task = await create_maintenance_task(
            f"Low disk space detected: {stats['disk_percent']:.1f}% used",
            {
                'maintenance_type': 'disk_cleanup',
                'disk_usage': stats['disk_percent'],
                'free_gb': stats['disk_free_gb'],
                'threshold': 90,
                'stats': stats
            }
        )
        await self.task_queue.add_task_object(task)

    async def _handle_high_memory_process(self, proc: Dict):
        """Handle process using too much memory"""
        task = await create_optimization_task(
            f"High memory process detected: {proc['name']} using {proc['memory_percent']:.1f}%",
            {
                'optimization_type': 'process_memory',
                'process': proc,
                'action': 'investigate_memory_leak'
            }
        )
        await self.task_queue.add_task_object(task)

    async def _handle_high_cpu_process(self, proc: Dict):
        """Handle process using too much CPU"""
        task = await create_optimization_task(
            f"High CPU process detected: {proc['name']} using {proc['cpu_percent']:.1f}%",
            {
                'optimization_type': 'process_cpu',
                'process': proc,
                'action': 'investigate_cpu_usage'
            }
        )
        await self.task_queue.add_task_object(task)

    def get_resource_stats(self) -> Dict:
        """Get latest resource statistics"""
        return self.last_stats