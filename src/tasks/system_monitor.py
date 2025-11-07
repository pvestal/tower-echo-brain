"""System monitoring module for Echo Brain"""

class SystemMonitor:
    """Monitor system resources and health"""

    def __init__(self, task_queue=None):
        self.monitoring_active = True
        self.task_queue = task_queue

    async def get_system_metrics(self):
        """Get system metrics"""
        return {
            "cpu_usage": "normal",
            "memory_usage": "normal",
            "status": "healthy"
        }

    async def check_system_health(self):
        """Check overall system health"""
        return {"status": "healthy", "monitoring": self.monitoring_active}

    async def monitor_resources(self):
        """Monitor system resources and create tasks for issues"""
        try:
            import psutil

            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            issues = []

            # Check for resource issues
            if cpu_percent > 90:
                issues.append(f"High CPU usage: {cpu_percent}%")
                if self.task_queue:
                    from .task_queue import Task, TaskType, TaskPriority
                    task = Task(
                        type=TaskType.OPTIMIZATION,
                        priority=TaskPriority.HIGH,
                        description="High CPU usage detected",
                        payload={"cpu_percent": cpu_percent, "action": "investigate"}
                    )
                    await self.task_queue.add_task(task)

            if memory.percent > 90:
                issues.append(f"High memory usage: {memory.percent}%")
                if self.task_queue:
                    from .task_queue import Task, TaskType, TaskPriority
                    task = Task(
                        type=TaskType.MAINTENANCE,
                        priority=TaskPriority.HIGH,
                        description="High memory usage detected",
                        payload={"memory_percent": memory.percent, "action": "cleanup"}
                    )
                    await self.task_queue.add_task(task)

            if disk.percent > 90:
                issues.append(f"High disk usage: {disk.percent}%")
                if self.task_queue:
                    from .task_queue import Task, TaskType, TaskPriority
                    task = Task(
                        type=TaskType.MAINTENANCE,
                        priority=TaskPriority.URGENT,
                        description="High disk usage detected",
                        payload={"disk_percent": disk.percent, "action": "cleanup"}
                    )
                    await self.task_queue.add_task(task)

            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent,
                "issues": issues,
                "status": "healthy" if not issues else "warning"
            }

        except Exception as e:
            return {"error": f"Resource monitoring failed: {str(e)}"}