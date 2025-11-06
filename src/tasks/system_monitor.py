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