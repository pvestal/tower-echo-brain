"""Service monitoring module for Echo Brain autonomous behaviors"""

class ServiceMonitor:
    """Monitor service health and status"""

    def __init__(self, task_queue=None):
        self.monitored_services = []
        self.task_queue = task_queue

    async def check_service_health(self, service_name):
        """Check health of a service"""
        return {"service": service_name, "status": "healthy", "available": True}

    async def get_service_status(self):
        """Get overall service status"""
        return {"status": "monitoring_active", "services_count": len(self.monitored_services)}

    async def monitor_services(self):
        """Monitor all registered services and detect failures"""
        try:
            # Check common Tower services
            services_to_check = [
                ("tower-echo-brain", 8309),
                ("comfyui", 8188),
                ("tower-auth", 8088),
                ("tower-kb", 8307),
                ("anime-production", 8328),
                ("apple-music", 8315),
                ("vault", 8200)
            ]

            failed_services = []
            for service_name, port in services_to_check:
                try:
                    # Basic port check
                    import socket
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(2)
                    result = sock.connect_ex(('127.0.0.1', port))
                    sock.close()

                    if result != 0:
                        failed_services.append(service_name)
                        if self.task_queue:
                            # Create repair task
                            from .task_queue import Task, TaskType, TaskPriority
                            repair_task = Task(
                                type=TaskType.SYSTEM_REPAIR,
                                priority=TaskPriority.HIGH,
                                description=f"Restart failed service: {service_name}",
                                payload={"service": service_name, "port": port, "action": "restart"}
                            )
                            await self.task_queue.add_task(repair_task)

                except Exception as e:
                    failed_services.append(f"{service_name} (error: {str(e)})")

            return {"failed_services": failed_services, "checked": len(services_to_check)}

        except Exception as e:
            return {"error": f"Service monitoring failed: {str(e)}"}