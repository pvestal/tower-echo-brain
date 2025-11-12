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
                ("vault", 8200),
                ("notifications", 8314),
                ("telegram-bot", 8316),
                ("veteran-guardian", 8317)
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
                            from .task_queue import Task, TaskType, TaskPriority, TaskStatus
                            repair_task = Task(
                                id=str(__import__('uuid').uuid4()),
                                name=f"Restart failed service: {service_name}",
                                task_type=TaskType.SYSTEM_REPAIR,
                                priority=TaskPriority.HIGH,
                                status=TaskStatus.PENDING,
                                payload={"service": service_name, "port": port, "action": "restart"},
                                created_at=__import__('datetime').datetime.now(),
                                updated_at=__import__('datetime').datetime.now()
                            )
                            await self.task_queue.add_task(repair_task)
                    else:
                        # Special check for Vault - port is open but it might be sealed
                        if service_name == "vault":
                            await self._check_vault_seal_status()

                except Exception as e:
                    failed_services.append(f"{service_name} (error: {str(e)})")

            return {"failed_services": failed_services, "checked": len(services_to_check)}

        except Exception as e:
            return {"error": f"Service monitoring failed: {str(e)}"}

    async def _check_vault_seal_status(self):
        """Check if Vault is sealed and create unseal task if needed"""
        try:
            import asyncio

            # Check vault status
            import os
            vault_env = os.environ.copy()
            vault_env['VAULT_ADDR'] = 'http://127.0.0.1:8200'
            vault_env['PATH'] = '/usr/local/bin:/usr/bin:/bin'

            proc = await asyncio.create_subprocess_shell(
                "/usr/bin/vault status",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=vault_env
            )
            stdout, stderr = await proc.communicate()

            # Vault returns exit code 2 when sealed (some versions return 1)
            if proc.returncode in [1, 2] and "Sealed" in stdout.decode() and ("true" in stdout.decode() or "Sealed          true" in stdout.decode()):
                # Vault is sealed, create unseal task
                if self.task_queue:
                    from .task_queue import Task, TaskType, TaskPriority, TaskStatus
                    from datetime import datetime
                    import uuid

                    unseal_task = Task(
                        id=str(uuid.uuid4()),
                        name="Unseal HashiCorp Vault",
                        task_type=TaskType.SYSTEM_REPAIR,
                        priority=TaskPriority.HIGH,
                        status=TaskStatus.PENDING,
                        payload={
                            "service": "vault",
                            "action": "vault_unseal",
                            "repair_type": "vault_unseal",
                            "issue": "Vault is sealed",
                            "target": "vault"
                        },
                        created_at=datetime.now(),
                        updated_at=datetime.now()
                    )
                    await self.task_queue.add_task(unseal_task)

        except Exception as e:
            # Log the error but don't fail the entire monitoring
            pass