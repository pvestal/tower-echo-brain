#!/usr/bin/env python3
"""
Echo Service Monitor & Self-Repair System
=========================================
Gives Echo the ability to monitor and fix his own services.
"""

import asyncio
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ServiceMonitor:
    """Monitors and repairs Tower services"""

    def __init__(self):
        self.services = {
            "telegram_bot": {
                "name": "Telegram Bot",
                "check_method": "systemd",
                "service_name": "echo-telegram-bot.service",
                "restart_command": "sudo systemctl restart echo-telegram-bot.service",
                "health_check": self.check_telegram_health,
                "repair_actions": [
                    "sudo systemctl stop echo-telegram-bot.service",
                    "pkill -f patricksecho",
                    "sleep 2",
                    "sudo systemctl start echo-telegram-bot.service",
                ],
            },
            "echo_brain": {
                "name": "AI Assist",
                "check_method": "port",
                "port": 8309,
                "restart_command": "cd /opt/tower-echo-brain && nohup python3 echo_working.py > echo.log 2>&1 &",
                "health_endpoint": "http://localhost:8309/api/echo/health",
            },
            "comfyui": {
                "name": "ComfyUI",
                "check_method": "port",
                "port": 8188,
                "health_endpoint": "http://localhost:8188/system_stats",
            },
            "knowledge_base": {
                "name": "Knowledge Base",
                "check_method": "port",
                "port": 8307,
                "health_endpoint": "http://localhost:8307/health",
            },
        }

        self.repair_history = []
        self.max_repair_attempts = 3

    async def check_telegram_health(self) -> Dict:
        """Check Telegram bot specific health"""
        try:
            # Check for 409 errors in logs
            result = subprocess.run(
                "tail -100 /opt/patricks-echo-bot/bot.log | grep -c '409'",
                shell=True,
                capture_output=True,
                text=True,
            )
            error_count = int(result.stdout.strip() or 0)

            # Check if process is running
            ps_result = subprocess.run(
                "ps aux | grep -c '[p]atricksecho'",
                shell=True,
                capture_output=True,
                text=True,
            )
            process_count = int(ps_result.stdout.strip() or 0)

            return {
                "healthy": error_count < 50 and process_count == 1,
                "error_count": error_count,
                "process_count": process_count,
                "issues": [],
            }
        except Exception as e:
            logger.error(f"Failed to check Telegram health: {e}")
            return {"healthy": False, "error": str(e)}

    async def check_service_port(
        self, port: int, endpoint: Optional[str] = None
    ) -> bool:
        """Check if service is responding on port"""
        try:
            if endpoint:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        endpoint, timeout=aiohttp.ClientTimeout(total=5)
                    ) as resp:
                        return resp.status == 200
            else:
                # Just check if port is listening
                result = subprocess.run(
                    f"lsof -i :{port} | grep LISTEN", shell=True, capture_output=True
                )
                return result.returncode == 0
        except:
            return False

    async def check_systemd_service(self, service_name: str) -> bool:
        """Check systemd service status"""
        try:
            result = subprocess.run(
                f"systemctl is-active {service_name}",
                shell=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip() == "active"
        except:
            return False

    async def diagnose_service(self, service_id: str) -> Dict:
        """Diagnose a specific service"""
        service = self.services.get(service_id)
        if not service:
            return {"error": f"Unknown service: {service_id}"}

        diagnosis = {
            "service": service["name"],
            "timestamp": datetime.now().isoformat(),
            "healthy": False,
            "issues": [],
        }

        # Check based on method
        if service["check_method"] == "systemd":
            is_active = await self.check_systemd_service(service["service_name"])
            diagnosis["healthy"] = is_active
            if not is_active:
                diagnosis["issues"].append("Service not active")

            # Additional health checks
            if "health_check" in service:
                health = await service["health_check"]()
                if not health.get("healthy"):
                    diagnosis["healthy"] = False
                    diagnosis["issues"].extend(health.get("issues", []))
                    if health.get("error_count", 0) > 50:
                        diagnosis["issues"].append(
                            f"High error count: {health['error_count']}"
                        )
                    if health.get("process_count", 0) > 1:
                        diagnosis["issues"].append(
                            f"Multiple processes: {health['process_count']}"
                        )

        elif service["check_method"] == "port":
            port_open = await self.check_service_port(
                service["port"], service.get("health_endpoint")
            )
            diagnosis["healthy"] = port_open
            if not port_open:
                diagnosis["issues"].append(
                    f"Port {service['port']} not responding")

        return diagnosis

    async def repair_service(self, service_id: str) -> Dict:
        """Attempt to repair a service"""
        service = self.services.get(service_id)
        if not service:
            return {"error": f"Unknown service: {service_id}"}

        repair_result = {
            "service": service["name"],
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "actions_taken": [],
        }

        # Execute repair actions
        if "repair_actions" in service:
            for action in service["repair_actions"]:
                logger.info(f"Executing: {action}")
                repair_result["actions_taken"].append(action)
                subprocess.run(action, shell=True)
                await asyncio.sleep(1)
        elif "restart_command" in service:
            logger.info(f"Restarting: {service['restart_command']}")
            repair_result["actions_taken"].append(service["restart_command"])
            subprocess.run(service["restart_command"], shell=True)

        # Wait for service to stabilize
        await asyncio.sleep(5)

        # Check if repair worked
        post_diagnosis = await self.diagnose_service(service_id)
        repair_result["success"] = post_diagnosis.get("healthy", False)
        repair_result["post_diagnosis"] = post_diagnosis

        # Save repair history
        self.repair_history.append(repair_result)

        return repair_result

    async def monitor_loop(self):
        """Main monitoring loop"""
        logger.info("🧠 Echo Service Monitor started")

        while True:
            try:
                # Check all services
                for service_id in self.services:
                    diagnosis = await self.diagnose_service(service_id)

                    if not diagnosis.get("healthy"):
                        logger.warning(
                            f"❌ {diagnosis['service']} unhealthy: {diagnosis.get('issues')}"
                        )

                        # Attempt repair
                        logger.info(
                            f"🔧 Attempting to repair {diagnosis['service']}")
                        repair_result = await self.repair_service(service_id)

                        if repair_result["success"]:
                            logger.info(
                                f"✅ {diagnosis['service']} repaired successfully"
                            )
                        else:
                            logger.error(
                                f"❌ Failed to repair {diagnosis['service']}")
                    else:
                        logger.debug(f"✅ {diagnosis['service']} healthy")

                # Save status to file for Echo to read
                status_file = Path(
                    "/opt/tower-echo-brain/data/service_status.json")
                status_file.parent.mkdir(parents=True, exist_ok=True)

                with open(status_file, "w") as f:
                    json.dump(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "services": {
                                service_id: await self.diagnose_service(service_id)
                                for service_id in self.services
                            },
                            "repair_history": self.repair_history[
                                -10:
                            ],  # Last 10 repairs
                        },
                        f,
                        indent=2,
                    )

            except Exception as e:
                logger.error(f"Monitor loop error: {e}")

            # Wait before next check
            await asyncio.sleep(60)


async def main():
    monitor = ServiceMonitor()
    await monitor.monitor_loop()


if __name__ == "__main__":
    asyncio.run(main())
