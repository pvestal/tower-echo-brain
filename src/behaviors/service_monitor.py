#!/usr/bin/env python3
"""
Service monitoring behavior for Echo Brain
"""
import asyncio
import logging
import subprocess
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List
from src.db.database import database
from src.tasks.task_queue import TaskQueue, create_monitoring_task

logger = logging.getLogger(__name__)

class ServiceMonitor:
    """Monitors Tower services and creates repair tasks"""

    def __init__(self, task_queue: TaskQueue):
        self.task_queue = task_queue
        self.service_name_map = {
            'echo': 'tower-echo-brain.service',
            'telegram_bot': 'echo-telegram-bot.service',
            'auth': 'tower-auth.service',
            'comfyui': 'comfyui.service',
            'knowledge_base': 'tower-kb.service',
            'anime': 'tower-anime-production.service',
            'apple_music': 'tower-apple-music.service',
            'notifications': 'tower-notification-bot.service',
            'vault': 'vault.service',
            'veteran_guardian': 'veteran-guardian-bot.service'
        }
        self.restart_cooldowns = {}
        self.last_checked = datetime.now()

    async def monitor_services(self):
        """Main monitoring loop for Tower services"""
        try:
            current_time = datetime.now()
            services_to_check = ['echo', 'comfyui', 'auth', 'knowledge_base', 'anime', 'apple_music', 'notifications', 'vault', 'veteran_guardian']

            for service in services_to_check:
                if await self._check_restart_cooldown(service):
                    continue  # Skip if in cooldown

                # Check systemd service status
                status = await self._check_service_status(service)
                if not status['active']:
                    await self._handle_failed_service(service, status)

                # Check service endpoints
                endpoint_status = await self._check_service_endpoint(service)
                if not endpoint_status['responsive']:
                    await self._handle_unresponsive_service(service, endpoint_status)

            self.last_checked = current_time
            logger.info(f"✅ Service monitoring completed at {current_time}")

        except Exception as e:
            logger.error(f"❌ Service monitoring failed: {e}")

    async def _check_service_status(self, service: str) -> Dict:
        """Check systemd service status"""
        try:
            systemd_name = self.service_name_map.get(service, f"{service}.service")
            result = subprocess.run(['systemctl', 'is-active', systemd_name], capture_output=True, text=True)

            return {
                'active': result.returncode == 0,
                'status': result.stdout.strip(),
                'systemd_name': systemd_name
            }
        except Exception as e:
            logger.error(f"Failed to check {service} status: {e}")
            return {'active': False, 'status': 'error', 'error': str(e)}

    async def _check_service_endpoint(self, service: str) -> Dict:
        """Check service HTTP endpoint"""
        endpoints = {
            'echo': 'http://localhost:8309/api/echo/health',
            'auth': 'http://localhost:8088/api/auth/health',
            'knowledge_base': 'http://localhost:8307/api/kb/health',
            'anime': 'http://localhost:8328/api/anime/health',
            'apple_music': 'http://localhost:8315/api/apple-music/health',
            'comfyui': 'http://localhost:8188/system_stats'
        }

        endpoint = endpoints.get(service)
        if not endpoint:
            return {'responsive': True, 'reason': 'no_endpoint_defined'}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(endpoint, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    return {
                        'responsive': response.status == 200,
                        'status_code': response.status,
                        'endpoint': endpoint
                    }
        except Exception as e:
            return {
                'responsive': False,
                'error': str(e),
                'endpoint': endpoint
            }

    async def _handle_failed_service(self, service: str, status: Dict):
        """Handle a failed service by creating repair task"""
        logger.warning(f"🚨 Service {service} is not active: {status}")

        task = await create_monitoring_task(
            f"Service {service} failed - status: {status.get('status', 'unknown')}",
            {'service': service, 'status': status, 'repair_type': 'service_restart'}
        )

        await self.task_queue.add_task_object(task)
        await self._record_restart(service)

    async def _handle_unresponsive_service(self, service: str, endpoint_status: Dict):
        """Handle an unresponsive service endpoint"""
        logger.warning(f"🚨 Service {service} endpoint unresponsive: {endpoint_status}")

        task = await create_monitoring_task(
            f"Service {service} endpoint unresponsive",
            {'service': service, 'endpoint_status': endpoint_status, 'repair_type': 'endpoint_restart'}
        )

        await self.task_queue.add_task_object(task)

    async def _check_restart_cooldown(self, service_name: str) -> bool:
        """Check if service is in restart cooldown period"""
        if service_name not in self.restart_cooldowns:
            return False

        last_restart = self.restart_cooldowns[service_name]
        cooldown_period = timedelta(minutes=5)

        if datetime.now() - last_restart < cooldown_period:
            logger.debug(f"⏰ Service {service_name} in cooldown period")
            return True
        else:
            del self.restart_cooldowns[service_name]
            return False

    async def _record_restart(self, service_name: str):
        """Record service restart timestamp"""
        self.restart_cooldowns[service_name] = datetime.now()

        try:
            await database.log_repair_action(
                'service_restart',
                service_name,
                {'timestamp': datetime.now().isoformat(), 'cooldown_applied': True}
            )
        except Exception as e:
            logger.error(f"Failed to log restart for {service_name}: {e}")

    def get_monitor_stats(self) -> Dict:
        """Get monitoring statistics"""
        return {
            'services_monitored': len(self.service_name_map),
            'active_cooldowns': len(self.restart_cooldowns),
            'last_check': self.last_checked.isoformat() if self.last_checked else None,
            'cooldown_services': list(self.restart_cooldowns.keys())
        }