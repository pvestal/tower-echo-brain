#!/usr/bin/env python3
"""
Autonomous Loop v2 - Fixed version with proper error handling and circuit breakers
"""

import asyncio
import os
import logging
import yaml
import aiohttp
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
import random
import subprocess

logger = logging.getLogger(__name__)


class ServiceState(Enum):
    """Service health states"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    CIRCUIT_OPEN = "circuit_open"


@dataclass
class ServiceStatus:
    """Track service status and failures"""
    name: str
    state: ServiceState = ServiceState.UNKNOWN
    last_check: Optional[datetime] = None
    consecutive_failures: int = 0
    last_failure: Optional[datetime] = None
    last_restart: Optional[datetime] = None
    restart_count: int = 0
    backoff_until: Optional[datetime] = None
    circuit_open_until: Optional[datetime] = None
    error_history: List[str] = field(default_factory=list)


class CircuitBreaker:
    """Circuit breaker pattern implementation"""

    def __init__(self, failure_threshold: int = 5, success_threshold: int = 2, timeout: int = 300):
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout
        self.failures = 0
        self.successes = 0
        self.state = "closed"
        self.open_until = None

    def record_success(self):
        """Record a successful operation"""
        self.failures = 0
        if self.state == "half-open":
            self.successes += 1
            if self.successes >= self.success_threshold:
                self.state = "closed"
                self.successes = 0
                logger.info("Circuit breaker closed")

    def record_failure(self):
        """Record a failed operation"""
        self.failures += 1
        self.successes = 0

        if self.failures >= self.failure_threshold:
            self.state = "open"
            self.open_until = datetime.now() + timedelta(seconds=self.timeout)
            logger.warning(f"Circuit breaker opened until {self.open_until}")

    def can_attempt(self) -> bool:
        """Check if we can attempt an operation"""
        if self.state == "closed":
            return True

        if self.state == "open":
            if datetime.now() >= self.open_until:
                self.state = "half-open"
                logger.info("Circuit breaker entering half-open state")
                return True
            return False

        return self.state == "half-open"


class AutonomousLoopV2:
    """Improved autonomous loop with proper error handling"""

    def __init__(self, config_path: str = "/opt/tower-echo-brain/config/autonomous_services.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
        self.service_status: Dict[str, ServiceStatus] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.running = False

        # Initialize service tracking
        self.initialize_services()

    def load_config(self) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            # Return minimal default config
            return {
                'services': {},
                'monitoring': {'check_interval': 60},
                'circuit_breaker': {
                    'failure_threshold': 5,
                    'success_threshold': 2,
                    'timeout': 300
                },
                'backoff': {
                    'max_delay': 3600,
                    'multiplier': 2,
                    'jitter': True
                }
            }

    def initialize_services(self):
        """Initialize service tracking structures"""
        for service_name, service_config in self.config.get('services', {}).items():
            if service_config.get('enabled', False):
                self.service_status[service_name] = ServiceStatus(name=service_name)

                # Create circuit breaker for each service
                cb_config = self.config.get('circuit_breaker', {})
                self.circuit_breakers[service_name] = CircuitBreaker(
                    failure_threshold=cb_config.get('failure_threshold', 5),
                    success_threshold=cb_config.get('success_threshold', 2),
                    timeout=cb_config.get('timeout', 300)
                )

                logger.info(f"Initialized monitoring for {service_name}")

    async def discover_services(self) -> Dict[str, bool]:
        """Dynamically discover which services are actually installed"""
        discovered = {}

        for service_name in self.service_status.keys():
            try:
                # Check if systemd service exists (use --all to see all units)
                result = subprocess.run(
                    ["systemctl", "list-unit-files", service_name],
                    capture_output=True, text=True, timeout=5
                )

                # Service exists if command succeeded and service appears in output
                discovered[service_name] = (result.returncode == 0 and service_name in result.stdout)

                if not discovered[service_name]:
                    # Also check if it's running as a process on a port
                    service_config = self.config['services'][service_name]
                    port = service_config.get('port')
                    if port:
                        # Check if something is listening on the port
                        port_check = subprocess.run(
                            ["ss", "-tlnp"],
                            capture_output=True, text=True, timeout=5
                        )
                        discovered[service_name] = f":{port}" in port_check.stdout

                if not discovered[service_name]:
                    logger.debug(f"Service {service_name} not found on this system")

            except Exception as e:
                logger.error(f"Error discovering {service_name}: {e}")
                discovered[service_name] = False

        return discovered

    def calculate_backoff(self, service_name: str) -> int:
        """Calculate exponential backoff with jitter"""
        status = self.service_status[service_name]
        service_config = self.config['services'][service_name]
        backoff_config = self.config['backoff']

        # Base delay from service config
        base_delay = service_config.get('backoff_base', 30)

        # Exponential backoff based on restart count
        delay = min(
            base_delay * (backoff_config['multiplier'] ** status.restart_count),
            backoff_config['max_delay']
        )

        # Add jitter if configured
        if backoff_config.get('jitter', True):
            delay = delay * (0.5 + random.random())

        return int(delay)

    async def check_service_health(self, service_name: str) -> ServiceState:
        """Check if a service is healthy"""
        service_config = self.config['services'][service_name]
        status = self.service_status[service_name]

        # Check if we're in backoff period
        if status.backoff_until and datetime.now() < status.backoff_until:
            return status.state

        # Check circuit breaker
        if not self.circuit_breakers[service_name].can_attempt():
            return ServiceState.CIRCUIT_OPEN

        try:
            port = service_config['port']
            endpoint = service_config.get('health_endpoint', '/health')
            url = f"http://localhost:{port}{endpoint}"

            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        self.circuit_breakers[service_name].record_success()
                        status.consecutive_failures = 0
                        return ServiceState.HEALTHY
                    else:
                        raise Exception(f"Unhealthy status: {response.status}")

        except asyncio.TimeoutError:
            error_msg = f"Health check timeout for {service_name}"
            logger.warning(error_msg)
            status.error_history.append(error_msg)
            status.consecutive_failures += 1
            self.circuit_breakers[service_name].record_failure()
            return ServiceState.UNHEALTHY

        except Exception as e:
            error_msg = f"Health check failed for {service_name}: {str(e)}"
            logger.debug(error_msg)  # Debug level since many services don't have health endpoints
            status.error_history.append(error_msg)

            # Keep only last 10 errors
            if len(status.error_history) > 10:
                status.error_history = status.error_history[-10:]

            status.consecutive_failures += 1
            self.circuit_breakers[service_name].record_failure()
            return ServiceState.UNHEALTHY

    async def apply_auto_fix(self, service_name: str) -> bool:
        """Apply automatic fixes for known issues"""
        service_config = self.config['services'][service_name]
        auto_fixes = service_config.get('auto_fix', [])

        if not auto_fixes:
            return False

        try:
            # Check recent logs for known issues
            result = subprocess.run(
                ["journalctl", "-u", service_name, "-n", "20", "--no-pager"],
                capture_output=True, text=True, timeout=5
            )

            for fix in auto_fixes:
                if fix['condition'] in result.stdout:
                    logger.info(f"Applying auto-fix '{fix['type']}' for {service_name}")

                    if fix['action'] == 'fix_kb_password':
                        # Fix Knowledge Base password
                        config_file = "/opt/tower-kb/bin/kb_postgresql.py"
                        subprocess.run([
                            "sed", "-i",
                            "s/'password': '.*'/'password': os.getenv("TOWER_DB_PASSWORD", "")/",
                            config_file
                        ], timeout=5)
                        return True

                    # Add more auto-fix actions as needed

        except Exception as e:
            logger.error(f"Auto-fix failed for {service_name}: {e}")

        return False

    async def restart_service(self, service_name: str) -> bool:
        """Restart a service with proper error handling"""
        service_config = self.config['services'][service_name]
        status = self.service_status[service_name]

        # Check restart limits
        max_retries = service_config.get('max_retries', 3)
        if status.restart_count >= max_retries:
            logger.error(f"Service {service_name} exceeded max retries ({max_retries})")
            # Set long backoff
            status.backoff_until = datetime.now() + timedelta(hours=1)
            return False

        try:
            # Try auto-fix first
            if await self.apply_auto_fix(service_name):
                logger.info(f"Applied auto-fix for {service_name}")

            # Restart the service
            restart_cmd = service_config.get('restart_command', f"sudo systemctl restart {service_name}")
            logger.info(f"Restarting {service_name}: {restart_cmd}")

            result = subprocess.run(
                restart_cmd.split(),
                capture_output=True, text=True, timeout=30
            )

            if result.returncode == 0:
                status.last_restart = datetime.now()
                status.restart_count += 1

                # Set backoff period
                backoff_seconds = self.calculate_backoff(service_name)
                status.backoff_until = datetime.now() + timedelta(seconds=backoff_seconds)

                logger.info(f"Successfully restarted {service_name}, backoff for {backoff_seconds}s")
                return True
            else:
                logger.error(f"Failed to restart {service_name}: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Exception restarting {service_name}: {e}")
            return False

    async def monitor_cycle(self):
        """Single monitoring cycle"""
        # First, discover what's actually installed
        discovered = await self.discover_services()

        for service_name, status in self.service_status.items():
            # Skip if service not installed
            if not discovered.get(service_name, False):
                continue

            # Skip if disabled
            service_config = self.config['services'][service_name]
            if not service_config.get('enabled', False):
                continue

            # Check health
            health = await self.check_service_health(service_name)
            old_state = status.state
            status.state = health
            status.last_check = datetime.now()

            # Handle state changes
            if health == ServiceState.UNHEALTHY and old_state != ServiceState.UNHEALTHY:
                logger.warning(f"Service {service_name} became unhealthy")

                if self.config['monitoring'].get('enable_auto_restart', True):
                    await self.restart_service(service_name)

            elif health == ServiceState.HEALTHY and old_state != ServiceState.HEALTHY:
                logger.info(f"Service {service_name} recovered")
                status.restart_count = 0  # Reset on recovery

    async def start(self):
        """Start the autonomous monitoring loop"""
        logger.info("Starting Autonomous Loop v2 with proper error handling")
        self.running = True

        check_interval = self.config['monitoring'].get('check_interval', 60)

        while self.running:
            try:
                await self.monitor_cycle()
                await asyncio.sleep(check_interval)

            except asyncio.CancelledError:
                logger.info("Autonomous loop cancelled")
                break

            except Exception as e:
                logger.error(f"Error in monitor cycle: {e}")
                await asyncio.sleep(check_interval)

        logger.info("Autonomous loop stopped")

    async def stop(self):
        """Stop the monitoring loop"""
        self.running = False

    def get_status(self) -> Dict[str, Any]:
        """Get current status of all services"""
        return {
            service_name: {
                'state': status.state.value,
                'last_check': status.last_check.isoformat() if status.last_check else None,
                'restart_count': status.restart_count,
                'consecutive_failures': status.consecutive_failures,
                'circuit_breaker': self.circuit_breakers[service_name].state,
                'backoff_until': status.backoff_until.isoformat() if status.backoff_until else None
            }
            for service_name, status in self.service_status.items()
        }


# For testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loop = AutonomousLoopV2()
    asyncio.run(loop.start())