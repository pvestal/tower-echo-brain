#!/usr/bin/env python3
"""
Tower Services Monitor - Lightweight autonomous monitoring
Monitors critical Tower services and auto-fixes common issues
"""

import asyncio
import os
import logging
import subprocess
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class TowerServicesMonitor:
    """Monitors Tower services health and performs auto-recovery"""

    def __init__(self):
        self.services = [
            "tower-dashboard",
            "tower-auth",
            "tower-kb",
            "tower-apple-music",
            "tower-echo-brain",
            "tower-anime-production"
        ]

        self.check_interval = 60  # Check every minute
        self.max_restart_attempts = 3
        self.restart_counts: Dict[str, int] = {}
        self.known_fixes = {
            "tower-kb": self._fix_kb_password,
            "tower-auth": self._fix_auth_service
        }

    async def _fix_kb_password(self) -> bool:
        """Fix common KB password issue"""
        try:
            # Check if it's a password issue
            result = subprocess.run(
                ["journalctl", "-u", "tower-kb", "-n", "5", "--no-pager"],
                capture_output=True, text=True
            )

            if "password authentication failed" in result.stdout:
                logger.info("🔧 Detected KB password issue, applying fix...")

                # Update password in config
                config_path = "/opt/tower-kb/bin/kb_postgresql.py"
                with open(config_path, 'r') as f:
                    content = f.read()

                if "RP78eIrW7cI2jYvL5akt1yurE" in content:
                    content = content.replace(
                        "'password': 'RP78eIrW7cI2jYvL5akt1yurE'",
                        "'password': os.getenv("TOWER_DB_PASSWORD", "RP78eIrW7cI2jYvL5akt1yurE")"
                    )
                    with open(config_path, 'w') as f:
                        f.write(content)
                    logger.info("✅ KB password fixed")
                    return True

        except Exception as e:
            logger.error(f"Failed to fix KB password: {e}")
        return False

    async def _fix_auth_service(self) -> bool:
        """Fix common auth service issues"""
        try:
            # Check if vault token is expired
            result = subprocess.run(
                ["journalctl", "-u", "tower-auth", "-n", "5", "--no-pager"],
                capture_output=True, text=True
            )

            if "vault" in result.stdout.lower() or "token" in result.stdout.lower():
                logger.info("🔧 Detected auth token issue, refreshing...")
                # Refresh vault token
                subprocess.run(["vault", "token", "renew"], check=False)
                return True

        except Exception as e:
            logger.error(f"Failed to fix auth service: {e}")
        return False

    async def check_service(self, service: str) -> bool:
        """Check if a service is healthy"""
        try:
            result = subprocess.run(
                ["systemctl", "is-active", service],
                capture_output=True, text=True
            )
            return result.stdout.strip() == "active"
        except Exception as e:
            logger.error(f"Error checking {service}: {e}")
            return False

    async def restart_service(self, service: str) -> bool:
        """Restart a failed service with auto-fix if available"""
        try:
            # Try known fixes first
            if service in self.known_fixes:
                logger.info(f"🔧 Attempting auto-fix for {service}")
                if await self.known_fixes[service]():
                    logger.info(f"✅ Auto-fix applied for {service}")

            # Restart the service
            logger.info(f"🔄 Restarting {service}...")
            result = subprocess.run(
                ["sudo", "systemctl", "restart", service],
                capture_output=True, text=True
            )

            if result.returncode == 0:
                await asyncio.sleep(3)  # Give service time to start
                if await self.check_service(service):
                    logger.info(f"✅ {service} restarted successfully")
                    self.restart_counts[service] = 0
                    return True

            logger.error(f"❌ Failed to restart {service}")
            return False

        except Exception as e:
            logger.error(f"Error restarting {service}: {e}")
            return False

    async def monitor_loop(self):
        """Main monitoring loop"""
        logger.info("🚀 Tower Services Monitor starting...")

        while True:
            try:
                for service in self.services:
                    is_healthy = await self.check_service(service)

                    if not is_healthy:
                        logger.warning(f"⚠️ {service} is not healthy")

                        # Check restart attempts
                        if service not in self.restart_counts:
                            self.restart_counts[service] = 0

                        if self.restart_counts[service] < self.max_restart_attempts:
                            self.restart_counts[service] += 1
                            await self.restart_service(service)
                        else:
                            logger.error(f"❌ {service} exceeded max restart attempts")

                await asyncio.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(self.check_interval)

    async def start(self):
        """Start the monitor with safety wrapper"""
        try:
            # Run with timeout to prevent blocking
            await asyncio.wait_for(
                self.monitor_loop(),
                timeout=None  # Run indefinitely but can be cancelled
            )
        except asyncio.CancelledError:
            logger.info("🛑 Tower Services Monitor stopped")
            raise
        except Exception as e:
            logger.error(f"Monitor failed: {e}")

# Standalone runner for testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    monitor = TowerServicesMonitor()
    asyncio.run(monitor.start())