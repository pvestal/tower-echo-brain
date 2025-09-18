#!/usr/bin/env python3
"""
Tower Testing Framework Integration for Echo Brain
"""

import asyncio
import subprocess
import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class TowerTestingFramework:
    """Tower Testing Framework Integration for Echo Brain"""

    def __init__(self):
        self.framework_path = "/home/patrick/Documents"
        self.tower_script = os.path.join(self.framework_path, "tower")
        self.universal_test_script = os.path.join(self.framework_path, "tower_universal_test.sh")
        self.debug_tools_script = os.path.join(self.framework_path, "tower_debug_tools.sh")
        self.tower_host = "***REMOVED***"

        # Ensure scripts are executable
        for script in [self.tower_script, self.universal_test_script, self.debug_tools_script]:
            if os.path.exists(script):
                os.chmod(script, 0o755)

    async def run_universal_test(self, target: str) -> Dict:
        """Run universal test on a target service"""
        logger.info(f"Running universal test on {target}")

        try:
            start_time = asyncio.get_event_loop().time()

            # Execute universal test script
            result = subprocess.run(
                [self.universal_test_script, target],
                capture_output=True,
                text=True,
                timeout=60
            )

            processing_time = asyncio.get_event_loop().time() - start_time

            return {
                "success": result.returncode == 0,
                "target": target,
                "output": result.stdout,
                "error": result.stderr if result.stderr else None,
                "exit_code": result.returncode,
                "processing_time": processing_time,
                "test_type": "universal"
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "target": target,
                "output": "",
                "error": "Test timed out after 60 seconds",
                "exit_code": -1,
                "processing_time": 60.0,
                "test_type": "universal"
            }
        except Exception as e:
            return {
                "success": False,
                "target": target,
                "output": "",
                "error": str(e),
                "exit_code": -1,
                "processing_time": 0.0,
                "test_type": "universal"
            }

    async def run_debug_analysis(self, target: str) -> Dict:
        """Run debug analysis on a target service"""
        logger.info(f"Running debug analysis on {target}")

        try:
            start_time = asyncio.get_event_loop().time()

            # Execute debug tools script
            result = subprocess.run(
                [self.debug_tools_script, target],
                capture_output=True,
                text=True,
                timeout=120  # Debug takes longer
            )

            processing_time = asyncio.get_event_loop().time() - start_time

            return {
                "success": result.returncode == 0,
                "target": target,
                "output": result.stdout,
                "error": result.stderr if result.stderr else None,
                "exit_code": result.returncode,
                "processing_time": processing_time,
                "test_type": "debug"
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "target": target,
                "output": "",
                "error": "Debug analysis timed out after 120 seconds",
                "exit_code": -1,
                "processing_time": 120.0,
                "test_type": "debug"
            }
        except Exception as e:
            return {
                "success": False,
                "target": target,
                "output": "",
                "error": str(e),
                "exit_code": -1,
                "processing_time": 0.0,
                "test_type": "debug"
            }

    async def run_tower_command(self, command: str, args: List[str] = None) -> Dict:
        """Run a tower framework command"""
        if args is None:
            args = []

        try:
            start_time = asyncio.get_event_loop().time()

            cmd = [self.tower_script, command] + args
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=180
            )

            processing_time = asyncio.get_event_loop().time() - start_time

            return {
                "success": result.returncode == 0,
                "command": " ".join(cmd),
                "output": result.stdout,
                "error": result.stderr if result.stderr else None,
                "exit_code": result.returncode,
                "processing_time": processing_time
            }

        except Exception as e:
            return {
                "success": False,
                "command": " ".join([self.tower_script, command] + args),
                "output": "",
                "error": str(e),
                "exit_code": -1,
                "processing_time": 0.0
            }

    async def get_tower_service_status(self) -> Dict:
        """Get status of all Tower services"""
        services = [
            "tower-dashboard", "tower-auth", "tower-kb", "tower-apple-music",
            "tower-echo-brain", "tower-anime-production", "tower-music-production",
            "tower-personal-media"
        ]

        service_status = {}
        for service in services:
            try:
                result = subprocess.run(
                    ["systemctl", "is-active", service],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                service_status[service] = {
                    "status": result.stdout.strip(),
                    "active": result.returncode == 0
                }
            except Exception as e:
                service_status[service] = {
                    "status": "error",
                    "active": False,
                    "error": str(e)
                }

        return service_status

    async def get_tower_health_summary(self) -> Dict:
        """Get overall Tower health summary"""
        service_status = await self.get_tower_service_status()

        total_services = len(service_status)
        active_services = sum(1 for s in service_status.values() if s.get("active", False))

        # Check key ports
        key_ports = {
            "8309": "echo-brain",
            "8305": "anime-production",
            "8307": "knowledge-base",
            "8088": "auth",
            "8188": "comfyui"
        }

        port_status = {}
        for port, service in key_ports.items():
            try:
                result = subprocess.run(
                    ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
                     f"http://127.0.0.1:{port}/api/health"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                port_status[service] = {
                    "port": port,
                    "accessible": result.stdout.strip() in ["200", "404"],  # 404 is ok if endpoint different
                    "http_code": result.stdout.strip()
                }
            except Exception as e:
                port_status[service] = {
                    "port": port,
                    "accessible": False,
                    "error": str(e)
                }

        return {
            "services": service_status,
            "ports": port_status,
            "summary": {
                "total_services": total_services,
                "active_services": active_services,
                "service_health": f"{active_services}/{total_services}",
                "overall_status": "healthy" if active_services >= total_services * 0.8 else "degraded"
            }
        }

# Global testing framework instance
testing_framework = TowerTestingFramework()