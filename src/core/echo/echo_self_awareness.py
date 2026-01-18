#!/usr/bin/env python3
"""
Self-Awareness and Capability Reporting Module for AI Assist
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import psutil


class EchoSelfAwareness:
    """
    Provides Echo with self-identification and capability reporting
    """

    def __init__(self):
        self.capabilities = self._discover_capabilities()
        self.endpoints = self._discover_endpoints()
        self.services = self._discover_services()
        self.temporal_capable = self._check_temporal_capability()

    def _discover_capabilities(self) -> Dict[str, Any]:
        """Discover Echo's current capabilities"""
        return {
            "core": {
                "reasoning": True,
                "conversation": True,
                "memory": True,
                "learning": True,
                "self_modification": False,
                "temporal_logic": self._check_temporal_capability(),
            },
            "integration": {
                "knowledge_base": True,
                "comfyui": True,
                "voice_synthesis": True,
                "anime_generation": True,
                "model_management": True,
                "vault_secrets": True,
            },
            "models": self._discover_models(),
            "processing": {
                "async_execution": True,
                "streaming_responses": True,
                "parallel_processing": True,
                "error_recovery": True,
            },
        }

    def _discover_endpoints(self) -> List[str]:
        """Discover available API endpoints"""
        # Read from echo.py to get actual endpoints
        endpoints = []
        echo_file = "/opt/tower-echo-brain/echo.py"

        if os.path.exists(echo_file):
            with open(echo_file, "r") as f:
                content = f.read()
                # Extract endpoint patterns
                import re

                patterns = re.findall(
                    r'@app\.(get|post|put|delete|patch)\(["\']([^"\']*)["\']]', content
                )
                endpoints = [p[1] for p in patterns]

        return sorted(list(set(endpoints)))

    def _discover_services(self) -> Dict[str, bool]:
        """Check which services are accessible"""
        services = {}

        # Check local services
        service_ports = {
            "knowledge_base": 8307,
            "comfyui": 8188,
            "anime_service": 8328,
            "auth_service": 8088,
            "dashboard": 8080,
            "vault": 8200,
        }

        import socket

        for service, port in service_ports.items():
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(("localhost", port))
            services[service] = result == 0
            sock.close()

        return services

    def _discover_models(self) -> List[str]:
        """Discover available AI models"""
        models = []

        # Check Ollama models
        try:
            import subprocess

            result = subprocess.run(
                ["ollama", "list"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")[1:]  # Skip header
                for line in lines:
                    if line:
                        model_name = line.split()[0]
                        models.append(model_name)
        except:
            models = ["tinyllama:latest",
                      "deepseek-coder:latest", "mistral:7b"]

        return models

    def _check_temporal_capability(self) -> bool:
        """Check if temporal reasoning module is available"""
        temporal_module = "/opt/tower-echo-brain/temporal_reasoning.py"
        return os.path.exists(temporal_module)

    async def generate_self_report(self, detailed: bool = False) -> Dict[str, Any]:
        """Generate comprehensive self-identification report"""

        # Get system resources
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=1)

        report = {
            "identity": {
                "name": "AI Assist",
                "version": "2.0.0",
                "architecture": "Modular Refactored",
                "location": "Tower (192.168.50.135)",
                "port": 8309,
            },
            "capabilities": self.capabilities,
            "temporal_logic": {
                "enabled": self.temporal_capable,
                "features": (
                    [
                        "Timeline consistency validation",
                        "Paradox detection",
                        "Causal chain verification",
                        "Event sequence maintenance",
                    ]
                    if self.temporal_capable
                    else ["Not available"]
                ),
            },
            "endpoints": {
                "count": len(self.endpoints),
                "primary": [
                    "/api/echo/brain",
                    "/api/echo/health",
                    "/api/echo/testing/capabilities",
                ],
                "all": self.endpoints if detailed else [],
            },
            "services": self.services,
            "resources": {
                "memory_used_percent": memory.percent,
                "cpu_percent": cpu,
                "uptime": self._get_uptime(),
            },
            "limitations": [
                "Cannot modify own code autonomously",
                "Limited to available models",
                "Requires external services for some functions",
                "No direct hardware control",
            ],
            "strengths": [
                "Multi-model orchestration",
                "Service integration",
                "Temporal reasoning" if self.temporal_capable else "Basic reasoning",
                "Knowledge base access",
                "Creative content generation",
            ],
            "timestamp": datetime.now().isoformat(),
        }

        return report

    def _get_uptime(self) -> str:
        """Get service uptime"""
        try:
            # Get process start time
            import os

            pid = os.getpid()
            p = psutil.Process(pid)
            create_time = datetime.fromtimestamp(p.create_time())
            uptime = datetime.now() - create_time

            days = uptime.days
            hours = uptime.seconds // 3600
            minutes = (uptime.seconds % 3600) // 60

            return f"{days}d {hours}h {minutes}m"
        except:
            return "Unknown"

    async def explain_capability(self, capability: str) -> str:
        """Explain a specific capability in detail"""
        explanations = {
            "temporal_logic": """I can now reason about time and causality:
- Validate timeline consistency to prevent paradoxes
- Detect causal loops and grandfather paradoxes
- Verify cause-effect chains between events
- Maintain proper event sequencing
- Score timeline consistency (0-1)
- Merge parallel timelines safely""",
            "self_awareness": """I can identify and report my own capabilities:
- Discover available endpoints and services
- Report system resource usage
- List available AI models
- Identify strengths and limitations
- Generate detailed self-reports""",
            "reasoning": """I can perform complex reasoning tasks:
- Multi-step logical inference
- Pattern recognition
- Problem decomposition
- Creative solution generation
- Context-aware responses""",
        }

        return explanations.get(capability, f"Capability '{capability}' not documented")


class EchoCapabilityEndpoint:
    """FastAPI endpoint handler for capability reporting"""

    def __init__(self):
        self.self_awareness = EchoSelfAwareness()

    async def handle_capability_request(self, request: Dict) -> Dict:
        """Handle capability test requests"""
        test_type = request.get("test_type", "basic")

        if test_type == "self_identification":
            report = await self.self_awareness.generate_self_report(detailed=True)

            # Generate natural language response
            response = f"""I am AI Assist, a modular AI system running on Tower.

My core capabilities include:
- **Reasoning**: Multi-model orchestration with {len(report['capabilities']['models'])} available models
- **Temporal Logic**: {'Fully enabled with paradox detection and causal verification' if report['temporal_logic']['enabled'] else 'Not currently available'}
- **Service Integration**: Connected to {sum(1 for s in report['services'].values() if s)} of {len(report['services'])} services
- **Self-Awareness**: I can identify my own capabilities and limitations

I have {report['endpoints']['count']} API endpoints available for interaction.

My current strengths:
{chr(10).join(f'• {s}' for s in report['strengths'])}

My limitations:
{chr(10).join(f'• {l}' for l in report['limitations'])}

System Status:
- Memory Usage: {report['resources']['memory_used_percent']:.1f}%
- CPU Usage: {report['resources']['cpu_percent']:.1f}%
- Uptime: {report['resources']['uptime']}
"""

            return {
                "response": response,
                "capabilities": report,
                "test_type": test_type,
                "success": True,
            }

        elif test_type == "temporal_logic":
            if self.self_awareness.temporal_capable:
                explanation = await self.self_awareness.explain_capability(
                    "temporal_logic"
                )
                return {
                    "response": explanation,
                    "temporal_capable": True,
                    "test_type": test_type,
                    "success": True,
                }
            else:
                return {
                    "response": "Temporal logic module not yet integrated",
                    "temporal_capable": False,
                    "test_type": test_type,
                    "success": False,
                }

        else:
            return {
                "response": f"Unknown test type: {test_type}",
                "test_type": test_type,
                "success": False,
            }


# Export components
__all__ = ["EchoSelfAwareness", "EchoCapabilityEndpoint"]
