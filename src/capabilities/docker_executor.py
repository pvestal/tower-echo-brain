"""
Docker Executor Module
Provides Docker-based code execution for autonomous operations
"""

import docker
import asyncio
import base64
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class DockerExecutor:
    """Execute code in Docker containers"""

    def __init__(self):
        try:
            self.docker_client = docker.from_env()
        except:
            self.docker_client = None
            logger.warning("Docker not available")

    async def execute_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Execute code in a Docker container"""

        if not self.docker_client:
            return {
                'success': False,
                'error': 'Docker not available'
            }

        try:
            # Determine image based on language
            images = {
                'python': 'python:3.9-slim',
                'javascript': 'node:16-slim',
                'bash': 'ubuntu:latest'
            }

            image = images.get(language, 'python:3.9-slim')

            # Use base64 encoding to pass code safely
            encoded_code = base64.b64encode(code.encode()).decode()

            # Prepare command based on language
            if language == 'python':
                command = f"python -c \"import base64; exec(base64.b64decode('{encoded_code}').decode())\""
            elif language == 'javascript':
                command = f"node -e \"eval(Buffer.from('{encoded_code}', 'base64').toString())\""
            else:
                command = f"bash -c \"eval $(echo {encoded_code} | base64 -d)\""

            # Run in container
            container = self.docker_client.containers.run(
                image=image,
                command=command,
                detach=False,
                remove=True,
                stdout=True,
                stderr=True,
                mem_limit='512m',
                cpu_period=100000,
                cpu_quota=50000
            )

            return {
                'success': True,
                'output': container.decode() if isinstance(container, bytes) else str(container),
                'language': language
            }

        except Exception as e:
            logger.error(f"Docker execution failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

# Also export as CodeExecutor for compatibility
CodeExecutor = DockerExecutor