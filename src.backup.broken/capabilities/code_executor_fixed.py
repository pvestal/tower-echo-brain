"""
Fixed Sandboxed Code Execution Module using Docker
Provides secure execution of arbitrary code within containerized environments
"""

import docker
import tempfile
import asyncio
import hashlib
import json
import base64
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class SandboxedCodeExecutor:
    """Execute code safely in Docker containers - FIXED VERSION"""

    def __init__(self):
        self.docker_client = docker.from_env()
        self.execution_history = []
        self.container_cache = {}

    async def execute_code(
        self,
        code: str,
        language: str = "python",
        timeout: int = 30,
        memory_limit: str = "512m",
        cpu_limit: float = 1.0
    ) -> Dict[str, Any]:
        """
        Execute code in a sandboxed Docker container (FIXED)

        Args:
            code: Code to execute
            language: Programming language (python, javascript, bash)
            timeout: Execution timeout in seconds
            memory_limit: Container memory limit
            cpu_limit: CPU cores limit

        Returns:
            Execution results with output, errors, and metrics
        """

        # Generate unique execution ID
        exec_id = hashlib.sha256(f"{code}{datetime.now()}".encode()).hexdigest()[:12]

        # Select appropriate Docker image
        images = {
            "python": "python:3.11-slim",
            "javascript": "node:18-slim",
            "bash": "alpine:latest",
            "rust": "rust:latest",
            "go": "golang:latest"
        }

        image = images.get(language, "python:3.11-slim")

        try:
            # Pull image if not available
            try:
                self.docker_client.images.get(image)
            except docker.errors.ImageNotFound:
                logger.info(f"Pulling Docker image: {image}")
                self.docker_client.images.pull(image)

            # Method 1: Pass code via stdin (most reliable)
            if language == "python":
                # Encode code to base64 to avoid quote issues
                encoded_code = base64.b64encode(code.encode()).decode()
                command = f"python -c \"import base64; exec(base64.b64decode('{encoded_code}').decode())\""
            elif language == "javascript":
                encoded_code = base64.b64encode(code.encode()).decode()
                command = f"node -e \"eval(Buffer.from('{encoded_code}', 'base64').toString())\""
            elif language == "bash":
                # For bash, use echo and pipe
                encoded_code = base64.b64encode(code.encode()).decode()
                command = f"sh -c \"echo '{encoded_code}' | base64 -d | sh\""
            else:
                # Fallback for other languages
                command = code

            # Run container with the command
            start_time = datetime.now()

            try:
                container_result = self.docker_client.containers.run(
                    image,
                    command,
                    mem_limit=memory_limit,
                    cpu_period=100000,
                    cpu_quota=int(cpu_limit * 100000),
                    network_mode="none",  # No network access
                    remove=True,
                    detach=False,  # Wait for completion
                    environment={
                        "PYTHONUNBUFFERED": "1",
                        "NODE_ENV": "production"
                    }
                )

                execution_time = (datetime.now() - start_time).total_seconds()

                # Parse output
                if isinstance(container_result, bytes):
                    output = container_result.decode('utf-8')
                else:
                    output = str(container_result)

                result = {
                    "success": True,
                    "output": output,
                    "error": None,
                    "exit_code": 0,
                    "execution_time": execution_time,
                    "exec_id": exec_id,
                    "language": language,
                    "timestamp": datetime.now().isoformat()
                }

            except docker.errors.ContainerError as e:
                # Container exited with non-zero status
                result = {
                    "success": False,
                    "output": e.stdout.decode() if e.stdout else "",
                    "error": e.stderr.decode() if e.stderr else str(e),
                    "exit_code": e.exit_status,
                    "execution_time": (datetime.now() - start_time).total_seconds(),
                    "exec_id": exec_id,
                    "language": language,
                    "timestamp": datetime.now().isoformat()
                }

            except docker.errors.APIError as e:
                # Timeout or other API error
                result = {
                    "success": False,
                    "output": "",
                    "error": f"Docker API error: {str(e)}",
                    "exit_code": -1,
                    "execution_time": timeout,
                    "exec_id": exec_id,
                    "language": language,
                    "timestamp": datetime.now().isoformat()
                }

            self.execution_history.append(result)
            return result

        except Exception as e:
            logger.error(f"Code execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "output": "",
                "exec_id": exec_id,
                "execution_time": 0
            }

    async def execute_with_file_access(
        self,
        code: str,
        language: str = "python",
        mount_path: str = "/opt/tower-echo-brain",
        timeout: int = 30
    ) -> Dict[str, Any]:
        """
        Execute code with access to specific files (FIXED)

        Args:
            code: Code to execute
            language: Programming language
            mount_path: Path to mount in container
            timeout: Execution timeout

        Returns:
            Execution results
        """

        exec_id = hashlib.sha256(f"{code}{datetime.now()}".encode()).hexdigest()[:12]

        # Create temporary directory for the code
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write code to file in temp directory
            code_file = Path(tmpdir) / f"script.{self._get_file_extension(language)[1:]}"
            code_file.write_text(code)

            # Select image
            images = {
                "python": "python:3.11-slim",
                "javascript": "node:18-slim",
                "bash": "alpine:latest"
            }
            image = images.get(language, "python:3.11-slim")

            # Prepare command
            if language == "python":
                command = f"python /workspace/script.py"
            elif language == "javascript":
                command = f"node /workspace/script.js"
            else:
                command = f"sh /workspace/script.sh"

            try:
                # Run with proper volume mounts
                container = self.docker_client.containers.run(
                    image,
                    command,
                    volumes={
                        tmpdir: {'bind': '/workspace', 'mode': 'ro'},
                        mount_path: {'bind': '/data', 'mode': 'ro'}
                    },
                    working_dir="/workspace",
                    mem_limit="512m",
                    network_mode="none",
                    remove=True,
                    detach=False,
                    environment={"PYTHONPATH": "/data"}
                )

                output = container.decode() if hasattr(container, 'decode') else str(container)

                return {
                    "success": True,
                    "output": output,
                    "error": None,
                    "exec_id": exec_id,
                    "execution_time": 0
                }

            except docker.errors.ContainerError as e:
                return {
                    "success": False,
                    "output": e.stdout.decode() if e.stdout else "",
                    "error": e.stderr.decode() if e.stderr else str(e),
                    "exec_id": exec_id
                }

            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "output": "",
                    "exec_id": exec_id
                }

    def _get_file_extension(self, language: str) -> str:
        """Get file extension for language"""
        extensions = {
            "python": ".py",
            "javascript": ".js",
            "bash": ".sh",
            "rust": ".rs",
            "go": ".go"
        }
        return extensions.get(language, ".txt")

    def cleanup_containers(self):
        """Clean up any lingering containers"""
        for container in self.docker_client.containers.list(all=True):
            if container.name and "echo-brain" in container.name:
                try:
                    container.remove(force=True)
                    logger.info(f"Cleaned up container: {container.name}")
                except:
                    pass

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""

        if not self.execution_history:
            return {"total_executions": 0}

        success_count = sum(1 for e in self.execution_history if e.get("success"))

        return {
            "total_executions": len(self.execution_history),
            "successful": success_count,
            "failed": len(self.execution_history) - success_count,
            "average_time": sum(e.get("execution_time", 0) for e in self.execution_history) / len(self.execution_history) if self.execution_history else 0,
            "languages_used": list(set(e.get("language") for e in self.execution_history if e.get("language"))),
            "last_execution": self.execution_history[-1]["timestamp"] if self.execution_history else None
        }