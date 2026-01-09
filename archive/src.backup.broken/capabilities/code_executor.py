"""
Sandboxed Code Execution Module using Docker
Provides secure execution of arbitrary code within containerized environments
"""

import docker
import tempfile
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class SandboxedCodeExecutor:
    """Execute code safely in Docker containers"""

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
        Execute code in a sandboxed Docker container

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

        # Create temporary file with code
        with tempfile.NamedTemporaryFile(mode='w', suffix=self._get_file_extension(language), delete=False) as f:
            f.write(code)
            code_path = f.name

        try:
            # Pull image if not available
            try:
                self.docker_client.images.get(image)
            except docker.errors.ImageNotFound:
                logger.info(f"Pulling Docker image: {image}")
                self.docker_client.images.pull(image)

            # Prepare execution command
            cmd = self._get_execution_command(language, "/code/script")

            # Run container with restrictions
            container = self.docker_client.containers.run(
                image,
                cmd,
                volumes={code_path: {'bind': f'/code/script{self._get_file_extension(language)}', 'mode': 'ro'}},
                mem_limit=memory_limit,
                cpu_period=100000,
                cpu_quota=int(cpu_limit * 100000),
                network_mode="none",  # No network access
                remove=False,
                detach=True,
                environment={
                    "PYTHONUNBUFFERED": "1",
                    "NODE_ENV": "production"
                }
            )

            # Wait for completion with timeout
            start_time = datetime.now()
            try:
                exit_code = container.wait(timeout=timeout)['StatusCode']
                execution_time = (datetime.now() - start_time).total_seconds()
            except docker.errors.APIError:
                container.kill()
                return {
                    "success": False,
                    "error": f"Execution timeout ({timeout}s)",
                    "output": "",
                    "execution_time": timeout,
                    "exec_id": exec_id
                }

            # Get output
            output = container.logs(stdout=True, stderr=False).decode('utf-8')
            errors = container.logs(stdout=False, stderr=True).decode('utf-8')

            # Clean up
            container.remove()
            Path(code_path).unlink()

            # Store execution history
            result = {
                "success": exit_code == 0,
                "output": output,
                "error": errors if errors else None,
                "exit_code": exit_code,
                "execution_time": execution_time,
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

    def _get_execution_command(self, language: str, script_path: str) -> str:
        """Get execution command for language"""
        commands = {
            "python": f"python {script_path}.py",
            "javascript": f"node {script_path}.js",
            "bash": f"sh {script_path}.sh",
            "rust": f"rustc {script_path}.rs -o /tmp/prog && /tmp/prog",
            "go": f"go run {script_path}.go"
        }
        return commands.get(language, f"python {script_path}.py")

    async def execute_with_dependencies(
        self,
        code: str,
        dependencies: List[str],
        language: str = "python"
    ) -> Dict[str, Any]:
        """
        Execute code with specific dependencies installed

        Args:
            code: Code to execute
            dependencies: List of packages to install
            language: Programming language

        Returns:
            Execution results
        """

        # Build custom Dockerfile
        dockerfile = self._build_dockerfile(language, dependencies)

        # Create temporary directory for build context
        with tempfile.TemporaryDirectory() as build_dir:
            build_path = Path(build_dir)

            # Write Dockerfile
            (build_path / "Dockerfile").write_text(dockerfile)

            # Write code file
            (build_path / f"script{self._get_file_extension(language)}").write_text(code)

            # Build custom image
            image_tag = f"echo-brain-exec-{hashlib.sha256(dockerfile.encode()).hexdigest()[:8]}"

            try:
                # Check if image exists in cache
                if image_tag not in self.container_cache:
                    logger.info(f"Building custom image with dependencies: {dependencies}")
                    self.docker_client.images.build(
                        path=str(build_path),
                        tag=image_tag,
                        rm=True
                    )
                    self.container_cache[image_tag] = True

                # Run code in custom image
                container = self.docker_client.containers.run(
                    image_tag,
                    f"{self._get_execution_command(language, '/app/script')}",
                    mem_limit="1g",
                    cpu_period=100000,
                    cpu_quota=200000,  # 2 CPU cores
                    network_mode="none",
                    remove=False,
                    detach=True
                )

                # Wait and get results
                exit_code = container.wait(timeout=60)['StatusCode']
                output = container.logs(stdout=True, stderr=False).decode('utf-8')
                errors = container.logs(stdout=False, stderr=True).decode('utf-8')

                container.remove()

                return {
                    "success": exit_code == 0,
                    "output": output,
                    "error": errors if errors else None,
                    "dependencies": dependencies,
                    "image_tag": image_tag
                }

            except Exception as e:
                logger.error(f"Execution with dependencies failed: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "output": ""
                }

    def _build_dockerfile(self, language: str, dependencies: List[str]) -> str:
        """Build Dockerfile with dependencies"""

        if language == "python":
            deps_cmd = f"RUN pip install --no-cache-dir {' '.join(dependencies)}" if dependencies else ""
            return f"""
FROM python:3.11-slim
WORKDIR /app
{deps_cmd}
COPY script.py /app/
CMD ["python", "/app/script.py"]
"""

        elif language == "javascript":
            deps_cmd = f"RUN npm install {' '.join(dependencies)}" if dependencies else ""
            return f"""
FROM node:18-slim
WORKDIR /app
{deps_cmd}
COPY script.js /app/
CMD ["node", "/app/script.js"]
"""

        else:
            return f"""
FROM alpine:latest
WORKDIR /app
COPY script.sh /app/
CMD ["sh", "/app/script.sh"]
"""

    def cleanup_old_images(self, keep_recent: int = 5):
        """Clean up old custom images"""

        # Get all images with our tag pattern
        custom_images = [
            img for img in self.docker_client.images.list()
            if any(tag.startswith("echo-brain-exec-") for tag in img.tags)
        ]

        # Sort by creation date and remove old ones
        if len(custom_images) > keep_recent:
            for img in custom_images[keep_recent:]:
                try:
                    self.docker_client.images.remove(img.id, force=True)
                    logger.info(f"Removed old image: {img.tags}")
                except Exception as e:
                    logger.warning(f"Failed to remove image: {e}")

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""

        if not self.execution_history:
            return {"total_executions": 0}

        success_count = sum(1 for e in self.execution_history if e.get("success"))

        return {
            "total_executions": len(self.execution_history),
            "successful": success_count,
            "failed": len(self.execution_history) - success_count,
            "average_time": sum(e.get("execution_time", 0) for e in self.execution_history) / len(self.execution_history),
            "languages_used": list(set(e.get("language") for e in self.execution_history if e.get("language"))),
            "last_execution": self.execution_history[-1]["timestamp"] if self.execution_history else None
        }