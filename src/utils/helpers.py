"""
Utility functions and helper classes for Echo Brain system
"""

import asyncio
import aiohttp
import subprocess
import os
import re
import shlex
import logging
import time
import json
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class SafeShellExecutor:
    """Safe shell command execution with security controls"""

    def __init__(self):
        self.allowed_commands = [
            'ls', 'pwd', 'echo', 'cat', 'grep', 'find', 'which',
            'python', 'python3', 'pip', 'pip3', 'node', 'npm', 'pnpm',
            'git', 'curl', 'wget', 'ps', 'kill', 'pkill',
            'df', 'du', 'free', 'top', 'htop', 'lsof', 'netstat',
            'sudo systemctl start', 'sudo systemctl stop', 'sudo systemctl restart',
            'sudo systemctl status', 'sudo systemctl enable', 'sudo systemctl disable',
            'sudo systemctl daemon-reload', 'sudo journalctl', 'sudo pkill',
            'sudo killall', 'sudo ufw', 'wc', 'head', 'tail', 'sort', 'uniq',
            'mkdir', 'touch', 'cp', 'mv', 'nano', 'vim', 'tee',
            'chmod', 'chown', 'ln', 'dirname', 'basename', 'realpath',
            'sh', 'bash'
        ]
        self.forbidden_patterns = [
            r';\s*rm\s+-rf',
            r'>\s*/dev/.*',
            r'mkfs',
            r'dd\s+if=.*of=/dev/',
            r'format\s+[cC]:',
            r'del\s+/[sS]'
        ]

    async def execute(self, command: str, timeout: int = 30, allow_all: bool = False) -> dict:
        """
        Execute shell command with safety controls

        Args:
            command: Shell command to execute
            timeout: Maximum execution time in seconds
            allow_all: Bypass safety checks (use with caution)

        Returns:
            Dictionary with execution results
        """
        start_time = asyncio.get_event_loop().time()

        # Safety checks
        safety_checks = {
            'command_allowed': True,
            'patterns_safe': True,
            'timeout_valid': timeout <= 300
        }

        if not allow_all:
            # Check if command or command prefix is allowed
            command_allowed = False
            for allowed_cmd in self.allowed_commands:
                if command.startswith(allowed_cmd + ' ') or command == allowed_cmd:
                    command_allowed = True
                    break

            if not command_allowed:
                logger.warning(f"Command '{command}' not in allowed list")
                safety_checks['command_allowed'] = False

            # Check for forbidden patterns
            for pattern in self.forbidden_patterns:
                if re.search(pattern, command):
                    logger.error(f"Forbidden pattern detected: {pattern}")
                    safety_checks['patterns_safe'] = False
                    break

            # Enforce safety
            if not all(safety_checks.values()):
                return {
                    'success': False,
                    'error': 'Command failed safety checks',
                    'safety_checks': safety_checks,
                    'stdout': '',
                    'stderr': '',
                    'exit_code': -1,
                    'processing_time': asyncio.get_event_loop().time() - start_time
                }

        try:
            # Check if command needs shell execution (contains shell operators)
            shell_operators = ['>', '>>', '|', '&&', '||', ';', '$(', '`']
            needs_shell = any(op in command for op in shell_operators)

            if needs_shell:
                # Execute through shell for redirection, pipes, etc.
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=os.getcwd()
                )
            else:
                # Parse command safely for direct execution
                if isinstance(command, str):
                    args = shlex.split(command)
                else:
                    args = command

                # Execute directly
                process = await asyncio.create_subprocess_exec(
                    *args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=os.getcwd()
                )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )

                return {
                    'success': process.returncode == 0,
                    'stdout': stdout.decode('utf-8', errors='replace'),
                    'stderr': stderr.decode('utf-8', errors='replace'),
                    'exit_code': process.returncode,
                    'processing_time': asyncio.get_event_loop().time() - start_time,
                    'safety_checks': safety_checks
                }

            except asyncio.TimeoutError:
                process.kill()
                await process.communicate()  # Clean up
                return {
                    'success': False,
                    'error': f'Command timed out after {timeout} seconds',
                    'stdout': '',
                    'stderr': '',
                    'exit_code': -1,
                    'processing_time': asyncio.get_event_loop().time() - start_time,
                    'safety_checks': safety_checks
                }

        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'stdout': '',
                'stderr': '',
                'exit_code': -1,
                'processing_time': asyncio.get_event_loop().time() - start_time,
                'safety_checks': safety_checks
            }

    async def execute_python(self, code: str, timeout: int = 30) -> dict:
        """Execute Python code safely"""
        # Write code to temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name

        try:
            result = await self.execute(f"python3 {temp_file}", timeout=timeout)
            return result
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file)
            except:
                pass

    async def execute_with_retry(self, command: str, max_retries: int = 3,
                                timeout: int = 30) -> dict:
        """Execute command with retry logic"""
        for attempt in range(max_retries):
            result = await self.execute(command, timeout=timeout)

            if result['success']:
                return result

            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                logger.info(f"Retrying command (attempt {attempt + 2}/{max_retries})")

        return result


# TowerOrchestrator now uses resilient implementation
# Old code archived in helpers_tower_orchestrator_archived.py
from src.orchestrators.resilient_orchestrator import ResilientOrchestrator as TowerOrchestrator


def format_response_for_web(response: str) -> str:
    """Format response text for web display"""
    # Add basic HTML formatting
    response = response.replace('\n', '<br>')
    response = response.replace('```', '<pre>')
    response = response.replace('**', '<strong>')
    return response


def extract_code_blocks(text: str) -> List[Dict]:
    """Extract code blocks from text"""
    code_blocks = []
    pattern = r'```(\w+)?\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)

    for match in matches:
        language = match[0] if match[0] else 'text'
        code = match[1].strip()
        code_blocks.append({
            'language': language,
            'code': code
        })

    return code_blocks


def validate_json_structure(data: dict, required_fields: List[str]) -> Tuple[bool, str]:
    """Validate JSON structure has required fields"""
    missing_fields = []
    for field in required_fields:
        if field not in data:
            missing_fields.append(field)

    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"

    return True, "Valid JSON structure"


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file operations"""
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')

    # Remove leading/trailing whitespace and dots
    filename = filename.strip('. ')

    # Limit length
    if len(filename) > 255:
        filename = filename[:255]

    return filename


# Global utility instances
safe_executor = SafeShellExecutor()
# Import the actual ResilientOrchestrator that has ComfyUI integration
from src.orchestrators.resilient_orchestrator import ResilientOrchestrator

# Use ResilientOrchestrator as tower_orchestrator for actual execution
tower_orchestrator = ResilientOrchestrator(firebase_config=None)