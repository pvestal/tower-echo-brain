#!/usr/bin/env python3
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
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class SafeShellExecutor:
    """Safe shell command execution with security controls"""

    def __init__(self):
        # Allowed commands for safe mode
        self.safe_commands = {
            'ls', 'pwd', 'echo', 'cat', 'head', 'tail', 'grep', 'find',
            'ps', 'top', 'df', 'free', 'uptime', 'whoami', 'id', 'date',
            'curl', 'wget', 'nc', 'ping', 'nmap', 'systemctl', 'journalctl',
            'git', 'docker', 'python3', 'pip3', 'npm', 'node', 'pnpm'
        }

        # SECURITY FIX: Enhanced dangerous patterns to block including directory traversal
        self.dangerous_patterns = [
            r'rm\s+-rf?\s*/', r'sudo\s+rm', r'>\s*/dev/', r'dd\s+if=',
            r'mkfs', r'fdisk', r'parted', r'format', r'del\s+/[qsf]',
            r'shutdown', r'reboot', r'halt', r'init\s+[06]',
            r':\(\)\{', r'fork\s*\(', r'while\s*true', r'yes\s*\|',
            # Directory traversal patterns
            r'\.\./', r'\.\.\\', r'\.\.%2f', r'\.\.%5c',
            r'%2e%2e%2f', r'%2e%2e%5c', r'%252e%252e%252f',
            r'\.\.\/\.\.\/', r'\.\.\\\.\.\\',
            # Path injection patterns
            r'/etc/passwd', r'/etc/shadow', r'/proc/', r'/sys/',
            r'C:\\Windows\\System32', r'C:\\Windows\\system32',
            # Command injection patterns
            r';\s*rm', r'&&\s*rm', r'\|\s*rm', r'`rm', r'\$\(rm',
            r';\s*cat\s+/etc', r'&&\s*cat\s+/etc', r'\|\s*cat\s+/etc'
        ]

    def is_command_safe(self, command: str, safe_mode: bool = True) -> Tuple[bool, str]:
        """SECURITY FIX: Enhanced command safety validation with path normalization"""
        if not safe_mode:
            return True, "Safe mode disabled"

        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return False, f"Command blocked by pattern: {pattern}"

        # SECURITY FIX: Validate file paths in command arguments
        try:
            args = shlex.split(command)
        except ValueError:
            return False, "Invalid command syntax"

        for arg in args:
            # Check for path traversal in arguments
            if self._contains_path_traversal(arg):
                return False, f"Path traversal detected in argument: {arg}"

        # Check if base command is in safe list
        base_cmd = args[0] if args else ""
        if base_cmd not in self.safe_commands:
            return False, f"Command '{base_cmd}' not in safe commands list"

        return True, "Command passed safety checks"

    def _contains_path_traversal(self, path: str) -> bool:
        """SECURITY FIX: Check if path contains directory traversal attempts"""
        # Normalize the path to resolve any relative components
        try:
            normalized = os.path.normpath(path)

            # Check for directory traversal indicators
            traversal_indicators = [
                '..', '../', '..\\', './', '.\\',
                '%2e%2e', '%2e%2e%2f', '%2e%2e%5c',
                '%252e%252e%252f', '%252e%252e%255c'
            ]

            for indicator in traversal_indicators:
                if indicator in path.lower():
                    return True

            # Check if normalized path goes outside intended boundaries
            if normalized.startswith('/etc/') or normalized.startswith('/proc/') or normalized.startswith('/sys/'):
                return True

            if normalized.startswith('C:\\Windows\\') or normalized.startswith('c:\\windows\\'):
                return True

            return False

        except (ValueError, OSError):
            # If path cannot be normalized, consider it suspicious
            return True

    async def execute_command(self, command: str, safe_mode: bool = True) -> Dict:
        """Execute a shell command safely"""
        start_time = asyncio.get_event_loop().time()

        # Safety check
        is_safe, safety_msg = self.is_command_safe(command, safe_mode)
        safety_checks = {
            "passed_safety_check": is_safe,
            "safe_mode_enabled": safe_mode,
            "safety_message": safety_msg
        }

        if not is_safe:
            return {
                "command": command,
                "success": False,
                "output": "",
                "error": f"Command blocked for security: {safety_msg}",
                "exit_code": -1,
                "processing_time": asyncio.get_event_loop().time() - start_time,
                "safety_checks": safety_checks
            }

        try:
            # Execute command with timeout
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=os.path.expanduser("~")
            )

            processing_time = asyncio.get_event_loop().time() - start_time

            return {
                "command": command,
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr if result.stderr else None,
                "exit_code": result.returncode,
                "processing_time": processing_time,
                "safety_checks": safety_checks
            }

        except subprocess.TimeoutExpired:
            return {
                "command": command,
                "success": False,
                "output": "",
                "error": "Command timed out after 30 seconds",
                "exit_code": -1,
                "processing_time": 30.0,
                "safety_checks": safety_checks
            }
        except Exception as e:
            return {
                "command": command,
                "success": False,
                "output": "",
                "error": str(e),
                "exit_code": -1,
                "processing_time": asyncio.get_event_loop().time() - start_time,
                "safety_checks": safety_checks
            }


class TowerOrchestrator:
    """Integration with Tower Orchestrator Service for task delegation"""

    def __init__(self):
        self.orchestrator_url = "http://localhost:8400"
        self.timeout = 30

    async def submit_task(self, task_type: str, description: str, requirements: dict = {}, priority: int = 5) -> dict:
        """Submit a task to the orchestrator for delegation"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "type": task_type,
                    "description": description,
                    "requirements": requirements,
                    "priority": priority
                }

                async with session.post(
                    f"{self.orchestrator_url}/submit_task",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "success": True,
                            "task_id": result.get("task_id"),
                            "status": result.get("status", "submitted")
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"Orchestrator returned status {response.status}"
                        }
        except Exception as e:
            logger.error(f"Failed to submit task to orchestrator: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def get_task_status(self, task_id: str) -> dict:
        """Get status of a submitted task"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.orchestrator_url}/task_status/{task_id}",
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return {
                            "success": False,
                            "error": f"Status check failed: {response.status}"
                        }
        except Exception as e:
            logger.error(f"Failed to get task status: {e}")
            return {
                "success": False,
                "error": str(e)
            }


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
tower_orchestrator = TowerOrchestrator()