#!/usr/bin/env python3
"""
Secure command execution system for Echo Brain
Replaces dangerous subprocess.shell execution with whitelisted, validated commands
"""

import asyncio
import subprocess
import shlex
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import re

logger = logging.getLogger(__name__)

class SafeCommandExecutor:
    """
    Secure command executor with strict whitelisting and input validation
    """

    # Whitelist of allowed commands - ONLY these can be executed
    ALLOWED_COMMANDS = {
        'ls': {
            'binary': '/bin/ls',
            'allowed_flags': ['-l', '-la', '-lh', '-a', '-h', '--help'],
            'description': 'List directory contents'
        },
        'pwd': {
            'binary': '/bin/pwd',
            'allowed_flags': [],
            'description': 'Print working directory'
        },
        'whoami': {
            'binary': '/usr/bin/whoami',
            'allowed_flags': [],
            'description': 'Display current user'
        },
        'date': {
            'binary': '/bin/date',
            'allowed_flags': ['-u', '--help'],
            'description': 'Display current date/time'
        },
        'df': {
            'binary': '/bin/df',
            'allowed_flags': ['-h', '-H', '--help'],
            'description': 'Display filesystem usage'
        },
        'free': {
            'binary': '/usr/bin/free',
            'allowed_flags': ['-h', '-m', '-g', '--help'],
            'description': 'Display memory usage'
        },
        'uptime': {
            'binary': '/usr/bin/uptime',
            'allowed_flags': [],
            'description': 'System uptime and load'
        },
        'ps': {
            'binary': '/bin/ps',
            'allowed_flags': ['aux', '-ef', '--help'],
            'description': 'List running processes'
        },
        'systemctl': {
            'binary': '/bin/systemctl',
            'allowed_flags': ['status', 'is-active', 'is-enabled', '--help'],
            'description': 'Check systemd service status (read-only)'
        },
        'journalctl': {
            'binary': '/bin/journalctl',
            'allowed_flags': ['-u', '-n', '-f', '--since', '--help'],
            'description': 'View system logs (read-only)'
        }
    }

    # Dangerous patterns that are never allowed
    DANGEROUS_PATTERNS = [
        r'[;&|`$]',     # Shell metacharacters (removed () to allow -la)
        r'\(.*\)',      # Subshell execution
        r'\.\./',       # Directory traversal
        r'>/dev/',      # Device access
        r'\bsudo\b|\bsu\s',   # Privilege escalation
        r'\brm\s|\bdel\s',  # File deletion
        r'\bchmod\b|\bchown\b', # Permission changes
        r'\bwget\b|\bcurl\b',   # Network access
        r'\bpython\b|\bperl\b|\bruby\b|\bnode\b|\bbash\b|\bsh\b|\bzsh\b', # Script interpreters
        r'\bnc\b|\bnetcat\b|\btelnet\b|\bssh\b',  # Network tools
        r'\bkill\b|\bkillall\b', # Process termination
        r'\bmount\b|\bumount\b', # Filesystem operations
        r'\bpasswd\b|\buser\b',  # User management
        r'\bcrontab\b|\bat\s', # Scheduled tasks
        r'\becho\b.*>',      # File writing
        r'\>\s*/\w+',       # Redirecting to files
        r'\<\s*/\w+',       # Input redirection
    ]

    def __init__(self):
        self.command_history = []
        self.max_history = 100

    def _extract_command_from_query(self, query_string: str) -> str:
        """
        Extract the actual command from a potentially memory-augmented query

        Memory system may prepend context like:
        "[Known facts]: ...; [Current message]: whoami"

        We need to extract just "whoami" for command validation.
        """

        # Look for the pattern "[Current message]: command"
        import re
        current_message_match = re.search(r'\[Current message\]:\s*(.+?)(?:\n|$)', query_string)
        if current_message_match:
            extracted = current_message_match.group(1).strip()
            logger.debug(f"🔍 Extracted command from memory context: '{extracted}'")
            return extracted

        # If no memory pattern, check if it looks like augmented query
        if ('[Known facts]:' in query_string or
            'Character:' in query_string or
            query_string.count('\n') > 3):
            # Try to get the last line as the actual command
            lines = query_string.strip().split('\n')
            last_line = lines[-1].strip()
            if last_line and not last_line.startswith('['):
                logger.debug(f"🔍 Extracted last line as command: '{last_line}'")
                return last_line

        # If no augmentation detected, return original
        logger.debug(f"🔍 No augmentation detected, using original: '{query_string}'")
        return query_string.strip()

    async def execute_command(self, command_string: str, username: str = "unknown") -> Dict:
        """
        Safely execute a whitelisted command with strict validation

        Args:
            command_string: The command to execute (may be augmented with context)
            username: User requesting the command (for logging)

        Returns:
            Dict with success status, output, and metadata
        """
        start_time = datetime.now()

        try:
            # Extract actual command from potential memory-augmented query
            actual_command = self._extract_command_from_query(command_string)

            # Log all command attempts
            logger.info(f"🔒 Command request from {username}: Original='{command_string[:100]}...', Extracted='{actual_command}'")
            self._log_command_attempt(actual_command, username, start_time)

            # Step 1: Basic input validation on extracted command
            validation_result = self._validate_input(actual_command)
            if not validation_result['valid']:
                return self._create_error_response(
                    f"Command blocked: {validation_result['reason']}",
                    actual_command, username, start_time
                )

            # Step 2: Parse and validate command components
            parsed = self._parse_command(actual_command)
            if not parsed['valid']:
                return self._create_error_response(
                    f"Invalid command format: {parsed['reason']}",
                    command_string, username, start_time
                )

            command = parsed['command']
            args = parsed['args']

            # Step 3: Check command whitelist
            if command not in self.ALLOWED_COMMANDS:
                return self._create_error_response(
                    f"Command '{command}' is not whitelisted. Allowed: {list(self.ALLOWED_COMMANDS.keys())}",
                    command_string, username, start_time
                )

            # Step 4: Validate arguments against command-specific rules
            validation_result = self._validate_command_args(command, args)
            if not validation_result['valid']:
                return self._create_error_response(
                    f"Invalid arguments for {command}: {validation_result['reason']}",
                    command_string, username, start_time
                )

            # Step 5: Execute the validated command
            return await self._execute_validated_command(command, args, username, start_time)

        except Exception as e:
            logger.error(f"❌ Command execution failed: {e}")
            return self._create_error_response(
                f"Internal error: {str(e)}",
                command_string, username, start_time
            )

    def _validate_input(self, command_string: str) -> Dict:
        """Validate basic input safety"""

        # Check for dangerous patterns
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, command_string, re.IGNORECASE):
                logger.warning(f"🚫 Command blocked by pattern '{pattern}' in '{command_string}' (repr: {repr(command_string)})")
                return {
                    'valid': False,
                    'reason': f'SECURITY_V2: Contains dangerous pattern: {pattern} (command was: {repr(command_string)})'
                }

        # Check length limits
        if len(command_string) > 500:
            return {
                'valid': False,
                'reason': 'Command too long (max 500 chars)'
            }

        # Check for null bytes
        if '\x00' in command_string:
            return {
                'valid': False,
                'reason': 'Null bytes not allowed'
            }

        return {'valid': True}

    def _parse_command(self, command_string: str) -> Dict:
        """Parse command into components using secure shell lexer"""
        try:
            # Use shlex for secure parsing (handles quotes properly)
            parts = shlex.split(command_string.strip())

            if not parts:
                return {
                    'valid': False,
                    'reason': 'Empty command'
                }

            command = parts[0].lower()
            args = parts[1:] if len(parts) > 1 else []

            # Validate command name format
            if not re.match(r'^[a-zA-Z0-9_-]+$', command):
                return {
                    'valid': False,
                    'reason': 'Invalid command name format'
                }

            return {
                'valid': True,
                'command': command,
                'args': args
            }

        except ValueError as e:
            return {
                'valid': False,
                'reason': f'Command parsing failed: {str(e)}'
            }

    def _validate_command_args(self, command: str, args: List[str]) -> Dict:
        """Validate arguments for specific command"""

        command_config = self.ALLOWED_COMMANDS[command]
        allowed_flags = command_config.get('allowed_flags', [])

        for arg in args:
            # Check if argument is a flag
            if arg.startswith('-'):
                if arg not in allowed_flags:
                    return {
                        'valid': False,
                        'reason': f'Flag {arg} not allowed. Allowed flags: {allowed_flags}'
                    }
            else:
                # Non-flag arguments - validate based on command
                if command == 'ls':
                    # Only allow safe directory paths for ls
                    if not self._is_safe_path(arg):
                        return {
                            'valid': False,
                            'reason': f'Unsafe path: {arg}'
                        }
                elif command == 'systemctl':
                    # Only allow specific service names for systemctl status
                    if not self._is_safe_service_name(arg):
                        return {
                            'valid': False,
                            'reason': f'Invalid service name: {arg}'
                        }
                elif command == 'journalctl':
                    # Special validation for journalctl
                    if not self._is_safe_journalctl_arg(arg):
                        return {
                            'valid': False,
                            'reason': f'Invalid journalctl argument: {arg}'
                        }

        return {'valid': True}

    def _is_safe_path(self, path: str) -> bool:
        """Check if path is safe for directory operations"""
        # Whitelist safe directories
        safe_dirs = [
            '/tmp', '/var/log', '/home', '/opt',
            '/usr', '/etc', '/var', '/proc'
        ]

        # Block dangerous patterns
        if '..' in path or path.startswith('/dev') or path.startswith('/sys'):
            return False

        # Allow relative paths in current directory
        if not path.startswith('/'):
            return True

        # Check if path starts with safe directory
        return any(path.startswith(safe_dir) for safe_dir in safe_dirs)

    def _is_safe_service_name(self, service_name: str) -> bool:
        """Check if service name is safe for systemctl"""
        # Only allow alphanumeric, dash, underscore, dot
        return re.match(r'^[a-zA-Z0-9._-]+$', service_name) is not None

    def _is_safe_journalctl_arg(self, arg: str) -> bool:
        """Check if journalctl argument is safe"""
        # Allow service names and time specifications
        if re.match(r'^[a-zA-Z0-9._-]+$', arg):
            return True
        # Allow time formats like "1 hour ago", "today"
        if re.match(r'^[0-9]+ (hour|minute|day)s? ago$', arg):
            return True
        if arg in ['today', 'yesterday']:
            return True
        return False

    async def _execute_validated_command(self, command: str, args: List[str], username: str, start_time: datetime) -> Dict:
        """Execute a validated command safely"""

        command_config = self.ALLOWED_COMMANDS[command]
        binary_path = command_config['binary']

        # Build command array (NOT shell string)
        cmd_array = [binary_path] + args

        try:
            # Execute with subprocess (shell=False for security)
            process = await asyncio.create_subprocess_exec(
                *cmd_array,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd='/tmp'  # Safe working directory
            )

            # Wait for completion with timeout
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=15  # 15 second timeout
            )

            stdout_text = stdout.decode('utf-8', errors='replace')
            stderr_text = stderr.decode('utf-8', errors='replace')

            processing_time = (datetime.now() - start_time).total_seconds()

            # Log successful execution
            logger.info(f"✅ Command executed successfully: {command} (user: {username}, time: {processing_time:.2f}s)")

            return {
                'success': True,
                'command': command,
                'args': args,
                'stdout': stdout_text,
                'stderr': stderr_text,
                'return_code': process.returncode,
                'processing_time': processing_time,
                'username': username,
                'timestamp': start_time.isoformat()
            }

        except asyncio.TimeoutError:
            logger.warning(f"⏰ Command timeout: {command} (user: {username})")
            return self._create_error_response(
                f"Command '{command}' timed out (15s limit)",
                f"{command} {' '.join(args)}", username, start_time
            )
        except Exception as e:
            logger.error(f"❌ Command execution error: {e}")
            return self._create_error_response(
                f"Execution failed: {str(e)}",
                f"{command} {' '.join(args)}", username, start_time
            )

    def _create_error_response(self, error_message: str, command: str, username: str, start_time: datetime) -> Dict:
        """Create standardized error response"""

        processing_time = (datetime.now() - start_time).total_seconds()

        logger.warning(f"🚫 Command blocked/failed: {command} (user: {username}) - {error_message}")

        return {
            'success': False,
            'error': error_message,
            'command': command,
            'username': username,
            'processing_time': processing_time,
            'timestamp': start_time.isoformat()
        }

    def _log_command_attempt(self, command: str, username: str, timestamp: datetime):
        """Log command attempt for security auditing"""

        # Add to in-memory history
        self.command_history.append({
            'command': command,
            'username': username,
            'timestamp': timestamp.isoformat(),
            'status': 'pending'
        })

        # Keep history size manageable
        if len(self.command_history) > self.max_history:
            self.command_history = self.command_history[-self.max_history:]

    def get_allowed_commands(self) -> Dict:
        """Return list of allowed commands with descriptions"""
        return {
            cmd: {
                'description': config['description'],
                'allowed_flags': config['allowed_flags']
            }
            for cmd, config in self.ALLOWED_COMMANDS.items()
        }

    def get_command_history(self, username: Optional[str] = None) -> List[Dict]:
        """Get command history (optionally filtered by user)"""
        if username:
            return [h for h in self.command_history if h['username'] == username]
        return self.command_history.copy()

# Global instance
safe_command_executor = SafeCommandExecutor()