"""
Action Executor for Echo Brain
Executes actions with safety controls.
"""

import asyncio
import asyncpg
import subprocess
import shutil
import json
import httpx
import logging
import tempfile
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import psutil

from .schemas import (
    ActionType, SafetyLevel, ShellResult, ModifyResult,
    APIResult, ServiceResult, QueryResult, ActionLog
)

logger = logging.getLogger(__name__)


class FileChange:
    """Represents a change to make to a file"""
    def __init__(self, line_number: int, old_content: str, new_content: str):
        self.line_number = line_number
        self.old_content = old_content
        self.new_content = new_content


class ActionExecutor:
    """
    Executes actions with safety controls.
    """

    # Safety classifications
    SAFE_COMMANDS = {
        'ls', 'cat', 'grep', 'find', 'head', 'tail', 'wc', 'pwd', 'whoami',
        'date', 'uptime', 'free', 'df', 'ps', 'systemctl status', 'journalctl',
        'docker ps', 'docker images', 'git status', 'git log', 'git diff'
    }

    NEEDS_CONFIRM_COMMANDS = {
        'systemctl restart', 'systemctl stop', 'systemctl start',
        'service restart', 'service stop', 'service start',
        'docker restart', 'docker stop', 'docker start',
        'git pull', 'git push', 'git reset', 'git checkout',
        'npm install', 'pip install', 'apt install'
    }

    DANGEROUS_COMMANDS = {
        'rm -rf', 'sudo rm', 'rmdir', 'delete', 'drop database',
        'truncate table', 'format', 'fdisk', 'mkfs',
        'shutdown', 'reboot', 'init 0', 'init 6'
    }

    def __init__(self, db_config: Dict[str, str] = None):
        self.db_config = db_config or {
            "host": "localhost",
            "database": "echo_brain",
            "user": "patrick",
            "password": "RP78eIrW7cI2jYvL5akt1yurE"
        }
        self._pool = None

    async def get_db_pool(self):
        """Get or create database connection pool"""
        if not self._pool:
            self._pool = await asyncpg.create_pool(
                **self.db_config,
                min_size=2,
                max_size=10,
                timeout=10
            )
        return self._pool

    async def close(self):
        """Clean up connections"""
        if self._pool:
            await self._pool.close()

    def _classify_command_safety(self, command: str) -> SafetyLevel:
        """Determine safety level of a command"""
        command_lower = command.lower().strip()

        # Check for dangerous patterns
        for dangerous in self.DANGEROUS_COMMANDS:
            if dangerous in command_lower:
                return SafetyLevel.DANGEROUS

        # Check for commands needing confirmation
        for confirm in self.NEEDS_CONFIRM_COMMANDS:
            if command_lower.startswith(confirm):
                return SafetyLevel.NEEDS_CONFIRM

        # Check for safe commands
        for safe in self.SAFE_COMMANDS:
            if command_lower.startswith(safe):
                return SafetyLevel.SAFE

        # Default to needs confirmation for unknown commands
        return SafetyLevel.NEEDS_CONFIRM

    async def _log_action(self, action_type: ActionType, target: str, command: str,
                         result: Dict[str, Any], success: bool, user_confirmed: bool = False,
                         execution_time_ms: int = 0):
        """Log action to database"""
        pool = await self.get_db_pool()

        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    """INSERT INTO action_log
                       (timestamp, action_type, target, command, result, success, user_confirmed, execution_time_ms)
                       VALUES ($1, $2, $3, $4, $5, $6, $7, $8)""",
                    datetime.now(), action_type.value, target, command,
                    json.dumps(result), success, user_confirmed, execution_time_ms
                )
        except Exception as e:
            logger.error(f"Error logging action: {e}")

    async def execute_shell(self, command: str, safe: bool = False, timeout: int = 30) -> ShellResult:
        """
        Run a shell command.
        - Timeout protection
        - Output capture
        - Error handling
        - Audit logging
        """
        start_time = datetime.now()

        # Check safety
        safety_level = self._classify_command_safety(command)

        if not safe and safety_level == SafetyLevel.DANGEROUS:
            result = ShellResult(
                command=command,
                exit_code=-1,
                stdout="",
                stderr="Command blocked: Dangerous operation requires explicit confirmation",
                execution_time_ms=0,
                success=False
            )

            await self._log_action(
                ActionType.SHELL, "shell", command,
                result.dict(), False, user_confirmed=False
            )

            return result

        try:
            # Execute command with timeout
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=os.getcwd()
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
                exit_code = process.returncode
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                stdout = b""
                stderr = b"Command timed out"
                exit_code = -1

            # Calculate execution time
            execution_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            result = ShellResult(
                command=command,
                exit_code=exit_code,
                stdout=stdout.decode('utf-8', errors='replace'),
                stderr=stderr.decode('utf-8', errors='replace'),
                execution_time_ms=execution_time_ms,
                success=exit_code == 0
            )

            # Log the action
            await self._log_action(
                ActionType.SHELL, "shell", command,
                result.dict(), result.success,
                user_confirmed=safe and safety_level != SafetyLevel.SAFE,
                execution_time_ms=execution_time_ms
            )

            # Trigger learning loop
            try:
                from .learner import get_learning_loop
                learner = get_learning_loop()
                await learner.on_action_complete(
                    action_type="shell",
                    command=command,
                    success=result.success,
                    result=result.stdout if result.success else result.stderr,
                    context={"execution_time_ms": execution_time_ms}
                )
            except Exception as e:
                logger.debug(f"Learning loop error: {e}")

            return result

        except Exception as e:
            execution_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            result = ShellResult(
                command=command,
                exit_code=-1,
                stdout="",
                stderr=f"Execution error: {str(e)}",
                execution_time_ms=execution_time_ms,
                success=False
            )

            await self._log_action(
                ActionType.SHELL, "shell", command,
                result.dict(), False, execution_time_ms=execution_time_ms
            )

            return result

    async def modify_file(self, path: str, changes: List[FileChange]) -> ModifyResult:
        """
        Safely modify a file:
        - Create backup first
        - Apply changes
        - Validate (syntax check for code)
        - Rollback on failure
        """
        start_time = datetime.now()
        backup_path = None

        try:
            # Create backup
            backup_path = f"{path}.backup.{int(start_time.timestamp())}"
            shutil.copy2(path, backup_path)

            # Read current content
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Apply changes
            changes_applied = 0
            for change in changes:
                if 0 <= change.line_number - 1 < len(lines):
                    if lines[change.line_number - 1].strip() == change.old_content.strip():
                        lines[change.line_number - 1] = change.new_content + '\n'
                        changes_applied += 1
                    else:
                        logger.warning(f"Line {change.line_number} content mismatch in {path}")

            # Write modified content
            with open(path, 'w', encoding='utf-8') as f:
                f.writelines(lines)

            # Validate for Python files
            validation_passed = True
            if path.endswith('.py'):
                validation_passed = await self._validate_python_syntax(path)

            success = changes_applied > 0 and validation_passed

            if not success and backup_path:
                # Restore backup
                shutil.copy2(backup_path, path)

            execution_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            result = ModifyResult(
                file_path=path,
                backup_path=backup_path,
                changes_applied=changes_applied,
                success=success,
                validation_passed=validation_passed
            )

            await self._log_action(
                ActionType.FILE_MODIFY, path, f"Modified {changes_applied} lines",
                result.dict(), success, execution_time_ms=execution_time_ms
            )

            return result

        except Exception as e:
            # Restore backup if available
            if backup_path and os.path.exists(backup_path):
                try:
                    shutil.copy2(backup_path, path)
                except:
                    pass

            execution_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            result = ModifyResult(
                file_path=path,
                backup_path=backup_path or "",
                changes_applied=0,
                success=False,
                validation_passed=False,
                error=str(e)
            )

            await self._log_action(
                ActionType.FILE_MODIFY, path, f"Failed modification",
                result.dict(), False, execution_time_ms=execution_time_ms
            )

            return result

    async def _validate_python_syntax(self, file_path: str) -> bool:
        """Validate Python syntax"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            compile(content, file_path, 'exec')
            return True
        except SyntaxError as e:
            logger.error(f"Syntax error in {file_path}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error validating {file_path}: {e}")
            return False

    async def call_api(self, url: str, method: str = "GET", data: Dict[str, Any] = None,
                      headers: Dict[str, str] = None, timeout: int = 30) -> APIResult:
        """Make HTTP requests to services"""
        start_time = datetime.now()

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                if method.upper() == "GET":
                    response = await client.get(url, headers=headers)
                elif method.upper() == "POST":
                    response = await client.post(url, json=data, headers=headers)
                elif method.upper() == "PUT":
                    response = await client.put(url, json=data, headers=headers)
                elif method.upper() == "DELETE":
                    response = await client.delete(url, headers=headers)
                else:
                    raise ValueError(f"Unsupported method: {method}")

                # Try to parse JSON response
                try:
                    response_data = response.json()
                except:
                    response_data = {"text": response.text}

                execution_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

                result = APIResult(
                    url=url,
                    method=method.upper(),
                    status_code=response.status_code,
                    response_data=response_data,
                    response_time_ms=execution_time_ms,
                    success=200 <= response.status_code < 300
                )

                await self._log_action(
                    ActionType.API_CALL, url, f"{method} {url}",
                    result.dict(), result.success, execution_time_ms=execution_time_ms
                )

                return result

        except Exception as e:
            execution_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            result = APIResult(
                url=url,
                method=method.upper(),
                status_code=0,
                response_data={"error": str(e)},
                response_time_ms=execution_time_ms,
                success=False
            )

            await self._log_action(
                ActionType.API_CALL, url, f"{method} {url}",
                result.dict(), False, execution_time_ms=execution_time_ms
            )

            return result

    async def manage_service(self, name: str, action: str, confirm_dangerous: bool = False) -> ServiceResult:
        """Start/stop/restart systemd services"""
        start_time = datetime.now()

        # Validate action
        valid_actions = ['start', 'stop', 'restart', 'reload', 'enable', 'disable']
        if action not in valid_actions:
            result = ServiceResult(
                service_name=name,
                action=action,
                success=False,
                new_status='unknown',
                message=f"Invalid action. Must be one of: {', '.join(valid_actions)}"
            )

            await self._log_action(
                ActionType.SERVICE_MANAGE, name, f"{action} {name}",
                result.dict(), False
            )

            return result

        # Check safety
        if action in ['stop', 'restart', 'disable'] and not confirm_dangerous:
            result = ServiceResult(
                service_name=name,
                action=action,
                success=False,
                new_status='unknown',
                message=f"Action '{action}' requires confirmation for safety"
            )

            await self._log_action(
                ActionType.SERVICE_MANAGE, name, f"{action} {name}",
                result.dict(), False
            )

            return result

        try:
            # Execute systemctl command
            command = f"systemctl {action} {name}"
            shell_result = await self.execute_shell(command, safe=confirm_dangerous)

            # Get new status
            status_result = await self.execute_shell(f"systemctl is-active {name}", safe=True)
            new_status = status_result.stdout.strip() if status_result.success else 'unknown'

            success = shell_result.success
            message = shell_result.stderr if not success else shell_result.stdout

            result = ServiceResult(
                service_name=name,
                action=action,
                success=success,
                new_status=new_status,
                message=message
            )

            execution_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            await self._log_action(
                ActionType.SERVICE_MANAGE, name, f"{action} {name}",
                result.dict(), success, user_confirmed=confirm_dangerous,
                execution_time_ms=execution_time_ms
            )

            return result

        except Exception as e:
            execution_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            result = ServiceResult(
                service_name=name,
                action=action,
                success=False,
                new_status='unknown',
                message=str(e)
            )

            await self._log_action(
                ActionType.SERVICE_MANAGE, name, f"{action} {name}",
                result.dict(), False, execution_time_ms=execution_time_ms
            )

            return result

    async def query_database(self, db: str, query: str, params: List[Any] = None) -> QueryResult:
        """Execute SQL queries safely (parameterized only)"""
        start_time = datetime.now()

        # Basic safety checks
        query_lower = query.lower().strip()

        # Block dangerous operations
        dangerous_keywords = ['drop', 'delete', 'truncate', 'update', 'insert', 'alter']
        if any(keyword in query_lower for keyword in dangerous_keywords):
            if not query_lower.startswith('select'):
                result = QueryResult(
                    query=query,
                    rows_affected=0,
                    data=[],
                    execution_time_ms=0,
                    success=False
                )

                await self._log_action(
                    ActionType.DB_QUERY, db, query,
                    result.dict(), False
                )

                return result

        try:
            # Use the shared connection pool instead of creating a new one each time
            pool = await self.get_db_pool()

            async with pool.acquire() as conn:
                # Switch to the target database if needed
                if db != 'echo_brain':
                    # For queries to other databases, we need to use a direct connection
                    # since the pool is configured for 'echo_brain'
                    conn = await asyncio.wait_for(
                        asyncpg.connect(
                            host=self.db_config['host'],
                            database=db,
                            user=self.db_config['user'],
                            password=self.db_config['password'],
                            timeout=10
                        ),
                        timeout=15
                    )

                    try:
                        if params:
                            rows = await asyncio.wait_for(conn.fetch(query, *params), timeout=10)
                        else:
                            rows = await asyncio.wait_for(conn.fetch(query), timeout=10)
                    finally:
                        await conn.close()
                else:
                    # Use the pooled connection for echo_brain database
                    if params:
                        rows = await asyncio.wait_for(conn.fetch(query, *params), timeout=10)
                    else:
                        rows = await asyncio.wait_for(conn.fetch(query), timeout=10)

                # Convert to list of dicts
                data = [dict(row) for row in rows]

                execution_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

                result = QueryResult(
                    query=query,
                    rows_affected=len(rows),
                    data=data,
                    execution_time_ms=execution_time_ms,
                    success=True
                )

                await self._log_action(
                    ActionType.DB_QUERY, db, query,
                    {"rows_returned": len(rows)}, True,
                    execution_time_ms=execution_time_ms
                )

                # Trigger learning loop
                try:
                    from .learner import get_learning_loop
                    learner = get_learning_loop()
                    await learner.on_action_complete(
                        action_type="db_query",
                        command=query,
                        success=result.success,
                        result=f"Retrieved {len(rows)} rows",
                        context={"database": db, "execution_time_ms": execution_time_ms}
                    )
                except Exception as e:
                    logger.debug(f"Learning loop error: {e}")

                return result

        except Exception as e:
            execution_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            result = QueryResult(
                query=query,
                rows_affected=0,
                data=[],
                execution_time_ms=execution_time_ms,
                success=False
            )

            await self._log_action(
                ActionType.DB_QUERY, db, query,
                {"error": str(e)}, False,
                execution_time_ms=execution_time_ms
            )

            return result

    async def get_recent_actions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent action log entries"""
        pool = await self.get_db_pool()

        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """SELECT timestamp, action_type, target, command, success, execution_time_ms
                       FROM action_log
                       ORDER BY timestamp DESC
                       LIMIT $1""",
                    limit
                )

                return [
                    {
                        'timestamp': row['timestamp'].isoformat(),
                        'action_type': row['action_type'],
                        'target': row['target'],
                        'command': row['command'],
                        'success': row['success'],
                        'execution_time_ms': row['execution_time_ms']
                    }
                    for row in rows
                ]

        except Exception as e:
            logger.error(f"Error getting recent actions: {e}")
            return []


# Singleton instance
_action_executor = None

def get_action_executor() -> ActionExecutor:
    """Get or create singleton instance"""
    global _action_executor
    if not _action_executor:
        _action_executor = ActionExecutor()
    return _action_executor