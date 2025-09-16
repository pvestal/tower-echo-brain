#!/usr/bin/env python3
"""
Sandbox Executor for Echo Brain Board of Directors
Provides secure, isolated execution environment for code validation and testing
"""

import logging
import json
import uuid
import subprocess
import tempfile
import shutil
import os
import signal
import time
import resource
import psutil
import ast
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import docker
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

class ASTSecurityValidator:
    """
    SECURITY FIX: Comprehensive AST validation for secure code execution
    Prevents dangerous operations through Abstract Syntax Tree analysis
    """

    DANGEROUS_BUILTINS = {
        'eval', 'exec', 'compile', 'open', '__import__', 'globals', 'locals',
        'vars', 'dir', 'hasattr', 'getattr', 'setattr', 'delattr',
        'input', 'raw_input', 'reload', 'breakpoint'
    }

    DANGEROUS_MODULES = {
        'os', 'sys', 'subprocess', 'shutil', 'tempfile', 'pickle', 'marshal',
        'imp', 'importlib', 'types', 'code', 'codeop', 'compileall',
        'socket', 'urllib', 'http', 'ftplib', 'smtplib', 'telnetlib',
        'ctypes', 'multiprocessing', 'threading', 'asyncio'
    }

    ALLOWED_MODULES = {
        'math', 'random', 'datetime', 'json', 're', 'string', 'itertools',
        'collections', 'functools', 'operator', 'heapq', 'bisect', 'array',
        'copy', 'decimal', 'fractions', 'statistics', 'uuid', 'hashlib',
        'base64', 'binascii', 'struct', 'zlib', 'gzip', 'bz2', 'lzma'
    }

    @classmethod
    def validate_code(cls, code: str) -> Tuple[bool, List[str]]:
        """
        Validate Python code using AST analysis

        Args:
            code: Python code to validate

        Returns:
            Tuple of (is_safe, violations_list)
        """
        violations = []

        try:
            # Parse code into AST
            tree = ast.parse(code)
        except SyntaxError as e:
            violations.append(f"Syntax error: {e}")
            return False, violations

        # Walk the AST and check for dangerous operations
        for node in ast.walk(tree):
            # Check for dangerous function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in cls.DANGEROUS_BUILTINS:
                        violations.append(f"Dangerous builtin function: {node.func.id}")
                elif isinstance(node.func, ast.Attribute):
                    if hasattr(node.func.value, 'id') and node.func.value.id in cls.DANGEROUS_MODULES:
                        violations.append(f"Dangerous module call: {node.func.value.id}.{node.func.attr}")

            # Check for imports
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name.split('.')[0]
                        if module_name in cls.DANGEROUS_MODULES:
                            violations.append(f"Dangerous module import: {module_name}")
                        elif module_name not in cls.ALLOWED_MODULES:
                            violations.append(f"Unauthorized module import: {module_name}")
                elif isinstance(node, ast.ImportFrom):
                    module_name = node.module.split('.')[0] if node.module else ''
                    if module_name in cls.DANGEROUS_MODULES:
                        violations.append(f"Dangerous module import: {module_name}")
                    elif module_name and module_name not in cls.ALLOWED_MODULES:
                        violations.append(f"Unauthorized module import: {module_name}")

            # Check for attribute access to dangerous attributes
            elif isinstance(node, ast.Attribute):
                dangerous_attrs = ['__globals__', '__locals__', '__dict__', '__class__', '__bases__']
                if node.attr in dangerous_attrs:
                    violations.append(f"Dangerous attribute access: {node.attr}")

            # Check for exec/eval in string literals
            elif isinstance(node, ast.Str):
                if 'eval(' in node.s or 'exec(' in node.s:
                    violations.append("String contains eval/exec code")

        return len(violations) == 0, violations

class ExecutionResult(Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    RESOURCE_LIMIT = "resource_limit"
    SECURITY_VIOLATION = "security_violation"
    BLOCKED = "blocked"

class SandboxType(Enum):
    DOCKER = "docker"
    PROCESS = "process"
    CHROOT = "chroot"
    MOCK = "mock"  # For testing

@dataclass
class ExecutionLimits:
    """Resource limits for sandbox execution"""
    max_memory_mb: int = 512
    max_cpu_seconds: int = 30
    max_disk_mb: int = 100
    max_network_calls: int = 0  # 0 = no network
    max_file_descriptors: int = 64
    max_processes: int = 10
    timeout_seconds: int = 60
    max_output_size: int = 1024 * 1024  # 1MB

@dataclass
class SecurityPolicy:
    """Security policy for sandbox execution"""
    allow_network: bool = False
    allow_file_system_write: bool = False
    allowed_imports: List[str] = None
    blocked_imports: List[str] = None
    allowed_system_calls: List[str] = None
    blocked_system_calls: List[str] = None
    allow_subprocess: bool = False
    allow_eval_exec: bool = False

@dataclass
class ExecutionContext:
    """Context for code execution"""
    execution_id: str
    code: str
    language: str
    input_data: Optional[str] = None
    working_directory: Optional[str] = None
    environment_variables: Dict[str, str] = None
    arguments: List[str] = None
    limits: ExecutionLimits = None
    security_policy: SecurityPolicy = None

@dataclass
class ExecutionOutput:
    """Result of code execution"""
    execution_id: str
    result: ExecutionResult
    stdout: str
    stderr: str
    exit_code: int
    execution_time: float
    memory_used: int
    cpu_time: float
    error_message: Optional[str] = None
    security_violations: List[str] = None
    resource_usage: Dict[str, Any] = None
    artifacts: List[str] = None  # Created files

class SandboxExecutor:
    """
    Secure sandbox execution environment for the Board of Directors
    Supports multiple isolation levels and security policies
    """

    def __init__(self, sandbox_type: SandboxType = SandboxType.DOCKER,
                 base_directory: str = "/tmp/echo_sandbox",
                 docker_image: str = "python:3.9-alpine"):
        """
        Initialize SandboxExecutor

        Args:
            sandbox_type: Type of sandbox to use
            base_directory: Base directory for sandbox operations
            docker_image: Docker image for containerized execution
        """
        self.sandbox_type = sandbox_type
        self.base_directory = Path(base_directory)
        self.docker_image = docker_image
        self.active_executions: Dict[str, threading.Thread] = {}
        self.docker_client = None

        # Default limits and policies
        self.default_limits = ExecutionLimits()
        self.default_security_policy = SecurityPolicy(
            allowed_imports=[
                "os", "sys", "json", "math", "random", "datetime", "time",
                "collections", "itertools", "functools", "operator",
                "re", "string", "urllib.parse", "base64", "hashlib",
                "uuid", "logging"
            ],
            blocked_imports=[
                "subprocess", "multiprocessing", "threading", "socket",
                "http", "urllib.request", "ftplib", "smtplib", "ssl",
                "ctypes", "importlib", "__import__"
            ],
            blocked_system_calls=[
                "exec", "eval", "compile", "open", "__import__",
                "getattr", "setattr", "delattr", "globals", "locals"
            ]
        )

        self._initialize_sandbox()

    def _initialize_sandbox(self):
        """Initialize sandbox environment"""
        try:
            # Create base directory
            self.base_directory.mkdir(parents=True, exist_ok=True)

            # Initialize Docker client if using Docker sandbox
            if self.sandbox_type == SandboxType.DOCKER:
                try:
                    import docker
                    self.docker_client = docker.from_env()
                    # Test connection
                    self.docker_client.ping()
                    logger.info("Docker client initialized successfully")
                except Exception as e:
                    logger.warning(f"Failed to initialize Docker client: {e}")
                    logger.info("Falling back to process sandbox")
                    self.sandbox_type = SandboxType.PROCESS

            # Set up signal handlers for cleanup
            signal.signal(signal.SIGTERM, self._cleanup_handler)
            signal.signal(signal.SIGINT, self._cleanup_handler)

            logger.info(f"Sandbox executor initialized with type: {self.sandbox_type.value}")

        except Exception as e:
            logger.error(f"Failed to initialize sandbox: {e}")
            raise

    def execute_code(self, context: ExecutionContext) -> ExecutionOutput:
        """
        Execute code in secure sandbox

        Args:
            context: Execution context with code and configuration

        Returns:
            ExecutionOutput: Result of execution
        """
        try:
            # Apply default limits and policy if not provided
            if context.limits is None:
                context.limits = self.default_limits
            if context.security_policy is None:
                context.security_policy = self.default_security_policy

            # Validate input
            validation_result = self._validate_execution_context(context)
            if validation_result.result != ExecutionResult.SUCCESS:
                return validation_result

            # Choose execution method based on sandbox type
            if self.sandbox_type == SandboxType.DOCKER:
                return self._execute_in_docker(context)
            elif self.sandbox_type == SandboxType.PROCESS:
                return self._execute_in_process(context)
            elif self.sandbox_type == SandboxType.MOCK:
                return self._execute_mock(context)
            else:
                return ExecutionOutput(
                    execution_id=context.execution_id,
                    result=ExecutionResult.ERROR,
                    stdout="",
                    stderr=f"Unsupported sandbox type: {self.sandbox_type}",
                    exit_code=-1,
                    execution_time=0.0,
                    memory_used=0,
                    cpu_time=0.0,
                    error_message=f"Unsupported sandbox type: {self.sandbox_type}"
                )

        except Exception as e:
            logger.error(f"Execution failed: {e}")
            return ExecutionOutput(
                execution_id=context.execution_id,
                result=ExecutionResult.ERROR,
                stdout="",
                stderr=str(e),
                exit_code=-1,
                execution_time=0.0,
                memory_used=0,
                cpu_time=0.0,
                error_message=str(e)
            )

    def _validate_execution_context(self, context: ExecutionContext) -> ExecutionOutput:
        """SECURITY FIX: Enhanced validation with AST analysis"""
        violations = []

        # Check code length
        if len(context.code) > 1024 * 1024:  # 1MB limit
            violations.append("Code too large (>1MB)")

        # SECURITY FIX: Use AST validation for comprehensive security analysis
        if context.language == "python":
            is_safe, ast_violations = ASTSecurityValidator.validate_code(context.code)
            if not is_safe:
                violations.extend(ast_violations)

        # Check for blocked imports (legacy fallback)
        if context.security_policy.blocked_imports:
            for blocked in context.security_policy.blocked_imports:
                if f"import {blocked}" in context.code or f"from {blocked}" in context.code:
                    violations.append(f"Blocked import: {blocked}")

        # Check for blocked system calls (enhanced pattern matching)
        if context.security_policy.blocked_system_calls:
            for blocked in context.security_policy.blocked_system_calls:
                # More sophisticated pattern matching
                import re
                pattern = rf'\b{re.escape(blocked)}\b'
                if re.search(pattern, context.code):
                    violations.append(f"Blocked system call: {blocked}")

        # Strict eval/exec checking
        if not context.security_policy.allow_eval_exec:
            import re
            # Check for various forms of dynamic execution
            dangerous_patterns = [
                r'\beval\s*\(',
                r'\bexec\s*\(',
                r'\bcompile\s*\(',
                r'__import__\s*\(',
                r'globals\s*\(\)',
                r'locals\s*\(\)'
            ]
            for pattern in dangerous_patterns:
                if re.search(pattern, context.code):
                    violations.append(f"Dynamic code execution not allowed: {pattern}")

        if violations:
            return ExecutionOutput(
                execution_id=context.execution_id,
                result=ExecutionResult.SECURITY_VIOLATION,
                stdout="",
                stderr="Security violations detected",
                exit_code=-1,
                execution_time=0.0,
                memory_used=0,
                cpu_time=0.0,
                error_message="Security policy violations",
                security_violations=violations
            )

        return ExecutionOutput(
            execution_id=context.execution_id,
            result=ExecutionResult.SUCCESS,
            stdout="",
            stderr="",
            exit_code=0,
            execution_time=0.0,
            memory_used=0,
            cpu_time=0.0
        )

    def _execute_in_docker(self, context: ExecutionContext) -> ExecutionOutput:
        """Execute code in Docker container"""
        if not self.docker_client:
            return ExecutionOutput(
                execution_id=context.execution_id,
                result=ExecutionResult.ERROR,
                stdout="",
                stderr="Docker client not available",
                exit_code=-1,
                execution_time=0.0,
                memory_used=0,
                cpu_time=0.0,
                error_message="Docker client not available"
            )

        execution_dir = None
        container = None
        start_time = time.time()

        try:
            # Create execution directory
            execution_dir = self.base_directory / context.execution_id
            execution_dir.mkdir(exist_ok=True)

            # Write code to file
            code_file = execution_dir / "code.py"
            with open(code_file, 'w') as f:
                f.write(context.code)

            # Write input data if provided
            if context.input_data:
                input_file = execution_dir / "input.txt"
                with open(input_file, 'w') as f:
                    f.write(context.input_data)

            # Prepare Docker run command
            volumes = {
                str(execution_dir): {
                    'bind': '/workspace',
                    'mode': 'rw' if context.security_policy.allow_file_system_write else 'ro'
                }
            }

            # Set resource limits
            mem_limit = f"{context.limits.max_memory_mb}m"
            cpu_quota = int(context.limits.max_cpu_seconds * 100000)  # Convert to microseconds

            # Environment variables
            environment = context.environment_variables or {}
            environment.update({
                'PYTHONPATH': '/workspace',
                'PYTHONUNBUFFERED': '1'
            })

            # SECURITY FIX: Run container with comprehensive security constraints
            container = self.docker_client.containers.run(
                self.docker_image,
                command=['python', '/workspace/code.py'],
                volumes=volumes,
                environment=environment,
                mem_limit=mem_limit,
                cpu_quota=cpu_quota,
                cpu_period=100000,
                network_disabled=not context.security_policy.allow_network,
                # SECURITY FIX: Enhanced Docker security constraints
                user='1000:1000',  # Run as non-root user
                read_only=True,    # Read-only filesystem
                security_opt=['no-new-privileges:true'],  # Prevent privilege escalation
                cap_drop=['ALL'],  # Drop all capabilities
                cap_add=['SETUID', 'SETGID'] if context.security_policy.allow_setuid else [],
                pids_limit=context.limits.max_processes,  # Limit process count
                privileged=False,  # Never run privileged containers
                # Mount /tmp as tmpfs to prevent persistent file access
                tmpfs={'/tmp': 'size=100m,noexec,nosuid,nodev'},
                # Additional security options
                shm_size='64m',    # Limit shared memory
                remove=False,
                detach=True,
                stdout=True,
                stderr=True
            )

            # Wait for completion with timeout
            try:
                container.wait(timeout=context.limits.timeout_seconds)
            except Exception as e:
                # Container timed out
                container.kill()
                container.wait()
                return ExecutionOutput(
                    execution_id=context.execution_id,
                    result=ExecutionResult.TIMEOUT,
                    stdout="",
                    stderr=f"Execution timed out after {context.limits.timeout_seconds}s",
                    exit_code=-1,
                    execution_time=time.time() - start_time,
                    memory_used=0,
                    cpu_time=0.0,
                    error_message="Execution timeout"
                )

            # Get results
            logs = container.logs(stdout=True, stderr=True).decode('utf-8')
            exit_code = container.attrs['State']['ExitCode']

            # Split stdout/stderr (Docker combines them)
            stdout_lines = []
            stderr_lines = []

            for line in logs.split('\n'):
                if line.startswith('ERROR:') or line.startswith('Traceback'):
                    stderr_lines.append(line)
                else:
                    stdout_lines.append(line)

            stdout = '\n'.join(stdout_lines)
            stderr = '\n'.join(stderr_lines)

            # Get resource usage stats
            stats = container.stats(stream=False)
            memory_used = stats['memory_stats'].get('usage', 0) // (1024 * 1024)  # MB
            cpu_stats = stats['cpu_stats']
            cpu_time = (cpu_stats.get('cpu_usage', {}).get('total_usage', 0) / 1000000000.0)  # seconds

            execution_time = time.time() - start_time

            # Determine result
            result = ExecutionResult.SUCCESS if exit_code == 0 else ExecutionResult.ERROR

            # Check for resource limits exceeded
            if memory_used > context.limits.max_memory_mb:
                result = ExecutionResult.RESOURCE_LIMIT
                stderr += f"\nMemory limit exceeded: {memory_used}MB > {context.limits.max_memory_mb}MB"

            if cpu_time > context.limits.max_cpu_seconds:
                result = ExecutionResult.RESOURCE_LIMIT
                stderr += f"\nCPU limit exceeded: {cpu_time}s > {context.limits.max_cpu_seconds}s"

            return ExecutionOutput(
                execution_id=context.execution_id,
                result=result,
                stdout=stdout[:context.limits.max_output_size],
                stderr=stderr[:context.limits.max_output_size],
                exit_code=exit_code,
                execution_time=execution_time,
                memory_used=memory_used,
                cpu_time=cpu_time,
                resource_usage={
                    'memory_stats': stats['memory_stats'],
                    'cpu_stats': cpu_stats
                }
            )

        except Exception as e:
            logger.error(f"Docker execution failed: {e}")
            return ExecutionOutput(
                execution_id=context.execution_id,
                result=ExecutionResult.ERROR,
                stdout="",
                stderr=str(e),
                exit_code=-1,
                execution_time=time.time() - start_time,
                memory_used=0,
                cpu_time=0.0,
                error_message=str(e)
            )

        finally:
            # Cleanup
            if container:
                try:
                    container.remove(force=True)
                except:
                    pass

            if execution_dir and execution_dir.exists():
                try:
                    shutil.rmtree(execution_dir)
                except:
                    pass

    def _execute_in_process(self, context: ExecutionContext) -> ExecutionOutput:
        """Execute code in isolated process"""
        execution_dir = None
        start_time = time.time()

        try:
            # Create execution directory
            execution_dir = self.base_directory / context.execution_id
            execution_dir.mkdir(exist_ok=True)

            # Write code to file
            code_file = execution_dir / "code.py"
            with open(code_file, 'w') as f:
                f.write(self._wrap_code_for_security(context))

            # Write input data if provided
            if context.input_data:
                input_file = execution_dir / "input.txt"
                with open(input_file, 'w') as f:
                    f.write(context.input_data)

            # Set up environment
            env = os.environ.copy()
            if context.environment_variables:
                env.update(context.environment_variables)

            env.update({
                'PYTHONPATH': str(execution_dir),
                'PYTHONUNBUFFERED': '1'
            })

            # Prepare command
            cmd = ['python', str(code_file)]
            if context.arguments:
                cmd.extend(context.arguments)

            # Execute with resource limits
            def preexec_function():
                # Set resource limits
                resource.setrlimit(resource.RLIMIT_AS,
                                 (context.limits.max_memory_mb * 1024 * 1024,
                                  context.limits.max_memory_mb * 1024 * 1024))
                resource.setrlimit(resource.RLIMIT_CPU,
                                 (context.limits.max_cpu_seconds,
                                  context.limits.max_cpu_seconds))
                resource.setrlimit(resource.RLIMIT_NPROC,
                                 (context.limits.max_processes,
                                  context.limits.max_processes))
                resource.setrlimit(resource.RLIMIT_NOFILE,
                                 (context.limits.max_file_descriptors,
                                  context.limits.max_file_descriptors))

            # Run process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
                cwd=str(execution_dir),
                env=env,
                preexec_fn=preexec_function,
                text=True
            )

            try:
                stdout, stderr = process.communicate(
                    input=context.input_data,
                    timeout=context.limits.timeout_seconds
                )
                exit_code = process.returncode
                result = ExecutionResult.SUCCESS if exit_code == 0 else ExecutionResult.ERROR

            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                exit_code = -1
                result = ExecutionResult.TIMEOUT
                stderr += f"\nExecution timed out after {context.limits.timeout_seconds}s"

            execution_time = time.time() - start_time

            # Get resource usage (approximate)
            memory_used = 0
            cpu_time = execution_time  # Approximate

            try:
                # Try to get actual resource usage if process is still alive
                if process.pid:
                    proc = psutil.Process(process.pid)
                    memory_used = proc.memory_info().rss // (1024 * 1024)  # MB
                    cpu_time = sum(proc.cpu_times())
            except:
                pass

            return ExecutionOutput(
                execution_id=context.execution_id,
                result=result,
                stdout=stdout[:context.limits.max_output_size],
                stderr=stderr[:context.limits.max_output_size],
                exit_code=exit_code,
                execution_time=execution_time,
                memory_used=memory_used,
                cpu_time=cpu_time
            )

        except Exception as e:
            logger.error(f"Process execution failed: {e}")
            return ExecutionOutput(
                execution_id=context.execution_id,
                result=ExecutionResult.ERROR,
                stdout="",
                stderr=str(e),
                exit_code=-1,
                execution_time=time.time() - start_time,
                memory_used=0,
                cpu_time=0.0,
                error_message=str(e)
            )

        finally:
            # Cleanup
            if execution_dir and execution_dir.exists():
                try:
                    shutil.rmtree(execution_dir)
                except:
                    pass

    def _execute_mock(self, context: ExecutionContext) -> ExecutionOutput:
        """Mock execution for testing"""
        time.sleep(0.1)  # Simulate execution time

        # Simple evaluation for testing
        output = "Mock execution completed"
        if "error" in context.code.lower():
            result = ExecutionResult.ERROR
            stderr = "Mock error occurred"
            exit_code = 1
        else:
            result = ExecutionResult.SUCCESS
            stderr = ""
            exit_code = 0

        return ExecutionOutput(
            execution_id=context.execution_id,
            result=result,
            stdout=output,
            stderr=stderr,
            exit_code=exit_code,
            execution_time=0.1,
            memory_used=10,
            cpu_time=0.05
        )

    def _wrap_code_for_security(self, context: ExecutionContext) -> str:
        """Wrap user code with security restrictions"""
        security_wrapper = f'''
import sys
import os

# Security restrictions
class SecurityError(Exception):
    pass

# Override dangerous builtins
original_import = __builtins__['__import__']
original_open = __builtins__['open']
original_eval = __builtins__['eval'] if 'eval' in __builtins__ else eval
original_exec = __builtins__['exec'] if 'exec' in __builtins__ else exec

def safe_import(name, *args, **kwargs):
    blocked_modules = {repr(context.security_policy.blocked_imports or [])}
    allowed_modules = {repr(context.security_policy.allowed_imports or [])}

    if blocked_modules and name in blocked_modules:
        raise SecurityError(f"Import of '{{name}}' is not allowed")

    if allowed_modules and name not in allowed_modules:
        raise SecurityError(f"Import of '{{name}}' is not allowed")

    return original_import(name, *args, **kwargs)

def safe_open(filename, *args, **kwargs):
    if not {context.security_policy.allow_file_system_write}:
        if len(args) > 0 and 'w' in args[0]:
            raise SecurityError("File writing is not allowed")
        if 'mode' in kwargs and 'w' in kwargs['mode']:
            raise SecurityError("File writing is not allowed")

    # Only allow access to current directory
    if os.path.isabs(filename) or '..' in filename:
        raise SecurityError(f"Access to '{{filename}}' is not allowed")

    return original_open(filename, *args, **kwargs)

def safe_eval(code, *args, **kwargs):
    if not {context.security_policy.allow_eval_exec}:
        raise SecurityError("Dynamic code execution is not allowed")
    return original_eval(code, *args, **kwargs)

def safe_exec(code, *args, **kwargs):
    if not {context.security_policy.allow_eval_exec}:
        raise SecurityError("Dynamic code execution is not allowed")
    return original_exec(code, *args, **kwargs)

# Apply restrictions
__builtins__['__import__'] = safe_import
__builtins__['open'] = safe_open
__builtins__['eval'] = safe_eval
__builtins__['exec'] = safe_exec

# User code starts here
try:
{self._indent_code(context.code, 4)}
except SecurityError as e:
    print(f"SECURITY VIOLATION: {{e}}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"EXECUTION ERROR: {{e}}", file=sys.stderr)
    sys.exit(1)
'''
        return security_wrapper

    def _indent_code(self, code: str, spaces: int) -> str:
        """Indent code by specified number of spaces"""
        indent = ' ' * spaces
        return '\n'.join(indent + line for line in code.split('\n'))

    def execute_code_async(self, context: ExecutionContext,
                          callback: callable = None) -> str:
        """
        Execute code asynchronously

        Args:
            context: Execution context
            callback: Optional callback function for results

        Returns:
            str: Execution ID for tracking
        """
        def execution_thread():
            try:
                result = self.execute_code(context)
                if callback:
                    callback(result)
            except Exception as e:
                logger.error(f"Async execution failed: {e}")
                if callback:
                    callback(ExecutionOutput(
                        execution_id=context.execution_id,
                        result=ExecutionResult.ERROR,
                        stdout="",
                        stderr=str(e),
                        exit_code=-1,
                        execution_time=0.0,
                        memory_used=0,
                        cpu_time=0.0,
                        error_message=str(e)
                    ))

        thread = threading.Thread(target=execution_thread)
        thread.daemon = True
        thread.start()

        self.active_executions[context.execution_id] = thread
        return context.execution_id

    def cancel_execution(self, execution_id: str) -> bool:
        """Cancel running execution"""
        if execution_id in self.active_executions:
            thread = self.active_executions[execution_id]
            if thread.is_alive():
                # Note: Python threads cannot be forcibly terminated
                # This would need OS-level process termination
                logger.warning(f"Cannot forcibly terminate execution {execution_id}")
                return False
            else:
                del self.active_executions[execution_id]
                return True
        return False

    def get_execution_status(self, execution_id: str) -> Optional[str]:
        """Get status of execution"""
        if execution_id in self.active_executions:
            thread = self.active_executions[execution_id]
            return "running" if thread.is_alive() else "completed"
        return None

    def cleanup(self):
        """Cleanup sandbox resources"""
        try:
            # Wait for active executions to complete
            for execution_id, thread in self.active_executions.items():
                if thread.is_alive():
                    logger.info(f"Waiting for execution {execution_id} to complete...")
                    thread.join(timeout=5.0)

            # Clean up temporary files
            if self.base_directory.exists():
                shutil.rmtree(self.base_directory, ignore_errors=True)

            # Clean up Docker resources
            if self.docker_client:
                try:
                    # Remove any leftover containers
                    containers = self.docker_client.containers.list(
                        filters={"ancestor": self.docker_image}
                    )
                    for container in containers:
                        container.remove(force=True)
                except:
                    pass

            logger.info("Sandbox cleanup completed")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

    def _cleanup_handler(self, signum, frame):
        """Signal handler for cleanup"""
        logger.info(f"Received signal {signum}, cleaning up...")
        self.cleanup()

    def create_execution_context(self, code: str, language: str = "python",
                                execution_id: str = None,
                                limits: ExecutionLimits = None,
                                security_policy: SecurityPolicy = None) -> ExecutionContext:
        """
        Factory method to create ExecutionContext

        Args:
            code: Code to execute
            language: Programming language
            execution_id: Optional execution ID
            limits: Optional execution limits
            security_policy: Optional security policy

        Returns:
            ExecutionContext: Configured execution context
        """
        return ExecutionContext(
            execution_id=execution_id or str(uuid.uuid4()),
            code=code,
            language=language,
            limits=limits or self.default_limits,
            security_policy=security_policy or self.default_security_policy
        )

    def validate_code_safety(self, code: str, language: str = "python") -> Tuple[bool, List[str]]:
        """
        Validate code for safety without execution

        Args:
            code: Code to validate
            language: Programming language

        Returns:
            Tuple of (is_safe, violations)
        """
        context = ExecutionContext(
            execution_id="validation",
            code=code,
            language=language,
            security_policy=self.default_security_policy
        )

        result = self._validate_execution_context(context)
        return (
            result.result == ExecutionResult.SUCCESS,
            result.security_violations or []
        )

    def get_supported_languages(self) -> List[str]:
        """Get list of supported programming languages"""
        return ["python"]  # Currently only Python is supported

    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution statistics"""
        return {
            "active_executions": len(self.active_executions),
            "sandbox_type": self.sandbox_type.value,
            "base_directory": str(self.base_directory),
            "docker_available": self.docker_client is not None,
            "supported_languages": self.get_supported_languages()
        }


# Factory functions for common configurations

def create_strict_sandbox() -> SandboxExecutor:
    """Create sandbox with strict security policy"""
    return SandboxExecutor(
        sandbox_type=SandboxType.DOCKER if docker else SandboxType.PROCESS
    )

def create_permissive_sandbox() -> SandboxExecutor:
    """Create sandbox with more permissive policy"""
    executor = SandboxExecutor(
        sandbox_type=SandboxType.DOCKER if docker else SandboxType.PROCESS
    )

    executor.default_security_policy = SecurityPolicy(
        allow_network=True,
        allow_file_system_write=True,
        allow_subprocess=True,
        allowed_imports=None,  # Allow all imports
        blocked_imports=[],
        blocked_system_calls=[]
    )

    executor.default_limits = ExecutionLimits(
        max_memory_mb=1024,
        max_cpu_seconds=60,
        timeout_seconds=120
    )

    return executor

def create_testing_sandbox() -> SandboxExecutor:
    """Create sandbox for testing (mock execution)"""
    return SandboxExecutor(sandbox_type=SandboxType.MOCK)