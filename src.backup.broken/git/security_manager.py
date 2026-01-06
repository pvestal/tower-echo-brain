#!/usr/bin/env python3
"""
Security and Credential Manager for Git Operations
Handles secure storage, rotation, and validation of git credentials and SSH keys
"""

import os
import asyncio
import logging
import json
import subprocess
import base64
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import yaml
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import paramiko

logger = logging.getLogger(__name__)

class CredentialType(Enum):
    """Types of credentials"""
    SSH_KEY = "ssh_key"
    HTTPS_TOKEN = "https_token"
    GITHUB_TOKEN = "github_token"
    GITLAB_TOKEN = "gitlab_token"
    API_KEY = "api_key"

class SecurityLevel(Enum):
    """Security levels for operations"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"

@dataclass
class Credential:
    """Represents a credential"""
    credential_id: str
    name: str
    credential_type: CredentialType
    encrypted_data: str
    metadata: Dict[str, Any]
    created_at: datetime
    expires_at: Optional[datetime]
    last_used: Optional[datetime]
    usage_count: int
    is_active: bool
    security_level: SecurityLevel

@dataclass
class SSHKey:
    """Represents an SSH key pair"""
    key_id: str
    name: str
    public_key: str
    private_key_encrypted: str
    fingerprint: str
    key_type: str  # rsa, ed25519, etc.
    key_size: int
    created_at: datetime
    last_used: Optional[datetime]
    associated_repos: List[str]
    is_active: bool

@dataclass
class SecurityAuditEntry:
    """Security audit log entry"""
    timestamp: datetime
    operation: str
    credential_id: Optional[str]
    user: str
    source_ip: Optional[str]
    success: bool
    details: Dict[str, Any]
    risk_level: SecurityLevel

class GitSecurityManager:
    """
    Manages security and credentials for git operations with enterprise-grade features.

    Features:
    - Encrypted credential storage
    - SSH key management and rotation
    - Access control and audit logging
    - Token validation and refresh
    - Security policy enforcement
    """

    def __init__(self, vault_path: Path = None):
        self.vault_path = vault_path or Path("/opt/tower-echo-brain/vault/git_credentials")
        self.ssh_keys_path = self.vault_path / "ssh_keys"
        self.credentials_path = self.vault_path / "credentials"
        self.audit_log_path = self.vault_path / "audit.log"
        self.security_config_path = self.vault_path / "security_config.yaml"

        # Ensure directories exist
        self.vault_path.mkdir(parents=True, exist_ok=True)
        self.ssh_keys_path.mkdir(parents=True, exist_ok=True)
        self.credentials_path.mkdir(parents=True, exist_ok=True)

        # Encryption setup
        self.encryption_key = self._get_or_create_encryption_key()
        self.fernet = Fernet(self.encryption_key)

        # Storage
        self.credentials: Dict[str, Credential] = {}
        self.ssh_keys: Dict[str, SSHKey] = {}
        self.audit_entries: List[SecurityAuditEntry] = []

        # Security configuration
        self.security_config = {
            'max_credential_age_days': 90,
            'ssh_key_rotation_days': 365,
            'failed_auth_lockout_threshold': 5,
            'audit_retention_days': 365,
            'require_2fa': False,
            'allowed_key_types': ['rsa', 'ed25519'],
            'minimum_key_size': 2048,
            'token_refresh_threshold_hours': 24
        }

    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key for credential storage"""
        key_file = self.vault_path / ".encryption_key"

        if key_file.exists():
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            # Generate new key
            key = Fernet.generate_key()

            # Secure the key file
            with open(key_file, 'wb') as f:
                f.write(key)
            os.chmod(key_file, 0o600)  # Owner read/write only

            return key

    async def initialize(self) -> bool:
        """Initialize the security manager"""
        try:
            logger.info("Initializing Git Security Manager...")

            # Load existing credentials and keys
            await self._load_credentials()
            await self._load_ssh_keys()
            await self._load_security_config()
            await self._load_audit_log()

            # Validate existing credentials
            await self._validate_credentials()

            # Setup SSH agent
            await self._setup_ssh_agent()

            logger.info(f"Security Manager initialized with {len(self.credentials)} credentials and {len(self.ssh_keys)} SSH keys")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Security Manager: {e}")
            return False

    async def _load_credentials(self):
        """Load encrypted credentials from storage"""
        try:
            credentials_file = self.credentials_path / "credentials.json"
            if credentials_file.exists():
                with open(credentials_file, 'r') as f:
                    encrypted_data = json.load(f)

                for cred_data in encrypted_data.get('credentials', []):
                    try:
                        # Decrypt credential data
                        decrypted_data = self.fernet.decrypt(cred_data['encrypted_data'].encode()).decode()

                        credential = Credential(
                            credential_id=cred_data['credential_id'],
                            name=cred_data['name'],
                            credential_type=CredentialType(cred_data['credential_type']),
                            encrypted_data=cred_data['encrypted_data'],
                            metadata=cred_data.get('metadata', {}),
                            created_at=datetime.fromisoformat(cred_data['created_at']),
                            expires_at=datetime.fromisoformat(cred_data['expires_at']) if cred_data.get('expires_at') else None,
                            last_used=datetime.fromisoformat(cred_data['last_used']) if cred_data.get('last_used') else None,
                            usage_count=cred_data.get('usage_count', 0),
                            is_active=cred_data.get('is_active', True),
                            security_level=SecurityLevel(cred_data.get('security_level', 'internal'))
                        )

                        self.credentials[credential.credential_id] = credential

                    except Exception as e:
                        logger.warning(f"Failed to decrypt credential {cred_data.get('credential_id')}: {e}")

                logger.info(f"Loaded {len(self.credentials)} credentials")

        except Exception as e:
            logger.warning(f"Failed to load credentials: {e}")

    async def _load_ssh_keys(self):
        """Load SSH keys from storage"""
        try:
            ssh_keys_file = self.ssh_keys_path / "ssh_keys.json"
            if ssh_keys_file.exists():
                with open(ssh_keys_file, 'r') as f:
                    keys_data = json.load(f)

                for key_data in keys_data.get('ssh_keys', []):
                    try:
                        ssh_key = SSHKey(
                            key_id=key_data['key_id'],
                            name=key_data['name'],
                            public_key=key_data['public_key'],
                            private_key_encrypted=key_data['private_key_encrypted'],
                            fingerprint=key_data['fingerprint'],
                            key_type=key_data['key_type'],
                            key_size=key_data['key_size'],
                            created_at=datetime.fromisoformat(key_data['created_at']),
                            last_used=datetime.fromisoformat(key_data['last_used']) if key_data.get('last_used') else None,
                            associated_repos=key_data.get('associated_repos', []),
                            is_active=key_data.get('is_active', True)
                        )

                        self.ssh_keys[ssh_key.key_id] = ssh_key

                    except Exception as e:
                        logger.warning(f"Failed to load SSH key {key_data.get('key_id')}: {e}")

                logger.info(f"Loaded {len(self.ssh_keys)} SSH keys")

        except Exception as e:
            logger.warning(f"Failed to load SSH keys: {e}")

    async def _load_security_config(self):
        """Load security configuration"""
        try:
            if self.security_config_path.exists():
                with open(self.security_config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                    self.security_config.update(config_data)

                logger.info("Loaded security configuration")
            else:
                await self._save_security_config()

        except Exception as e:
            logger.warning(f"Failed to load security config: {e}")

    async def _save_security_config(self):
        """Save security configuration"""
        try:
            with open(self.security_config_path, 'w') as f:
                yaml.dump(self.security_config, f, default_flow_style=False)
        except Exception as e:
            logger.error(f"Failed to save security config: {e}")

    async def _load_audit_log(self):
        """Load audit log entries"""
        try:
            if self.audit_log_path.exists():
                with open(self.audit_log_path, 'r') as f:
                    for line in f:
                        try:
                            entry_data = json.loads(line.strip())
                            entry = SecurityAuditEntry(
                                timestamp=datetime.fromisoformat(entry_data['timestamp']),
                                operation=entry_data['operation'],
                                credential_id=entry_data.get('credential_id'),
                                user=entry_data['user'],
                                source_ip=entry_data.get('source_ip'),
                                success=entry_data['success'],
                                details=entry_data.get('details', {}),
                                risk_level=SecurityLevel(entry_data.get('risk_level', 'internal'))
                            )
                            self.audit_entries.append(entry)
                        except Exception as e:
                            logger.warning(f"Failed to parse audit entry: {e}")

                # Keep only recent entries
                cutoff_date = datetime.now() - timedelta(days=self.security_config['audit_retention_days'])
                self.audit_entries = [e for e in self.audit_entries if e.timestamp > cutoff_date]

                logger.info(f"Loaded {len(self.audit_entries)} audit entries")

        except Exception as e:
            logger.warning(f"Failed to load audit log: {e}")

    async def _validate_credentials(self):
        """Validate and cleanup expired credentials"""
        now = datetime.now()
        expired_credentials = []

        for cred_id, credential in self.credentials.items():
            if credential.expires_at and credential.expires_at < now:
                expired_credentials.append(cred_id)
                logger.info(f"Credential {credential.name} has expired")

        # Deactivate expired credentials
        for cred_id in expired_credentials:
            self.credentials[cred_id].is_active = False
            await self._audit_log("credential_expired", cred_id, "system", True, {
                'credential_name': self.credentials[cred_id].name
            })

        if expired_credentials:
            await self._save_credentials()

    async def _setup_ssh_agent(self):
        """Setup SSH agent with managed keys"""
        try:
            # Check if SSH agent is running
            ssh_auth_sock = os.environ.get('SSH_AUTH_SOCK')
            if not ssh_auth_sock:
                logger.warning("SSH agent not running")
                return

            # Add active SSH keys to agent
            for key_id, ssh_key in self.ssh_keys.items():
                if ssh_key.is_active:
                    await self._add_key_to_ssh_agent(ssh_key)

        except Exception as e:
            logger.warning(f"Failed to setup SSH agent: {e}")

    async def _add_key_to_ssh_agent(self, ssh_key: SSHKey):
        """Add SSH key to SSH agent"""
        try:
            # Decrypt private key
            private_key_data = self.fernet.decrypt(ssh_key.private_key_encrypted.encode()).decode()

            # Write to temporary file
            temp_key_file = f"/tmp/ssh_key_{ssh_key.key_id}"
            with open(temp_key_file, 'w') as f:
                f.write(private_key_data)
            os.chmod(temp_key_file, 0o600)

            # Add to SSH agent
            result = subprocess.run(
                ['ssh-add', temp_key_file],
                capture_output=True,
                text=True
            )

            # Remove temporary file
            os.unlink(temp_key_file)

            if result.returncode == 0:
                logger.info(f"Added SSH key {ssh_key.name} to agent")
            else:
                logger.warning(f"Failed to add SSH key {ssh_key.name} to agent: {result.stderr}")

        except Exception as e:
            logger.error(f"Failed to add SSH key {ssh_key.key_id} to agent: {e}")

    async def _save_credentials(self):
        """Save encrypted credentials to storage"""
        try:
            credentials_data = {
                'credentials': [
                    {
                        'credential_id': cred.credential_id,
                        'name': cred.name,
                        'credential_type': cred.credential_type.value,
                        'encrypted_data': cred.encrypted_data,
                        'metadata': cred.metadata,
                        'created_at': cred.created_at.isoformat(),
                        'expires_at': cred.expires_at.isoformat() if cred.expires_at else None,
                        'last_used': cred.last_used.isoformat() if cred.last_used else None,
                        'usage_count': cred.usage_count,
                        'is_active': cred.is_active,
                        'security_level': cred.security_level.value
                    }
                    for cred in self.credentials.values()
                ],
                'last_updated': datetime.now().isoformat()
            }

            credentials_file = self.credentials_path / "credentials.json"
            with open(credentials_file, 'w') as f:
                json.dump(credentials_data, f, indent=2)

            # Secure the file
            os.chmod(credentials_file, 0o600)
            logger.info(f"Saved {len(self.credentials)} credentials")

        except Exception as e:
            logger.error(f"Failed to save credentials: {e}")

    async def _save_ssh_keys(self):
        """Save SSH keys to storage"""
        try:
            keys_data = {
                'ssh_keys': [
                    {
                        'key_id': key.key_id,
                        'name': key.name,
                        'public_key': key.public_key,
                        'private_key_encrypted': key.private_key_encrypted,
                        'fingerprint': key.fingerprint,
                        'key_type': key.key_type,
                        'key_size': key.key_size,
                        'created_at': key.created_at.isoformat(),
                        'last_used': key.last_used.isoformat() if key.last_used else None,
                        'associated_repos': key.associated_repos,
                        'is_active': key.is_active
                    }
                    for key in self.ssh_keys.values()
                ],
                'last_updated': datetime.now().isoformat()
            }

            ssh_keys_file = self.ssh_keys_path / "ssh_keys.json"
            with open(ssh_keys_file, 'w') as f:
                json.dump(keys_data, f, indent=2)

            # Secure the file
            os.chmod(ssh_keys_file, 0o600)
            logger.info(f"Saved {len(self.ssh_keys)} SSH keys")

        except Exception as e:
            logger.error(f"Failed to save SSH keys: {e}")

    async def _audit_log(
        self,
        operation: str,
        credential_id: Optional[str],
        user: str,
        success: bool,
        details: Dict[str, Any],
        risk_level: SecurityLevel = SecurityLevel.INTERNAL
    ):
        """Add entry to audit log"""
        try:
            entry = SecurityAuditEntry(
                timestamp=datetime.now(),
                operation=operation,
                credential_id=credential_id,
                user=user,
                source_ip=None,  # Could be enhanced to detect source IP
                success=success,
                details=details,
                risk_level=risk_level
            )

            self.audit_entries.append(entry)

            # Write to log file
            log_data = {
                'timestamp': entry.timestamp.isoformat(),
                'operation': entry.operation,
                'credential_id': entry.credential_id,
                'user': entry.user,
                'source_ip': entry.source_ip,
                'success': entry.success,
                'details': entry.details,
                'risk_level': entry.risk_level.value
            }

            with open(self.audit_log_path, 'a') as f:
                f.write(json.dumps(log_data) + '\n')

        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")

    # Public API Methods

    async def create_ssh_key_pair(
        self,
        name: str,
        key_type: str = "ed25519",
        key_size: int = None,
        comment: str = None
    ) -> Tuple[bool, Optional[str]]:
        """Create a new SSH key pair"""
        try:
            # Validate key type
            if key_type not in self.security_config['allowed_key_types']:
                return False, f"Key type {key_type} not allowed"

            # Set default key size
            if not key_size:
                key_size = 4096 if key_type == "rsa" else 256

            if key_type == "rsa" and key_size < self.security_config['minimum_key_size']:
                return False, f"Key size {key_size} below minimum {self.security_config['minimum_key_size']}"

            # Generate key pair
            if key_type == "rsa":
                private_key = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=key_size
                )
                public_key = private_key.public_key()

                # Serialize private key
                private_pem = private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.OpenSSH,
                    encryption_algorithm=serialization.NoEncryption()
                ).decode()

                # Serialize public key
                public_ssh = public_key.public_bytes(
                    encoding=serialization.Encoding.OpenSSH,
                    format=serialization.PublicFormat.OpenSSH
                ).decode()

            elif key_type == "ed25519":
                # Use ssh-keygen for ed25519
                key_id = f"echo_brain_{name}_{int(datetime.now().timestamp())}"
                temp_key_path = f"/tmp/{key_id}"

                result = subprocess.run([
                    'ssh-keygen', '-t', 'ed25519', '-f', temp_key_path, '-N', '',
                    '-C', comment or f"Echo Brain {name}"
                ], capture_output=True, text=True)

                if result.returncode != 0:
                    return False, f"Failed to generate key: {result.stderr}"

                # Read generated keys
                with open(temp_key_path, 'r') as f:
                    private_pem = f.read()
                with open(f"{temp_key_path}.pub", 'r') as f:
                    public_ssh = f.read()

                # Remove temporary files
                os.unlink(temp_key_path)
                os.unlink(f"{temp_key_path}.pub")

            else:
                return False, f"Unsupported key type: {key_type}"

            # Calculate fingerprint
            fingerprint = self._calculate_ssh_fingerprint(public_ssh)

            # Encrypt private key
            encrypted_private_key = self.fernet.encrypt(private_pem.encode()).decode()

            # Create SSH key object
            ssh_key = SSHKey(
                key_id=key_id if key_type == "ed25519" else f"echo_brain_{name}_{int(datetime.now().timestamp())}",
                name=name,
                public_key=public_ssh.strip(),
                private_key_encrypted=encrypted_private_key,
                fingerprint=fingerprint,
                key_type=key_type,
                key_size=key_size,
                created_at=datetime.now(),
                last_used=None,
                associated_repos=[],
                is_active=True
            )

            # Store SSH key
            self.ssh_keys[ssh_key.key_id] = ssh_key
            await self._save_ssh_keys()

            # Add to SSH agent
            await self._add_key_to_ssh_agent(ssh_key)

            # Audit log
            await self._audit_log(
                "ssh_key_created",
                ssh_key.key_id,
                "echo_brain",
                True,
                {'name': name, 'key_type': key_type, 'key_size': key_size}
            )

            logger.info(f"Created SSH key pair: {name} ({key_type})")
            return True, ssh_key.key_id

        except Exception as e:
            await self._audit_log(
                "ssh_key_create_failed",
                None,
                "echo_brain",
                False,
                {'name': name, 'error': str(e)}
            )
            logger.error(f"Failed to create SSH key pair: {e}")
            return False, str(e)

    def _calculate_ssh_fingerprint(self, public_key: str) -> str:
        """Calculate SSH key fingerprint"""
        try:
            # Extract key data (remove type and comment)
            key_parts = public_key.strip().split()
            if len(key_parts) >= 2:
                key_data = base64.b64decode(key_parts[1])
                fingerprint = hashlib.sha256(key_data).digest()
                return base64.b64encode(fingerprint).decode().rstrip('=')
            else:
                return "unknown"
        except Exception:
            return "unknown"

    async def store_credential(
        self,
        name: str,
        credential_type: CredentialType,
        credential_data: str,
        metadata: Optional[Dict[str, Any]] = None,
        expires_at: Optional[datetime] = None,
        security_level: SecurityLevel = SecurityLevel.INTERNAL
    ) -> Tuple[bool, Optional[str]]:
        """Store a new credential"""
        try:
            credential_id = f"{credential_type.value}_{hashlib.md5(name.encode()).hexdigest()[:8]}"

            # Encrypt credential data
            encrypted_data = self.fernet.encrypt(credential_data.encode()).decode()

            # Create credential object
            credential = Credential(
                credential_id=credential_id,
                name=name,
                credential_type=credential_type,
                encrypted_data=encrypted_data,
                metadata=metadata or {},
                created_at=datetime.now(),
                expires_at=expires_at,
                last_used=None,
                usage_count=0,
                is_active=True,
                security_level=security_level
            )

            # Store credential
            self.credentials[credential_id] = credential
            await self._save_credentials()

            # Audit log
            await self._audit_log(
                "credential_stored",
                credential_id,
                "echo_brain",
                True,
                {'name': name, 'type': credential_type.value},
                security_level
            )

            logger.info(f"Stored credential: {name} ({credential_type.value})")
            return True, credential_id

        except Exception as e:
            await self._audit_log(
                "credential_store_failed",
                None,
                "echo_brain",
                False,
                {'name': name, 'error': str(e)}
            )
            logger.error(f"Failed to store credential: {e}")
            return False, str(e)

    async def get_credential(self, credential_id: str) -> Tuple[bool, Optional[str]]:
        """Get decrypted credential data"""
        try:
            credential = self.credentials.get(credential_id)
            if not credential:
                return False, "Credential not found"

            if not credential.is_active:
                return False, "Credential is inactive"

            if credential.expires_at and credential.expires_at < datetime.now():
                return False, "Credential has expired"

            # Decrypt credential data
            decrypted_data = self.fernet.decrypt(credential.encrypted_data.encode()).decode()

            # Update usage statistics
            credential.last_used = datetime.now()
            credential.usage_count += 1
            await self._save_credentials()

            # Audit log
            await self._audit_log(
                "credential_accessed",
                credential_id,
                "echo_brain",
                True,
                {'name': credential.name}
            )

            return True, decrypted_data

        except Exception as e:
            await self._audit_log(
                "credential_access_failed",
                credential_id,
                "echo_brain",
                False,
                {'error': str(e)}
            )
            logger.error(f"Failed to get credential {credential_id}: {e}")
            return False, str(e)

    async def rotate_ssh_key(self, key_id: str) -> Tuple[bool, Optional[str]]:
        """Rotate an SSH key pair"""
        try:
            old_key = self.ssh_keys.get(key_id)
            if not old_key:
                return False, "SSH key not found"

            # Create new key with same parameters
            success, new_key_id = await self.create_ssh_key_pair(
                name=f"{old_key.name}_rotated",
                key_type=old_key.key_type,
                key_size=old_key.key_size
            )

            if not success:
                return False, new_key_id

            # Transfer repository associations
            new_key = self.ssh_keys[new_key_id]
            new_key.associated_repos = old_key.associated_repos.copy()

            # Deactivate old key
            old_key.is_active = False

            await self._save_ssh_keys()

            # Audit log
            await self._audit_log(
                "ssh_key_rotated",
                key_id,
                "echo_brain",
                True,
                {'old_key': old_key.name, 'new_key': new_key.name}
            )

            logger.info(f"Rotated SSH key: {old_key.name} -> {new_key.name}")
            return True, new_key_id

        except Exception as e:
            await self._audit_log(
                "ssh_key_rotation_failed",
                key_id,
                "echo_brain",
                False,
                {'error': str(e)}
            )
            logger.error(f"Failed to rotate SSH key {key_id}: {e}")
            return False, str(e)

    async def validate_github_token(self, token: str) -> Tuple[bool, Dict[str, Any]]:
        """Validate GitHub token and get permissions"""
        try:
            import aiohttp
            headers = {
                'Authorization': f'token {token}',
                'Accept': 'application/vnd.github.v3+json'
            }

            async with aiohttp.ClientSession() as session:
                async with session.get('https://api.github.com/user', headers=headers) as response:
                    if response.status == 200:
                        user_data = await response.json()

                        # Check token scopes
                        scopes = response.headers.get('X-OAuth-Scopes', '').split(', ')

                        return True, {
                            'username': user_data.get('login'),
                            'user_id': user_data.get('id'),
                            'scopes': scopes,
                            'rate_limit_remaining': response.headers.get('X-RateLimit-Remaining'),
                            'valid': True
                        }
                    else:
                        return False, {'error': f'HTTP {response.status}', 'valid': False}

        except Exception as e:
            return False, {'error': str(e), 'valid': False}

    async def get_security_status(self) -> Dict[str, Any]:
        """Get security status and statistics"""
        now = datetime.now()

        # Count credentials by type and status
        active_credentials = len([c for c in self.credentials.values() if c.is_active])
        expired_credentials = len([
            c for c in self.credentials.values()
            if c.expires_at and c.expires_at < now
        ])

        # Count SSH keys
        active_ssh_keys = len([k for k in self.ssh_keys.values() if k.is_active])

        # Recent audit events
        recent_events = [
            e for e in self.audit_entries
            if e.timestamp > now - timedelta(days=7)
        ]

        failed_events = [e for e in recent_events if not e.success]

        return {
            'credentials': {
                'total': len(self.credentials),
                'active': active_credentials,
                'expired': expired_credentials
            },
            'ssh_keys': {
                'total': len(self.ssh_keys),
                'active': active_ssh_keys
            },
            'audit': {
                'total_entries': len(self.audit_entries),
                'recent_events': len(recent_events),
                'failed_events': len(failed_events)
            },
            'security_config': {
                'credential_max_age': self.security_config['max_credential_age_days'],
                'ssh_rotation_period': self.security_config['ssh_key_rotation_days'],
                'audit_retention': self.security_config['audit_retention_days']
            }
        }

    async def export_public_keys(self, output_path: Optional[Path] = None) -> Tuple[bool, str]:
        """Export all active public keys"""
        try:
            if not output_path:
                output_path = self.vault_path / "authorized_keys"

            public_keys = []
            for ssh_key in self.ssh_keys.values():
                if ssh_key.is_active:
                    public_keys.append(f"# {ssh_key.name} ({ssh_key.key_type})")
                    public_keys.append(ssh_key.public_key)
                    public_keys.append("")

            with open(output_path, 'w') as f:
                f.write('\n'.join(public_keys))

            logger.info(f"Exported {len([k for k in self.ssh_keys.values() if k.is_active])} public keys to {output_path}")
            return True, str(output_path)

        except Exception as e:
            logger.error(f"Failed to export public keys: {e}")
            return False, str(e)

    async def cleanup_expired_credentials(self) -> Dict[str, int]:
        """Cleanup expired credentials and old SSH keys"""
        now = datetime.now()

        # Remove expired credentials
        expired_creds = [
            cred_id for cred_id, cred in self.credentials.items()
            if cred.expires_at and cred.expires_at < now
        ]

        for cred_id in expired_creds:
            del self.credentials[cred_id]

        # Remove old inactive SSH keys (older than 1 year)
        cutoff_date = now - timedelta(days=365)
        old_keys = [
            key_id for key_id, key in self.ssh_keys.items()
            if not key.is_active and key.created_at < cutoff_date
        ]

        for key_id in old_keys:
            del self.ssh_keys[key_id]

        # Cleanup audit log
        audit_cutoff = now - timedelta(days=self.security_config['audit_retention_days'])
        old_entries = len(self.audit_entries)
        self.audit_entries = [e for e in self.audit_entries if e.timestamp > audit_cutoff]

        # Save changes
        await self._save_credentials()
        await self._save_ssh_keys()

        return {
            'expired_credentials_removed': len(expired_creds),
            'old_ssh_keys_removed': len(old_keys),
            'audit_entries_cleaned': old_entries - len(self.audit_entries)
        }


# Global instance
git_security_manager = GitSecurityManager()

async def test_security_manager():
    """Test the security manager"""
    security_manager = git_security_manager

    # Initialize
    success = await security_manager.initialize()
    if not success:
        print("❌ Failed to initialize security manager")
        return

    print("✅ Security manager initialized")

    # Create SSH key pair
    success, key_id = await security_manager.create_ssh_key_pair(
        "test_key",
        key_type="ed25519"
    )

    if success:
        print(f"✅ Created SSH key pair: {key_id}")
    else:
        print(f"❌ Failed to create SSH key: {key_id}")

    # Store a credential
    success, cred_id = await security_manager.store_credential(
        "github_token",
        CredentialType.GITHUB_TOKEN,
        "ghp_example_token_data",
        metadata={"scope": "repo"},
        expires_at=datetime.now() + timedelta(days=90)
    )

    if success:
        print(f"✅ Stored credential: {cred_id}")
    else:
        print(f"❌ Failed to store credential: {cred_id}")

    # Get security status
    status = await security_manager.get_security_status()
    print(f"\n📊 Security Status:")
    print(f"  Credentials: {status['credentials']['active']}/{status['credentials']['total']} active")
    print(f"  SSH Keys: {status['ssh_keys']['active']}/{status['ssh_keys']['total']} active")
    print(f"  Audit Events: {status['audit']['recent_events']} recent")

    print("\n✅ Security manager test complete")


if __name__ == "__main__":
    asyncio.run(test_security_manager())