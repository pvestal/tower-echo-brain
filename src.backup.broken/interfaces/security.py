#!/usr/bin/env python3
"""
Security Interface Protocols
Defines contracts for authentication, authorization, and security systems
"""

from typing import Protocol, runtime_checkable, List, Dict, Optional, Any
from datetime import datetime
from enum import Enum

class UserRole(Enum):
    """User role enumeration"""
    GUEST = "guest"
    USER = "user"
    ADMIN = "admin"
    SYSTEM = "system"
    DEVELOPER = "developer"

class PermissionLevel(Enum):
    """Permission level enumeration"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    SYSTEM = "system"

class SecurityEventType(Enum):
    """Security event type enumeration"""
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    PERMISSION_DENIED = "permission_denied"
    DATA_ACCESS = "data_access"
    CONFIGURATION_CHANGE = "configuration_change"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"

@runtime_checkable
class AuthenticationInterface(Protocol):
    """
    Protocol for authentication systems

    Defines standardized methods for user authentication,
    session management, and credential validation.
    """

    async def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """
        Authenticate user credentials

        Args:
            username: Username or email
            password: User password

        Returns:
            Optional[Dict]: User data if authenticated, None otherwise
        """
        ...

    async def create_session(self, user_id: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create user session

        Args:
            user_id: User identifier
            metadata: Optional session metadata

        Returns:
            str: Session token
        """
        ...

    async def validate_session(self, session_token: str) -> Optional[Dict[str, Any]]:
        """
        Validate session token

        Args:
            session_token: Session token to validate

        Returns:
            Optional[Dict]: Session data if valid, None otherwise
        """
        ...

    async def refresh_session(self, session_token: str) -> Optional[str]:
        """
        Refresh session token

        Args:
            session_token: Current session token

        Returns:
            Optional[str]: New session token or None if invalid
        """
        ...

    async def logout_user(self, session_token: str) -> bool:
        """
        Logout user and invalidate session

        Args:
            session_token: Session token to invalidate

        Returns:
            bool: True if logout successful
        """
        ...

    async def create_user(self, username: str, email: str, password: str,
                         role: UserRole = UserRole.USER) -> Optional[str]:
        """
        Create new user account

        Args:
            username: Username
            email: Email address
            password: Password
            role: User role

        Returns:
            Optional[str]: User ID if created successfully, None otherwise
        """
        ...

    async def update_password(self, user_id: str, old_password: str,
                            new_password: str) -> bool:
        """
        Update user password

        Args:
            user_id: User identifier
            old_password: Current password
            new_password: New password

        Returns:
            bool: True if password updated successfully
        """
        ...

    async def reset_password(self, email: str) -> bool:
        """
        Initiate password reset process

        Args:
            email: User email address

        Returns:
            bool: True if reset initiated successfully
        """
        ...

    async def validate_password_reset(self, reset_token: str, new_password: str) -> bool:
        """
        Validate password reset token and set new password

        Args:
            reset_token: Password reset token
            new_password: New password

        Returns:
            bool: True if password reset successfully
        """
        ...

    async def enable_two_factor(self, user_id: str) -> Dict[str, Any]:
        """
        Enable two-factor authentication for user

        Args:
            user_id: User identifier

        Returns:
            Dict: 2FA setup information (QR code, backup codes, etc.)
        """
        ...

    async def verify_two_factor(self, user_id: str, code: str) -> bool:
        """
        Verify two-factor authentication code

        Args:
            user_id: User identifier
            code: 2FA code

        Returns:
            bool: True if code is valid
        """
        ...


@runtime_checkable
class AuthorizationInterface(Protocol):
    """
    Protocol for authorization and permission systems
    """

    async def check_permission(self, user_id: str, resource: str,
                             permission: PermissionLevel) -> bool:
        """
        Check if user has permission for resource

        Args:
            user_id: User identifier
            resource: Resource identifier
            permission: Required permission level

        Returns:
            bool: True if user has permission
        """
        ...

    async def grant_permission(self, user_id: str, resource: str,
                             permission: PermissionLevel) -> bool:
        """
        Grant permission to user for resource

        Args:
            user_id: User identifier
            resource: Resource identifier
            permission: Permission level to grant

        Returns:
            bool: True if permission granted successfully
        """
        ...

    async def revoke_permission(self, user_id: str, resource: str,
                              permission: PermissionLevel) -> bool:
        """
        Revoke permission from user for resource

        Args:
            user_id: User identifier
            resource: Resource identifier
            permission: Permission level to revoke

        Returns:
            bool: True if permission revoked successfully
        """
        ...

    async def list_user_permissions(self, user_id: str) -> List[Dict[str, Any]]:
        """
        List all permissions for a user

        Args:
            user_id: User identifier

        Returns:
            List[Dict]: List of user permissions
        """
        ...

    async def list_resource_permissions(self, resource: str) -> List[Dict[str, Any]]:
        """
        List all permissions for a resource

        Args:
            resource: Resource identifier

        Returns:
            List[Dict]: List of resource permissions
        """
        ...

    async def create_role(self, role_name: str, permissions: List[str]) -> bool:
        """
        Create a new role with specified permissions

        Args:
            role_name: Name of the role
            permissions: List of permission identifiers

        Returns:
            bool: True if role created successfully
        """
        ...

    async def assign_role(self, user_id: str, role_name: str) -> bool:
        """
        Assign role to user

        Args:
            user_id: User identifier
            role_name: Role name to assign

        Returns:
            bool: True if role assigned successfully
        """
        ...

    async def remove_role(self, user_id: str, role_name: str) -> bool:
        """
        Remove role from user

        Args:
            user_id: User identifier
            role_name: Role name to remove

        Returns:
            bool: True if role removed successfully
        """
        ...


@runtime_checkable
class SecurityAuditInterface(Protocol):
    """
    Protocol for security auditing and monitoring
    """

    async def log_security_event(self, event_type: SecurityEventType,
                                user_id: Optional[str], resource: Optional[str],
                                details: Dict[str, Any]) -> str:
        """
        Log a security event

        Args:
            event_type: Type of security event
            user_id: Optional user identifier
            resource: Optional resource identifier
            details: Event details

        Returns:
            str: Event ID
        """
        ...

    async def get_security_events(self, user_id: Optional[str] = None,
                                event_type: Optional[SecurityEventType] = None,
                                start_date: Optional[datetime] = None,
                                end_date: Optional[datetime] = None,
                                limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get security events with optional filtering

        Args:
            user_id: Optional user filter
            event_type: Optional event type filter
            start_date: Optional start date filter
            end_date: Optional end date filter
            limit: Maximum events to return

        Returns:
            List[Dict]: List of security events
        """
        ...

    async def detect_suspicious_activity(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Detect suspicious activity for a user

        Args:
            user_id: User identifier

        Returns:
            List[Dict]: List of suspicious activities detected
        """
        ...

    async def generate_security_report(self, start_date: datetime,
                                     end_date: datetime) -> Dict[str, Any]:
        """
        Generate security report for date range

        Args:
            start_date: Report start date
            end_date: Report end date

        Returns:
            Dict: Security report data
        """
        ...


@runtime_checkable
class EncryptionInterface(Protocol):
    """
    Protocol for encryption and data protection
    """

    def encrypt_data(self, data: str, key: Optional[str] = None) -> str:
        """
        Encrypt data

        Args:
            data: Data to encrypt
            key: Optional encryption key

        Returns:
            str: Encrypted data
        """
        ...

    def decrypt_data(self, encrypted_data: str, key: Optional[str] = None) -> str:
        """
        Decrypt data

        Args:
            encrypted_data: Data to decrypt
            key: Optional decryption key

        Returns:
            str: Decrypted data
        """
        ...

    def hash_password(self, password: str) -> str:
        """
        Hash password securely

        Args:
            password: Password to hash

        Returns:
            str: Hashed password
        """
        ...

    def verify_password(self, password: str, hashed_password: str) -> bool:
        """
        Verify password against hash

        Args:
            password: Plain text password
            hashed_password: Hashed password

        Returns:
            bool: True if password matches hash
        """
        ...

    def generate_token(self, data: Dict[str, Any], expiry_hours: int = 24) -> str:
        """
        Generate secure token

        Args:
            data: Data to encode in token
            expiry_hours: Token expiry time in hours

        Returns:
            str: Generated token
        """
        ...

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify and decode token

        Args:
            token: Token to verify

        Returns:
            Optional[Dict]: Decoded token data or None if invalid
        """
        ...