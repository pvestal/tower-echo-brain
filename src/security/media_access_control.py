#!/usr/bin/env python3
"""
Media Access Control System - Privacy Protection Implementation
Based on expert security reviews from qwen and deepseek

This module implements:
1. User consent management for personal media access
2. Access controls for photos, videos, and documents
3. Data retention policies
4. Privacy audit logging

CRITICAL: All personal media access requires explicit user consent
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ConsentLevel(Enum):
    """User consent levels for different types of data access"""
    DENIED = "denied"
    BASIC_METADATA = "basic_metadata"  # File size, creation date only
    CONTENT_ANALYSIS = "content_analysis"  # AI analysis of content
    FULL_ACCESS = "full_access"  # Complete access including training data

class MediaType(Enum):
    """Types of media that require consent"""
    PHOTOS = "photos"
    VIDEOS = "videos"
    DOCUMENTS = "documents"
    AUDIO = "audio"
    CLOUD_STORAGE = "cloud_storage"

@dataclass
class UserConsent:
    """User consent record"""
    user_id: str
    media_type: MediaType
    consent_level: ConsentLevel
    granted_at: datetime
    expires_at: Optional[datetime]
    specific_paths: Optional[List[str]] = None
    restrictions: Optional[Dict] = None

class MediaAccessControl:
    """Manages user consent and media access controls"""

    def __init__(self, consent_file_path: str = "/opt/tower-echo-brain/data/user_consents.json"):
        self.consent_file_path = consent_file_path
        self.consents: Dict[str, List[UserConsent]] = {}
        self.audit_log_path = "/opt/tower-echo-brain/logs/media_access_audit.log"

        # Protected directories that require explicit consent
        self.protected_directories = {
            "/home/patrick/Pictures",
            "/home/patrick/Videos",
            "/home/patrick/Documents",
            "/home/patrick/Downloads",
            "/home/patrick/Desktop",
            "/mnt/",  # External drives
            # Cloud storage paths (if mounted)
            "/home/patrick/Google Drive",
            "/home/patrick/OneDrive",
            "/home/patrick/Dropbox"
        }

        self.load_consents()
        self._setup_audit_logging()

    def _setup_audit_logging(self):
        """Setup audit logging for media access"""
        os.makedirs(os.path.dirname(self.audit_log_path), exist_ok=True)

        # Create audit logger
        self.audit_logger = logging.getLogger('media_access_audit')
        if not self.audit_logger.handlers:
            handler = logging.FileHandler(self.audit_log_path)
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.audit_logger.addHandler(handler)
            self.audit_logger.setLevel(logging.INFO)

    def load_consents(self):
        """Load user consents from file"""
        try:
            if os.path.exists(self.consent_file_path):
                with open(self.consent_file_path, 'r') as f:
                    data = json.load(f)

                # Convert loaded data back to UserConsent objects
                for user_id, consents_data in data.items():
                    self.consents[user_id] = []
                    for consent_data in consents_data:
                        consent = UserConsent(
                            user_id=consent_data['user_id'],
                            media_type=MediaType(consent_data['media_type']),
                            consent_level=ConsentLevel(consent_data['consent_level']),
                            granted_at=datetime.fromisoformat(consent_data['granted_at']),
                            expires_at=datetime.fromisoformat(consent_data['expires_at']) if consent_data.get('expires_at') else None,
                            specific_paths=consent_data.get('specific_paths'),
                            restrictions=consent_data.get('restrictions')
                        )
                        self.consents[user_id].append(consent)
        except Exception as e:
            logger.error(f"Failed to load consents: {e}")
            self.consents = {}

    def save_consents(self):
        """Save user consents to file"""
        try:
            os.makedirs(os.path.dirname(self.consent_file_path), exist_ok=True)

            # Convert UserConsent objects to serializable dict
            data = {}
            for user_id, consents in self.consents.items():
                data[user_id] = []
                for consent in consents:
                    consent_data = {
                        'user_id': consent.user_id,
                        'media_type': consent.media_type.value,
                        'consent_level': consent.consent_level.value,
                        'granted_at': consent.granted_at.isoformat(),
                        'expires_at': consent.expires_at.isoformat() if consent.expires_at else None,
                        'specific_paths': consent.specific_paths,
                        'restrictions': consent.restrictions
                    }
                    data[user_id].append(consent_data)

            with open(self.consent_file_path, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save consents: {e}")

    def request_consent(self, user_id: str, media_type: MediaType,
                       consent_level: ConsentLevel, duration_days: Optional[int] = None,
                       specific_paths: Optional[List[str]] = None) -> bool:
        """Request user consent for media access"""

        # Log consent request
        self.audit_logger.info(
            f"CONSENT_REQUEST - User: {user_id}, Type: {media_type.value}, "
            f"Level: {consent_level.value}, Paths: {specific_paths}"
        )

        # In a real system, this would present a consent UI to the user
        # For now, we'll require explicit approval via configuration

        print(f"\n🔒 MEDIA ACCESS CONSENT REQUEST")
        print(f"User: {user_id}")
        print(f"Media Type: {media_type.value}")
        print(f"Access Level: {consent_level.value}")
        if specific_paths:
            print(f"Specific Paths: {', '.join(specific_paths)}")
        if duration_days:
            print(f"Duration: {duration_days} days")

        # For production, replace this with actual user consent UI
        response = input("\nDo you grant this consent? (yes/no): ").lower().strip()

        if response == 'yes':
            self.grant_consent(user_id, media_type, consent_level, duration_days, specific_paths)
            return True
        else:
            self.audit_logger.info(f"CONSENT_DENIED - User: {user_id}, Type: {media_type.value}")
            return False

    def grant_consent(self, user_id: str, media_type: MediaType,
                     consent_level: ConsentLevel, duration_days: Optional[int] = None,
                     specific_paths: Optional[List[str]] = None,
                     restrictions: Optional[Dict] = None):
        """Grant user consent for media access"""

        expires_at = None
        if duration_days:
            expires_at = datetime.now() + timedelta(days=duration_days)

        consent = UserConsent(
            user_id=user_id,
            media_type=media_type,
            consent_level=consent_level,
            granted_at=datetime.now(),
            expires_at=expires_at,
            specific_paths=specific_paths,
            restrictions=restrictions
        )

        if user_id not in self.consents:
            self.consents[user_id] = []

        # Remove any existing consent for this media type
        self.consents[user_id] = [
            c for c in self.consents[user_id]
            if c.media_type != media_type
        ]

        # Add new consent
        self.consents[user_id].append(consent)
        self.save_consents()

        self.audit_logger.info(
            f"CONSENT_GRANTED - User: {user_id}, Type: {media_type.value}, "
            f"Level: {consent_level.value}, Expires: {expires_at}"
        )

    def revoke_consent(self, user_id: str, media_type: MediaType):
        """Revoke user consent for media access"""
        if user_id in self.consents:
            original_count = len(self.consents[user_id])
            self.consents[user_id] = [
                c for c in self.consents[user_id]
                if c.media_type != media_type
            ]

            if len(self.consents[user_id]) < original_count:
                self.save_consents()
                self.audit_logger.info(
                    f"CONSENT_REVOKED - User: {user_id}, Type: {media_type.value}"
                )
                return True

        return False

    def check_access_permission(self, user_id: str, file_path: str,
                               requested_access: ConsentLevel) -> bool:
        """Check if user has permission to access a file"""

        # Determine media type from file path
        media_type = self._get_media_type(file_path)
        if not media_type:
            return True  # Not a protected media type

        # Check if path is in protected directories
        if not self._is_protected_path(file_path):
            return True  # Not a protected path

        # Get user consent for this media type
        consent = self._get_active_consent(user_id, media_type)
        if not consent:
            self.audit_logger.warning(
                f"ACCESS_DENIED - No consent: User: {user_id}, Path: {file_path}"
            )
            return False

        # Check if consent level is sufficient
        if not self._consent_level_sufficient(consent.consent_level, requested_access):
            self.audit_logger.warning(
                f"ACCESS_DENIED - Insufficient consent: User: {user_id}, "
                f"Required: {requested_access.value}, Have: {consent.consent_level.value}"
            )
            return False

        # Check specific path restrictions
        if consent.specific_paths and file_path not in consent.specific_paths:
            if not any(file_path.startswith(path) for path in consent.specific_paths):
                self.audit_logger.warning(
                    f"ACCESS_DENIED - Path not in consent: User: {user_id}, Path: {file_path}"
                )
                return False

        # Log successful access
        self.audit_logger.info(
            f"ACCESS_GRANTED - User: {user_id}, Path: {file_path}, "
            f"Level: {requested_access.value}"
        )

        return True

    def _get_media_type(self, file_path: str) -> Optional[MediaType]:
        """Determine media type from file path"""
        file_path_lower = file_path.lower()

        # Image extensions
        if any(file_path_lower.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.heic', '.raw']):
            return MediaType.PHOTOS

        # Video extensions
        if any(file_path_lower.endswith(ext) for ext in ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.3gp']):
            return MediaType.VIDEOS

        # Document extensions
        if any(file_path_lower.endswith(ext) for ext in ['.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt']):
            return MediaType.DOCUMENTS

        # Audio extensions
        if any(file_path_lower.endswith(ext) for ext in ['.mp3', '.wav', '.flac', '.m4a', '.ogg']):
            return MediaType.AUDIO

        # Check for cloud storage paths
        if any(cloud in file_path_lower for cloud in ['google drive', 'onedrive', 'dropbox', 'icloud']):
            return MediaType.CLOUD_STORAGE

        return None

    def _is_protected_path(self, file_path: str) -> bool:
        """Check if file path is in protected directories"""
        file_path = os.path.abspath(file_path)
        return any(file_path.startswith(protected) for protected in self.protected_directories)

    def _get_active_consent(self, user_id: str, media_type: MediaType) -> Optional[UserConsent]:
        """Get active consent for user and media type"""
        if user_id not in self.consents:
            return None

        for consent in self.consents[user_id]:
            if consent.media_type == media_type:
                # Check if consent is still valid
                if consent.expires_at and consent.expires_at < datetime.now():
                    self.audit_logger.info(
                        f"CONSENT_EXPIRED - User: {user_id}, Type: {media_type.value}"
                    )
                    continue
                return consent

        return None

    def _consent_level_sufficient(self, granted: ConsentLevel, requested: ConsentLevel) -> bool:
        """Check if granted consent level is sufficient for requested access"""
        levels = {
            ConsentLevel.DENIED: 0,
            ConsentLevel.BASIC_METADATA: 1,
            ConsentLevel.CONTENT_ANALYSIS: 2,
            ConsentLevel.FULL_ACCESS: 3
        }

        return levels[granted] >= levels[requested]

    def get_user_consents(self, user_id: str) -> List[Dict]:
        """Get all consents for a user"""
        if user_id not in self.consents:
            return []

        result = []
        for consent in self.consents[user_id]:
            result.append({
                'media_type': consent.media_type.value,
                'consent_level': consent.consent_level.value,
                'granted_at': consent.granted_at.isoformat(),
                'expires_at': consent.expires_at.isoformat() if consent.expires_at else None,
                'specific_paths': consent.specific_paths,
                'restrictions': consent.restrictions
            })

        return result

    def cleanup_expired_consents(self):
        """Remove expired consents"""
        now = datetime.now()
        cleaned_count = 0

        for user_id in self.consents:
            original_count = len(self.consents[user_id])
            self.consents[user_id] = [
                c for c in self.consents[user_id]
                if not c.expires_at or c.expires_at > now
            ]
            cleaned_count += original_count - len(self.consents[user_id])

        if cleaned_count > 0:
            self.save_consents()
            self.audit_logger.info(f"CLEANUP - Removed {cleaned_count} expired consents")

        return cleaned_count

# Global instance
media_access_control = MediaAccessControl()

# Security decorator for functions that access media
def require_media_consent(media_type: MediaType, consent_level: ConsentLevel):
    """Decorator to require media consent for function access"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Extract user_id and file_path from function arguments
            user_id = kwargs.get('user_id') or (args[0] if args else 'default')
            file_path = kwargs.get('file_path') or (args[1] if len(args) > 1 else None)

            if file_path and not media_access_control.check_access_permission(
                user_id, file_path, consent_level
            ):
                raise PermissionError(
                    f"Access denied: User {user_id} lacks consent for {media_type.value} "
                    f"access at level {consent_level.value}"
                )

            return func(*args, **kwargs)
        return wrapper
    return decorator

if __name__ == "__main__":
    # Example usage and testing
    control = MediaAccessControl()

    # Test consent request
    control.request_consent(
        user_id="patrick",
        media_type=MediaType.PHOTOS,
        consent_level=ConsentLevel.BASIC_METADATA,
        duration_days=30
    )

    # Test access check
    allowed = control.check_access_permission(
        user_id="patrick",
        file_path="/home/patrick/Pictures/test.jpg",
        requested_access=ConsentLevel.BASIC_METADATA
    )

    print(f"Access allowed: {allowed}")