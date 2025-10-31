#!/usr/bin/env python3
"""
Secure Media Scanner - Privacy-Protected Version
Replaces the unauthorized echo_media_scanner.py with security controls

This module implements:
1. Explicit user consent before any media scanning
2. Granular permissions for different types of analysis
3. Data retention policies
4. Privacy audit trails

CRITICAL: No media scanning without explicit user consent
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Import security controls
sys.path.append('/opt/tower-echo-brain/src')
from security.media_access_control import (
    MediaAccessControl, MediaType, ConsentLevel,
    require_media_consent, media_access_control
)

logger = logging.getLogger(__name__)

class SecureMediaScanner:
    """Privacy-protected media scanner with explicit consent requirements"""

    def __init__(self):
        self.access_control = media_access_control
        self.scan_results = {}

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        logger.info("Secure Media Scanner initialized with privacy protections")

    @require_media_consent(MediaType.PHOTOS, ConsentLevel.BASIC_METADATA)
    def scan_photos_metadata(self, user_id: str, directory_path: str) -> Dict:
        """Scan photo metadata only (with consent)"""
        logger.info(f"Scanning photo metadata for user {user_id} in {directory_path}")

        if not self.access_control.check_access_permission(
            user_id, directory_path, ConsentLevel.BASIC_METADATA
        ):
            raise PermissionError(f"No consent for metadata access: {directory_path}")

        results = {
            "directory": directory_path,
            "scanned_files": 0,
            "total_size": 0,
            "file_types": {},
            "date_range": {"earliest": None, "latest": None}
        }

        try:
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if self._is_image_file(file_path):
                        # Only basic metadata - no content analysis
                        try:
                            stat = os.stat(file_path)
                            results["scanned_files"] += 1
                            results["total_size"] += stat.st_size

                            # File type counting
                            ext = Path(file_path).suffix.lower()
                            results["file_types"][ext] = results["file_types"].get(ext, 0) + 1

                            # Date tracking
                            mod_time = datetime.fromtimestamp(stat.st_mtime)
                            if not results["date_range"]["earliest"] or mod_time < results["date_range"]["earliest"]:
                                results["date_range"]["earliest"] = mod_time
                            if not results["date_range"]["latest"] or mod_time > results["date_range"]["latest"]:
                                results["date_range"]["latest"] = mod_time

                        except Exception as e:
                            logger.error(f"Error processing {file_path}: {e}")

        except Exception as e:
            logger.error(f"Error scanning directory {directory_path}: {e}")

        return results

    @require_media_consent(MediaType.PHOTOS, ConsentLevel.CONTENT_ANALYSIS)
    def analyze_photo_content(self, user_id: str, file_path: str) -> Dict:
        """Analyze photo content with AI (requires higher consent level)"""
        logger.info(f"Analyzing photo content for user {user_id}: {file_path}")

        if not self.access_control.check_access_permission(
            user_id, file_path, ConsentLevel.CONTENT_ANALYSIS
        ):
            raise PermissionError(f"No consent for content analysis: {file_path}")

        # This would integrate with Ollama for content analysis
        # For now, returning a placeholder
        return {
            "file_path": file_path,
            "analysis_type": "content_analysis",
            "consent_verified": True,
            "analyzed_at": datetime.now().isoformat(),
            "content": "AI analysis would go here (with consent)"
        }

    def request_scanning_consent(self, user_id: str) -> bool:
        """Request comprehensive consent for media scanning"""
        print("\n🔒 MEDIA SCANNING CONSENT REQUEST")
        print("="*50)
        print("The system is requesting permission to scan and analyze your personal media.")
        print("\nTypes of access requested:")
        print("1. Photos - Basic metadata (file size, date, type)")
        print("2. Photos - Content analysis (AI description of images)")
        print("3. Videos - Basic metadata only")
        print("\nYour rights:")
        print("- You can revoke consent at any time")
        print("- You can limit access to specific directories")
        print("- All access is logged for audit purposes")
        print("- No data is shared with third parties")

        consent_granted = False

        # Request photo metadata consent
        print(f"\n📸 PHOTO METADATA ACCESS")
        if self.access_control.request_consent(
            user_id=user_id,
            media_type=MediaType.PHOTOS,
            consent_level=ConsentLevel.BASIC_METADATA,
            duration_days=30,
            specific_paths=["/home/patrick/Pictures"]
        ):
            consent_granted = True

        # Request photo content analysis consent (optional)
        print(f"\n🔍 PHOTO CONTENT ANALYSIS (Optional)")
        print("This allows AI to describe what's in your photos")
        self.access_control.request_consent(
            user_id=user_id,
            media_type=MediaType.PHOTOS,
            consent_level=ConsentLevel.CONTENT_ANALYSIS,
            duration_days=7  # Shorter duration for sensitive access
        )

        # Request video metadata consent
        print(f"\n🎥 VIDEO METADATA ACCESS")
        self.access_control.request_consent(
            user_id=user_id,
            media_type=MediaType.VIDEOS,
            consent_level=ConsentLevel.BASIC_METADATA,
            duration_days=30,
            specific_paths=["/home/patrick/Videos"]
        )

        return consent_granted

    def show_current_consents(self, user_id: str):
        """Display user's current consents"""
        consents = self.access_control.get_user_consents(user_id)

        print(f"\n📋 CURRENT CONSENTS FOR USER: {user_id}")
        print("="*50)

        if not consents:
            print("No active consents found.")
            return

        for consent in consents:
            print(f"\nMedia Type: {consent['media_type']}")
            print(f"Access Level: {consent['consent_level']}")
            print(f"Granted: {consent['granted_at']}")
            if consent['expires_at']:
                print(f"Expires: {consent['expires_at']}")
            if consent['specific_paths']:
                print(f"Paths: {', '.join(consent['specific_paths'])}")

    def revoke_all_consents(self, user_id: str):
        """Revoke all media access consents for user"""
        print(f"\n🚫 REVOKING ALL MEDIA CONSENTS FOR USER: {user_id}")

        for media_type in MediaType:
            if self.access_control.revoke_consent(user_id, media_type):
                print(f"Revoked: {media_type.value}")

        print("All consents have been revoked.")

    def _is_image_file(self, file_path: str) -> bool:
        """Check if file is an image"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.heic', '.raw'}
        return Path(file_path).suffix.lower() in image_extensions

    def _is_video_file(self, file_path: str) -> bool:
        """Check if file is a video"""
        video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.3gp'}
        return Path(file_path).suffix.lower() in video_extensions

def disable_unauthorized_scanner():
    """Disable the original unauthorized media scanner"""
    old_scanner_path = "/opt/tower-echo-brain/echo_media_scanner.py"

    if os.path.exists(old_scanner_path):
        # Rename to prevent accidental execution
        backup_path = f"{old_scanner_path}.DISABLED_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.rename(old_scanner_path, backup_path)
        print(f"🔒 Disabled unauthorized scanner: {old_scanner_path} -> {backup_path}")

        # Create warning file
        with open(old_scanner_path, 'w') as f:
            f.write("""#!/usr/bin/env python3
# SECURITY WARNING: This unauthorized media scanner has been disabled
#
# The original media scanner violated privacy by scanning personal photos/videos
# without explicit user consent. It has been replaced with a secure version
# that requires proper consent management.
#
# Use: /opt/tower-echo-brain/src/security/secure_media_scanner.py
#
# Date disabled: {date}
# Reason: Privacy violation - unauthorized personal media access
# Expert reviews: qwen and deepseek identified this as critical security issue

import sys
print("❌ ERROR: Unauthorized media scanner disabled for privacy protection")
print("✅ Use: /opt/tower-echo-brain/src/security/secure_media_scanner.py")
sys.exit(1)
""".format(date=datetime.now().isoformat()))

        print(f"✅ Created warning file at original location")

if __name__ == "__main__":
    # Disable unauthorized scanner first
    disable_unauthorized_scanner()

    # Initialize secure scanner
    scanner = SecureMediaScanner()

    if len(sys.argv) > 1:
        command = sys.argv[1]
        user_id = sys.argv[2] if len(sys.argv) > 2 else "patrick"

        if command == "consent":
            scanner.request_scanning_consent(user_id)
        elif command == "status":
            scanner.show_current_consents(user_id)
        elif command == "revoke":
            scanner.revoke_all_consents(user_id)
        elif command == "scan":
            # Only proceed if consent exists
            try:
                results = scanner.scan_photos_metadata(user_id, "/home/patrick/Pictures")
                print(f"\n📊 SCAN RESULTS:")
                print(f"Files scanned: {results['scanned_files']}")
                print(f"Total size: {results['total_size']:,} bytes")
                print(f"File types: {results['file_types']}")
            except PermissionError as e:
                print(f"❌ {e}")
                print("💡 Run with 'consent' command first to grant permissions")
        else:
            print("Usage: python secure_media_scanner.py [consent|status|revoke|scan] [user_id]")
    else:
        print("\n🔒 SECURE MEDIA SCANNER")
        print("="*50)
        print("This is the privacy-protected replacement for the unauthorized media scanner.")
        print("\nCommands:")
        print("  consent  - Request media scanning consent")
        print("  status   - Show current consents")
        print("  revoke   - Revoke all consents")
        print("  scan     - Scan media (requires consent)")
        print("\nExample: python secure_media_scanner.py consent patrick")