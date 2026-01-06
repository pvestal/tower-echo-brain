#!/usr/bin/env python3
"""
Telegram Security Module - User ID Verification
Ensures only authorized users can interact with Echo Brain via Telegram
"""

import os
import logging
from typing import Set, Optional
from fastapi import HTTPException

logger = logging.getLogger(__name__)

class TelegramSecurity:
    """Handle Telegram user authentication and authorization"""
    
    def __init__(self):
        # Patrick's Telegram user ID (set via env var or Vault)
        self.patrick_user_id = os.getenv('PATRICK_TELEGRAM_USER_ID', '')
        
        # Authorized user IDs (comma-separated in env)
        authorized_ids_str = os.getenv('TELEGRAM_AUTHORIZED_USERS', self.patrick_user_id)
        self.authorized_users: Set[str] = set(
            uid.strip() for uid in authorized_ids_str.split(',') if uid.strip()
        )
        
        if not self.authorized_users:
            logger.warning("⚠️ No authorized Telegram users configured!")
        else:
            logger.info(f"✅ Telegram security initialized with {len(self.authorized_users)} authorized users")
    
    def is_authorized(self, user_id: int) -> bool:
        """Check if user ID is authorized"""
        if not self.authorized_users:
            logger.warning(f"⚠️ No whitelist configured - rejecting user {user_id}")
            return False
        
        user_id_str = str(user_id)
        authorized = user_id_str in self.authorized_users
        
        if not authorized:
            logger.warning(f"🚫 Unauthorized Telegram access attempt from user ID: {user_id}")
        
        return authorized
    
    def verify_or_raise(self, user_id: int, username: str = "Unknown") -> None:
        """Verify user is authorized or raise HTTPException"""
        if not self.is_authorized(user_id):
            logger.error(f"🚫 BLOCKED unauthorized Telegram user: {username} (ID: {user_id})")
            raise HTTPException(
                status_code=403,
                detail="Unauthorized: This bot is for authorized users only"
            )
        
        logger.info(f"✅ Authorized Telegram user: {username} (ID: {user_id})")

# Global security instance
telegram_security = TelegramSecurity()
