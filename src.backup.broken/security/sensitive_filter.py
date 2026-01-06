#!/usr/bin/env python3
"""
Sensitive Information Filter for Telegram Bot
Only Patrick gets sensitive system information
"""

import os
import logging

logger = logging.getLogger(__name__)

# Patrick's Telegram user ID
PATRICK_USER_ID = int(os.getenv('PATRICK_TELEGRAM_USER_ID', '605288143'))

def is_patrick(user_id: int) -> bool:
    """Check if user is Patrick"""
    return user_id == PATRICK_USER_ID

def filter_sensitive_response(response: str, user_id: int, username: str = "Unknown") -> str:
    """
    Filter sensitive information from responses for non-Patrick users
    
    Sensitive info includes:
    - System paths (/opt/, /home/, etc.)
    - IP addresses
    - Passwords, tokens, secrets
    - Database names/credentials
    - Service ports
    - Internal URLs
    """
    if is_patrick(user_id):
        logger.info(f"✅ Returning full response to Patrick (ID: {user_id})")
        return response
    
    logger.info(f"🔒 Filtering sensitive info for user {username} (ID: {user_id})")
    
    # For non-Patrick users, remove sensitive patterns
    import re
    filtered = response
    
    # Remove system paths
    filtered = re.sub(r'/opt/[^\s]+', '[SYSTEM_PATH]', filtered)
    filtered = re.sub(r'/home/[^\s]+', '[SYSTEM_PATH]', filtered)
    filtered = re.sub(r'/var/[^\s]+', '[SYSTEM_PATH]', filtered)
    
    # Remove IP addresses
    filtered = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '[IP_ADDRESS]', filtered)
    
    # Remove ports
    filtered = re.sub(r':(\d{4,5})', ':[PORT]', filtered)
    
    # Remove potential secrets/passwords
    filtered = re.sub(r'(password|token|secret|key)[=:]\s*[^\s]+', r'\1=[REDACTED]', filtered, flags=re.IGNORECASE)
    
    # Remove database names
    filtered = re.sub(r'(database|db)[=:]\s*[^\s]+', r'\1=[REDACTED]', filtered, flags=re.IGNORECASE)
    
    return filtered

def should_answer_query(query: str, user_id: int) -> tuple[bool, str]:
    """
    Determine if query requests sensitive information
    Returns (allow, reason)
    """
    if is_patrick(user_id):
        return (True, "Patrick authorized")
    
    # Queries that request sensitive info
    sensitive_keywords = [
        'password', 'secret', 'token', 'key', 'credential',
        'system status', 'service status', 'logs', 'database',
        'ip address', 'port', 'configuration', 'env', '.env'
    ]
    
    query_lower = query.lower()
    for keyword in sensitive_keywords:
        if keyword in query_lower:
            logger.warning(f"🚫 User {user_id} requested sensitive info: {keyword}")
            return (False, f"That information is restricted. General questions are welcome!")
    
    return (True, "General query allowed")

