#!/usr/bin/env python3
"""
Authentication Manager for Echo Brain
Handles Apple Music, Google Calendar, and other OAuth integrations
"""

import os
import json
import jwt
import time
import asyncio
import logging
import aiohttp
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

logger = logging.getLogger(__name__)

class AuthenticationManager:
    """Manages OAuth tokens and API access for various services"""

    def __init__(self):
        self.config_dir = Path("/opt/tower-echo-brain/config/auth")
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.tokens = {}
        self.apple_jwt_token = None
        self.google_credentials = None
        self._load_configurations()

    def _load_configurations(self):
        """Load authentication configurations from files"""
        # Apple Music configuration
        apple_config_path = Path("/opt/tower-apple-music/config/apple_music_config.env")
        if apple_config_path.exists():
            self.apple_config = self._parse_env_file(apple_config_path)
            logger.info("✅ Apple Music configuration loaded")
        else:
            logger.warning("❌ Apple Music configuration not found")

        # Google credentials
        google_creds_path = self.config_dir / "google_credentials.json"
        if google_creds_path.exists():
            self.google_credentials = Credentials.from_authorized_user_file(
                str(google_creds_path),
                ['https://www.googleapis.com/auth/calendar',
                 'https://www.googleapis.com/auth/gmail.readonly']
            )
            logger.info("✅ Google credentials loaded")
        else:
            logger.warning("❌ Google credentials not found")

    def _parse_env_file(self, path: Path) -> Dict[str, str]:
        """Parse .env file into dictionary"""
        config = {}
        with open(path, 'r') as f:
            for line in f:
                if '=' in line and not line.strip().startswith('#'):
                    key, value = line.strip().split('=', 1)
                    config[key] = value.strip('"\'')
        return config

    async def get_apple_music_token(self) -> Optional[str]:
        """Generate Apple Music JWT token"""
        if not hasattr(self, 'apple_config'):
            logger.error("Apple Music not configured")
            return None

        try:
            # Check if current token is still valid
            if self.apple_jwt_token and self._is_token_valid(self.apple_jwt_token):
                return self.apple_jwt_token

            # Generate new JWT token
            team_id = self.apple_config.get('APPLE_TEAM_ID')
            key_id = self.apple_config.get('APPLE_KEY_ID')
            key_path = self.apple_config.get('APPLE_PRIVATE_KEY_PATH')

            if not all([team_id, key_id, key_path]):
                logger.error("Missing Apple Music credentials")
                return None

            # Read private key
            with open(key_path, 'r') as f:
                private_key = f.read()

            # Create JWT payload
            time_now = int(time.time())
            time_expires = time_now + (12 * 3600)  # 12 hours

            headers = {
                'alg': 'ES256',
                'kid': key_id
            }

            payload = {
                'iss': team_id,
                'iat': time_now,
                'exp': time_expires
            }

            # Generate token
            self.apple_jwt_token = jwt.encode(
                payload,
                private_key,
                algorithm='ES256',
                headers=headers
            )

            logger.info("✅ Apple Music JWT token generated")
            return self.apple_jwt_token

        except Exception as e:
            logger.error(f"Failed to generate Apple Music token: {e}")
            return None

    def _is_token_valid(self, token: str) -> bool:
        """Check if JWT token is still valid"""
        try:
            # Decode without verification to check expiry
            payload = jwt.decode(token, options={"verify_signature": False})
            exp = payload.get('exp', 0)
            return exp > time.time()
        except Exception:
            return False

    async def get_google_calendar_service(self):
        """Get authenticated Google Calendar service"""
        if not self.google_credentials:
            logger.error("Google credentials not available")
            return None

        # Refresh token if expired
        if self.google_credentials.expired and self.google_credentials.refresh_token:
            self.google_credentials.refresh(Request())
            # Save updated credentials
            creds_path = self.config_dir / "google_credentials.json"
            with open(creds_path, 'w') as f:
                f.write(self.google_credentials.to_json())

        # Return credentials for use with Google API client
        return self.google_credentials

    async def search_apple_music(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Search Apple Music catalog"""
        token = await self.get_apple_music_token()
        if not token:
            return {"error": "Apple Music authentication failed"}

        storefront = self.apple_config.get('APPLE_STOREFRONT', 'us')

        async with aiohttp.ClientSession() as session:
            headers = {
                'Authorization': f'Bearer {token}',
                'Music-User-Token': ''  # Would need user token for personal library
            }

            url = f"https://api.music.apple.com/v1/catalog/{storefront}/search"
            params = {
                'term': query,
                'limit': limit,
                'types': 'songs,albums,playlists'
            }

            try:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('results', {})
                    else:
                        error = await response.text()
                        logger.error(f"Apple Music search failed: {error}")
                        return {"error": f"Search failed: {response.status}"}
            except Exception as e:
                logger.error(f"Apple Music search error: {e}")
                return {"error": str(e)}

    async def get_google_calendar_events(self, max_results: int = 10):
        """Get upcoming Google Calendar events"""
        creds = await self.get_google_calendar_service()
        if not creds:
            return {"error": "Google Calendar authentication failed"}

        # This would use Google Calendar API client
        # For now, return placeholder
        return {
            "message": "Google Calendar integration ready",
            "credentials_valid": creds is not None
        }

    async def setup_google_auth(self, client_secrets_path: str):
        """Setup Google OAuth authentication"""
        SCOPES = [
            'https://www.googleapis.com/auth/calendar',
            'https://www.googleapis.com/auth/gmail.readonly',
            'https://www.googleapis.com/auth/drive.file'
        ]

        flow = InstalledAppFlow.from_client_secrets_file(
            client_secrets_path, SCOPES
        )

        # Run OAuth flow
        creds = flow.run_local_server(port=0)

        # Save credentials
        creds_path = self.config_dir / "google_credentials.json"
        with open(creds_path, 'w') as f:
            f.write(creds.to_json())

        self.google_credentials = creds
        logger.info("✅ Google authentication configured")
        return True

    async def grant_echo_permissions(self):
        """Grant Echo Brain necessary permissions to access auth services"""
        permissions = {
            'apple_music': {
                'search': True,
                'playlists': True,
                'library': False,  # Requires user token
                'recommendations': True
            },
            'google': {
                'calendar_read': True,
                'calendar_write': True,
                'gmail_read': True,
                'drive_files': True
            }
        }

        # Save permissions config
        permissions_path = self.config_dir / "permissions.json"
        with open(permissions_path, 'w') as f:
            json.dump(permissions, f, indent=2)

        logger.info("✅ Echo permissions configured")
        return permissions

# Global instance
auth_manager = AuthenticationManager()