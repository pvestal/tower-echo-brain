#!/usr/bin/env python3
"""
Echo Brain Auth Management Module
Handles authentication for all Tower services
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import aiofiles
import httpx
from google.oauth2.credentials import Credentials
import google.auth.transport.requests

class EchoAuthManager:
    """Central authentication manager for all Tower services"""

    def __init__(self):
        self.token_dir = "/opt/tower-auth/tokens"
        self.oauth_service_url = "http://192.168.50.135:8088"
        self.tokens_cache = {}

    async def get_token(self, provider: str = "google") -> Optional[Dict[str, Any]]:
        """Get valid token for provider, refresh if needed"""
        token_file = os.path.join(self.token_dir, f"{provider}_latest.json")

        if not os.path.exists(token_file):
            print(f"❌ No token found for {provider}")
            return None

        try:
            async with aiofiles.open(token_file, "r") as f:
                token_data = json.loads(await f.read())

            # Check if token needs refresh
            if "expiry" in token_data and token_data["expiry"]:
                expiry = datetime.fromisoformat(token_data["expiry"].replace("Z", "+00:00"))
                if expiry - datetime.now(expiry.tzinfo) < timedelta(minutes=10):
                    # Refresh token
                    print(f"🔄 Refreshing {provider} token...")
                    token_data = await self.refresh_token(provider, token_data)

            return token_data

        except Exception as e:
            print(f"Error getting token: {e}")
            return None

    async def refresh_token(self, provider: str, token_data: Dict) -> Dict:
        """Refresh an expired token"""
        if provider == "google":
            creds = Credentials.from_authorized_user_info(token_data)
            if creds.refresh_token:
                request = google.auth.transport.requests.Request()
                creds.refresh(request)

                new_token_data = {
                    "token": creds.token,
                    "refresh_token": creds.refresh_token,
                    "token_uri": creds.token_uri,
                    "client_id": creds.client_id,
                    "client_secret": creds.client_secret,
                    "scopes": creds.scopes,
                    "expiry": creds.expiry.isoformat() if creds.expiry else None,
                    "provider": provider,
                    "refreshed": datetime.now().isoformat()
                }

                # Save refreshed token
                token_file = os.path.join(self.token_dir, f"{provider}_latest.json")
                async with aiofiles.open(token_file, "w") as f:
                    await f.write(json.dumps(new_token_data, indent=2))

                print(f"✅ Token refreshed for {provider}")
                return new_token_data

        return token_data

    async def get_google_headers(self) -> Dict[str, str]:
        """Get authorization headers for Google API calls"""
        token_data = await self.get_token("google")
        if token_data and "token" in token_data:
            return {"Authorization": f"Bearer {token_data['token']}"}
        return {}

    async def check_auth_status(self) -> Dict[str, Any]:
        """Check authentication status for all providers"""
        status = {}

        for token_file in os.listdir(self.token_dir):
            if token_file.endswith("_latest.json"):
                provider = token_file.replace("_latest.json", "")
                try:
                    async with aiofiles.open(os.path.join(self.token_dir, token_file), "r") as f:
                        token_data = json.loads(await f.read())

                    is_valid = True
                    if "expiry" in token_data and token_data["expiry"]:
                        expiry = datetime.fromisoformat(token_data["expiry"].replace("Z", "+00:00"))
                        is_valid = expiry > datetime.now(expiry.tzinfo)

                    status[provider] = {
                        "authenticated": True,
                        "valid": is_valid,
                        "expiry": token_data.get("expiry"),
                        "scopes": len(token_data.get("scopes", []))
                    }
                except:
                    status[provider] = {"authenticated": False}

        return status

    async def require_auth(self, provider: str = "google") -> bool:
        """Check if authentication is required for provider"""
        token = await self.get_token(provider)
        return token is None or "token" not in token

    def get_auth_url(self) -> str:
        """Get the authentication URL for users"""
        return f"{self.oauth_service_url}/login"


# Echo Brain integration
class EchoWithAuth:
    """Echo Brain with integrated auth management"""

    def __init__(self):
        self.auth_manager = EchoAuthManager()

    async def handle_google_request(self, endpoint: str, method: str = "GET", data: Dict = None):
        """Make authenticated Google API request"""
        headers = await self.auth_manager.get_google_headers()

        if not headers:
            return {"error": "Not authenticated. Visit http://192.168.50.135:8088 to login"}

        async with httpx.AsyncClient() as client:
            if method == "GET":
                response = await client.get(endpoint, headers=headers)
            elif method == "POST":
                response = await client.post(endpoint, headers=headers, json=data)

            return response.json()

    async def get_google_photos(self):
        """Get Google Photos using auth"""
        return await self.handle_google_request(
            "https://photoslibrary.googleapis.com/v1/mediaItems"
        )

    async def send_gmail(self, to: str, subject: str, body: str):
        """Send email via Gmail using auth"""
        import base64
        from email.mime.text import MIMEText

        message = MIMEText(body)
        message["to"] = to
        message["subject"] = subject
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

        return await self.handle_google_request(
            "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
            method="POST",
            data={"raw": raw_message}
        )

    async def get_auth_status(self):
        """Get authentication status for Echo"""
        status = await self.auth_manager.check_auth_status()
        return {
            "services": status,
            "auth_url": self.auth_manager.get_auth_url() if any(
                not s.get("authenticated") for s in status.values()
            ) else None
        }


# Integration with Echo Brain main service
async def integrate_with_echo():
    """Add auth capabilities to Echo Brain"""
    echo = EchoWithAuth()

    # Check auth status
    status = await echo.get_auth_status()
    print(f"🔐 Echo Auth Status: {json.dumps(status, indent=2)}")

    # Example: Get photos if authenticated
    if status["services"].get("google", {}).get("authenticated"):
        photos = await echo.get_google_photos()
        print(f"📷 Found {len(photos.get('mediaItems', []))} photos")

    return echo


if __name__ == "__main__":
    import asyncio

    async def test():
        echo = await integrate_with_echo()
        # Test auth status
        status = await echo.get_auth_status()
        print(json.dumps(status, indent=2))

    asyncio.run(test())