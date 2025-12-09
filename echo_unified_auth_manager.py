#!/usr/bin/env python3
"""
Echo Brain Unified Authentication Manager
Integrates with ALL existing auth systems and tokens:
- Google OAuth (Calendar, Drive, Gmail)
- GitHub API (Repositories, Issues, Actions)
- Apple Music API (Search, Library)
- Plaid Financial (Accounts, Transactions)
- Anthropic Claude API
"""

import os
import json
import jwt
import asyncio
import aiohttp
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import logging
from typing import Dict, List, Any, Optional
import subprocess
import base64
import hvac

logger = logging.getLogger(__name__)

class EchoUnifiedAuthManager:
    """Unified authentication manager for all Tower services"""

    def __init__(self):
        self.credentials = {}
        self.tokens = {}
        self.vault_client = None
        self.load_existing_credentials()

    def load_existing_credentials(self):
        """Load all existing credentials from Tower auth system"""
        try:
            # Load Google/GitHub OAuth credentials
            oauth_config_path = Path('/opt/tower-auth/credentials/oauth_config.json')
            if oauth_config_path.exists():
                with open(oauth_config_path) as f:
                    self.credentials.update(json.load(f))

            # Load working Google OAuth client
            working_oauth_path = Path('/opt/tower-auth/working_oauth_client.json')
            if working_oauth_path.exists():
                with open(working_oauth_path) as f:
                    working_oauth = json.load(f)
                    self.credentials['google_working'] = working_oauth['installed']

            # Initialize Vault client for secure token storage
            self.init_vault_client()

            logger.info("✅ Loaded existing credentials from Tower auth system")

        except Exception as e:
            logger.error(f"❌ Failed to load credentials: {e}")

    def init_vault_client(self):
        """Initialize Vault client if available"""
        try:
            self.vault_client = hvac.Client(url='http://127.0.0.1:8200')
            token_path = Path('/opt/vault/.vault-token')
            if token_path.exists():
                vault_token = token_path.read_text().strip()
                self.vault_client.token = vault_token
                logger.info("✅ Vault client initialized")
        except Exception as e:
            logger.warning(f"⚠️ Vault not available: {e}")

    async def get_google_access_token(self) -> Optional[str]:
        """Get Google access token using existing OAuth credentials"""
        try:
            # Check if we have stored tokens in Vault
            if self.vault_client:
                try:
                    tokens = self.vault_client.secrets.kv.v2.read_secret_version(
                        path='google/tokens/patrick'
                    )
                    google_token = tokens['data']['data']['access_token']
                    logger.info("✅ Retrieved Google token from Vault")
                    return google_token
                except Exception:
                    logger.info("No Google tokens in Vault, need to authenticate")

            # Use existing Google OAuth credentials for new authentication
            google_creds = self.credentials.get('google')
            if google_creds:
                logger.info("Google OAuth credentials available for authentication")
                # In a real implementation, this would trigger the OAuth flow
                return None

        except Exception as e:
            logger.error(f"❌ Failed to get Google token: {e}")
            return None

    def get_github_token(self) -> Optional[str]:
        """Get GitHub token from CLI authentication"""
        try:
            # GitHub CLI is already authenticated
            result = subprocess.run(['gh', 'auth', 'token'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                token = result.stdout.strip()
                logger.info("✅ Retrieved GitHub token from CLI")
                return token
            return None
        except Exception as e:
            logger.error(f"❌ Failed to get GitHub token: {e}")
            return None

    def get_apple_music_token(self) -> Optional[str]:
        """Generate Apple Music JWT token"""
        try:
            apple_creds = self.credentials.get('apple', {})
            team_id = apple_creds.get('team_id')
            key_id = apple_creds.get('key_id')

            if not team_id or not key_id:
                logger.error("❌ Apple Music credentials not found")
                return None

            # Try to load private key
            key_path = Path('/opt/tower-apple-music/config/AuthKey.p8')
            if not key_path.exists():
                # Try alternative path
                key_path = Path('/home/{os.getenv("TOWER_USER", "patrick")}/Downloads/AuthKey_9M85DX285V.p8')

            if key_path.exists():
                with open(key_path, 'rb') as f:
                    private_key = f.read()

                # Generate JWT token
                now = datetime.utcnow()
                payload = {
                    'iss': team_id,
                    'iat': now,
                    'exp': now + timedelta(hours=6),
                    'aud': 'appstoreconnect-v1'
                }

                token = jwt.encode(payload, private_key, algorithm='ES256',
                                 headers={'kid': key_id})
                logger.info("✅ Generated Apple Music JWT token")
                return token
            else:
                logger.error("❌ Apple Music private key not found")
                return None

        except Exception as e:
            logger.error(f"❌ Failed to generate Apple Music token: {e}")
            return None

    def get_plaid_client_id(self) -> Optional[str]:
        """Get Plaid client ID"""
        return self.credentials.get('plaid', {}).get('client_id')

    async def fetch_google_calendar_data(self) -> Dict[str, Any]:
        """Fetch Google Calendar data for Echo training"""
        token = await self.get_google_access_token()
        if not token:
            return {"error": "No Google access token"}

        try:
            headers = {"Authorization": f"Bearer {token}"}

            async with aiohttp.ClientSession() as session:
                # Get calendar list
                async with session.get(
                    "https://www.googleapis.com/calendar/v3/users/me/calendarList",
                    headers=headers
                ) as response:
                    if response.status == 200:
                        calendars = await response.json()

                        # Get events from primary calendar
                        now = datetime.utcnow().isoformat() + 'Z'
                        async with session.get(
                            f"https://www.googleapis.com/calendar/v3/calendars/primary/events"
                            f"?timeMin={now}&maxResults=50&singleEvents=true&orderBy=startTime",
                            headers=headers
                        ) as events_response:
                            if events_response.status == 200:
                                events = await events_response.json()
                                return {
                                    "calendars": calendars.get("items", []),
                                    "events": events.get("items", []),
                                    "status": "success"
                                }
            return {"error": "Failed to fetch calendar data"}
        except Exception as e:
            logger.error(f"❌ Failed to fetch Google Calendar: {e}")
            return {"error": str(e)}

    async def fetch_github_data(self) -> Dict[str, Any]:
        """Fetch GitHub data for Echo training"""
        token = self.get_github_token()
        if not token:
            return {"error": "No GitHub token"}

        try:
            headers = {"Authorization": f"token {token}"}

            async with aiohttp.ClientSession() as session:
                # Get user repositories
                async with session.get(
                    "https://api.github.com/user/repos?sort=updated&per_page=50",
                    headers=headers
                ) as response:
                    if response.status == 200:
                        repos = await response.json()

                        # Get recent activity
                        async with session.get(
                            "https://api.github.com/user/events?per_page=30",
                            headers=headers
                        ) as events_response:
                            if events_response.status == 200:
                                events = await events_response.json()
                                return {
                                    "repositories": repos,
                                    "recent_activity": events,
                                    "status": "success"
                                }
            return {"error": "Failed to fetch GitHub data"}
        except Exception as e:
            logger.error(f"❌ Failed to fetch GitHub data: {e}")
            return {"error": str(e)}

    async def fetch_apple_music_data(self) -> Dict[str, Any]:
        """Fetch Apple Music data for Echo training"""
        token = self.get_apple_music_token()
        if not token:
            return {"error": "No Apple Music token"}

        try:
            headers = {"Authorization": f"Bearer {token}"}

            async with aiohttp.ClientSession() as session:
                # Search for popular music (example)
                async with session.get(
                    "https://api.music.apple.com/v1/catalog/us/search?term=popular&limit=25",
                    headers=headers
                ) as response:
                    if response.status == 200:
                        results = await response.json()
                        return {
                            "search_results": results,
                            "status": "success"
                        }
            return {"error": "Failed to fetch Apple Music data"}
        except Exception as e:
            logger.error(f"❌ Failed to fetch Apple Music data: {e}")
            return {"error": str(e)}

    async def fetch_plaid_financial_data(self) -> Dict[str, Any]:
        """Fetch Plaid financial data for Echo training"""
        # Use existing Plaid integration
        try:
            # Import existing Plaid integration
            import sys
            sys.path.append('/opt/tower-echo-brain/auth_service')
            from plaid_auth_api import PlaidAuthAPI

            plaid_api = PlaidAuthAPI()
            # Get account data if available
            financial_data = await plaid_api.get_accounts_summary()
            return {
                "financial_data": financial_data,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"❌ Failed to fetch Plaid data: {e}")
            return {"error": str(e)}

    async def collect_all_data_for_echo_training(self) -> Dict[str, Any]:
        """Collect all data from all services for Echo Brain training"""
        logger.info("🧠 Collecting data from all services for Echo training...")

        # Collect data from all services concurrently
        tasks = [
            self.fetch_google_calendar_data(),
            self.fetch_github_data(),
            self.fetch_apple_music_data(),
            self.fetch_plaid_financial_data()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        training_data = {
            "google_calendar": results[0] if not isinstance(results[0], Exception) else {"error": str(results[0])},
            "github": results[1] if not isinstance(results[1], Exception) else {"error": str(results[1])},
            "apple_music": results[2] if not isinstance(results[2], Exception) else {"error": str(results[2])},
            "plaid_financial": results[3] if not isinstance(results[3], Exception) else {"error": str(results[3])},
            "collected_at": datetime.utcnow().isoformat(),
            "collection_summary": self.generate_collection_summary(results)
        }

        # Store training data in Echo Brain database
        await self.store_training_data(training_data)

        return training_data

    def generate_collection_summary(self, results: List[Any]) -> Dict[str, Any]:
        """Generate summary of data collection"""
        successful = sum(1 for r in results if not isinstance(r, Exception) and not r.get("error"))
        total = len(results)

        return {
            "successful_integrations": successful,
            "total_integrations": total,
            "success_rate": f"{(successful/total)*100:.1f}%",
            "services": {
                "google_calendar": "success" if not isinstance(results[0], Exception) and not results[0].get("error") else "failed",
                "github": "success" if not isinstance(results[1], Exception) and not results[1].get("error") else "failed",
                "apple_music": "success" if not isinstance(results[2], Exception) and not results[2].get("error") else "failed",
                "plaid_financial": "success" if not isinstance(results[3], Exception) and not results[3].get("error") else "failed"
            }
        }

    async def store_training_data(self, training_data: Dict[str, Any]):
        """Store collected data in Echo Brain database for training"""
        try:
            db_path = Path('/opt/tower-echo-brain/data/echo_brain.db')
            if not db_path.exists():
                db_path.parent.mkdir(parents=True, exist_ok=True)

            with sqlite3.connect(db_path) as conn:
                # Create training data table if not exists
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS training_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        source TEXT,
                        data_type TEXT,
                        content TEXT,
                        collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        processed BOOLEAN DEFAULT FALSE
                    )
                ''')

                # Store data from each service
                for service, data in training_data.items():
                    if service != "collected_at" and service != "collection_summary":
                        conn.execute('''
                            INSERT INTO training_data (source, data_type, content)
                            VALUES (?, ?, ?)
                        ''', (service, "api_data", json.dumps(data)))

                conn.commit()
                logger.info("✅ Training data stored in Echo Brain database")

        except Exception as e:
            logger.error(f"❌ Failed to store training data: {e}")

    def get_credentials_status(self) -> Dict[str, Any]:
        """Get status of all credential systems"""
        return {
            "google_oauth": "configured" if self.credentials.get("google") else "missing",
            "github_token": "authenticated" if self.get_github_token() else "missing",
            "apple_music": "configured" if self.credentials.get("apple") else "missing",
            "plaid_client": "configured" if self.credentials.get("plaid") else "missing",
            "vault_connection": "connected" if self.vault_client else "disconnected",
            "total_services": len([k for k in self.credentials.keys() if k != "anthropic"]),
            "loaded_credentials": list(self.credentials.keys())
        }

# Example usage and testing
async def main():
    """Test the unified auth manager"""
    auth_manager = EchoUnifiedAuthManager()

    print("🔐 Echo Unified Auth Manager")
    print("=" * 60)

    # Check credentials status
    status = auth_manager.get_credentials_status()
    print(f"📊 Credentials Status: {json.dumps(status, indent=2)}")

    # Collect all data for training
    print("\n🧠 Collecting data for Echo training...")
    training_data = await auth_manager.collect_all_data_for_echo_training()

    print(f"\n📈 Collection Summary:")
    summary = training_data.get("collection_summary", {})
    print(f"  • Success Rate: {summary.get('success_rate', 'Unknown')}")
    print(f"  • Services Status: {json.dumps(summary.get('services', {}), indent=4)}")

if __name__ == "__main__":
    asyncio.run(main())