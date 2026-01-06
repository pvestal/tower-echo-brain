#!/usr/bin/env python3
"""
AI Assist Apple Music Authentication Provider
Integrates with existing Apple Music service for authentication and data access
"""

import os
import jwt
import json
import time
import asyncio
import aiohttp
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class AppleMusicAuthProvider:
    """Apple Music authentication provider for AI Assist"""

    def __init__(self):
        # Apple credentials from existing setup
        self.team_id = "CNXX42ZGF8"  # From apple_music_config.env
        self.key_id = "9M85DX285V"   # From apple_music_config.env
        self.private_key_path = Path("/opt/tower-apple-music/config/AuthKey.p8")
        self.apple_music_api_base = "https://api.music.apple.com/v1"
        self.storefront = "us"

        # Alternative credentials (from oauth_config.json)
        self.alt_team_id = "7XY5SYJMAP"
        self.alt_key_path = Path("/home/{os.getenv("TOWER_USER", "patrick")}/Downloads/AuthKey_9M85DX285V.p8")

        # Local service endpoint
        self.local_service_url = "http://localhost:8315"

        # Token cache
        self.developer_token = None
        self.token_expiry = None

    def generate_developer_token(self, use_alt_creds: bool = False) -> str:
        """Generate Apple Music JWT developer token"""
        try:
            # Select credentials
            if use_alt_creds and self.alt_key_path.exists():
                key_path = self.alt_key_path
                team_id = self.alt_team_id
                key_id = self.key_id
            else:
                key_path = self.private_key_path
                team_id = self.team_id
                key_id = self.key_id

            # Check if we have a valid cached token
            if self.developer_token and self.token_expiry:
                if datetime.now() < self.token_expiry:
                    return self.developer_token

            # Read private key
            if not key_path.exists():
                logger.error(f"Private key not found at {key_path}")
                return None

            with open(key_path, 'rb') as f:
                private_key = f.read()

            # Generate JWT token
            now = datetime.utcnow()
            headers = {
                'alg': 'ES256',
                'kid': key_id
            }

            payload = {
                'iss': team_id,
                'iat': int(now.timestamp()),
                'exp': int((now + timedelta(hours=12)).timestamp()),
                'aud': 'appstoreconnect-v1'
            }

            token = jwt.encode(payload, private_key, algorithm='ES256', headers=headers)

            # Cache the token
            self.developer_token = token
            self.token_expiry = now + timedelta(hours=11)  # Refresh 1 hour before expiry

            logger.info(f"✅ Generated Apple Music JWT token (Team: {team_id})")
            return token

        except Exception as e:
            logger.error(f"❌ Failed to generate Apple Music token: {e}")
            return None

    async def search_music(self, query: str, types: List[str] = None) -> Dict[str, Any]:
        """Search Apple Music catalog"""
        token = self.generate_developer_token()
        if not token:
            return {"error": "Failed to generate token"}

        if not types:
            types = ["songs", "albums", "artists", "playlists"]

        try:
            headers = {
                "Authorization": f"Bearer {token}",
                "Music-User-Token": ""  # Would need user token for personal library
            }

            params = {
                "term": query,
                "types": ",".join(types),
                "limit": 25
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.apple_music_api_base}/catalog/{self.storefront}/search",
                    headers=headers,
                    params=params
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "results": data.get("results", {}),
                            "query": query,
                            "status": "success"
                        }
                    else:
                        return {"error": f"Search failed: {response.status}"}

        except Exception as e:
            logger.error(f"❌ Apple Music search failed: {e}")
            return {"error": str(e)}

    async def get_charts(self, chart_type: str = "songs") -> Dict[str, Any]:
        """Get Apple Music charts"""
        token = self.generate_developer_token()
        if not token:
            return {"error": "Failed to generate token"}

        try:
            headers = {"Authorization": f"Bearer {token}"}

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.apple_music_api_base}/catalog/{self.storefront}/charts",
                    headers=headers,
                    params={"types": chart_type, "limit": 50}
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "charts": data.get("results", {}),
                            "status": "success"
                        }
                    else:
                        return {"error": f"Charts request failed: {response.status}"}

        except Exception as e:
            logger.error(f"❌ Failed to get charts: {e}")
            return {"error": str(e)}

    async def get_local_library(self) -> Dict[str, Any]:
        """Get local music library from Apple Music service"""
        try:
            async with aiohttp.ClientSession() as session:
                # Get playlists
                async with session.get(f"{self.local_service_url}/api/playlists") as response:
                    if response.status == 200:
                        playlists = await response.json()
                    else:
                        playlists = []

                # Get library songs
                async with session.get(f"{self.local_service_url}/api/library") as response:
                    if response.status == 200:
                        library = await response.json()
                    else:
                        library = {"songs": []}

                return {
                    "playlists": playlists,
                    "library": library,
                    "status": "success"
                }

        except Exception as e:
            logger.error(f"❌ Failed to get local library: {e}")
            return {"error": str(e)}

    async def get_recommendations(self) -> Dict[str, Any]:
        """Get music recommendations based on listening history"""
        token = self.generate_developer_token()
        if not token:
            return {"error": "Failed to generate token"}

        try:
            # Get recommendations would require user token for personalized results
            # For now, return popular charts as recommendations
            return await self.get_charts("songs")

        except Exception as e:
            logger.error(f"❌ Failed to get recommendations: {e}")
            return {"error": str(e)}

    async def collect_training_data_for_echo(self) -> Dict[str, Any]:
        """Collect Apple Music data for AI Assist training"""
        logger.info("🎵 Collecting Apple Music data for Echo training...")

        # Generate token first
        token = self.generate_developer_token()
        if not token:
            return {"error": "Failed to generate Apple Music token"}

        training_data = {
            "service": "apple_music",
            "has_valid_token": bool(token),
            "collected_at": datetime.utcnow().isoformat()
        }

        # Collect various data types
        tasks = [
            ("search_results", self.search_music("popular songs 2025")),
            ("charts", self.get_charts()),
            ("local_library", self.get_local_library()),
            ("recommendations", self.get_recommendations())
        ]

        for name, task in tasks:
            try:
                result = await task
                training_data[name] = result
            except Exception as e:
                training_data[name] = {"error": str(e)}

        # Calculate success metrics
        successful = sum(1 for k, v in training_data.items()
                        if isinstance(v, dict) and v.get("status") == "success")

        training_data["summary"] = {
            "successful_calls": successful,
            "total_calls": len(tasks),
            "token_generated": bool(token),
            "team_id": self.team_id,
            "key_id": self.key_id
        }

        return training_data

    def get_status(self) -> Dict[str, Any]:
        """Get Apple Music authentication status"""
        return {
            "provider": "apple_music",
            "configured": self.private_key_path.exists() or self.alt_key_path.exists(),
            "team_id": self.team_id,
            "alt_team_id": self.alt_team_id,
            "key_id": self.key_id,
            "private_key_exists": self.private_key_path.exists(),
            "alt_key_exists": self.alt_key_path.exists(),
            "local_service_url": self.local_service_url,
            "token_cached": bool(self.developer_token),
            "token_expiry": self.token_expiry.isoformat() if self.token_expiry else None
        }

# Test the provider
async def test_apple_music_provider():
    """Test Apple Music authentication provider"""
    print("🎵 Testing Apple Music Authentication Provider")
    print("=" * 60)

    provider = AppleMusicAuthProvider()

    # Check status
    status = provider.get_status()
    print(f"📊 Status: {json.dumps(status, indent=2)}")

    # Generate token
    token = provider.generate_developer_token()
    if token:
        print(f"✅ Token generated successfully")
        print(f"   Token preview: {token[:50]}...")
    else:
        print("❌ Failed to generate token")
        return

    # Test search
    print("\n🔍 Testing search...")
    search_results = await provider.search_music("Taylor Swift")
    if search_results.get("status") == "success":
        songs = search_results.get("results", {}).get("songs", {}).get("data", [])
        print(f"✅ Found {len(songs)} songs")
        if songs:
            print(f"   First result: {songs[0].get('attributes', {}).get('name', 'Unknown')}")
    else:
        print(f"❌ Search failed: {search_results.get('error')}")

    # Test charts
    print("\n📊 Testing charts...")
    charts = await provider.get_charts()
    if charts.get("status") == "success":
        chart_songs = charts.get("charts", {}).get("songs", [])
        print(f"✅ Got {len(chart_songs)} chart entries")
    else:
        print(f"❌ Charts failed: {charts.get('error')}")

    # Test local library
    print("\n📚 Testing local library access...")
    library = await provider.get_local_library()
    if library.get("status") == "success":
        print(f"✅ Got {len(library.get('playlists', []))} playlists")
        print(f"✅ Got {len(library.get('library', {}).get('songs', []))} library songs")
    else:
        print(f"❌ Library access failed: {library.get('error')}")

    # Collect training data
    print("\n🧠 Collecting training data for Echo...")
    training_data = await provider.collect_training_data_for_echo()
    summary = training_data.get("summary", {})
    print(f"📈 Collection Summary:")
    print(f"   • Success rate: {summary.get('successful_calls', 0)}/{summary.get('total_calls', 0)}")
    print(f"   • Token generated: {summary.get('token_generated', False)}")
    print(f"   • Team ID: {summary.get('team_id', 'Unknown')}")

if __name__ == "__main__":
    asyncio.run(test_apple_music_provider())