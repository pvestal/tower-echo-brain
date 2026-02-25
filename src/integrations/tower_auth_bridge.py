#!/usr/bin/env python3
"""
Tower Auth Bridge for Echo Brain
Integrates with existing Tower Auth service for SSO and OAuth
"""

import logging
import httpx
import asyncio
from typing import Dict, Optional, Any, List
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

class TowerAuthBridge:
    """Bridge between Echo Brain and Tower Auth Service"""

    def __init__(self):
        self.auth_service_url = "http://localhost:8088"
        self.vault_url = "http://localhost:8200"
        self.session = None
        self.cached_tokens = {}
        self._provider_disabled: Dict[str, str] = {}  # provider -> reason (skip refresh until re-auth)

    async def initialize(self):
        """Initialize connection to Tower Auth service"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.auth_service_url}/health")
                if response.status_code == 200:
                    logger.info("✅ Connected to Tower Auth service")
                    return True
        except Exception as e:
            logger.error(f"Failed to connect to Tower Auth: {e}")
            return False

    async def initiate_oauth_flow(self, provider: str, scopes: List[str] = None) -> Dict[str, Any]:
        """
        Initiate OAuth flow through Tower Auth service

        Supported providers:
        - google: Full Google Workspace integration
        - github: Repository and user management
        - apple: Apple ID and Apple Music
        """
        provider_configs = {
            'google': {
                'endpoint': '/oauth/google',
                'default_scopes': [
                    'https://www.googleapis.com/auth/calendar',
                    'https://www.googleapis.com/auth/calendar.events',
                    'https://www.googleapis.com/auth/gmail.modify',
                    'https://www.googleapis.com/auth/photoslibrary.readonly',
                    'https://www.googleapis.com/auth/drive.file',
                    'https://www.googleapis.com/auth/userinfo.profile',
                    'https://www.googleapis.com/auth/userinfo.email'
                ]
            },
            'github': {
                'endpoint': '/oauth/github',
                'default_scopes': ['repo', 'user', 'gist', 'notifications']
            },
            'apple': {
                'endpoint': '/oauth/apple',
                'default_scopes': ['name', 'email']
            }
        }

        if provider not in provider_configs:
            raise ValueError(f"Unsupported OAuth provider: {provider}")

        config = provider_configs[provider]
        use_scopes = scopes or config['default_scopes']

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.auth_service_url}{config['endpoint']}",
                    params={'scopes': ' '.join(use_scopes)}
                )

                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"✅ OAuth flow initiated for {provider}")
                    return {
                        'auth_url': data.get('auth_url'),
                        'state': data.get('state'),
                        'provider': provider
                    }
                else:
                    logger.error(f"OAuth initiation failed: {response.text}")
                    return {'error': 'Failed to initiate OAuth flow'}

        except Exception as e:
            logger.error(f"OAuth flow error: {e}")
            return {'error': str(e)}

    async def handle_oauth_callback(self, provider: str, code: str, state: str) -> Dict[str, Any]:
        """Handle OAuth callback and store tokens"""
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    f"{self.auth_service_url}/oauth/{provider}/callback",
                    json={'code': code, 'state': state}
                )

                if response.status_code == 200:
                    token_data = response.json()

                    # Cache tokens
                    self.cached_tokens[provider] = {
                        'access_token': token_data.get('access_token'),
                        'refresh_token': token_data.get('refresh_token'),
                        'expires_at': datetime.now() + timedelta(seconds=token_data.get('expires_in', 3600)),
                        'user_info': token_data.get('user_info', {})
                    }

                    logger.info(f"✅ OAuth tokens stored for {provider}")
                    return token_data
                else:
                    logger.error(f"Callback handling failed: {response.text}")
                    return {'error': 'Failed to handle OAuth callback'}

        except Exception as e:
            logger.error(f"Callback error: {e}")
            return {'error': str(e)}

    async def get_valid_token(self, provider: str) -> Optional[str]:
        """Get valid access token, refreshing if necessary"""
        # If provider was marked disabled (e.g. no refresh token), don't spam retries
        if provider in self._provider_disabled:
            return None

        if provider not in self.cached_tokens:
            # Try to load from Tower Auth database
            await self.load_existing_tokens()

        if provider in self.cached_tokens:
            token_info = self.cached_tokens[provider]

            # Check if token expired or missing expiry (assume expired)
            needs_refresh = False
            if token_info.get('expires_at'):
                needs_refresh = token_info['expires_at'] < datetime.now()
            elif provider == 'google':
                # No expiry tracked — test the token
                needs_refresh = not await self._test_google_token(token_info.get('access_token', ''))

            if needs_refresh:
                refreshed = await self.refresh_token(provider)
                if not refreshed:
                    logger.warning(f"Token refresh failed for {provider}")
                    return None

            return self.cached_tokens.get(provider, {}).get('access_token')

        return None

    async def _test_google_token(self, token: str) -> bool:
        """Quick test if a Google access token is still valid."""
        if not token:
            return False
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(
                    "https://www.googleapis.com/oauth2/v1/tokeninfo",
                    params={"access_token": token},
                )
                return resp.status_code == 200
        except Exception:
            return False

    async def refresh_token(self, provider: str) -> bool:
        """Refresh OAuth token through Tower Auth"""
        if provider == 'google':
            return await self._refresh_google_token()

        if provider not in self.cached_tokens:
            return False

        refresh_token = self.cached_tokens[provider].get('refresh_token')
        if not refresh_token:
            logger.error(f"No refresh token for {provider}")
            return False

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.auth_service_url}/oauth/{provider}/refresh",
                    json={'refresh_token': refresh_token}
                )

                if response.status_code == 200:
                    token_data = response.json()
                    self.cached_tokens[provider].update({
                        'access_token': token_data['access_token'],
                        'expires_at': datetime.now() + timedelta(seconds=token_data.get('expires_in', 3600))
                    })
                    logger.info(f"✅ Token refreshed for {provider}")
                    return True

        except Exception as e:
            logger.error(f"Token refresh failed: {e}")

        return False

    async def _refresh_google_token(self) -> bool:
        """Refresh Google token via tower-auth's Vault-backed refresh endpoint."""
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.post(
                    f"{self.auth_service_url}/api/auth/oauth/google/refresh"
                )
                if response.status_code == 200:
                    data = response.json()
                    self.cached_tokens['google'] = {
                        'access_token': data['access_token'],
                        'expires_at': datetime.now() + timedelta(seconds=data.get('expires_in', 3600)),
                    }
                    self._provider_disabled.pop('google', None)
                    logger.info("Google token refreshed via tower-auth")
                    return True
                else:
                    body = response.text[:200]
                    if "No refresh token stored" in body or "re-authenticate" in body:
                        # No refresh token on server — stop retrying until manual re-auth
                        self._provider_disabled['google'] = "No refresh token — re-authenticate via /api/auth/oauth/google/login"
                        logger.warning("Google OAuth disabled: no refresh token stored. Re-authenticate to re-enable.")
                    else:
                        logger.error(f"Google refresh returned {response.status_code}: {body}")
        except Exception as e:
            logger.error(f"Google token refresh failed: {e}")
        return False

    async def load_existing_tokens(self) -> Dict[str, Any]:
        """Load existing OAuth tokens from Tower Auth database"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.auth_service_url}/tokens/list")

                if response.status_code == 200:
                    tokens = response.json()
                    for provider, token_info in tokens.items():
                        if token_info.get('access_token'):
                            self.cached_tokens[provider] = token_info

                    logger.info(f"✅ Loaded {len(self.cached_tokens)} OAuth tokens")
                    return self.cached_tokens

        except Exception as e:
            logger.error(f"Failed to load existing tokens: {e}")

        return {}

    # Google-specific methods
    async def get_google_calendar_service(self):
        """Get authenticated Google Calendar service"""
        token = await self.get_valid_token('google')
        if not token:
            logger.debug("No valid Google token available")
            return None

        from googleapiclient.discovery import build
        from google.oauth2.credentials import Credentials

        creds = Credentials(token=token, scopes=[
            "https://www.googleapis.com/auth/calendar",
            "https://www.googleapis.com/auth/calendar.events",
        ])
        service = build('calendar', 'v3', credentials=creds)
        return service

    async def sync_google_calendars(self, calendars: List[str]) -> Dict[str, Any]:
        """Sync multiple Google calendars"""
        service = await self.get_google_calendar_service()
        if not service:
            return {'error': 'Failed to get calendar service'}

        all_events = []
        for calendar_id in calendars:
            try:
                events_result = service.events().list(
                    calendarId=calendar_id,
                    timeMin=datetime.utcnow().isoformat() + 'Z',
                    maxResults=100,
                    singleEvents=True,
                    orderBy='startTime'
                ).execute()

                events = events_result.get('items', [])
                for event in events:
                    event['calendar_id'] = calendar_id
                    all_events.append(event)

            except Exception as e:
                logger.error(f"Failed to sync calendar {calendar_id}: {e}")

        # Sort all events by start time
        all_events.sort(key=lambda x: x.get('start', {}).get('dateTime', ''))

        return {
            'events': all_events,
            'calendars_synced': len(calendars),
            'total_events': len(all_events)
        }

    async def get_gmail_service(self):
        """Get authenticated Gmail service with full access"""
        token = await self.get_valid_token('google')
        if not token:
            logger.debug("No valid Google token available")
            return None

        from googleapiclient.discovery import build
        from google.oauth2.credentials import Credentials

        creds = Credentials(token=token)
        service = build('gmail', 'v1', credentials=creds)
        return service

    async def sync_gmail(self, query: str = "is:unread", max_results: int = 50) -> Dict[str, Any]:
        """Sync Gmail messages"""
        service = await self.get_gmail_service()
        if not service:
            return {'error': 'Failed to get Gmail service'}

        try:
            results = service.users().messages().list(
                userId='me',
                q=query,
                maxResults=max_results
            ).execute()

            messages = results.get('messages', [])
            full_messages = []

            for msg in messages:
                msg_data = service.users().messages().get(
                    userId='me',
                    id=msg['id']
                ).execute()
                full_messages.append(msg_data)

            return {
                'messages': full_messages,
                'count': len(full_messages),
                'query': query
            }

        except Exception as e:
            logger.error(f"Gmail sync failed: {e}")
            return {'error': str(e)}

    async def get_google_photos_service(self):
        """Get authenticated Google Photos service"""
        token = await self.get_valid_token('google')
        if not token:
            return None

        # Google Photos uses REST API, not discovery service
        return {'token': token, 'base_url': 'https://photoslibrary.googleapis.com/v1'}

    async def batch_sync_google_photos(self, batch_size: int = 100) -> Dict[str, Any]:
        """Batch sync Google Photos"""
        photos_service = await self.get_google_photos_service()
        if not photos_service:
            return {'error': 'Failed to get Photos service'}

        token = photos_service['token']
        base_url = photos_service['base_url']

        headers = {'Authorization': f'Bearer {token}'}
        all_media = []
        next_page_token = None

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                while True:
                    params = {'pageSize': batch_size}
                    if next_page_token:
                        params['pageToken'] = next_page_token

                    response = await client.get(
                        f"{base_url}/mediaItems",
                        headers=headers,
                        params=params
                    )

                    if response.status_code == 200:
                        data = response.json()
                        media_items = data.get('mediaItems', [])
                        all_media.extend(media_items)

                        next_page_token = data.get('nextPageToken')
                        if not next_page_token:
                            break

                        # Add small delay to avoid rate limiting
                        await asyncio.sleep(0.5)
                    else:
                        logger.error(f"Photos sync error: {response.text}")
                        break

            return {
                'media_items': all_media,
                'count': len(all_media),
                'status': 'success'
            }

        except Exception as e:
            logger.error(f"Photos batch sync failed: {e}")
            return {'error': str(e)}

    # Apple Music integration via tower-auth
    async def get_apple_music_developer_token(self) -> Optional[str]:
        """Get Apple Music developer token via tower-auth"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.auth_service_url}/api/auth/apple-music/developer-token")
                if response.status_code == 200:
                    return response.json().get('token')
        except Exception as e:
            logger.error(f"Failed to get Apple Music developer token: {e}")
        return None

    async def get_apple_music_user_token(self, user_id: str = "patrick") -> Optional[str]:
        """Get Apple Music user token via tower-auth"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.auth_service_url}/api/auth/apple-music/token",
                    params={"user_id": user_id}
                )
                if response.status_code == 200:
                    return response.json().get('user_token')
        except Exception as e:
            logger.error(f"Failed to get Apple Music user token: {e}")
        return None

    async def get_apple_music_status(self, user_id: str = "patrick") -> Dict[str, Any]:
        """Check Apple Music authorization status via tower-auth"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.auth_service_url}/api/auth/apple-music/status",
                    params={"user_id": user_id}
                )
                if response.status_code == 200:
                    return response.json()
        except Exception as e:
            logger.error(f"Apple Music status check failed: {e}")
        return {"authorized": False, "error": "tower-auth unreachable"}

    async def search_apple_music(self, query: str, types: str = "songs,albums,playlists", limit: int = 10) -> Dict[str, Any]:
        """Search Apple Music catalog via developer token"""
        developer_token = await self.get_apple_music_developer_token()
        if not developer_token:
            return {"error": "No Apple Music developer token available"}

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.get(
                    "https://api.music.apple.com/v1/catalog/us/search",
                    headers={"Authorization": f"Bearer {developer_token}"},
                    params={"term": query, "types": types, "limit": limit}
                )
                if response.status_code == 200:
                    return response.json().get('results', {})
                else:
                    return {"error": f"Apple Music API returned {response.status_code}"}
        except Exception as e:
            logger.error(f"Apple Music search failed: {e}")
            return {"error": str(e)}

    async def get_apple_music_playlists(self, user_id: str = "patrick") -> Dict[str, Any]:
        """Get user's Apple Music playlists (requires Music User Token)"""
        developer_token = await self.get_apple_music_developer_token()
        user_token = await self.get_apple_music_user_token(user_id)

        if not developer_token:
            return {"error": "No Apple Music developer token available"}
        if not user_token:
            return {"error": "No Apple Music user token - user must authorize via tower-auth"}

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.get(
                    "https://api.music.apple.com/v1/me/library/playlists",
                    headers={
                        "Authorization": f"Bearer {developer_token}",
                        "Music-User-Token": user_token
                    }
                )
                if response.status_code == 200:
                    return response.json()
                else:
                    return {"error": f"Apple Music API returned {response.status_code}"}
        except Exception as e:
            logger.error(f"Apple Music playlists failed: {e}")
            return {"error": str(e)}

    async def get_apple_music_playlist_tracks(self, playlist_id: str, user_id: str = "patrick") -> Dict[str, Any]:
        """Get tracks from a user's Apple Music library playlist, enriched with catalog preview URLs.

        Uses batch catalog lookups (up to 10 concurrent) for speed.
        """
        developer_token = await self.get_apple_music_developer_token()
        user_token = await self.get_apple_music_user_token(user_id)

        if not developer_token:
            return {"error": "No Apple Music developer token available"}
        if not user_token:
            return {"error": "No Apple Music user token - user must authorize via /apple-music-auth"}

        headers = {
            "Authorization": f"Bearer {developer_token}",
            "Music-User-Token": user_token,
        }

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    f"https://api.music.apple.com/v1/me/library/playlists/{playlist_id}/tracks",
                    headers=headers,
                )
                if resp.status_code != 200:
                    return {"error": f"Apple Music API returned {resp.status_code}: {resp.text[:200]}"}

                library_data = resp.json()

                # Build track list from library data
                tracks = []
                catalog_ids = []
                for item in library_data.get("data", []):
                    attrs = item.get("attributes", {})
                    play_params = attrs.get("playParams", {})
                    catalog_id = play_params.get("catalogId")

                    artwork = attrs.get("artwork", {})
                    artwork_url = None
                    if artwork.get("url"):
                        artwork_url = artwork["url"].replace("{w}", "300").replace("{h}", "300")

                    tracks.append({
                        "library_id": item.get("id"),
                        "catalog_id": catalog_id,
                        "name": attrs.get("name"),
                        "artist": attrs.get("artistName"),
                        "album": attrs.get("albumName"),
                        "duration_ms": attrs.get("durationInMillis"),
                        "artwork_url": artwork_url,
                        "preview_url": None,
                    })
                    if catalog_id:
                        catalog_ids.append(catalog_id)

                # Batch catalog lookups — Apple allows up to 300 IDs per request
                if catalog_ids:
                    catalog_map = {}
                    for i in range(0, len(catalog_ids), 25):
                        batch = catalog_ids[i:i + 25]
                        ids_param = ",".join(batch)
                        try:
                            cat_resp = await client.get(
                                f"https://api.music.apple.com/v1/catalog/us/songs",
                                headers={"Authorization": f"Bearer {developer_token}"},
                                params={"ids": ids_param},
                            )
                            if cat_resp.status_code == 200:
                                for song in cat_resp.json().get("data", []):
                                    sid = song.get("id")
                                    sa = song.get("attributes", {})
                                    previews = sa.get("previews", [])
                                    art = sa.get("artwork", {})
                                    catalog_map[sid] = {
                                        "preview_url": previews[0].get("url") if previews else None,
                                        "artwork_url": art["url"].replace("{w}", "300").replace("{h}", "300") if art.get("url") else None,
                                    }
                        except Exception as batch_err:
                            logger.warning(f"Catalog batch lookup failed: {batch_err}")

                    # Enrich tracks with catalog data
                    for track in tracks:
                        cid = track.get("catalog_id")
                        if cid and cid in catalog_map:
                            cat = catalog_map[cid]
                            track["preview_url"] = cat.get("preview_url")
                            if not track["artwork_url"] and cat.get("artwork_url"):
                                track["artwork_url"] = cat["artwork_url"]

                return {
                    "playlist_id": playlist_id,
                    "tracks": tracks,
                    "total": len(tracks),
                }

        except Exception as e:
            logger.error(f"Apple Music playlist tracks failed: {e}")
            return {"error": str(e)}

    async def get_apple_music_catalog_track(self, catalog_id: str) -> Dict[str, Any]:
        """Look up a single track in the Apple Music catalog to get preview URL and metadata."""
        developer_token = await self.get_apple_music_developer_token()
        if not developer_token:
            return {"error": "No Apple Music developer token available"}

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    f"https://api.music.apple.com/v1/catalog/us/songs/{catalog_id}",
                    headers={"Authorization": f"Bearer {developer_token}"},
                )
                if resp.status_code != 200:
                    return {"error": f"Catalog lookup returned {resp.status_code}"}

                data = resp.json()
                items = data.get("data", [])
                if not items:
                    return {"error": "Track not found in catalog"}

                attrs = items[0].get("attributes", {})
                previews = attrs.get("previews", [])
                preview_url = previews[0].get("url") if previews else None

                artwork = attrs.get("artwork", {})
                artwork_url = None
                if artwork.get("url"):
                    artwork_url = artwork["url"].replace("{w}", "300").replace("{h}", "300")

                return {
                    "catalog_id": catalog_id,
                    "name": attrs.get("name"),
                    "artist": attrs.get("artistName"),
                    "album": attrs.get("albumName"),
                    "duration_ms": attrs.get("durationInMillis"),
                    "genre": (attrs.get("genreNames") or [None])[0],
                    "preview_url": preview_url,
                    "artwork_url": artwork_url,
                    "release_date": attrs.get("releaseDate"),
                    "isrc": attrs.get("isrc"),
                }

        except Exception as e:
            logger.error(f"Apple Music catalog lookup failed: {e}")
            return {"error": str(e)}

    # Plaid financial integration — direct Vault + Plaid API (bypasses tower-auth session requirement)

    async def _get_plaid_credentials(self) -> Optional[Dict[str, str]]:
        """Read Plaid credentials and access token from Vault."""
        import os
        try:
            import hvac
            token = os.getenv("VAULT_TOKEN")
            if not token:
                # Fall back to ~/.vault-token file
                token_file = os.path.expanduser("~/.vault-token")
                if os.path.exists(token_file):
                    with open(token_file) as f:
                        token = f.read().strip()
            vault = hvac.Client(
                url=self.vault_url,
                token=token,
            )
            # Client credentials
            creds = vault.secrets.kv.v2.read_secret_version(path="plaid")
            creds_data = creds["data"]["data"]
            # Access token + item_id
            tokens = vault.secrets.kv.v2.read_secret_version(path="tokens/plaid")
            tokens_data = tokens["data"]["data"]

            return {
                "client_id": creds_data["client_id"],
                "secret": creds_data["secret"],
                "access_token": tokens_data["access_token"],
                "item_id": tokens_data.get("item_id", ""),
                "environment": creds_data.get("environment", "production"),
            }
        except Exception as e:
            logger.error(f"Failed to read Plaid credentials from Vault: {e}")
            return None

    def _plaid_base_url(self, env: str) -> str:
        urls = {
            "sandbox": "https://sandbox.plaid.com",
            "development": "https://development.plaid.com",
            "production": "https://production.plaid.com",
        }
        return urls.get(env, "https://production.plaid.com")

    async def get_plaid_accounts(self) -> Dict[str, Any]:
        """Get linked Plaid accounts directly via Plaid API."""
        creds = await self._get_plaid_credentials()
        if not creds:
            return {"error": "Plaid credentials unavailable", "accounts": []}
        try:
            base = self._plaid_base_url(creds["environment"])
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    f"{base}/accounts/get",
                    json={
                        "client_id": creds["client_id"],
                        "secret": creds["secret"],
                        "access_token": creds["access_token"],
                    },
                )
                if resp.status_code == 200:
                    return resp.json()
                logger.error(f"Plaid accounts/get: {resp.status_code} {resp.text[:200]}")
                return {"error": f"Plaid API {resp.status_code}", "accounts": []}
        except Exception as e:
            logger.error(f"Plaid accounts failed: {e}")
            return {"error": str(e), "accounts": []}

    async def get_plaid_transactions(self, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """Get Plaid transactions directly via Plaid API."""
        creds = await self._get_plaid_credentials()
        if not creds:
            return {"error": "Plaid credentials unavailable", "transactions": []}
        from datetime import datetime, timedelta
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        try:
            base = self._plaid_base_url(creds["environment"])
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{base}/transactions/get",
                    json={
                        "client_id": creds["client_id"],
                        "secret": creds["secret"],
                        "access_token": creds["access_token"],
                        "start_date": start_date,
                        "end_date": end_date,
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return {
                        "transactions": data.get("transactions", []),
                        "total_transactions": data.get("total_transactions", 0),
                    }
                logger.error(f"Plaid transactions/get: {resp.status_code} {resp.text[:200]}")
                return {"error": f"Plaid API {resp.status_code}", "transactions": []}
        except Exception as e:
            logger.error(f"Plaid transactions failed: {e}")
            return {"error": str(e), "transactions": []}

    async def get_plaid_balances(self) -> Dict[str, Any]:
        """Get Plaid account balances directly via Plaid API."""
        creds = await self._get_plaid_credentials()
        if not creds:
            return {"error": "Plaid credentials unavailable", "balances": []}
        try:
            base = self._plaid_base_url(creds["environment"])
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    f"{base}/accounts/balance/get",
                    json={
                        "client_id": creds["client_id"],
                        "secret": creds["secret"],
                        "access_token": creds["access_token"],
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return {"accounts": data.get("accounts", [])}
                logger.error(f"Plaid balance/get: {resp.status_code} {resp.text[:200]}")
                return {"error": f"Plaid API {resp.status_code}", "balances": []}
        except Exception as e:
            logger.error(f"Plaid balances failed: {e}")
            return {"error": str(e), "balances": []}

    # Apple SSO implementation
    async def initiate_apple_sso(self) -> Dict[str, Any]:
        """Initiate Apple Sign-In flow"""
        # Apple Sign-In requires specific setup
        apple_config = {
            'client_id': 'com.tower.echo',  # Your Service ID
            'redirect_uri': 'https://192.168.50.135:8088/oauth/apple/callback',
            'response_type': 'code id_token',
            'scope': 'name email',
            'response_mode': 'form_post'
        }

        auth_url = 'https://appleid.apple.com/auth/authorize'
        params = '&'.join([f"{k}={v}" for k, v in apple_config.items()])

        return {
            'auth_url': f"{auth_url}?{params}",
            'provider': 'apple',
            'config': apple_config
        }

    async def handle_apple_callback(self, code: str, id_token: str) -> Dict[str, Any]:
        """Handle Apple Sign-In callback"""
        # Decode the ID token to get user info
        import jwt as pyjwt

        try:
            # Apple's public keys would need to be fetched
            user_info = pyjwt.decode(id_token, options={"verify_signature": False})

            # Exchange code for tokens
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    'https://appleid.apple.com/auth/token',
                    data={
                        'client_id': 'com.tower.echo',
                        'client_secret': await self._generate_apple_client_secret(),
                        'code': code,
                        'grant_type': 'authorization_code'
                    }
                )

                if response.status_code == 200:
                    token_data = response.json()
                    self.cached_tokens['apple'] = {
                        'access_token': token_data.get('access_token'),
                        'refresh_token': token_data.get('refresh_token'),
                        'id_token': token_data.get('id_token'),
                        'user_info': user_info
                    }

                    logger.info("✅ Apple SSO successful")
                    return self.cached_tokens['apple']

        except Exception as e:
            logger.error(f"Apple SSO callback failed: {e}")
            return {'error': str(e)}

    async def _generate_apple_client_secret(self) -> str:
        """Generate Apple client secret JWT"""
        # This requires your Apple Developer credentials
        # Normally loaded from secure storage
        import jwt as pyjwt
        from datetime import datetime, timedelta

        headers = {
            'kid': 'YOUR_KEY_ID',  # From Apple Developer
            'alg': 'ES256'
        }

        payload = {
            'iss': 'YOUR_TEAM_ID',  # From Apple Developer
            'iat': datetime.utcnow(),
            'exp': datetime.utcnow() + timedelta(days=180),
            'aud': 'https://appleid.apple.com',
            'sub': 'com.tower.echo'
        }

        # Would load private key from secure storage
        private_key = Path('/opt/tower-auth/keys/apple_private.pem').read_text()

        client_secret = pyjwt.encode(payload, private_key, algorithm='ES256', headers=headers)
        return client_secret

# Global instance
tower_auth = TowerAuthBridge()