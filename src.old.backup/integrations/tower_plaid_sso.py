#!/usr/bin/env python3
"""
Tower SSO Integration for Plaid with MFA Support
Integrates Plaid Link with Tower's auth system at port 8088
"""
import asyncio
import json
import aiohttp
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging
import hvac
from pathlib import Path
import uuid

logger = logging.getLogger(__name__)

class TowerPlaidSSO:
    """Integrate Plaid with Tower's SSO auth system"""

    def __init__(self):
        self.vault_client = hvac.Client(url='http://127.0.0.1:8200')
        self.tower_auth_url = "http://127.0.0.1:8088"
        self.plaid_client_id = None
        self.plaid_secret = None
        self._load_credentials()

    def _load_credentials(self):
        """Load credentials from Vault"""
        try:
            token_path = Path('/opt/vault/.vault-token')
            if token_path.exists():
                self.vault_client.token = token_path.read_text().strip()
            else:
                self.vault_client.token = '***REMOVED***'

            # Load Plaid credentials
            plaid_data = self.vault_client.secrets.kv.v2.read_secret_version(
                path='plaid/credentials',
                raise_on_deleted_version=False
            )
            if plaid_data:
                data = plaid_data['data']['data']
                self.plaid_client_id = data['client_id']
                self.plaid_secret = data['secret']

        except Exception as e:
            logger.error(f"Failed to load credentials: {e}")
            raise

    async def create_authenticated_link_session(self, user_id: str) -> Dict[str, Any]:
        """Create a Plaid Link session authenticated through Tower SSO"""
        session_id = str(uuid.uuid4())

        # Step 1: Create Tower auth session
        tower_session = await self._create_tower_auth_session(user_id, session_id)
        if not tower_session.get('success'):
            return {'error': 'Failed to create Tower auth session', 'details': tower_session}

        # Step 2: Create Plaid Link token
        plaid_link = await self._create_plaid_link_token(user_id, session_id)
        if 'error' in plaid_link:
            return {'error': 'Failed to create Plaid Link token', 'details': plaid_link}

        # Step 3: Store session mapping in Vault
        await self._store_session_mapping(session_id, tower_session, plaid_link)

        return {
            'session_id': session_id,
            'tower_auth_token': tower_session['auth_token'],
            'plaid_link_token': plaid_link['link_token'],
            'auth_url': f"{self.tower_auth_url}/api/auth/plaid/start/{session_id}",
            'link_url': f"https://***REMOVED***/plaid/link/{session_id}",
            'expiration': plaid_link['expiration']
        }

    async def _create_tower_auth_session(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """Create auth session with Tower SSO"""
        async with aiohttp.ClientSession() as session:
            try:
                # Check if Tower auth is running
                async with session.get(f"{self.tower_auth_url}/api/auth/status") as resp:
                    if resp.status != 200:
                        # Tower auth in bypass mode, create mock session
                        return {
                            'success': True,
                            'auth_token': f"tower_bypass_{session_id}",
                            'user_id': user_id,
                            'session_id': session_id,
                            'mode': 'bypass'
                        }

                # Tower auth is running, create real session
                auth_data = {
                    'user_id': user_id,
                    'session_id': session_id,
                    'service': 'plaid',
                    'scope': ['financial:read', 'financial:transactions'],
                    'redirect_uri': f"https://***REMOVED***/plaid/callback/{session_id}"
                }

                async with session.post(f"{self.tower_auth_url}/api/auth/create_session", json=auth_data) as resp:
                    result = await resp.json()
                    if resp.status == 200:
                        return {
                            'success': True,
                            'auth_token': result['auth_token'],
                            'user_id': user_id,
                            'session_id': session_id,
                            'mode': 'active'
                        }
                    else:
                        return {'success': False, 'error': result}

            except aiohttp.ClientError:
                # Fallback to bypass mode
                return {
                    'success': True,
                    'auth_token': f"tower_bypass_{session_id}",
                    'user_id': user_id,
                    'session_id': session_id,
                    'mode': 'bypass'
                }

    async def _create_plaid_link_token(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """Create Plaid Link token with webhook and auth integration"""
        async with aiohttp.ClientSession() as session:
            url = "https://production.plaid.com/link/token/create"
            headers = {"Content-Type": "application/json"}

            data = {
                "client_id": self.plaid_client_id,
                "secret": self.plaid_secret,
                "user": {
                    "client_user_id": user_id,
                    "legal_name": "Patrick Vestal",
                    "email_address": "patrick.vestal.digital@gmail.com"
                },
                "client_name": "Tower Financial Intelligence",
                "products": ["transactions", "auth"],
                "country_codes": ["US"],
                "language": "en",
                "webhook": f"https://***REMOVED***/api/plaid/webhook/{session_id}",
                "link_customization_name": "default",
                "redirect_uri": f"https://***REMOVED***/plaid/success/{session_id}"
            }

            async with session.post(url, json=data, headers=headers) as resp:
                result = await resp.json()
                if resp.status == 200:
                    return {
                        'link_token': result['link_token'],
                        'expiration': result['expiration'],
                        'request_id': result['request_id']
                    }
                else:
                    return {'error': result.get('error_message'), 'details': result}

    async def _store_session_mapping(self, session_id: str, tower_session: Dict, plaid_link: Dict):
        """Store session mapping in Vault"""
        session_data = {
            'session_id': session_id,
            'tower_auth': tower_session,
            'plaid_link': plaid_link,
            'created_at': datetime.now().isoformat(),
            'status': 'pending'
        }

        self.vault_client.secrets.kv.v2.create_or_update_secret(
            path=f'plaid/sessions/{session_id}',
            secret=session_data
        )

    async def handle_plaid_callback(self, session_id: str, public_token: str, metadata: Dict) -> Dict[str, Any]:
        """Handle Plaid Link callback and complete auth flow"""
        try:
            # Get session data
            session_data = self.vault_client.secrets.kv.v2.read_secret_version(
                path=f'plaid/sessions/{session_id}',
                raise_on_deleted_version=False
            )
            if not session_data:
                return {'error': 'Session not found'}

            session_info = session_data['data']['data']

            # Exchange public token for access token
            access_token_result = await self._exchange_public_token(public_token)
            if 'error' in access_token_result:
                return {'error': 'Token exchange failed', 'details': access_token_result}

            # Store access token
            item_id = access_token_result['item_id']
            access_token = access_token_result['access_token']

            self.vault_client.secrets.kv.v2.create_or_update_secret(
                path=f'plaid/access_tokens/{item_id}',
                secret={
                    'access_token': access_token,
                    'item_id': item_id,
                    'session_id': session_id,
                    'user_id': session_info['tower_auth']['user_id'],
                    'accounts': metadata.get('accounts', []),
                    'institution': metadata.get('institution', {}),
                    'created_at': datetime.now().isoformat(),
                    'status': 'active'
                }
            )

            # Update session status
            session_info['status'] = 'completed'
            session_info['access_token_item_id'] = item_id
            session_info['completed_at'] = datetime.now().isoformat()

            self.vault_client.secrets.kv.v2.create_or_update_secret(
                path=f'plaid/sessions/{session_id}',
                secret=session_info
            )

            # Notify Tower auth system
            await self._notify_tower_auth_completion(session_id, item_id, metadata)

            return {
                'success': True,
                'item_id': item_id,
                'accounts_connected': len(metadata.get('accounts', [])),
                'institution': metadata.get('institution', {}).get('name', 'Unknown')
            }

        except Exception as e:
            logger.error(f"Callback handling failed: {e}")
            return {'error': 'Callback processing failed', 'details': str(e)}

    async def _exchange_public_token(self, public_token: str) -> Dict[str, Any]:
        """Exchange public token for access token"""
        async with aiohttp.ClientSession() as session:
            url = "https://production.plaid.com/item/public_token/exchange"
            data = {
                "client_id": self.plaid_client_id,
                "secret": self.plaid_secret,
                "public_token": public_token
            }

            async with session.post(url, json=data) as resp:
                result = await resp.json()
                if resp.status == 200:
                    return {
                        'access_token': result['access_token'],
                        'item_id': result['item_id'],
                        'request_id': result['request_id']
                    }
                else:
                    return {'error': result.get('error_message'), 'details': result}

    async def _notify_tower_auth_completion(self, session_id: str, item_id: str, metadata: Dict):
        """Notify Tower auth system of successful completion"""
        async with aiohttp.ClientSession() as session:
            try:
                notification_data = {
                    'session_id': session_id,
                    'service': 'plaid',
                    'status': 'completed',
                    'item_id': item_id,
                    'accounts_count': len(metadata.get('accounts', [])),
                    'institution': metadata.get('institution', {}).get('name', 'Unknown'),
                    'completed_at': datetime.now().isoformat()
                }

                async with session.post(
                    f"{self.tower_auth_url}/api/auth/service_completion",
                    json=notification_data
                ) as resp:
                    if resp.status == 200:
                        logger.info(f"✅ Tower auth notified of completion for session {session_id}")
                    else:
                        logger.warning(f"⚠️ Tower auth notification failed: {resp.status}")

            except aiohttp.ClientError:
                logger.warning("⚠️ Could not notify Tower auth (service may be in bypass mode)")

    async def get_auth_status(self, session_id: str) -> Dict[str, Any]:
        """Get authentication status for a session"""
        try:
            session_data = self.vault_client.secrets.kv.v2.read_secret_version(
                path=f'plaid/sessions/{session_id}',
                raise_on_deleted_version=False
            )
            if session_data:
                return session_data['data']['data']
            else:
                return {'error': 'Session not found'}
        except Exception as e:
            return {'error': f'Failed to get session status: {e}'}

    def list_connected_accounts(self) -> List[Dict[str, Any]]:
        """List all connected Plaid accounts"""
        try:
            tokens_list = self.vault_client.secrets.kv.v2.list_secrets(
                path='plaid/access_tokens'
            )
            keys = tokens_list.get('data', {}).get('keys', [])

            connected_accounts = []
            for token_key in keys:
                token_data = self.vault_client.secrets.kv.v2.read_secret_version(
                    path=f'plaid/access_tokens/{token_key}',
                    raise_on_deleted_version=False
                )
                if token_data:
                    data = token_data['data']['data']
                    connected_accounts.append({
                        'item_id': data['item_id'],
                        'institution': data.get('institution', {}).get('name', 'Unknown'),
                        'accounts_count': len(data.get('accounts', [])),
                        'connected_at': data.get('created_at', 'Unknown'),
                        'status': data.get('status', 'unknown')
                    })

            return connected_accounts

        except Exception as e:
            logger.error(f"Failed to list accounts: {e}")
            return []


# Demonstration and testing
async def demonstrate_tower_plaid_sso():
    """Demonstrate the complete Tower-Plaid SSO integration"""
    print("🔐 TOWER-PLAID SSO INTEGRATION")
    print("=" * 60)

    sso = TowerPlaidSSO()

    # Test 1: Create authenticated Link session
    print("\n1️⃣ Creating Authenticated Link Session:")
    session_result = await sso.create_authenticated_link_session("patrick")

    if 'error' in session_result:
        print(f"❌ Error: {session_result['error']}")
        return

    print(f"✅ Session created: {session_result['session_id']}")
    print(f"🔗 Auth URL: {session_result['auth_url']}")
    print(f"🔗 Link URL: {session_result['link_url']}")
    print(f"🎫 Tower Auth Token: {session_result['tower_auth_token'][:20]}...")
    print(f"🎫 Plaid Link Token: {session_result['plaid_link_token'][:20]}...")

    # Test 2: Check session status
    print(f"\n2️⃣ Checking Session Status:")
    status = await sso.get_auth_status(session_result['session_id'])
    print(f"📊 Status: {status.get('status', 'unknown')}")
    print(f"🏛️ Tower Auth Mode: {status.get('tower_auth', {}).get('mode', 'unknown')}")

    # Test 3: List existing connected accounts
    print(f"\n3️⃣ Connected Accounts:")
    accounts = sso.list_connected_accounts()
    if accounts:
        for acc in accounts:
            print(f"  🏦 {acc['institution']}: {acc['accounts_count']} accounts ({acc['status']})")
    else:
        print("  📭 No accounts connected yet")

    print(f"\n🚀 NEXT STEPS:")
    print(f"1. User visits: {session_result['link_url']}")
    print(f"2. Completes MFA with their bank")
    print(f"3. System exchanges tokens automatically")
    print(f"4. Access token stored in Vault")
    print(f"5. AgenticPersona begins monitoring")

    return session_result

if __name__ == "__main__":
    asyncio.run(demonstrate_tower_plaid_sso())