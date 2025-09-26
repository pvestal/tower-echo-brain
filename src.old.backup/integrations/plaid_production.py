#!/usr/bin/env python3
"""
Production Plaid Integration for Echo Brain
Using real Plaid API with production credentials
"""
import asyncio
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
import hvac
from pathlib import Path

logger = logging.getLogger(__name__)

class PlaidProductionIntegration:
    """Production Plaid API integration with real credentials"""

    def __init__(self):
        self.vault_client = hvac.Client(url='http://127.0.0.1:8200')
        self.client = None
        self.client_id = None
        self.secret = None
        self.environment = 'production'
        self._initialized = False
        self._load_credentials()
        self._init_plaid_client()

    def _load_credentials(self):
        """Load production Plaid credentials from Vault"""
        try:
            # Load Vault token
            token_path = Path('/opt/vault/.vault-token')
            if token_path.exists():
                self.vault_client.token = token_path.read_text().strip()
            else:
                self.vault_client.token = '***REMOVED***'

            # Get Plaid credentials from Vault
            plaid_data = self.vault_client.secrets.kv.v2.read_secret_version(
                path='plaid/credentials',
                raise_on_deleted_version=False
            )
            if plaid_data:
                data = plaid_data['data']['data']
                self.client_id = data['client_id']
                self.secret = data['secret']
                self.environment = data.get('environment', 'production')
                logger.info(f"✅ Plaid production credentials loaded (Client ID: {self.client_id})")

        except Exception as e:
            logger.error(f"Failed to load Plaid credentials: {e}")
            raise

    def _init_plaid_client(self):
        """Initialize Plaid Python SDK client"""
        try:
            import plaid
            from plaid.api import plaid_api
            from plaid.configuration import Configuration
            from plaid.api_client import ApiClient

            # Configure for production
            configuration = Configuration(
                host=plaid.Environment.production,
                api_key={
                    'clientId': self.client_id,
                    'secret': self.secret,
                }
            )

            api_client = ApiClient(configuration)
            self.client = plaid_api.PlaidApi(api_client)
            self._initialized = True
            logger.info("✅ Plaid client initialized for production")

        except ImportError:
            logger.warning("Plaid SDK not installed. Using HTTP API fallback.")
            self._initialized = False

    async def create_link_token(self, user_id: str) -> Dict[str, Any]:
        """Create a Link token for Plaid Link initialization"""
        if not self._initialized:
            # Use HTTP API directly
            import aiohttp
            async with aiohttp.ClientSession() as session:
                url = "https://production.plaid.com/link/token/create"
                headers = {
                    "Content-Type": "application/json",
                }
                data = {
                    "client_id": self.client_id,
                    "secret": self.secret,
                    "user": {
                        "client_user_id": user_id
                    },
                    "client_name": "Tower Echo Brain",
                    "products": ["accounts", "transactions", "investments"],
                    "country_codes": ["US"],
                    "language": "en",
                    "android_package_name": None,
                    "redirect_uri": None
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
                        logger.error(f"Failed to create link token: {result}")
                        return {'error': result.get('error_message', 'Unknown error')}

        else:
            # Use SDK
            from plaid.model.link_token_create_request import LinkTokenCreateRequest
            from plaid.model.country_code import CountryCode
            from plaid.model.products import Products

            request = LinkTokenCreateRequest(
                products=[Products('accounts'), Products('transactions')],
                client_name="Tower Echo Brain",
                country_codes=[CountryCode('US')],
                language='en',
                user={
                    'client_user_id': user_id
                }
            )

            response = self.client.link_token_create(request)
            return {
                'link_token': response['link_token'],
                'expiration': response['expiration']
            }

    async def exchange_public_token(self, public_token: str) -> str:
        """Exchange a public token for an access token"""
        if not self._initialized:
            # Use HTTP API
            import aiohttp
            async with aiohttp.ClientSession() as session:
                url = "https://production.plaid.com/item/public_token/exchange"
                data = {
                    "client_id": self.client_id,
                    "secret": self.secret,
                    "public_token": public_token
                }

                async with session.post(url, json=data) as resp:
                    result = await resp.json()
                    if resp.status == 200:
                        # Store access token in Vault
                        self.vault_client.secrets.kv.v2.create_or_update_secret(
                            path=f'plaid/access_tokens/{result["item_id"]}',
                            secret={'access_token': result['access_token']}
                        )
                        return result['access_token']
                    else:
                        raise Exception(f"Token exchange failed: {result}")

        else:
            # Use SDK
            from plaid.model.item_public_token_exchange_request import ItemPublicTokenExchangeRequest

            request = ItemPublicTokenExchangeRequest(
                public_token=public_token
            )
            response = self.client.item_public_token_exchange(request)

            # Store in Vault
            self.vault_client.secrets.kv.v2.create_or_update_secret(
                path=f'plaid/access_tokens/{response["item_id"]}',
                secret={'access_token': response['access_token']}
            )

            return response['access_token']

    async def get_accounts(self, access_token: str) -> List[Dict[str, Any]]:
        """Get real account data using access token"""
        if not self._initialized:
            # Use HTTP API
            import aiohttp
            async with aiohttp.ClientSession() as session:
                url = "https://production.plaid.com/accounts/get"
                data = {
                    "client_id": self.client_id,
                    "secret": self.secret,
                    "access_token": access_token
                }

                async with session.post(url, json=data) as resp:
                    result = await resp.json()
                    if resp.status == 200:
                        return result['accounts']
                    else:
                        logger.error(f"Failed to get accounts: {result}")
                        return []

        else:
            # Use SDK
            from plaid.model.accounts_get_request import AccountsGetRequest

            request = AccountsGetRequest(access_token=access_token)
            response = self.client.accounts_get(request)
            return response['accounts']

    async def get_transactions(
        self,
        access_token: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get real transactions using access token"""
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)
        if not end_date:
            end_date = datetime.now()

        if not self._initialized:
            # Use HTTP API
            import aiohttp
            async with aiohttp.ClientSession() as session:
                url = "https://production.plaid.com/transactions/get"
                data = {
                    "client_id": self.client_id,
                    "secret": self.secret,
                    "access_token": access_token,
                    "start_date": start_date.date().isoformat(),
                    "end_date": end_date.date().isoformat()
                }

                async with session.post(url, json=data) as resp:
                    result = await resp.json()
                    if resp.status == 200:
                        return result['transactions']
                    else:
                        logger.error(f"Failed to get transactions: {result}")
                        return []

        else:
            # Use SDK
            from plaid.model.transactions_get_request import TransactionsGetRequest

            request = TransactionsGetRequest(
                access_token=access_token,
                start_date=start_date.date(),
                end_date=end_date.date()
            )
            response = self.client.transactions_get(request)
            return response['transactions']

    async def get_investment_holdings(self, access_token: str) -> Dict[str, Any]:
        """Get investment holdings"""
        if not self._initialized:
            # Use HTTP API
            import aiohttp
            async with aiohttp.ClientSession() as session:
                url = "https://production.plaid.com/investments/holdings/get"
                data = {
                    "client_id": self.client_id,
                    "secret": self.secret,
                    "access_token": access_token
                }

                async with session.post(url, json=data) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    else:
                        logger.error(f"Failed to get holdings: {await resp.text()}")
                        return {}

        else:
            # Use SDK
            from plaid.model.investments_holdings_get_request import InvestmentsHoldingsGetRequest

            request = InvestmentsHoldingsGetRequest(access_token=access_token)
            response = self.client.investments_holdings_get(request)
            return response

    def get_stored_access_tokens(self) -> List[str]:
        """Get all stored access tokens from Vault"""
        try:
            # List all access tokens
            tokens = self.vault_client.secrets.kv.v2.list_secrets(
                path='plaid/access_tokens'
            )
            access_tokens = []
            for token_key in tokens.get('data', {}).get('keys', []):
                token_data = self.vault_client.secrets.kv.v2.read_secret_version(
                    path=f'plaid/access_tokens/{token_key}',
                    raise_on_deleted_version=False
                )
                if token_data:
                    access_tokens.append(token_data['data']['data']['access_token'])

            return access_tokens

        except Exception as e:
            logger.error(f"Failed to retrieve access tokens: {e}")
            return []


# Demo and test
async def test_plaid_production():
    """Test production Plaid integration"""
    print("🏦 PLAID PRODUCTION INTEGRATION TEST")
    print("=" * 60)

    plaid = PlaidProductionIntegration()

    # Check for existing access tokens
    existing_tokens = plaid.get_stored_access_tokens()
    print(f"\n📌 Found {len(existing_tokens)} stored access tokens")

    if existing_tokens:
        # Use first token for demo
        access_token = existing_tokens[0]
        print("Using existing access token for demonstration...")

        # Get accounts
        print("\n💳 ACCOUNTS:")
        accounts = await plaid.get_accounts(access_token)
        for acc in accounts:
            balance = acc.get('balances', {}).get('current', 0)
            print(f"  - {acc['name']}: ${balance:,.2f}")

        # Get recent transactions
        print("\n💸 RECENT TRANSACTIONS:")
        transactions = await plaid.get_transactions(access_token)
        for txn in transactions[:5]:
            print(f"  - {txn['date']}: {txn['name']} (${txn['amount']:.2f})")

    else:
        print("\n⚠️ No access tokens found. To connect accounts:")
        print("1. Create a Link token:")
        link_token = await plaid.create_link_token("patrick")
        print(f"   Link token: {link_token.get('link_token', 'Failed to create')[:20]}...")

        print("\n2. Open Plaid Link in browser with this token")
        print("3. User authenticates with their bank")
        print("4. Exchange public token for access token")
        print("5. Store access token and start monitoring")

    print("\n✅ Plaid production integration ready!")
    print(f"  Client ID: {plaid.client_id}")
    print(f"  Environment: {plaid.environment}")

if __name__ == "__main__":
    asyncio.run(test_plaid_production())