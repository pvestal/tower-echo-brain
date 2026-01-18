"""
Financial Integration Module
Provides access to financial data via Plaid API
"""

import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

try:
    from plaid.api import plaid_api
    from plaid import Configuration, ApiClient
    from plaid.configuration import Environment
    from plaid.model.accounts_get_request import AccountsGetRequest
    from plaid.model.transactions_get_request import TransactionsGetRequest
    from plaid.model.accounts_balance_get_request import AccountsBalanceGetRequest
    from plaid.model.investments_holdings_get_request import InvestmentsHoldingsGetRequest
    from plaid.model.link_token_create_request import LinkTokenCreateRequest
    PLAID_AVAILABLE = True
except ImportError:
    PLAID_AVAILABLE = False
    logger.warning("Plaid module not available - financial features disabled")
    # Create dummy Environment class
    class Environment:
        sandbox = "https://sandbox.plaid.com"
        development = "https://development.plaid.com"
        production = "https://production.plaid.com"

class PlaidFinancialIntegration:
    """Integration with Plaid for financial data access"""

    def __init__(self, vault_path: str = "/home/patrick/.tower_credentials/vault.json"):
        self.vault_path = vault_path
        self.client = None
        self.access_tokens = {}
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Plaid client with credentials"""

        if not PLAID_AVAILABLE:
            logger.warning("Plaid module not installed - financial features will be limited")
            self.client = None
            return

        try:
            # Load credentials from vault
            with open(self.vault_path, 'r') as f:
                vault = json.load(f)

            plaid_config = vault.get("plaid", {})

            if not plaid_config:
                # Use known credentials from VueBudgetFire
                plaid_config = {
                    "client_id": "67b7532c37f3d10023aba53e",
                    "secret": os.getenv("PLAID_SECRET", ""),
                    "environment": "sandbox"  # Start with sandbox
                }

            # Configure Plaid
            env_name = plaid_config.get("environment", "sandbox")
            if env_name == "sandbox":
                host = Environment.Sandbox
            elif env_name == "production":
                host = Environment.Production
            else:
                host = Environment.Sandbox

            configuration = Configuration(
                host=host,
                api_key={
                    'clientId': plaid_config['client_id'],
                    'secret': plaid_config.get('secret', '')
                }
            )

            api_client = ApiClient(configuration)
            self.client = plaid_api.PlaidApi(api_client)

            logger.info("Plaid client initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Plaid client: {e}")
            raise

    async def create_link_token(self, user_id: str = "echo_brain_user") -> Dict[str, Any]:
        """
        Create a Link token for Plaid authentication

        Args:
            user_id: User identifier

        Returns:
            Link token response
        """

        try:
            request = LinkTokenCreateRequest(
                products=['accounts', 'transactions', 'investments'],
                client_name="Echo Brain Financial",
                country_codes=['US'],
                language='en',
                user={
                    'client_user_id': user_id
                }
            )

            response = self.client.link_token_create(request)

            return {
                "success": True,
                "link_token": response['link_token'],
                "expiration": response['expiration']
            }

        except Exception as e:
            logger.error(f"Failed to create link token: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def exchange_public_token(self, public_token: str, user_id: str = "echo_brain_user") -> bool:
        """
        Exchange public token for access token

        Args:
            public_token: Public token from Plaid Link
            user_id: User identifier

        Returns:
            True if successful
        """

        try:
            response = self.client.item_public_token_exchange(
                ItemPublicTokenExchangeRequest(public_token=public_token)
            )

            self.access_tokens[user_id] = response['access_token']

            # Save to vault for persistence
            self._save_access_token(user_id, response['access_token'])

            logger.info(f"Successfully exchanged token for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to exchange token: {e}")
            return False

    async def get_accounts(self, user_id: str = "echo_brain_user") -> Dict[str, Any]:
        """
        Get all accounts for a user

        Args:
            user_id: User identifier

        Returns:
            Account information
        """

        access_token = self._get_access_token(user_id)
        if not access_token:
            return {
                "success": False,
                "error": "No access token found. Please authenticate first."
            }

        try:
            request = AccountsGetRequest(access_token=access_token)
            response = self.client.accounts_get(request)

            accounts = []
            for account in response['accounts']:
                accounts.append({
                    "account_id": account['account_id'],
                    "name": account['name'],
                    "type": account['type'],
                    "subtype": account['subtype'],
                    "balance": {
                        "current": account['balances']['current'],
                        "available": account['balances'].get('available'),
                        "currency": account['balances']['iso_currency_code']
                    }
                })

            return {
                "success": True,
                "accounts": accounts,
                "total_accounts": len(accounts)
            }

        except Exception as e:
            logger.error(f"Failed to get accounts: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def get_balances(self, user_id: str = "echo_brain_user") -> Dict[str, Any]:
        """
        Get real-time balance information

        Args:
            user_id: User identifier

        Returns:
            Balance information
        """

        access_token = self._get_access_token(user_id)
        if not access_token:
            return {
                "success": False,
                "error": "No access token found"
            }

        try:
            request = AccountsBalanceGetRequest(access_token=access_token)
            response = self.client.accounts_balance_get(request)

            total_balance = 0
            balances_by_type = {}

            for account in response['accounts']:
                balance = account['balances']['current']
                account_type = account['type']

                if account_type not in balances_by_type:
                    balances_by_type[account_type] = 0

                balances_by_type[account_type] += balance
                total_balance += balance

            return {
                "success": True,
                "total_balance": total_balance,
                "by_type": balances_by_type,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to get balances: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def get_transactions(
        self,
        user_id: str = "echo_brain_user",
        days_back: int = 30
    ) -> Dict[str, Any]:
        """
        Get recent transactions

        Args:
            user_id: User identifier
            days_back: Number of days of history

        Returns:
            Transaction data
        """

        access_token = self._get_access_token(user_id)
        if not access_token:
            return {
                "success": False,
                "error": "No access token found"
            }

        try:
            start_date = (datetime.now() - timedelta(days=days_back)).date()
            end_date = datetime.now().date()

            request = TransactionsGetRequest(
                access_token=access_token,
                start_date=start_date,
                end_date=end_date
            )

            response = self.client.transactions_get(request)

            transactions = []
            for txn in response['transactions']:
                transactions.append({
                    "transaction_id": txn['transaction_id'],
                    "amount": txn['amount'],
                    "date": txn['date'],
                    "name": txn['name'],
                    "category": txn.get('category', []),
                    "merchant": txn.get('merchant_name'),
                    "account_id": txn['account_id']
                })

            # Calculate spending insights
            total_spending = sum(t['amount'] for t in transactions if t['amount'] > 0)
            total_income = abs(sum(t['amount'] for t in transactions if t['amount'] < 0))

            categories = {}
            for txn in transactions:
                if txn['amount'] > 0:  # Spending
                    cat = txn['category'][0] if txn['category'] else 'Other'
                    categories[cat] = categories.get(cat, 0) + txn['amount']

            return {
                "success": True,
                "transactions": transactions[:50],  # Return top 50
                "total_transactions": len(transactions),
                "insights": {
                    "total_spending": total_spending,
                    "total_income": total_income,
                    "net_cashflow": total_income - total_spending,
                    "top_categories": dict(sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5])
                }
            }

        except Exception as e:
            logger.error(f"Failed to get transactions: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def get_investments(self, user_id: str = "echo_brain_user") -> Dict[str, Any]:
        """
        Get investment holdings

        Args:
            user_id: User identifier

        Returns:
            Investment data
        """

        access_token = self._get_access_token(user_id)
        if not access_token:
            return {
                "success": False,
                "error": "No access token found"
            }

        try:
            request = InvestmentsHoldingsGetRequest(access_token=access_token)
            response = self.client.investments_holdings_get(request)

            holdings = []
            total_value = 0

            for holding in response['holdings']:
                value = holding['institution_value'] or (holding['quantity'] * holding['institution_price'])
                holdings.append({
                    "security_id": holding['security_id'],
                    "quantity": holding['quantity'],
                    "value": value,
                    "cost_basis": holding.get('cost_basis')
                })
                total_value += value

            # Get security details
            securities = {}
            for security in response['securities']:
                securities[security['security_id']] = {
                    "name": security['name'],
                    "ticker": security.get('ticker_symbol'),
                    "type": security['type']
                }

            # Enhance holdings with security details
            for holding in holdings:
                if holding['security_id'] in securities:
                    holding.update(securities[holding['security_id']])

            return {
                "success": True,
                "holdings": holdings,
                "total_value": total_value,
                "account_count": len(response['accounts'])
            }

        except Exception as e:
            logger.error(f"Failed to get investments: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _get_access_token(self, user_id: str) -> Optional[str]:
        """Get access token for user"""

        # Check memory first
        if user_id in self.access_tokens:
            return self.access_tokens[user_id]

        # Check vault
        try:
            with open(self.vault_path, 'r') as f:
                vault = json.load(f)

            tokens = vault.get("plaid_tokens", {})
            return tokens.get(user_id)

        except:
            return None

    def _save_access_token(self, user_id: str, token: str):
        """Save access token to vault"""

        try:
            with open(self.vault_path, 'r') as f:
                vault = json.load(f)

            if "plaid_tokens" not in vault:
                vault["plaid_tokens"] = {}

            vault["plaid_tokens"][user_id] = token

            with open(self.vault_path, 'w') as f:
                json.dump(vault, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save token: {e}")

    async def analyze_spending_patterns(self, user_id: str = "echo_brain_user") -> Dict[str, Any]:
        """
        Analyze spending patterns and provide insights

        Args:
            user_id: User identifier

        Returns:
            Spending analysis
        """

        transactions = await self.get_transactions(user_id, days_back=90)

        if not transactions.get("success"):
            return transactions

        # Analyze patterns
        txns = transactions["transactions"]

        # Group by day of week
        weekday_spending = {}
        for txn in txns:
            if txn['amount'] > 0:
                date = datetime.fromisoformat(txn['date'])
                weekday = date.strftime('%A')
                weekday_spending[weekday] = weekday_spending.get(weekday, 0) + txn['amount']

        # Find unusual transactions
        amounts = [t['amount'] for t in txns if t['amount'] > 0]
        if amounts:
            avg_amount = sum(amounts) / len(amounts)
            unusual = [
                t for t in txns
                if t['amount'] > avg_amount * 3
            ]
        else:
            unusual = []

        # Merchant frequency
        merchants = {}
        for txn in txns:
            if txn.get('merchant'):
                merchants[txn['merchant']] = merchants.get(txn['merchant'], 0) + 1

        return {
            "success": True,
            "patterns": {
                "highest_spending_day": max(weekday_spending.items(), key=lambda x: x[1]) if weekday_spending else None,
                "unusual_transactions": unusual[:5],
                "frequent_merchants": dict(sorted(merchants.items(), key=lambda x: x[1], reverse=True)[:10])
            }
        }