"""
Plaid Integration for Echo Brain (using Plaid Python SDK v2)
Mocked for demonstration until credentials are found
"""
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
import hvac
from pathlib import Path

logger = logging.getLogger(__name__)

class PlaidIntegration:
    """Manages Plaid financial account connections"""

    def __init__(self):
        self.vault_client = hvac.Client(url='http://127.0.0.1:8200')
        self.client_id = None
        self.secret = None
        self.environment = 'sandbox'
        self._load_credentials()

    def _load_credentials(self):
        """Load Plaid credentials from Vault or use mock"""
        try:
            # Load Vault token
            token_path = Path('/opt/vault/.vault-token')
            if token_path.exists():
                self.vault_client.token = token_path.read_text().strip()

            # Try to get Plaid credentials from Vault
            try:
                plaid_data = self.vault_client.secrets.kv.v2.read_secret_version(
                    path='plaid/credentials'
                )
                if plaid_data:
                    data = plaid_data['data']['data']
                    self.client_id = data['client_id']
                    self.secret = data['secret']
                    self.environment = data.get('environment', 'sandbox')
                    logger.info("Plaid credentials loaded from Vault")
                    return
            except:
                pass

            # Use mock/demo mode
            logger.info("Plaid running in demo mode (no credentials found)")
            self.client_id = 'demo_client_id'
            self.secret = 'demo_secret'

        except Exception as e:
            logger.error(f"Failed to load Plaid credentials: {e}")

    async def get_accounts(self, access_token: str = None) -> List[Dict[str, Any]]:
        """Get mock account data for demonstration"""
        # Mock accounts for demonstration
        return [
            {
                'account_id': 'checking_001',
                'name': 'Chase Checking',
                'type': 'depository',
                'subtype': 'checking',
                'balance': {
                    'current': 5234.67,
                    'available': 5234.67,
                    'limit': None
                },
                'mask': '4532'
            },
            {
                'account_id': 'savings_001',
                'name': 'Chase Savings',
                'type': 'depository',
                'subtype': 'savings',
                'balance': {
                    'current': 15678.90,
                    'available': 15678.90,
                    'limit': None
                },
                'mask': '9876'
            },
            {
                'account_id': 'credit_001',
                'name': 'Chase Freedom',
                'type': 'credit',
                'subtype': 'credit card',
                'balance': {
                    'current': 1234.56,
                    'available': 8765.44,
                    'limit': 10000
                },
                'mask': '1234'
            }
        ]

    async def get_transactions(
        self,
        access_token: str = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        account_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get mock transaction data for demonstration"""
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)
        if not end_date:
            end_date = datetime.now()

        # Mock transactions for demonstration
        transactions = [
            {
                'transaction_id': 'tx_001',
                'account_id': 'checking_001',
                'amount': 89.99,
                'date': (datetime.now() - timedelta(days=1)).isoformat(),
                'name': 'Target',
                'merchant_name': 'Target',
                'category': ['Shopping', 'General Merchandise'],
                'pending': False
            },
            {
                'transaction_id': 'tx_002',
                'account_id': 'checking_001',
                'amount': 45.67,
                'date': (datetime.now() - timedelta(days=2)).isoformat(),
                'name': 'Whole Foods',
                'merchant_name': 'Whole Foods Market',
                'category': ['Food and Drink', 'Groceries'],
                'pending': False
            },
            {
                'transaction_id': 'tx_003',
                'account_id': 'credit_001',
                'amount': 156.78,
                'date': (datetime.now() - timedelta(days=3)).isoformat(),
                'name': 'Dick\'s Sporting Goods',
                'merchant_name': 'Dick\'s Sporting Goods',
                'category': ['Shopping', 'Sporting Goods'],
                'pending': False
            },
            {
                'transaction_id': 'tx_004',
                'account_id': 'checking_001',
                'amount': -2500.00,
                'date': (datetime.now() - timedelta(days=5)).isoformat(),
                'name': 'Direct Deposit - Payroll',
                'merchant_name': None,
                'category': ['Transfer', 'Payroll'],
                'pending': False
            }
        ]

        return transactions

    async def get_financial_insights(self, access_token: str = None) -> Dict[str, Any]:
        """Generate financial insights from account data"""
        accounts = await self.get_accounts(access_token)
        transactions = await self.get_transactions(access_token)

        # Calculate spending by category
        category_spending = {}
        total_spending = 0
        for txn in transactions:
            if txn['amount'] > 0:  # Positive amounts are debits
                total_spending += txn['amount']
                for category in txn['category']:
                    if category not in category_spending:
                        category_spending[category] = 0
                    category_spending[category] += txn['amount']

        # Calculate account totals
        total_balance = sum(acc['balance']['current'] for acc in accounts if acc['type'] == 'depository')
        total_debt = sum(acc['balance']['current'] for acc in accounts if acc['type'] == 'credit')

        return {
            'total_balance': total_balance,
            'total_debt': total_debt,
            'net_worth': total_balance - total_debt,
            'accounts_count': len(accounts),
            'recent_transactions_count': len(transactions),
            'spending_by_category': dict(sorted(category_spending.items(), key=lambda x: x[1], reverse=True)),
            'total_spending_30d': total_spending,
            'insights': {
                'average_daily_spending': total_spending / 30,
                'most_expensive_category': max(category_spending.items(), key=lambda x: x[1])[0] if category_spending else None,
                'savings_rate': (2500 - total_spending) / 2500 * 100 if total_spending < 2500 else 0
            },
            'demo_mode': True  # Indicate this is demo data
        }

    async def create_link_token(self, user_id: str = 'patrick') -> Dict[str, Any]:
        """Create a Plaid Link token (mock for demo)"""
        return {
            'success': True,
            'link_token': 'link-sandbox-demo-token',
            'expiration': (datetime.now() + timedelta(hours=4)).isoformat(),
            'demo_mode': True
        }