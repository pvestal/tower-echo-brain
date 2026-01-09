#!/usr/bin/env python3
"""
Plaid Webhook Handlers with Tower MFA Authentication
Handles transaction updates, income refreshes, and wallet events
"""
import os
import asyncio
import json
import hashlib
import hmac
import secrets
from fastapi import FastAPI, HTTPException, Request, Depends, status
from src.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import aiohttp
import hvac
from pathlib import Path
import logging
import pyotp
import qrcode
import io
import base64

logger = logging.getLogger(__name__)

# Pydantic models for webhooks
class PlaidWebhook(BaseModel):
    webhook_type: str
    webhook_code: str
    item_id: str
    error: Optional[Dict] = None
    new_transactions: Optional[int] = None
    removed_transactions: Optional[List[str]] = None

class IncomeWebhook(BaseModel):
    webhook_type: str
    webhook_code: str
    item_id: str
    income: Optional[Dict] = None
    status: Optional[str] = None

class TransferWebhook(BaseModel):
    webhook_type: str
    webhook_code: str
    transfer_id: str
    account_id: str
    amount: float
    status: str

# Tower MFA Authentication
class TowerMFA:
    """Handle MFA for Tower access"""

    def __init__(self):
        self.vault_client = hvac.Client(url='http://127.0.0.1:8200')
        self.vault_client.token = self._get_vault_token()
        self.totp_secrets = {}
        self._load_mfa_secrets()

    def _get_vault_token(self):
        """Get Vault token"""
        token_path = Path('/opt/vault/.vault-token')
        if token_path.exists():
            return token_path.read_text().strip()
        return '***REMOVED***'

    def _load_mfa_secrets(self):
        """Load or create MFA secrets"""
        try:
            mfa_data = self.vault_client.secrets.kv.v2.read_secret_version(
                path='tower/mfa/secrets',
                raise_on_deleted_version=False
            )
            if mfa_data:
                self.totp_secrets = mfa_data['data']['data']
        except:
            # Generate new MFA secret for Patrick
            self.create_mfa_user(os.getenv("TOWER_USER", "patrick"))

    def create_mfa_user(self, user_id: str):
        """Create new MFA user"""
        secret = pyotp.random_base32()
        self.totp_secrets[user_id] = secret

        # Store in Vault
        self.vault_client.secrets.kv.v2.create_or_update_secret(
            path='tower/mfa/secrets',
            secret=self.totp_secrets
        )

        # Generate QR code
        totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            name=user_id,
            issuer_name='Tower Financial'
        )

        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(totp_uri)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")
        buf = io.BytesIO()
        img.save(buf, format='PNG')

        qr_base64 = base64.b64encode(buf.getvalue()).decode()

        return {
            'secret': secret,
            'qr_code': f"data:image/png;base64,{qr_base64}",
            'uri': totp_uri
        }

    def verify_totp(self, user_id: str, token: str) -> bool:
        """Verify TOTP token"""
        if user_id not in self.totp_secrets:
            return False

        totp = pyotp.TOTP(self.totp_secrets[user_id])
        return totp.verify(token, valid_window=1)

    def generate_session_token(self, user_id: str) -> str:
        """Generate authenticated session token"""
        session_token = secrets.token_urlsafe(32)

        # Store session in Vault
        self.vault_client.secrets.kv.v2.create_or_update_secret(
            path=f'tower/sessions/{session_token}',
            secret={
                'user_id': user_id,
                'created_at': datetime.now().isoformat(),
                'expires_at': (datetime.now() + timedelta(hours=24)).isoformat()
            }
        )

        return session_token

    def verify_session(self, token: str) -> Optional[str]:
        """Verify session token"""
        try:
            session_data = self.vault_client.secrets.kv.v2.read_secret_version(
                path=f'tower/sessions/{token}',
                raise_on_deleted_version=False
            )
            if session_data:
                data = session_data['data']['data']
                expires_at = datetime.fromisoformat(data['expires_at'])
                if expires_at > datetime.now():
                    return data['user_id']
        except:
            pass
        return None

# Webhook Handler
class PlaidWebhookHandler:
    """Handle Plaid webhooks with AgenticPersona integration"""

    def __init__(self):
        self.vault_client = hvac.Client(url='http://127.0.0.1:8200')
        self.vault_client.token = self._get_vault_token()
        self.plaid_client_id = None
        self.plaid_secret = None
        self._load_credentials()

    def _get_vault_token(self):
        """Get Vault token"""
        token_path = Path('/opt/vault/.vault-token')
        if token_path.exists():
            return token_path.read_text().strip()
        return '***REMOVED***'

    def _load_credentials(self):
        """Load Plaid credentials"""
        try:
            plaid_data = self.vault_client.secrets.kv.v2.read_secret_version(
                path='plaid/credentials',
                raise_on_deleted_version=False
            )
            if plaid_data:
                data = plaid_data['data']['data']
                self.plaid_client_id = data['client_id']
                self.plaid_secret = data['secret']
        except Exception as e:
            logger.error(f"Failed to load Plaid credentials: {e}")

    async def handle_transactions_webhook(self, webhook: PlaidWebhook):
        """Handle transaction webhooks"""
        logger.info(f"📊 Transaction webhook: {webhook.webhook_code} for item {webhook.item_id}")

        if webhook.webhook_code == "INITIAL_UPDATE":
            # Initial transaction pull complete
            await self.fetch_new_transactions(webhook.item_id, webhook.new_transactions)
            await self.notify_agentic_persona('financial', {
                'event': 'initial_transactions_loaded',
                'item_id': webhook.item_id,
                'count': webhook.new_transactions
            })

        elif webhook.webhook_code == "HISTORICAL_UPDATE":
            # Historical transactions ready
            await self.fetch_historical_transactions(webhook.item_id)
            await self.notify_agentic_persona('financial', {
                'event': 'historical_transactions_loaded',
                'item_id': webhook.item_id
            })

        elif webhook.webhook_code == "DEFAULT_UPDATE":
            # New transactions available
            await self.fetch_new_transactions(webhook.item_id, webhook.new_transactions)
            await self.analyze_spending_patterns(webhook.item_id)
            await self.notify_agentic_persona('financial', {
                'event': 'new_transactions',
                'item_id': webhook.item_id,
                'count': webhook.new_transactions
            })

        elif webhook.webhook_code == "TRANSACTIONS_REMOVED":
            # Transactions removed (duplicates/corrections)
            await self.handle_removed_transactions(webhook.item_id, webhook.removed_transactions)

        return {'status': 'processed', 'webhook_code': webhook.webhook_code}

    async def handle_income_webhook(self, webhook: IncomeWebhook):
        """Handle income verification webhooks"""
        logger.info(f"💰 Income webhook: {webhook.webhook_code} for item {webhook.item_id}")

        if webhook.webhook_code == "INCOME_VERIFICATION_REFRESH_STARTED":
            # Income refresh started
            await self.notify_agentic_persona('financial', {
                'event': 'income_refresh_started',
                'item_id': webhook.item_id,
                'status': 'processing'
            })

        elif webhook.webhook_code == "INCOME_VERIFICATION_REFRESH_COMPLETE":
            # Income refresh complete
            income_data = await self.fetch_income_data(webhook.item_id)
            await self.notify_agentic_persona('financial', {
                'event': 'income_refresh_complete',
                'item_id': webhook.item_id,
                'income': income_data,
                'status': 'complete'
            })

            # Store income data in Vault
            self.vault_client.secrets.kv.v2.create_or_update_secret(
                path=f'plaid/income/{webhook.item_id}',
                secret={
                    'income_data': income_data,
                    'updated_at': datetime.now().isoformat()
                }
            )

        elif webhook.webhook_code == "INCOME_VERIFICATION_RISK_SIGNALS":
            # Risk signals detected
            await self.handle_income_risk_signals(webhook.item_id, webhook.income)

        return {'status': 'processed', 'webhook_code': webhook.webhook_code}

    async def handle_transfer_webhook(self, webhook: TransferWebhook):
        """Handle wallet/transfer webhooks"""
        logger.info(f"💸 Transfer webhook: {webhook.webhook_code} for transfer {webhook.transfer_id}")

        if webhook.webhook_code == "TRANSFER_EVENTS_UPDATE":
            # Transfer status update
            if webhook.status == "posted":
                # Transfer completed
                await self.notify_agentic_persona('financial', {
                    'event': 'transfer_complete',
                    'transfer_id': webhook.transfer_id,
                    'amount': webhook.amount,
                    'account_id': webhook.account_id
                })

            elif webhook.status == "failed":
                # Transfer failed
                await self.notify_agentic_persona('financial', {
                    'event': 'transfer_failed',
                    'transfer_id': webhook.transfer_id,
                    'amount': webhook.amount,
                    'reason': 'Check transfer details'
                })

        return {'status': 'processed', 'webhook_code': webhook.webhook_code}

    async def fetch_new_transactions(self, item_id: str, count: int):
        """Fetch new transactions from Plaid"""
        # Get access token
        token_data = self.vault_client.secrets.kv.v2.read_secret_version(
            path=f'plaid/access_tokens/{item_id}',
            raise_on_deleted_version=False
        )

        if token_data:
            access_token = token_data['data']['data']['access_token']

            # Fetch transactions
            async with aiohttp.ClientSession() as session:
                url = "https://production.plaid.com/transactions/get"
                data = {
                    "client_id": self.plaid_client_id,
                    "secret": self.plaid_secret,
                    "access_token": access_token,
                    "start_date": (datetime.now() - timedelta(days=7)).date().isoformat(),
                    "end_date": datetime.now().date().isoformat(),
                    "count": count
                }

                async with session.post(url, json=data) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        transactions = result['transactions']

                        # Check for large transactions
                        for txn in transactions:
                            if abs(txn['amount']) > 500:
                                await self.alert_large_transaction(txn)

                        return transactions

        return []

    async def fetch_income_data(self, item_id: str):
        """Fetch income verification data"""
        # Get access token
        token_data = self.vault_client.secrets.kv.v2.read_secret_version(
            path=f'plaid/access_tokens/{item_id}',
            raise_on_deleted_version=False
        )

        if token_data:
            access_token = token_data['data']['data']['access_token']

            # Fetch income data
            async with aiohttp.ClientSession() as session:
                url = "https://production.plaid.com/income/verification/get"
                data = {
                    "client_id": self.plaid_client_id,
                    "secret": self.plaid_secret,
                    "access_token": access_token
                }

                async with session.post(url, json=data) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        return result.get('income', {})

        return {}

    async def analyze_spending_patterns(self, item_id: str):
        """Analyze spending patterns for insights"""
        # This would integrate with AgenticPersona for pattern analysis
        logger.info(f"🔍 Analyzing spending patterns for item {item_id}")

    async def alert_large_transaction(self, transaction: Dict):
        """Alert on large transactions"""
        logger.warning(f"⚠️ Large transaction: {transaction['name']} - ${abs(transaction['amount']):,.2f}")

        # Send notification via Echo
        await self.notify_agentic_persona('financial', {
            'event': 'large_transaction_alert',
            'transaction': transaction,
            'threshold_exceeded': True
        })

    async def notify_agentic_persona(self, persona_type: str, data: Dict):
        """Notify AgenticPersona of events"""
        logger.info(f"🤖 Notifying {persona_type} persona: {data['event']}")

        # Store event for AgenticPersona processing
        event_id = secrets.token_hex(8)
        self.vault_client.secrets.kv.v2.create_or_update_secret(
            path=f'agentic_events/{persona_type}/{event_id}',
            secret={
                'data': data,
                'timestamp': datetime.now().isoformat(),
                'processed': False
            }
        )

    async def handle_removed_transactions(self, item_id: str, transaction_ids: List[str]):
        """Handle removed/corrected transactions"""
        logger.info(f"🗑️ Removing {len(transaction_ids)} transactions for item {item_id}")

# FastAPI app with MFA
app = FastAPI(title="Tower Plaid Webhooks with MFA", version="2.0.0")

# Initialize handlers
mfa_handler = TowerMFA()
webhook_handler = PlaidWebhookHandler()
security = HTTPBearer()

# MFA Authentication dependency
async def verify_mfa(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify MFA token"""
    user_id = mfa_handler.verify_session(credentials.credentials)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session"
        )
    return user_id

# MFA Setup endpoints
@app.get("/mfa/setup/{user_id}")
async def setup_mfa(user_id: str):
    """Setup MFA for user"""
    result = mfa_handler.create_mfa_user(user_id)

    return HTMLResponse(f"""
    <html>
        <head><title>Tower MFA Setup</title></head>
        <body style="font-family: Arial; padding: 20px;">
            <h1>🔐 Tower MFA Setup</h1>
            <p>Scan this QR code with your authenticator app:</p>
            <img src="{result['qr_code']}" />
            <p><strong>Manual Entry:</strong> {result['secret']}</p>
            <p>Once configured, use the code from your app to login.</p>
        </body>
    </html>
    """)

@app.post("/mfa/verify")
async def verify_mfa_login(user_id: str, totp_code: str):
    """Verify MFA login"""
    if mfa_handler.verify_totp(user_id, totp_code):
        session_token = mfa_handler.generate_session_token(user_id)
        return {
            'success': True,
            'session_token': session_token,
            'expires_in': 86400
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid MFA code"
        )

# Protected webhook endpoints
@app.post("/api/webhooks/transactions", dependencies=[Depends(verify_mfa)])
async def handle_transactions(webhook: PlaidWebhook):
    """Handle transaction webhooks (MFA protected)"""
    return await webhook_handler.handle_transactions_webhook(webhook)

@app.post("/api/webhooks/income", dependencies=[Depends(verify_mfa)])
async def handle_income(webhook: IncomeWebhook):
    """Handle income webhooks (MFA protected)"""
    return await webhook_handler.handle_income_webhook(webhook)

@app.post("/api/webhooks/transfers", dependencies=[Depends(verify_mfa)])
async def handle_transfers(webhook: TransferWebhook):
    """Handle transfer/wallet webhooks (MFA protected)"""
    return await webhook_handler.handle_transfer_webhook(webhook)

# Public webhook endpoint for Plaid (uses webhook verification)
@app.post("/api/plaid/webhook/{session_id}")
async def plaid_webhook_public(session_id: str, request: Request):
    """Public webhook endpoint for Plaid callbacks"""
    try:
        body = await request.json()
        webhook_type = body.get('webhook_type')
        webhook_code = body.get('webhook_code')

        logger.info(f"📨 Plaid webhook: {webhook_type}/{webhook_code} for session {session_id}")

        # Route to appropriate handler
        if webhook_type == "TRANSACTIONS":
            webhook = PlaidWebhook(**body)
            result = await webhook_handler.handle_transactions_webhook(webhook)

        elif webhook_type == "INCOME":
            webhook = IncomeWebhook(**body)
            result = await webhook_handler.handle_income_webhook(webhook)

        elif webhook_type == "TRANSFER":
            webhook = TransferWebhook(**body)
            result = await webhook_handler.handle_transfer_webhook(webhook)

        else:
            result = {'status': 'unhandled', 'type': webhook_type}

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Webhook error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check"""
    return {"status": "healthy", "service": "tower-plaid-webhooks", "mfa": "enabled"}

if __name__ == "__main__":
    import uvicorn

    # Install required packages
    import subprocess
    subprocess.run(["pip", "install", "-q", "pyotp", "qrcode", "pillow"], check=False)

    print("🔐 Tower Plaid Webhooks with MFA")
    print("=" * 60)
    print("📍 Webhook Endpoints:")
    print("  • Transactions: /api/webhooks/transactions (MFA required)")
    print("  • Income: /api/webhooks/income (MFA required)")
    print("  • Transfers: /api/webhooks/transfers (MFA required)")
    print("  • Public: /api/plaid/webhook/{session_id}")
    print("\n🔒 MFA Setup:")
    print("  • Setup: /mfa/setup/patrick")
    print("  • Verify: POST /mfa/verify")
    print("\n🚀 Starting server on port 8090...")

    uvicorn.run(app, host="0.0.0.0", port=8090)