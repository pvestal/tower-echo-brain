#!/usr/bin/env python3
"""
Plaid Authentication API for Tower
FastAPI backend to support the auth UI with MFA
"""
import os
import asyncio
import json
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid
import aiohttp
import hvac
from pathlib import Path

# Pydantic models
class PlaidExchangeRequest(BaseModel):
    public_token: str
    metadata: Dict[str, Any]
    session_id: str

class MFARequest(BaseModel):
    session_id: str
    mfa_type: str
    responses: List[str]

class PlaidAuth:
    """Plaid authentication handler"""

    def __init__(self):
        self.vault_client = hvac.Client(url='http://127.0.0.1:8200')
        self.client_id = None
        self.secret = None
        self._load_credentials()

    def _load_credentials(self):
        """Load Plaid credentials from Vault"""
        try:
            token_path = Path('/opt/vault/.vault-token')
            if token_path.exists():
                self.vault_client.token = token_path.read_text().strip()
            else:
                self.vault_client.token = '***REMOVED***'

            plaid_data = self.vault_client.secrets.kv.v2.read_secret_version(
                path='plaid/credentials',
                raise_on_deleted_version=False
            )
            if plaid_data:
                data = plaid_data['data']['data']
                self.client_id = data['client_id']
                self.secret = data['secret']

        except Exception as e:
            print(f"Failed to load Plaid credentials: {e}")
            raise

    async def create_session(self, user_id: str) -> Dict[str, Any]:
        """Create new auth session"""
        session_id = str(uuid.uuid4())

        # Create Plaid Link token
        async with aiohttp.ClientSession() as session:
            url = "https://production.plaid.com/link/token/create"
            data = {
                "client_id": self.client_id,
                "secret": self.secret,
                "user": {
                    "client_user_id": user_id,
                    "legal_name": "Patrick Vestal",
                    "email_address": "patrick.vestal.digital@gmail.com"
                },
                "client_name": "Tower Financial Intelligence",
                "products": ["transactions", "auth"],
                "country_codes": ["US"],
                "language": "en",
                "webhook": f"https://vestal-garcia.duckdns.org:8089/api/plaid/webhook/{session_id}"
            }

            async with session.post(url, json=data) as resp:
                result = await resp.json()
                if resp.status == 200:
                    # Store session in Vault
                    session_data = {
                        'session_id': session_id,
                        'user_id': user_id,
                        'plaid_link_token': result['link_token'],
                        'status': 'pending',
                        'created_at': datetime.now().isoformat()
                    }

                    self.vault_client.secrets.kv.v2.create_or_update_secret(
                        path=f'plaid/sessions/{session_id}',
                        secret=session_data
                    )

                    return {
                        'session_id': session_id,
                        'plaid_link_token': result['link_token'],
                        'auth_url': f"https://vestal-garcia.duckdns.org:8089/plaid/auth?session={session_id}"
                    }
                else:
                    raise HTTPException(status_code=400, detail=result.get('error_message'))

    async def exchange_token(self, request: PlaidExchangeRequest) -> Dict[str, Any]:
        """Exchange public token for access token"""
        async with aiohttp.ClientSession() as session:
            url = "https://production.plaid.com/item/public_token/exchange"
            data = {
                "client_id": self.client_id,
                "secret": self.secret,
                "public_token": request.public_token
            }

            async with session.post(url, json=data) as resp:
                result = await resp.json()
                if resp.status == 200:
                    # Store access token
                    item_id = result['item_id']
                    access_token = result['access_token']

                    self.vault_client.secrets.kv.v2.create_or_update_secret(
                        path=f'plaid/access_tokens/{item_id}',
                        secret={
                            'access_token': access_token,
                            'item_id': item_id,
                            'session_id': request.session_id,
                            'user_id': os.getenv("TOWER_USER", "patrick"),
                            'accounts': request.metadata.get('accounts', []),
                            'institution': request.metadata.get('institution', {}),
                            'created_at': datetime.now().isoformat(),
                            'status': 'active'
                        }
                    )

                    # Update session status
                    session_data = self.vault_client.secrets.kv.v2.read_secret_version(
                        path=f'plaid/sessions/{request.session_id}',
                        raise_on_deleted_version=False
                    )
                    if session_data:
                        session_info = session_data['data']['data']
                        session_info['status'] = 'completed'
                        session_info['item_id'] = item_id
                        session_info['completed_at'] = datetime.now().isoformat()

                        self.vault_client.secrets.kv.v2.create_or_update_secret(
                            path=f'plaid/sessions/{request.session_id}',
                            secret=session_info
                        )

                    return {
                        'success': True,
                        'item_id': item_id,
                        'accounts_connected': len(request.metadata.get('accounts', [])),
                        'institution': request.metadata.get('institution', {}).get('name', 'Unknown')
                    }
                else:
                    raise HTTPException(status_code=400, detail=result.get('error_message'))

    async def get_session(self, session_id: str) -> Dict[str, Any]:
        """Get session information"""
        try:
            session_data = self.vault_client.secrets.kv.v2.read_secret_version(
                path=f'plaid/sessions/{session_id}',
                raise_on_deleted_version=False
            )
            if session_data:
                return session_data['data']['data']
            else:
                raise HTTPException(status_code=404, detail="Session not found")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def list_connected_accounts(self) -> List[Dict[str, Any]]:
        """List all connected accounts"""
        try:
            tokens_list = self.vault_client.secrets.kv.v2.list_secrets(
                path='plaid/access_tokens'
            )
            keys = tokens_list.get('data', {}).get('keys', [])

            accounts = []
            for token_key in keys:
                token_data = self.vault_client.secrets.kv.v2.read_secret_version(
                    path=f'plaid/access_tokens/{token_key}',
                    raise_on_deleted_version=False
                )
                if token_data:
                    data = token_data['data']['data']
                    accounts.append({
                        'item_id': data['item_id'],
                        'institution': data.get('institution', {}).get('name', 'Unknown'),
                        'accounts_count': len(data.get('accounts', [])),
                        'connected_at': data.get('created_at', 'Unknown'),
                        'status': data.get('status', 'unknown')
                    })

            return accounts

        except Exception as e:
            return []

# Initialize FastAPI app
app = FastAPI(title="Tower Plaid Auth API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://vestal-garcia.duckdns.org:8089", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Plaid auth
plaid_auth = PlaidAuth()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the auth UI"""
    try:
        with open('/opt/tower-echo-brain/static/plaid_auth.html', 'r') as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse("""
        <html>
            <body>
                <h1>Tower Plaid Auth</h1>
                <p>Auth UI not found. Please ensure plaid_auth.html is in the static directory.</p>
            </body>
        </html>
        """)

@app.get("/plaid/auth", response_class=HTMLResponse)
async def plaid_auth_page():
    """Serve the Plaid auth page"""
    return await read_root()

@app.post("/api/auth/plaid/create_session")
async def create_plaid_session(user_id: str = os.getenv("TOWER_USER", os.getenv("TOWER_USER", "patrick"))):
    """Create new Plaid auth session"""
    try:
        result = await plaid_auth.create_session(user_id)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/auth/plaid/session/{session_id}")
async def get_plaid_session(session_id: str):
    """Get Plaid auth session"""
    try:
        result = await plaid_auth.get_session(session_id)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/auth/plaid/exchange")
async def exchange_plaid_token(request: PlaidExchangeRequest):
    """Exchange Plaid public token for access token"""
    try:
        result = await plaid_auth.exchange_token(request)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/auth/plaid/accounts")
async def list_plaid_accounts():
    """List connected Plaid accounts"""
    try:
        accounts = plaid_auth.list_connected_accounts()
        return JSONResponse(content={'accounts': accounts})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/plaid/webhook/{session_id}")
async def plaid_webhook(session_id: str, request: Request):
    """Handle Plaid webhooks"""
    try:
        body = await request.json()
        print(f"📨 Plaid webhook for session {session_id}: {body}")

        # Handle different webhook types
        webhook_type = body.get('webhook_type')
        webhook_code = body.get('webhook_code')

        if webhook_type == 'TRANSACTIONS':
            # Handle transaction webhooks
            if webhook_code == 'INITIAL_UPDATE':
                print(f"✅ Initial transactions loaded for session {session_id}")
            elif webhook_code == 'HISTORICAL_UPDATE':
                print(f"📈 Historical transactions updated for session {session_id}")
            elif webhook_code == 'DEFAULT_UPDATE':
                print(f"🔄 New transactions available for session {session_id}")

        elif webhook_type == 'ITEM':
            # Handle item webhooks
            if webhook_code == 'WEBHOOK_UPDATE_ACKNOWLEDGED':
                print(f"✅ Webhook acknowledged for session {session_id}")
            elif webhook_code == 'ERROR':
                print(f"❌ Item error for session {session_id}: {body.get('error')}")

        return JSONResponse(content={'status': 'acknowledged'})

    except Exception as e:
        print(f"❌ Webhook error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "tower-plaid-auth"}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Tower Plaid Auth API on port 8089")
    print("🔗 Access auth UI at: https://vestal-garcia.duckdns.org:8089:8089/plaid/auth")
    uvicorn.run(app, host="0.0.0.0", port=8089)
@app.post("/api/plaid/webhook")
async def plaid_webhook_universal(request: Request):
    """Handle Plaid webhooks without requiring session_id"""
    try:
        body = await request.json()
        print(f"📨 Plaid webhook received: {json.dumps(body, indent=2)}")

        webhook_type = body.get('webhook_type')
        webhook_code = body.get('webhook_code')
        item_id = body.get('item_id')

        # Store webhook event in Vault
        webhook_data = {
            'webhook_type': webhook_type,
            'webhook_code': webhook_code,
            'item_id': item_id,
            'received_at': datetime.now().isoformat(),
            'full_body': body
        }

        try:
            # Store in Vault under webhooks path
            vault_client.secrets.kv.v2.create_or_update_secret(
                path=f'plaid/webhooks/{item_id}/{webhook_type}/{datetime.now().isoformat()}',
                secret=webhook_data
            )
        except Exception as e:
            print(f"⚠️ Failed to store webhook in Vault: {e}")

        # Process different webhook types
        if webhook_type == 'TRANSACTIONS':
            if webhook_code in ['SYNC_UPDATES', 'INITIAL_UPDATE', 'HISTORICAL_UPDATE', 'DEFAULT_UPDATE']:
                print(f"💰 Transaction update for item {item_id}")

        elif webhook_type == 'INCOME':
            if webhook_code == 'INCOME_VERIFICATION_REFRESH_COMPLETE':
                print(f"💵 Income verification complete for item {item_id}")

        elif webhook_type == 'ITEM':
            if webhook_code == 'ERROR':
                error = body.get('error', {})
                print(f"❌ Item error: {error.get('error_message')}")

        return {"status": "success", "message": f"Webhook processed for {item_id}"}

    except Exception as e:
        print(f"❌ Webhook error: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.post("/plaid/exchange")
async def plaid_exchange_redirect(request: PlaidExchangeRequest):
    """Redirect for UI that calls wrong endpoint"""
    return await exchange_plaid_token(request)

@app.get("/api/auth/plaid/exchange")
async def exchange_method_not_allowed():
    """Handle GET request to exchange endpoint"""
    raise HTTPException(status_code=405, detail="Method Not Allowed - Use POST with public_token")
