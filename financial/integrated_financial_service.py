#!/usr/bin/env python3
"""
Integrated Financial Service
Connects AI Assist with Plaid, Vestal Estate, Loan Search, and Board systems
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Optional
import asyncio
import httpx
from datetime import datetime
from decimal import Decimal

# Import our ecosystem
import sys
sys.path.append('/opt/tower-echo-brain/financial')
from echo_financial_ecosystem import EchoFinancialAssistant

app = FastAPI(title="Echo Financial Services")

# CORS for dashboard access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://***REMOVED***", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize Echo Financial Assistant
echo_assistant = EchoFinancialAssistant()

# Service endpoints
SERVICES = {
    'plaid': 'http://localhost:8089',
    'vestal_estate': 'http://localhost:8400',
    'loan_search': 'http://localhost:8401',
    'board': 'http://localhost:8402'
}

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "echo-financial-ecosystem",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/financial/connect-bank")
async def connect_bank(user_id: str, public_token: str):
    """Connect bank account via Plaid"""

    # Exchange public token for access token with Plaid
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{SERVICES['plaid']}/api/plaid/exchange-token",
            json={"public_token": public_token}
        )

        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to exchange token")

        plaid_data = response.json()

    # Store in our ecosystem
    result = await echo_assistant.ecosystem.connect_plaid_account(
        user_id,
        plaid_data['access_token']
    )

    return result

@app.get("/api/financial/overview/{user_id}")
async def get_financial_overview(user_id: str):
    """Get consolidated financial overview"""

    # Check permissions (in production, verify auth token)
    overview = await echo_assistant.ecosystem.get_consolidated_view(user_id)

    return overview

@app.post("/api/financial/board/submit-decision")
async def submit_board_decision(proposal: Dict):
    """Submit a decision to the board"""

    result = await echo_assistant.ecosystem.submit_board_decision(proposal)

    # Notify board members
    await notify_board_members(result['decision_id'])

    return result

@app.post("/api/financial/board/vote")
async def cast_vote(member_id: str, decision_id: str, vote: str):
    """Cast a board vote"""

    result = await echo_assistant.ecosystem.cast_board_vote(
        member_id,
        decision_id,
        vote
    )

    return result

@app.get("/api/financial/trust/{trust_id}")
async def get_trust_info(trust_id: str, user_id: str):
    """Get trust information"""

    # Verify user has access to this trust
    # ... verification logic ...

    # Get trust details from vestal-estate
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{SERVICES['vestal_estate']}/api/trust/{trust_id}",
            headers={"X-User-Id": user_id}
        )

    return response.json() if response.status_code == 200 else {"error": "Trust not found"}

@app.post("/api/financial/trust/distribution")
async def request_distribution(
    trust_id: str,
    beneficiary_id: str,
    amount: float,
    purpose: str
):
    """Request a trust distribution"""

    # Check trust rules
    # ... rule checking logic ...

    # If amount > threshold, needs board approval
    if amount > 25000:
        proposal = {
            'submitted_by': beneficiary_id,
            'type': 'trust_distribution',
            'amount': amount,
            'description': f'Trust distribution for {purpose}',
            'beneficiary': beneficiary_id,
            'purpose': purpose
        }

        board_result = await submit_board_decision(proposal)

        return {
            'status': 'pending_board_approval',
            'decision_id': board_result['decision_id']
        }

    # Auto-approve small distributions
    return {
        'status': 'approved',
        'amount': amount,
        'purpose': purpose
    }

@app.post("/api/financial/loan/search")
async def search_loans(user_id: str):
    """Search for loan options"""

    # Get user's financial profile
    profile = await echo_assistant.ecosystem.get_consolidated_view(user_id)

    # Search loans via loan-search service
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{SERVICES['loan_search']}/api/search",
            json={
                'profile': profile,
                'user_id': user_id
            }
        )

    return response.json() if response.status_code == 200 else []

@app.post("/api/financial/echo/query")
async def echo_financial_query(request: Dict):
    """Process financial query through Echo"""

    user_id = request.get("user_id")
    query = request.get("query")

    if not user_id or not query:
        raise HTTPException(status_code=400, detail="Missing user_id or query")

    result = await echo_assistant.process_financial_query(user_id, query)

    return result

@app.websocket("/ws/financial/{user_id}")
async def financial_websocket(websocket, user_id: str):
    """WebSocket for real-time financial updates"""

    await websocket.accept()

    try:
        while True:
            # Send periodic updates
            overview = await echo_assistant.ecosystem.get_consolidated_view(user_id)
            await websocket.send_json({
                'type': 'overview_update',
                'data': overview
            })

            await asyncio.sleep(60)  # Update every minute

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()

async def notify_board_members(decision_id: str):
    """Notify board members of new decision"""

    # In production, send email/SMS notifications
    print(f"📧 Notifying board members about decision {decision_id}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8403)
