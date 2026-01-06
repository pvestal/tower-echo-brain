#!/usr/bin/env python3
"""
Simple Plaid Service for Echo Financial Ecosystem
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
from datetime import datetime
from decimal import Decimal
import json

app = FastAPI(title="Plaid Banking Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Mock account data (in production, this comes from real banks)
MOCK_ACCOUNTS = {
    "patrick": {
        "institution": "Chase Bank",
        "accounts": [
            {
                "account_id": "chase_checking_001",
                "name": "Patrick's Checking",
                "type": "checking",
                "balance": 25000.00,
                "available": 24500.00
            },
            {
                "account_id": "chase_savings_001",
                "name": "Patrick's Savings",
                "type": "savings",
                "balance": 75000.00,
                "available": 75000.00
            },
            {
                "account_id": "chase_business_001",
                "name": "Vestal Tech LLC Operating",
                "type": "business_checking",
                "balance": 85000.00,
                "available": 82000.00
            }
        ]
    },
    "partner": {
        "institution": "Wells Fargo",
        "accounts": [
            {
                "account_id": "wells_checking_001",
                "name": "Partner's Checking",
                "type": "checking",
                "balance": 15000.00,
                "available": 14200.00
            },
            {
                "account_id": "wells_savings_001",
                "name": "Partner's Savings",
                "type": "savings",
                "balance": 45000.00,
                "available": 45000.00
            }
        ]
    }
}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "plaid-banking",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/plaid/exchange-token")
async def exchange_token(token_data: Dict):
    """Exchange public token for access token (mock)"""

    public_token = token_data.get("public_token")

    if not public_token:
        raise HTTPException(status_code=400, detail="Missing public token")

    # In real Plaid, this exchanges the token with the bank
    # For demo, we return a mock access token
    return {
        "access_token": f"access-token-{public_token}",
        "item_id": f"item-{datetime.now().timestamp()}",
        "institution_id": "ins_109508",
        "institution_name": "Chase Bank"
    }

@app.get("/api/plaid/accounts/{user_id}")
async def get_accounts(user_id: str):
    """Get bank accounts for user"""

    if user_id not in MOCK_ACCOUNTS:
        raise HTTPException(status_code=404, detail="User not found")

    user_data = MOCK_ACCOUNTS[user_id]

    return {
        "user_id": user_id,
        "institution": user_data["institution"],
        "accounts": user_data["accounts"],
        "total_balance": sum(acc["balance"] for acc in user_data["accounts"]),
        "last_updated": datetime.now().isoformat()
    }

@app.get("/api/plaid/transactions/{account_id}")
async def get_transactions(account_id: str, limit: int = 10):
    """Get transactions for an account"""

    # Mock transactions
    mock_transactions = [
        {
            "transaction_id": f"trans_{account_id}_001",
            "amount": -45.67,
            "date": "2025-09-25",
            "name": "GROCERY STORE",
            "category": ["Food and Drink", "Groceries"]
        },
        {
            "transaction_id": f"trans_{account_id}_002",
            "amount": -125.00,
            "date": "2025-09-24",
            "name": "ELECTRIC COMPANY",
            "category": ["Payment", "Utilities"]
        },
        {
            "transaction_id": f"trans_{account_id}_003",
            "amount": 2500.00,
            "date": "2025-09-23",
            "name": "SALARY DEPOSIT",
            "category": ["Deposit", "Payroll"]
        }
    ]

    return {
        "account_id": account_id,
        "transactions": mock_transactions[:limit],
        "count": len(mock_transactions)
    }

@app.get("/api/plaid/net-worth/{user_id}")
async def get_net_worth(user_id: str):
    """Calculate user's net worth from connected accounts"""

    if user_id not in MOCK_ACCOUNTS:
        return {"net_worth": 0, "error": "No accounts connected"}

    accounts = MOCK_ACCOUNTS[user_id]["accounts"]

    # Calculate net worth (assets - liabilities)
    assets = sum(acc["balance"] for acc in accounts if acc["balance"] > 0)

    return {
        "user_id": user_id,
        "net_worth": assets,
        "accounts_count": len(accounts),
        "institution": MOCK_ACCOUNTS[user_id]["institution"],
        "calculated_at": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8089)