#!/usr/bin/env python3
"""
Financial Intelligence API
Standalone service for Patrick's financial data and patterns
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
from echo_financial_context import EchoFinancialContext

app = FastAPI(title="Financial Intelligence API", version="1.0")

# Initialize financial context
try:
    financial = EchoFinancialContext()
    print("✅ Financial Intelligence API started")
except Exception as e:
    print(f"❌ Failed to initialize: {e}")
    financial = None

class CategoryQuery(BaseModel):
    category: str
    days: Optional[int] = 30

@app.get("/")
async def root():
    return {
        "service": "Financial Intelligence API",
        "version": "1.0",
        "endpoints": {
            "GET /health": "Service health",
            "GET /api/financial/summary": "30-day spending summary",
            "GET /api/financial/summary/{days}": "Custom period summary",
            "GET /api/financial/recurring": "Recurring bills",
            "GET /api/financial/category/{category}": "Category spending",
            "GET /api/financial/preferences": "Learned preferences",
            "GET /api/financial/context": "Full context for LLM prompts"
        }
    }

@app.get("/health")
async def health():
    if financial is None:
        raise HTTPException(status_code=503, detail="Financial context not available")
    return {"status": "healthy", "service": "financial_intelligence"}

@app.get("/api/financial/summary")
async def get_summary():
    """Get 30-day spending summary"""
    if not financial:
        raise HTTPException(status_code=503, detail="Service unavailable")
    
    try:
        data = financial.get_spending_summary(30)
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/financial/summary/{days}")
async def get_summary_custom(days: int):
    """Get custom period spending summary"""
    if not financial:
        raise HTTPException(status_code=503, detail="Service unavailable")
    
    if days < 1 or days > 365:
        raise HTTPException(status_code=400, detail="Days must be between 1 and 365")
    
    try:
        data = financial.get_spending_summary(days)
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/financial/recurring")
async def get_recurring():
    """Get recurring bills"""
    if not financial:
        raise HTTPException(status_code=503, detail="Service unavailable")
    
    try:
        bills = financial.get_recurring_bills()
        total = sum(b['amount'] for b in bills)
        return {
            "recurring_bills": bills,
            "total_monthly": round(total, 2),
            "count": len(bills)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/financial/category/{category}")
async def get_category(category: str, days: int = 30):
    """Get spending for specific category"""
    if not financial:
        raise HTTPException(status_code=503, detail="Service unavailable")
    
    try:
        data = financial.get_category_spending(category, days)
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/financial/preferences")
async def get_preferences():
    """Get learned spending preferences"""
    if not financial:
        raise HTTPException(status_code=503, detail="Service unavailable")
    
    try:
        data = financial.get_financial_preferences()
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/financial/context")
async def get_context():
    """Get full financial context formatted for LLM prompts"""
    if not financial:
        raise HTTPException(status_code=503, detail="Service unavailable")
    
    try:
        context = financial.build_echo_context()
        return {
            "context": context,
            "format": "plain_text",
            "usage": "Insert into LLM system prompt"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8092)
