#!/usr/bin/env python3
"""
Simple Echo Brain - Actually working service
No bullshit, just conversation persistence and health checks
"""

import asyncio
import psycopg2
import json
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List

# Database connection
def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        database="echo_brain",
        user="patrick",
        password="tower_echo_brain_secret_key_2025"
    )

# FastAPI app
app = FastAPI(title="Simple Echo Brain", version="0.1.0")

class QueryRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None

class QueryResponse(BaseModel):
    response: str
    conversation_id: str
    timestamp: datetime

@app.get("/api/echo/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "simple_echo", "timestamp": datetime.now()}

@app.get("/api/echo/session-context")
async def get_session_context():
    """Get session context for Claude continuity - THE ACTUAL SOLUTION"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get recent conversations
        cursor.execute("""
            SELECT query_text, response_text, intent, timestamp
            FROM conversations
            WHERE timestamp >= NOW() - INTERVAL '24 hours'
            ORDER BY timestamp DESC
            LIMIT 5
        """)
        recent_conversations = cursor.fetchall()

        # Get learned facts
        cursor.execute("""
            SELECT learned_fact, fact_type, created_at
            FROM learning_history
            ORDER BY created_at DESC
            LIMIT 10
        """)
        learned_facts = cursor.fetchall()

        cursor.close()
        conn.close()

        return {
            "user_profile": {
                "name": "Patrick",
                "preferences": ["direct responses", "no promotional naming", "technical accuracy"],
                "session_themes": ["persistence fixes", "architecture cleanup", "honest assessment"]
            },
            "recent_activity": [
                {
                    "topic": row[0][:80] if row[0] else "System query",
                    "intent": row[2],
                    "when": row[3].strftime("%H:%M") if row[3] else "unknown"
                }
                for row in recent_conversations
            ],
            "key_learnings": [
                {
                    "fact": row[0],
                    "type": row[1],
                    "when": row[2].strftime("%m-%d %H:%M") if row[2] else "unknown"
                }
                for row in learned_facts
            ],
            "current_issues": [
                "19,570 Python files in echo brain",
                "Multiple 2,800+ line files doing same job",
                "Main echo.py has syntax errors and missing imports",
                "Session amnesia between Claude instances"
            ],
            "working_fixes": [
                "Created simple_echo.py",
                "Renamed promotional files to descriptive names",
                "Fixed conversations table name"
            ]
        }

    except Exception as e:
        return {
            "error": f"Database error: {str(e)}",
            "fallback_context": {
                "user_profile": {"name": "Patrick", "focus": "persistence and cleanup"},
                "current_issues": ["File chaos", "Session amnesia", "Import errors"],
                "status": "degraded"
            }
        }

@app.post("/api/echo/query", response_model=QueryResponse)
async def simple_query(request: QueryRequest):
    """Simple query endpoint that just logs and responds"""
    conversation_id = request.conversation_id or f"conv_{int(datetime.now().timestamp())}"

    # Simple response
    response_text = f"Echo received: {request.query}"

    # Log to database if possible
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO conversations (conversation_id, query_text, response, intent, timestamp)
            VALUES (%s, %s, %s, %s, %s)
        """, (conversation_id, request.query, response_text, "simple", datetime.now()))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Database logging failed: {e}")

    return QueryResponse(
        response=response_text,
        conversation_id=conversation_id,
        timestamp=datetime.now()
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8309)