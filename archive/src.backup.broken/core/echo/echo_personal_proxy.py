#!/usr/bin/env python3
"""
Echo Personal Context Proxy
Intercepts requests to Echo and adds Patrick's personal context
"""
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Echo Personal Context Proxy")

class QueryRequest(BaseModel):
    query: str
    intelligence_level: str = "professional"
    conversation_id: str = ""

# Patrick's personal context profile
PATRICK_CONTEXT = {
    "name": "Patrick",
    "role": "Senior Full-Stack Developer",
    "expertise": ["Python", "FastAPI", "Docker", "Kubernetes", "PostgreSQL", "React"],
    "current_projects": ["Tower AI System", "Financial Automation", "DevOps Optimization"],
    "communication_style": "Direct, technical, solution-focused",
    "technical_background": {
        "browser_profile": "709MB Chrome profile with developer-focused research",
        "google_ecosystem": "725 files, technical tutorials and educational content",
        "financial_docs": "762 documents, investment-focused professional",
        "personal_files": "523,789 files, extensive code projects and documentation"
    },
    "work_environment": "Home office, Linux primary, terminal-heavy workflow",
    "preferences": {
        "response_style": "Technical detail matching senior developer expertise",
        "explanation_depth": "Assume advanced technical knowledge", 
        "code_examples": "Always include when relevant",
        "communication": "Minimal small talk, direct solutions"
    }
}

def enhance_query_with_personal_context(query: str) -> str:
    """Enhance query with Patrick's personal context"""
    query_lower = query.lower()
    
    # Detect Patrick-specific queries
    patrick_indicators = [
        os.getenv("TOWER_USER", "patrick"), "my", "i am", "help me", "what should i", "based on my",
        "recommend", "suggest", "what would you", "how should i"
    ]
    
    is_personal_query = any(indicator in query_lower for indicator in patrick_indicators)
    
    if is_personal_query:
        # Add comprehensive personal context
        context_prefix = f"""
Context: You are responding to Patrick, a {PATRICK_CONTEXT['role']} with expertise in {', '.join(PATRICK_CONTEXT['expertise'])}. 

Patrick's Profile:
- Currently working on: {', '.join(PATRICK_CONTEXT['current_projects'])}
- Communication style: {PATRICK_CONTEXT['communication_style']}
- Technical background: Extensive experience with full-stack development and DevOps
- Preferences: {PATRICK_CONTEXT['preferences']['response_style']}

Please provide senior-level technical responses with specific implementation details.

User Query: {query}"""
        
        return context_prefix
    
    # For non-personal queries, return as-is
    return query

@app.post("/api/echo/query")
async def enhanced_query(request: QueryRequest):
    """Enhanced Echo query with personal context"""
    try:
        # Enhance query with personal context
        enhanced_query = enhance_query_with_personal_context(request.query)
        
        # Forward to actual Echo service
        echo_request = {
            "query": enhanced_query,
            "intelligence_level": request.intelligence_level,
            "conversation_id": request.conversation_id
        }
        
        # Send to Echo on Tower
        response = requests.post(
            "http://127.0.0.1:8309/api/echo/query",
            json=echo_request,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
            
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Echo service error: {e}")

@app.get("/api/echo/health")
async def health_check():
    """Health check with personal context status"""
    try:
        # Check Echo service health
        echo_health = requests.get("http://127.0.0.1:8309/api/echo/health", timeout=5)
        
        if echo_health.status_code == 200:
            echo_status = echo_health.json()
            return {
                **echo_status,
                "personal_context": "enabled",
                "patrick_profile": "loaded",
                "enhanced_responses": True
            }
        else:
            raise HTTPException(status_code=503, detail="Echo service unavailable")
            
    except requests.exceptions.RequestException:
        raise HTTPException(status_code=503, detail="Cannot connect to Echo service")

@app.get("/api/personal/profile")
async def get_patrick_profile():
    """Get Patrick's personal profile"""
    return PATRICK_CONTEXT

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8313)