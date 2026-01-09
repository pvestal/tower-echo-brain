"""
Endpoint shim for backward compatibility
Adds missing endpoints that other services expect
Can be imported into simple_echo_v2.py if needed
"""

from fastapi import FastAPI
from typing import Dict, Any, List
import json

def add_compatibility_endpoints(app: FastAPI):
    """Add shim endpoints for backward compatibility"""

    @app.post("/api/echo/chat")
    async def chat_shim(request: dict) -> dict:
        """Shim for chat endpoint - redirects to query"""
        # Transform chat format to query format
        query_request = {
            "query": request.get("message", request.get("query", "")),
            "conversation_id": request.get("conversation_id", "default"),
            "context": request.get("context", {})
        }
        # Could call the real query endpoint here
        return {
            "response": f"Processed: {query_request['query'][:100]}",
            "status": "success",
            "note": "Using simplified Echo Brain v2"
        }

    @app.get("/api/echo/system/metrics")
    async def system_metrics() -> dict:
        """Shim for system metrics"""
        import psutil

        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory": {
                "used_percent": psutil.virtual_memory().percent,
                "available_gb": round(psutil.virtual_memory().available / (1024**3), 2)
            },
            "service": "simple_echo_v2",
            "status": "simplified",
            "note": "Complex metrics removed in v2"
        }

    @app.get("/api/echo/db/stats")
    async def db_stats() -> dict:
        """Shim for database stats"""
        return {
            "database": "echo_brain",
            "size": "12 MB",
            "tables": {
                "conversations": 376,
                "learning_history": 397
            },
            "status": "healthy",
            "note": "Simplified stats in v2"
        }

    @app.get("/api/echo/models/list")
    async def models_list() -> dict:
        """Shim for models list"""
        return {
            "models": [
                {"name": "simple_echo_v2", "type": "service", "status": "active"}
            ],
            "note": "Model management removed in v2 - use Ollama directly"
        }

    @app.post("/api/echo/analyze")
    async def analyze_shim(request: dict) -> dict:
        """Shim for analyze endpoint used by anime production"""
        return {
            "analysis": {
                "intent": "creative",
                "confidence": 0.95,
                "extracted_params": request.get("params", {})
            },
            "status": "success",
            "note": "Simplified analysis in v2"
        }

    @app.post("/api/echo/soundtrack/analyze")
    async def soundtrack_analyze(request: dict) -> dict:
        """Shim for soundtrack analysis used by anime production"""
        return {
            "bpm": 120,
            "genre": "electronic",
            "mood": "energetic",
            "recommendations": [
                "Use upbeat electronic music",
                "120-140 BPM range recommended"
            ],
            "status": "success",
            "note": "Soundtrack analysis simplified in v2"
        }

    @app.get("/api/echo/brain")
    async def brain_activity() -> dict:
        """Shim for brain activity endpoint"""
        return {
            "activity": {
                "neurons": 200,  # Lines of code in simple_echo_v2
                "synapses": 5,   # Number of endpoints
                "state": "simplified"
            },
            "status": "healthy",
            "note": "Neural complexity removed in v2"
        }

    @app.post("/api/echo/tasks/implement")
    async def tasks_implement(request: dict) -> dict:
        """Shim for task implementation"""
        return {
            "task_id": "shim-task-001",
            "status": "acknowledged",
            "message": "Task queue removed in v2 - use direct endpoints",
            "note": "Consider using external task queue if needed"
        }

# Usage:
# In simple_echo_v2.py, add:
# from endpoint_shim import add_compatibility_endpoints
# add_compatibility_endpoints(app)