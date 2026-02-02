"""
Echo Brain - MINIMAL WORKING VERSION
Only essential features, no duplicate systems
"""
from fastapi import FastAPI
from src.api.endpoints.echo_frontend_router import router as echo_frontend_router
from datetime import datetime
import logging
import os
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Tower Echo Brain",
    version="4.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Basic health endpoint
@app.get("/")
async def root():
    return {
        "service": "Echo Brain",
        "version": "4.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "echo-brain",
        "database_user": os.getenv("DB_USER", "not_set"),
        "timestamp": datetime.now().isoformat()
    }

# Mount essential routers only
try:
    # IMPORTANT: Include the echo frontend router
    app.include_router(echo_frontend_router)
    logger.info("✅ Echo frontend router mounted")
except Exception as e:
    logger.error(f"❌ Echo frontend router failed: {e}")

try:
    from src.api.endpoints.system_router import router as system_router
    app.include_router(system_router, prefix="/api/system")
    logger.info("✅ System router mounted")
except Exception as e:
    logger.error(f"❌ System router failed: {e}")

try:
    from src.api.endpoints.conversation_minimal_router import router as conv_router
    app.include_router(conv_router, prefix="/api/conversations")
    logger.info("✅ Conversation router mounted")
except Exception as e:
    logger.error(f"❌ Conversation router failed: {e}")

# Mount Moltbook router with proper error handling
try:
    from src.api.endpoints.moltbook_router import router as moltbook_router
    app.include_router(moltbook_router, prefix="/api/echo/moltbook")
    logger.info("✅ Moltbook router mounted")
except ImportError as e:
    logger.warning(f"⚠️ Moltbook router not available: {e}")
except Exception as e:
    logger.error(f"❌ Moltbook router failed: {e}")

# Mount Agent API with proper async context
try:
    from src.api.agents import router as agent_router
    app.include_router(agent_router)
    logger.info("✅ Agent router mounted")
except ImportError as e:
    logger.warning(f"⚠️ Agent router not available: {e}")
except Exception as e:
    logger.error(f"❌ Agent router failed: {e}")

# Mount Autonomous API with initialization check
try:
    from src.api.autonomous import router as autonomous_router
    app.include_router(autonomous_router)
    logger.info("✅ Autonomous router mounted")
except ImportError as e:
    logger.warning(f"⚠️ Autonomous router not available: {e}")
except Exception as e:
    logger.error(f"❌ Autonomous router failed: {e}")

# Create minimal MCP endpoints inline for now
@app.post("/mcp")
async def mcp_handler(request: dict):
    """Handle MCP requests with Qdrant integration"""
    from src.integrations.mcp_service import mcp_service

    method = request.get("method", "")

    if method == "tools/list":
        return {
            "tools": [
                {"name": "search_memory", "description": "Search Echo Brain memories"},
                {"name": "get_facts", "description": "Get facts from Echo Brain"},
                {"name": "store_fact", "description": "Store a fact in Echo Brain"}
            ]
        }
    elif method == "tools/call":
        tool_name = request.get("params", {}).get("name")
        arguments = request.get("params", {}).get("arguments", {})

        try:
            if tool_name == "search_memory":
                query = arguments.get("query", "")
                limit = arguments.get("limit", 10)
                results = await mcp_service.search_memory(query, limit)
                return results

            elif tool_name == "get_facts":
                topic = arguments.get("topic")
                limit = arguments.get("limit", 100)
                facts = await mcp_service.get_facts(topic, limit)
                return facts

            elif tool_name == "store_fact":
                subject = arguments.get("subject", "")
                predicate = arguments.get("predicate", "")
                object_ = arguments.get("object", "")
                confidence = arguments.get("confidence", 1.0)
                fact_id = await mcp_service.store_fact(subject, predicate, object_, confidence)
                return {"fact_id": fact_id, "stored": True}

            else:
                return {"error": f"Unknown tool: {tool_name}"}

        except Exception as e:
            logger.error(f"MCP handler error: {e}")
            return {"error": str(e)}

    return {"error": f"Unknown method: {method}"}

@app.get("/mcp/health")
async def mcp_health():
    """MCP health check"""
    return {
        "status": "ok",
        "service": "echo-brain-mcp",
        "version": "1.0.0"
    }

# Simple request logging
@app.middleware("http")
async def log_requests(request, call_next):
    logger.info(f"📡 {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"📤 Response: {response.status_code}")
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8309)
