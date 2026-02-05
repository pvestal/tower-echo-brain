"""
Echo Brain - MINIMAL WORKING VERSION
Only essential features, no duplicate systems
"""
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from datetime import datetime
import logging
import os
import time
import uuid
import traceback
from pathlib import Path
from contextlib import asynccontextmanager
from collections import defaultdict, deque

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Request metrics tracking
REQUEST_METRICS = {
    "total_requests": 0,
    "errors_4xx": 0,
    "errors_5xx": 0,
    "response_times": deque(maxlen=1000),  # Keep last 1000 response times
    "requests_by_endpoint": defaultdict(int),
    "errors_by_endpoint": defaultdict(int),
    "slowest_requests": deque(maxlen=10)  # Keep 10 slowest requests
}

# Error log for dashboard
ERROR_LOG = deque(maxlen=100)  # Keep last 100 errors

app = FastAPI(
    title="Tower Echo Brain",
    version="0.4.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Basic health endpoint
@app.get("/api")
async def api_root():
    return {
        "service": "Echo Brain",
        "version": "0.4.0",
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

# Mount the consolidated Echo Brain router - ALL endpoints in one place
try:
    from src.api.endpoints.echo_main_router import router as echo_main_router
    app.include_router(echo_main_router, prefix="/api/echo")
    logger.info("✅ Echo main router mounted at /api/echo - ALL endpoints consolidated")
except ImportError as e:
    logger.error(f"❌ Echo main router not available: {e}")
except Exception as e:
    logger.error(f"❌ Echo main router failed: {e}")

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

# Mount Unified Memory System API
try:
    from src.api.endpoints.memory_router import router as memory_router
    app.include_router(memory_router, prefix="/api/echo")
    logger.info("✅ Memory router mounted at /api/echo/memory")
except ImportError as e:
    logger.warning(f"⚠️ Memory router not available: {e}")
except Exception as e:
    logger.error(f"❌ Memory router failed: {e}")

# Mount Self-Test API
try:
    from src.api.endpoints.self_test_router import router as self_test_router
    app.include_router(self_test_router, prefix="/api/echo")
    logger.info("✅ Self-test router mounted at /api/echo/self-test")
except ImportError as e:
    logger.warning(f"⚠️ Self-test router not available: {e}")
except Exception as e:
    logger.error(f"❌ Self-test router failed: {e}")

# Mount Intelligence API - This is where Echo Brain THINKS
try:
    from src.api.endpoints.intelligence_router import router as intelligence_router
    app.include_router(intelligence_router, prefix="/api/echo")
    logger.info("✅ Intelligence router mounted at /api/echo/intelligence")
except ImportError as e:
    logger.warning(f"⚠️ Intelligence router not available: {e}")
except Exception as e:
    logger.error(f"❌ Intelligence router failed: {e}")

# Mount Reasoning API - This adds LLM synthesis for natural responses
try:
    from src.api.endpoints.reasoning_router import router as reasoning_router
    app.include_router(reasoning_router, prefix="/api/echo")
    logger.info("✅ Reasoning router mounted at /api/echo/reasoning")
except ImportError as e:
    logger.warning(f"⚠️ Reasoning router not available: {e}")
except Exception as e:
    logger.error(f"❌ Reasoning router failed: {e}")

# Mount Search API - Direct PostgreSQL conversation search
try:
    from src.api.endpoints.search_router import router as search_router
    app.include_router(search_router, prefix="/api/echo")
    logger.info("✅ Search router mounted at /api/echo/search")
except ImportError as e:
    logger.warning(f"⚠️ Search router not available: {e}")
except Exception as e:
    logger.error(f"❌ Search router failed: {e}")

# Mount Reasoning API - Transparent multi-stage reasoning
try:
    from src.api.endpoints.echo_reasoning_router import router as reasoning_router
    app.include_router(reasoning_router, prefix="/api/echo")
    logger.info("✅ Reasoning router mounted at /api/echo/reasoning")
except ImportError as e:
    logger.warning(f"⚠️ Reasoning router not available: {e}")
except Exception as e:
    logger.error(f"❌ Reasoning router failed: {e}")

# Health endpoints are now in the consolidated echo_main_router

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

# Mount static files for Vue frontend - MUST be last to serve as catch-all
frontend_path = Path("/opt/tower-echo-brain/frontend/dist")
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")
    logger.info(f"✅ Frontend mounted from {frontend_path}")
else:
    logger.warning(f"⚠️ Frontend not found at {frontend_path}")

# Enhanced request logging with metrics
@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Skip logging for static assets
    if request.url.path.startswith("/assets/") or request.url.path.endswith(".js") or request.url.path.endswith(".css"):
        return await call_next(request)

    # Generate request ID
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id

    # Track request start time
    start_time = time.time()

    # Log incoming request
    client_ip = request.client.host if request.client else "unknown"
    logger.info(f"[{request_id}] → {request.method} {request.url.path} from {client_ip}")

    try:
        # Process request
        response = await call_next(request)

        # Calculate response time
        duration_ms = (time.time() - start_time) * 1000

        # Update metrics
        REQUEST_METRICS["total_requests"] += 1
        REQUEST_METRICS["response_times"].append(duration_ms)
        REQUEST_METRICS["requests_by_endpoint"][request.url.path] += 1

        # Track errors
        if 400 <= response.status_code < 500:
            REQUEST_METRICS["errors_4xx"] += 1
            REQUEST_METRICS["errors_by_endpoint"][request.url.path] += 1
        elif response.status_code >= 500:
            REQUEST_METRICS["errors_5xx"] += 1
            REQUEST_METRICS["errors_by_endpoint"][request.url.path] += 1

        # Track slow requests
        if duration_ms > 1000:  # Over 1 second
            REQUEST_METRICS["slowest_requests"].append({
                "path": request.url.path,
                "method": request.method,
                "duration_ms": duration_ms,
                "timestamp": datetime.now().isoformat(),
                "request_id": request_id
            })

        # Log response
        logger.info(f"[{request_id}] ← {response.status_code} ({duration_ms:.2f}ms)")

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"

        return response

    except Exception as e:
        # Log exception
        duration_ms = (time.time() - start_time) * 1000
        logger.error(f"[{request_id}] ❌ Exception after {duration_ms:.2f}ms: {str(e)}")

        # Track error
        REQUEST_METRICS["errors_5xx"] += 1
        REQUEST_METRICS["errors_by_endpoint"][request.url.path] += 1

        # Store error for dashboard
        ERROR_LOG.append({
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method,
            "error": str(e),
            "traceback": traceback.format_exc()
        })

        # Return error response
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "request_id": request_id}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8309)
