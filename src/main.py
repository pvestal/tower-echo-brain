#!/usr/bin/env python3
"""
Minimal Vector Search API
Only exposes POST /api/context endpoint for Qdrant search
"""
import logging
from typing import Dict, List, Optional, Literal
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import asyncio

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create minimal app
app = FastAPI(
    title="Echo Brain Minimal API",
    description="Minimal vector search endpoint only",
    version="1.0.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ContextSearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 10
    collection: Optional[str] = "echo_memory"

class ContextSearchResult(BaseModel):
    text: str
    score: float
    metadata: Dict
    collection: str  # Added to show which collection the result came from

class ContextSearchResponse(BaseModel):
    results: List[ContextSearchResult]
    query: str
    total: int

# Agent request/response models
class AgentRequest(BaseModel):
    query: str
    agent: Optional[Literal["auto", "coding", "reasoning", "narration"]] = "auto"
    include_context: Optional[bool] = True

class AgentResponse(BaseModel):
    task: str
    response: str
    agent_used: str
    routing_method: str
    model: str
    context_used: Optional[Dict] = None

class ContextRequest(BaseModel):
    query: str
    limit: Optional[int] = 10

# Initialize connections
qdrant_memory = None
agent_router = None
context_provider = None

def get_qdrant(collection_name: str = "echo_memories"):
    """Get Qdrant connection for specific collection"""
    try:
        from src.qdrant_memory import QdrantMemory
        # Create new instance for each collection (lightweight, just changes collection name)
        memory = QdrantMemory(collection_name=collection_name)
        logger.info(f"✅ Qdrant memory initialized for collection: {collection_name}")
        return memory
    except Exception as e:
        logger.error(f"❌ Failed to initialize Qdrant: {e}")
        return None

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("🚀 Starting Echo Brain API with unified agents...")
    get_qdrant()  # Pre-load Qdrant connection

    # Initialize agent router and context provider
    global agent_router, context_provider
    try:
        from src.core.agent_router import get_agent_router
        from src.core.unified_context import get_context_provider

        agent_router = get_agent_router()
        context_provider = get_context_provider()
        await context_provider.initialize()

        logger.info("✅ Agent router and context provider initialized")
    except Exception as e:
        logger.warning(f"⚠️ Agent initialization failed (non-critical): {e}")

@app.get("/health")
async def health_check():
    """Enhanced health check with agent status"""
    health_status = {
        "status": "healthy",
        "service": "echo-brain-unified",
        "agents_available": agent_router is not None,
        "context_provider_available": context_provider is not None
    }

    if agent_router:
        try:
            agent_status = await agent_router.get_agent_status()
            health_status["agents"] = agent_status.get("agents", [])
        except:
            health_status["agents"] = "error"

    return health_status

@app.get("/api/collections")
async def list_collections():
    """List available Qdrant collections with vector counts"""
    try:
        memory = get_qdrant()
        if not memory:
            return {"collections": [], "error": "Qdrant not available"}

        # Get collections from Qdrant
        response = memory.client.get_collections()
        collections = []

        for collection in response.collections:
            # Get collection info including vector count
            info = memory.client.get_collection(collection.name)
            collections.append({
                "name": collection.name,
                "vectors_count": info.points_count,
                "status": str(info.status)
            })

        return {"collections": collections, "total": len(collections)}
    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        return {"collections": [], "error": str(e)}

@app.post("/api/context", response_model=ContextSearchResponse)
async def search_context(request: ContextSearchRequest):
    """
    Search vector memory for relevant context
    """
    try:
        # Use the collection specified in the request
        memory = get_qdrant(request.collection)
        if not memory:
            raise HTTPException(status_code=503, detail="Vector memory not available")

        # Perform search
        logger.info(f"Searching for: {request.query[:100]}...")

        # Search in Qdrant (async method)
        results = await memory.search_memories(
            query=request.query,
            limit=request.limit
        )

        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append(ContextSearchResult(
                text=result.get("text", ""),
                score=result.get("score", 0.0),
                metadata=result.get("metadata", {}),
                collection=request.collection  # Include collection name
            ))

        return ContextSearchResponse(
            results=formatted_results,
            query=request.query,
            total=len(formatted_results)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/api/agent", response_model=AgentResponse)
async def unified_agent_endpoint(request: AgentRequest):
    """
    Single endpoint that routes to appropriate agent with context.
    Automatically selects the best agent or uses the specified one.
    """
    if not agent_router:
        raise HTTPException(status_code=503, detail="Agent router not available")

    try:
        logger.info(f"Agent request: {request.query[:50]}... (agent: {request.agent})")

        # Route to appropriate agent
        result = await agent_router.route(
            query=request.query,
            agent=request.agent,
            include_context=request.include_context
        )

        return AgentResponse(**result)

    except Exception as e:
        logger.error(f"Agent processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Agent processing failed: {str(e)}")

@app.post("/api/context/unified")
async def get_unified_context(request: ContextRequest):
    """
    Get context from all sources without agent processing.
    Useful for external consumers who want raw context.
    """
    if not context_provider:
        raise HTTPException(status_code=503, detail="Context provider not available")

    try:
        logger.info(f"Context request: {request.query[:50]}...")

        # Get context from all sources
        context = await context_provider.get_context(
            query=request.query,
            limit=request.limit
        )

        return context

    except Exception as e:
        logger.error(f"Context retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Context retrieval failed: {str(e)}")

@app.get("/api/agents/status")
async def get_agents_status():
    """
    Get status and statistics for all available agents.
    """
    if not agent_router:
        return {"error": "Agent router not available", "agents": []}

    try:
        return await agent_router.get_agent_status()
    except Exception as e:
        logger.error(f"Failed to get agent status: {e}")
        return {"error": str(e), "agents": []}

# Include the existing agent routes if available
try:
    from src.api.agents import router as agents_router
    app.include_router(agents_router)
    logger.info("✅ Individual agent endpoints loaded")
except ImportError:
    logger.info("ℹ️ Individual agent endpoints not available")

# Include autonomous system routes
try:
    from src.api.autonomous import router as autonomous_router
    app.include_router(autonomous_router)
    logger.info("✅ Autonomous system endpoints loaded")
except ImportError as e:
    logger.info(f"ℹ️ Autonomous system endpoints not available: {e}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8309,
        reload=False,
        log_level="info"
    )