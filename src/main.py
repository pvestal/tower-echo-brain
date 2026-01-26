#!/usr/bin/env python3
"""
Echo Brain Unified Service - Single Entry Point
Consolidates all Echo Brain functionality into one organized service
"""

import os
import sys
import logging
import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import httpx
import asyncpg

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Echo Brain Unified API",
    description="Unified Echo Brain service with all functionality consolidated",
    version="3.0.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= Configuration =============
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'echo_brain',
    'user': 'patrick',
    'password': os.getenv('ECHO_BRAIN_DB_PASSWORD', os.getenv("TOWER_DB_PASSWORD", "RP78eIrW7cI2jYvL5akt1yurE"))
}

# ============= Request/Response Models =============

class ContextSearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 10
    collection: Optional[str] = "echo_memory"

class ContextSearchResult(BaseModel):
    text: str
    score: float
    metadata: Dict
    collection: str

class ContextSearchResponse(BaseModel):
    results: List[ContextSearchResult]
    query: str
    total: int

class ChatRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None
    include_context: Optional[bool] = True
    model: Optional[str] = "llama3.1:8b"
    temperature: Optional[float] = 0.7

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    context_used: Optional[List[Dict]] = None
    model_used: str

class AgentRequest(BaseModel):
    task: str
    agent_type: Optional[str] = "general"
    context: Optional[Dict] = {}

class AgentResponse(BaseModel):
    result: str
    agent_used: str
    task_id: str
    status: str

# ============= Import Modules =============

# Import main Echo API with intent classification and routing
try:
    from .api.echo import router as echo_router
    app.include_router(echo_router)  # No prefix - routes already have /api/echo/
    logger.info("✅ Echo Brain API with intent classification loaded")
except Exception as e:
    logger.warning(f"Could not load Echo API: {e}")

# Import agents router for coding, reasoning, narration
try:
    from .api.agents import router as agents_router
    app.include_router(agents_router)  # Routes have /api/echo/agents/ prefix
    logger.info("✅ Agents router loaded (coding, reasoning, narration)")
except Exception as e:
    logger.warning(f"Could not load agents router: {e}")

# Import autonomous operations router
try:
    from .api.autonomous import router as autonomous_router
    app.include_router(autonomous_router)
    logger.info("✅ Autonomous operations router loaded")
except Exception as e:
    logger.warning(f"Could not load autonomous router: {e}")

# Import codebase interaction router
try:
    from .api.codebase import router as codebase_router
    app.include_router(codebase_router)  # Routes have /api/echo/codebase/ prefix
    logger.info("✅ Codebase interaction router loaded")
except Exception as e:
    logger.warning(f"Could not load codebase router: {e}")

# Import git operations router
try:
    from .api.git_operations import router as git_router
    app.include_router(git_router)  # Check what prefix it needs
    logger.info("✅ Git operations router loaded")
except Exception as e:
    logger.warning(f"Could not load git router: {e}")

# Import anime module
try:
    from .modules.anime.anime_router import router as anime_router
    app.include_router(anime_router)
    logger.info("✅ Anime module loaded")
except Exception as e:
    logger.warning(f"Could not load anime module: {e}")

# Import anime production API
try:
    from .api.anime import router as anime_api_router
    app.include_router(anime_api_router)
    logger.info("✅ Anime Production API loaded (LTX video generation)")
except Exception as e:
    logger.warning(f"Could not load anime production API: {e}")

# Import diagnostics module
try:
    from .api.diagnostics import router as diagnostics_router, echo_diagnostics, handle_diagnosis_request
    app.include_router(diagnostics_router)
    logger.info("✅ Diagnostics module loaded")
except Exception as e:
    logger.warning(f"Could not load diagnostics module: {e}")

# Import autonomous features
try:
    from .autonomous.core import AutonomousCore
    autonomous_core = AutonomousCore()
    logger.info("✅ Autonomous core loaded")
except Exception as e:
    logger.warning(f"Could not load autonomous core: {e}")
    autonomous_core = None

# Import agent manager
try:
    from .managers.agent_manager import AgentManager
    agent_manager = AgentManager()
    logger.info("✅ Agent manager loaded")
except Exception as e:
    logger.warning(f"Could not load agent manager: {e}")
    agent_manager = None

# Import Qdrant client
try:
    from .qdrant_client_resilient import ResilientQdrantMemory
    qdrant_client = ResilientQdrantMemory()
    logger.info("✅ Qdrant client loaded")
except Exception as e:
    logger.warning(f"Could not load Qdrant client: {e}")
    qdrant_client = None

# Mount all API routers
mounted_count = 0
failed_routers = []

# List of all routers to mount
# Empty prefix means router has its own prefix
routers_to_mount = [
    ("agents", ""),  # Has own prefix
    ("autonomous", ""),  # Has own prefix
    ("claude_bridge", ""),  # Has own prefix
    ("codebase", ""),  # Has own prefix
    ("db_metrics", ""),  # Has own prefix
    ("delegation_routes", ""),  # Has own prefix
    ("diagnostics", ""),  # Has own prefix
    ("echo", ""),
    ("git_operations", ""),  # Has own prefix
    ("google_calendar_api", ""),  # Has own prefix
    ("health", ""),  # Has own prefix
    ("home_assistant_api", ""),  # Has own prefix
    ("integrations", ""),  # Has own prefix
    ("knowledge", ""),  # Has own prefix
    ("models", ""),
    ("models_manager", ""),  # Has own prefix
    ("notifications_api", ""),  # Has own prefix
    ("preferences", ""),  # Has own prefix
    ("repair_api", ""),  # Has own prefix
    ("resilience_status", ""),  # Has own prefix
    ("solutions", ""),  # Has own prefix
    ("system_metrics", ""),  # Has own prefix
    ("tasks", ""),  # Has own prefix
    ("training_status", ""),
    ("vault", ""),  # Has own prefix
]

for router_name, prefix in routers_to_mount:
    try:
        module = __import__(f"src.api.{router_name}", fromlist=["router"])
        router = getattr(module, "router")
        app.include_router(router, prefix=prefix)
        mounted_count += 1
        logger.info(f"✅ Mounted {router_name} router at {prefix if prefix else '/'}")
    except Exception as e:
        failed_routers.append(router_name)
        logger.warning(f"❌ Could not mount {router_name}: {e}")

logger.info(f"✅ Successfully mounted {mounted_count}/{len(routers_to_mount)} API routers")
if failed_routers:
    logger.warning(f"⚠️ Failed to mount: {', '.join(failed_routers)}")

# Mount additional routers from src/
additional_routers = [
    ("echo_settings_api", ""),  # Has prefix /api/echo
    ("photo_comparison", ""),  # Has own routes
    ("integrate_board_api", ""),  # Has prefix /api/board
    ("services.testing", ""),  # Has prefix /test
]

for router_module, prefix in additional_routers:
    try:
        if "." in router_module:
            parts = router_module.split(".")
            module = __import__(f"src.{router_module}", fromlist=["router"])
        else:
            module = __import__(f"src.{router_module}", fromlist=["router"])
        router = getattr(module, "router")
        app.include_router(router, prefix=prefix)
        mounted_count += 1
        logger.info(f"✅ Mounted {router_module} router at {prefix}")
    except Exception as e:
        logger.warning(f"❌ Could not mount {router_module}: {e}")

# ============= Core Endpoints =============

@app.get("/")
async def root():
    """Root endpoint with service info"""
    return {
        "service": "Echo Brain Unified",
        "version": "3.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "chat": "/api/echo/chat",
            "context": "/api/context",
            "anime": "/api/echo/anime/*",
            "diagnostics": "/api/diagnostics/*",
            "agents": "/api/agent",
            "status": "/api/status"
        }
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    health_status = {
        "status": "healthy",
        "service": "echo-brain-unified",
        "timestamp": datetime.now().isoformat(),
        "components": {}
    }

    # Check agent manager
    if agent_manager:
        try:
            agents_status = await agent_manager.get_all_agents_status()
            health_status["agents_available"] = True
            health_status["agents"] = agents_status
        except:
            health_status["agents_available"] = False

    # Check Qdrant
    if qdrant_client:
        try:
            collections = await qdrant_client.get_collections()
            health_status["components"]["vector_store"] = {
                "status": "healthy",
                "collections": len(collections)
            }
        except:
            health_status["components"]["vector_store"] = {"status": "unhealthy"}

    # Check autonomous core
    if autonomous_core:
        health_status["components"]["autonomous"] = {"status": "healthy"}

    return health_status

@app.get("/api/status")
async def api_status():
    """Detailed API status"""
    try:
        # Perform quick diagnosis if available
        if 'echo_diagnostics' in globals():
            diagnosis = await echo_diagnostics.perform_comprehensive_diagnosis()
            return diagnosis
    except:
        pass

    # Fallback status
    return {
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "modules": {
            "chat": "available",
            "context": "available",
            "anime": "available" if 'anime_router' in globals() else "not loaded",
            "diagnostics": "available" if 'diagnostics_router' in globals() else "not loaded",
            "agents": "available" if agent_manager else "not loaded"
        }
    }

# ============= Simple Chat Endpoint (Backup/Fallback) =============
# NOTE: Main chat endpoint is /api/echo/chat from echo.py router
# This is a simplified fallback endpoint

@app.post("/api/chat/simple", response_model=ChatResponse)
async def simple_chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint for Echo Brain
    Used by Telegram bot and other chat interfaces
    """
    try:
        # Generate conversation ID if not provided
        conversation_id = request.conversation_id or f"chat_{uuid.uuid4().hex[:8]}"

        # Check if this is a diagnosis request
        if 'handle_diagnosis_request' in globals():
            diagnosis_response = await handle_diagnosis_request(request.query)
            if diagnosis_response:
                return ChatResponse(
                    response=diagnosis_response,
                    conversation_id=conversation_id,
                    model_used="diagnostics"
                )

        # Import conversation manager for history management
        from src.services.conversation import conversation_manager
        from src.core.user_context_manager import get_user_context_manager

        # Check for anime generation intent FIRST
        query_lower = request.query.lower()
        if any(pattern in query_lower for pattern in [
            'generate anime', 'create anime', 'make anime',
            'anime generation', 'anime video', 'anime scene'
        ]):
            # Route to anime generation service
            try:
                import aiohttp
                prompt_text = request.query.replace('generate anime', '').replace('create anime', '').strip()
                if not prompt_text:
                    prompt_text = "goblin slayer battle scene"  # Use user's known preference

                async with aiohttp.ClientSession() as session:
                    payload = {"prompt": prompt_text}
                    async with session.post("http://localhost:8328/api/anime/generate",
                                           json=payload, timeout=60) as resp:
                        if resp.status == 200:
                            result = await resp.json()
                            return ChatResponse(
                                response=f"🎬 Generated anime video: {result.get('output_path', 'Video created successfully!')}",
                                conversation_id=conversation_id,
                                model_used="anime_generator"
                            )
                        else:
                            return ChatResponse(
                                response="❌ Anime generation service unavailable. The Tower Anime Production service is not responding.",
                                conversation_id=conversation_id,
                                model_used="error_handler"
                            )
            except Exception as e:
                logger.error(f"Anime generation failed: {e}")
                return ChatResponse(
                    response=f"❌ Anime generation failed: {str(e)}",
                    conversation_id=conversation_id,
                    model_used="error_handler"
                )

        # Get user context for preferences and styles
        user_manager = await get_user_context_manager()
        username = request.user_id or "anonymous"
        user_context = await user_manager.get_or_create_context(username)

        # Get conversation history from conversation manager
        conversation_context = await conversation_manager.get_conversation_context(conversation_id)
        conversation_history = conversation_context.get("history", [])

        # Get context from vector store if requested
        context = []
        if request.include_context and qdrant_client:
            try:
                search_results = await qdrant_client.search(
                    collection="echo_memory",
                    query=request.query,
                    limit=5
                )
                context = [{"text": r.payload.get("text", ""), "score": r.score} for r in search_results]
            except Exception as e:
                logger.warning(f"Context search failed: {e}")

        # Prepare context-enhanced prompt with user preferences, conversation history, and context
        enhanced_prompt = ""

        # Add user context (preferences and styles)
        if user_context:
            user_info = f"""You are talking to {username}.
Their preferences:
- Response style: {user_context.preferences.get('response_style', 'balanced')}
- Communication style: {user_context.communication_style.get('technical_level', 'advanced')} technical level
- Verbosity: {user_context.communication_style.get('verbosity', 'normal')}
- Personality preference: {user_context.preferences.get('personality', 'professional')}

"""
            # Add personal facts if any exist
            if user_context.personal_facts:
                facts = []
                for key, fact_data in list(user_context.personal_facts.items())[:5]:
                    facts.append(f"- {key}: {fact_data['value']}")
                if facts:
                    user_info += "Known facts about them:\n" + "\n".join(facts) + "\n\n"

            enhanced_prompt = user_info

        # Add conversation history (from conversation manager format)
        if conversation_history:
            history_text = "\n".join([
                f"User: {msg['query_text']}\nAssistant: {msg['response']}"
                for msg in conversation_history[-10:]
            ])  # Last 10 messages
            enhanced_prompt += f"Previous conversation:\n{history_text}\n\n"

        # Add vector context
        if context:
            context_text = "\n".join([f"- {c['text'][:200]}" for c in context[:3]])
            enhanced_prompt += f"Relevant context:\n{context_text}\n\n"

        enhanced_prompt += f"User: {request.query}"

        # Try to get response from Ollama
        response_text = await generate_ollama_response(enhanced_prompt, request.model, request.temperature)

        # If Ollama fails, use fallback
        if not response_text:
            response_text = await generate_fallback_response(request.query, context)

        # Update conversation with response
        conversation_manager.update_conversation(
            conversation_id,
            request.query,
            "chat",  # intent
            response_text,
            False  # requires_clarification
        )

        # Also update user context with conversation
        await user_manager.add_conversation(username, "user", request.query)
        await user_manager.add_conversation(username, "assistant", response_text)

        return ChatResponse(
            response=response_text,
            conversation_id=conversation_id,
            context_used=context[:3] if context else None,
            model_used=request.model
        )

    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def generate_ollama_response(prompt: str, model: str = "llama3.1:8b", temperature: float = 0.7) -> Optional[str]:
    """Generate response using Ollama"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "temperature": temperature,
                    "stream": False
                }
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("response", "")
    except Exception as e:
        logger.warning(f"Ollama generation failed: {e}")

    return None

async def generate_fallback_response(query: str, context: List[Dict]) -> str:
    """Generate fallback response when AI is unavailable"""
    query_lower = query.lower()

    # Check for common queries
    if "hello" in query_lower or "hi" in query_lower:
        return "Hello! I'm Echo Brain. How can I assist you today?"

    if "help" in query_lower:
        return """I can help you with:
• Anime production and scene planning
• System diagnostics and health checks
• Context search from memory
• General conversation and assistance

What would you like to know more about?"""

    if "status" in query_lower or "health" in query_lower:
        return "System is operational. Use /api/diagnostics/full for detailed status."

    # Use context if available
    if context:
        return f"Based on my memory, here's what I found relevant:\n\n{context[0]['text'][:300]}..."

    return "I understand your query but my AI models are currently unavailable. Please try again later or check system status."

# ============= Context Search Endpoint =============

@app.post("/api/context", response_model=ContextSearchResponse)
async def search_context(request: ContextSearchRequest):
    """Search vector memory for relevant context"""
    try:
        if not qdrant_client:
            # Fallback to direct Qdrant API
            async with httpx.AsyncClient(timeout=10.0) as client:
                # First, get embedding from Ollama
                embed_response = await client.post(
                    f"{OLLAMA_URL}/api/embeddings",
                    json={"model": "nomic-embed-text", "prompt": request.query}
                )

                if embed_response.status_code != 200:
                    raise HTTPException(status_code=503, detail="Embedding service unavailable")

                embedding = embed_response.json()["embedding"]

                # Search Qdrant
                search_response = await client.post(
                    f"{QDRANT_URL}/collections/{request.collection}/points/search",
                    json={
                        "vector": embedding,
                        "limit": request.limit,
                        "with_payload": True
                    }
                )

                if search_response.status_code == 200:
                    results = []
                    for point in search_response.json()["result"]:
                        results.append(ContextSearchResult(
                            text=point["payload"].get("text", ""),
                            score=point["score"],
                            metadata=point["payload"],
                            collection=request.collection
                        ))

                    return ContextSearchResponse(
                        results=results,
                        query=request.query,
                        total=len(results)
                    )

        # Use integrated client if available
        if not qdrant_client:
            raise HTTPException(status_code=503, detail="Qdrant client not available")

        # Get embedding first
        async with httpx.AsyncClient(timeout=10.0) as client:
            embed_response = await client.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": "nomic-embed-text", "prompt": request.query}
            )
            if embed_response.status_code != 200:
                raise HTTPException(status_code=503, detail="Embedding service unavailable")
            embedding = embed_response.json()["embedding"]

        # Search with qdrant_client
        search_results = await qdrant_client.search_similar(
            collection_name=request.collection,
            query_vector=embedding,
            limit=request.limit
        )

        results = []
        for result in search_results:
            results.append(ContextSearchResult(
                text=result.get("payload", {}).get("text", ""),
                score=result.get("score", 0.0),
                metadata=result.get("payload", {}),
                collection=request.collection
            ))

        return ContextSearchResponse(
            results=results,
            query=request.query,
            total=len(results)
        )

    except Exception as e:
        logger.error(f"Context search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============= Agent Endpoint =============

@app.post("/api/agent", response_model=AgentResponse)
async def execute_agent_task(request: AgentRequest):
    """Execute task using agent system"""
    try:
        if not agent_manager:
            return AgentResponse(
                result="Agent system not available",
                agent_used="none",
                task_id=str(uuid.uuid4()),
                status="failed"
            )

        # Execute task
        result = await agent_manager.execute_task(
            task=request.task,
            agent_type=request.agent_type,
            context=request.context
        )

        return AgentResponse(
            result=result.get("output", "No output"),
            agent_used=result.get("agent", request.agent_type),
            task_id=result.get("task_id", str(uuid.uuid4())),
            status=result.get("status", "completed")
        )

    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        return AgentResponse(
            result=str(e),
            agent_used=request.agent_type,
            task_id=str(uuid.uuid4()),
            status="error"
        )

# ============= Autonomous Status =============

if autonomous_core:
    @app.get("/api/autonomous/status")
    async def autonomous_status():
        """Get autonomous system status"""
        try:
            status = await autonomous_core.get_status()
            return status
        except Exception as e:
            return {"status": "error", "message": str(e)}

# ============= Unified Search =============

@app.post("/api/context/unified")
async def unified_search(request: Dict[str, Any]):
    """Unified search across all collections"""
    try:
        query = request.get("query", "")
        collections = request.get("collections", ["echo_memory", "claude_conversations"])
        limit_per = request.get("limit_per_collection", 5)

        all_results = []
        for collection in collections:
            try:
                search_req = ContextSearchRequest(
                    query=query,
                    limit=limit_per,
                    collection=collection
                )
                response = await search_context(search_req)
                all_results.extend(response.results)
            except:
                continue

        # Sort by score
        all_results.sort(key=lambda x: x.score, reverse=True)

        return {
            "query": query,
            "results": all_results[:request.get("total_limit", 20)],
            "collections_searched": collections,
            "total_found": len(all_results)
        }

    except Exception as e:
        logger.error(f"Unified search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============= Collections Info =============

@app.get("/api/collections")
async def get_collections():
    """Get information about vector collections"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{QDRANT_URL}/collections")
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        logger.error(f"Failed to get collections: {e}")

    return {"collections": [], "error": "Could not retrieve collections"}

# ============= Agent Status =============

@app.get("/api/agents/status")
async def get_agents_status():
    """Get status of all agents"""
    if agent_manager:
        return await agent_manager.get_all_agents_status()
    return {"agents": [], "message": "Agent system not available"}

# ============= Startup Event =============

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("=" * 60)
    logger.info("Echo Brain Unified Service Starting")
    logger.info("=" * 60)

    # Autonomous core and agent manager are initialized in their constructors
    if autonomous_core:
        logger.info("✅ Autonomous core ready")
    if agent_manager:
        logger.info("✅ Agent manager ready")

    # Initialize capability coordinator for system actions
    try:
        from .capabilities.capability_registry import CapabilityRegistry, CapabilityType
        from .capabilities.echo_capability_coordinator import initialize_coordinator
        import subprocess
        import asyncio

        capability_registry = CapabilityRegistry()

        # Register service monitoring capability
        async def check_services_status(**kwargs):
            """Check Tower services using systemctl"""
            try:
                # Check key Tower services
                services = ['tower-echo-brain', 'ollama', 'nginx', 'postgresql']
                healthy_services = []
                failed_services = []

                for service in services:
                    try:
                        result = subprocess.run(['systemctl', 'is-active', service],
                                              capture_output=True, text=True)
                        if result.stdout.strip() == 'active':
                            healthy_services.append(service)
                        else:
                            failed_services.append(service)
                    except:
                        failed_services.append(service)

                total = len(services)
                healthy = len(healthy_services)

                # Check Ollama specifically
                ollama_models = []
                try:
                    result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
                    if result.returncode == 0:
                        ollama_models = result.stdout.split('\n')[1:]  # Skip header
                        ollama_models = [line.split()[0] for line in ollama_models if line.strip()]
                except:
                    pass

                return {
                    "capability": "service_monitoring",
                    "success": True,
                    "summary": {
                        "total": total,
                        "healthy": healthy,
                        "health_rate": (healthy/total)*100,
                        "healthy_services": healthy_services,
                        "failed_services": failed_services,
                        "ollama_models": len(ollama_models),
                        "ollama_model_list": ollama_models[:5]  # First 5 models
                    }
                }
            except Exception as e:
                return {
                    "capability": "service_monitoring",
                    "success": False,
                    "error": str(e)
                }

        capability_registry.register_capability(
            name="service_monitoring",
            capability_type=CapabilityType.ANALYSIS,
            description="Monitor Tower service health",
            handler=check_services_status,
            requirements=[],
            permissions=["system_read"]
        )

        # Register autonomous repair capability
        async def autonomous_repair(**kwargs):
            """System diagnosis and repair operations"""
            issue_type = kwargs.get('issue_type', 'diagnose')
            try:
                if issue_type == 'diagnose':
                    # Run basic system diagnosis
                    result = subprocess.run(['df', '-h'], capture_output=True, text=True)
                    disk_usage = result.stdout if result.returncode == 0 else "Could not check disk"

                    return {
                        "capability": "autonomous_repair",
                        "success": True,
                        "diagnosis": f"System diagnosis completed. Disk usage: {disk_usage[:100]}..."
                    }
                else:
                    return {
                        "capability": "autonomous_repair",
                        "success": False,
                        "error": "Unknown repair type"
                    }
            except Exception as e:
                return {
                    "capability": "autonomous_repair",
                    "success": False,
                    "error": str(e)
                }

        capability_registry.register_capability(
            name="autonomous_repair",
            capability_type=CapabilityType.ANALYSIS,
            description="System diagnosis and repair",
            handler=autonomous_repair,
            requirements=[],
            permissions=["system_write"]
        )

        # Register model management capability
        async def model_management(**kwargs):
            """Manage AI models including Ollama"""
            try:
                # Check Ollama models
                result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
                if result.returncode == 0:
                    models = result.stdout.split('\n')[1:]  # Skip header
                    model_list = []
                    for line in models:
                        if line.strip():
                            parts = line.split()
                            if parts:
                                model_list.append(parts[0])

                    # For update requests, we would run ollama pull here
                    # But for safety, we'll just report current status
                    return {
                        "capability": "model_management",
                        "success": True,
                        "current_models": model_list,
                        "total_models": len(model_list),
                        "message": f"Found {len(model_list)} Ollama models. To update, run: ollama pull <model_name>"
                    }
                else:
                    return {
                        "capability": "model_management",
                        "success": False,
                        "error": "Could not access Ollama"
                    }
            except Exception as e:
                return {
                    "capability": "model_management",
                    "success": False,
                    "error": str(e)
                }

        capability_registry.register_capability(
            name="model_management",
            capability_type=CapabilityType.ANALYSIS,
            description="Manage AI models and Ollama",
            handler=model_management,
            requirements=[],
            permissions=["system_read"]
        )

        capability_coordinator = initialize_coordinator(capability_registry)
        logger.info("✅ Echo Capability Coordinator initialized with system monitoring")
    except Exception as e:
        logger.warning(f"Could not initialize capability coordinator: {e}")

    logger.info("Available endpoints:")
    logger.info("  - /health - System health check")
    logger.info("  - /api/echo/chat - Chat interface (Telegram compatible)")
    logger.info("  - /api/context - Vector search")
    logger.info("  - /api/echo/anime/* - Anime AI Director")
    logger.info("  - /api/diagnostics/* - System diagnostics")
    logger.info("  - /api/agent - Agent execution")
    logger.info("  - /api/status - Detailed status")
    logger.info("=" * 60)

# ============= Run Server =============

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8309,
        reload=False,
        log_level="info"
    )