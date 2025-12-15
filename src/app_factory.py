#!/usr/bin/env python3
"""
Application factory for Echo Brain
"""
import os
import logging
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

# Core routers
from src.api.routes import router as main_router
from src.api.legacy.feedback_routes import router as feedback_router
from src.api.legacy.learning_pipeline_routes import router as learning_pipeline_router

# Verified execution routes
try:
    from src.api.legacy.verified_execution_routes import router as verified_execution_router
    verified_execution_available = True
    print("✅ Verified execution router imported successfully")
except ImportError as e:
    verified_execution_available = False
    print(f"❌ Failed to import verified execution router: {e}")
    verified_execution_router = None
from src.api.legacy.system_metrics import router as system_metrics_router
from src.api.legacy.takeout_routes import router as takeout_router
from src.api.legacy.enhanced_system_metrics import router as enhanced_system_metrics_router
from src.api.legacy.gpu_monitor import router as gpu_monitor_router
from src.api.legacy.neural_metrics import router as neural_metrics_router
from src.api.legacy.learning_routes import router as learning_router
from src.api.legacy.autonomous_routes import router as autonomous_router
from src.api.legacy.coordination_routes import router as coordination_router
from src.api.legacy.integration_testing_routes import integration_router
from src.api.legacy.task_routes import router as task_router
from src.photo_comparison import router as photo_router
from src.api.legacy.improvement_metrics import router as improvement_router
from src.api.delegation_routes import router as delegation_router

# External integrations
from src.modules.agents.agent_development_endpoints import agent_dev_router
from src.misc.veteran_guardian_endpoints import veteran_router
from src.misc.telegram_general_chat import general_telegram_router
from src.misc.telegram_integration import telegram_router

# Enhanced Telegram Executor
try:
    from src.integrations.telegram_echo_executor import telegram_executor_router
    telegram_executor_available = True
    print("✅ Telegram executor router imported successfully")
except ImportError as e:
    telegram_executor_available = False
    print(f"❌ Failed to import telegram executor: {e}")
    telegram_executor_router = None

# Enhanced Telegram Image Handler
try:
    from src.misc.telegram_image_handler import enhanced_telegram_router
    telegram_image_available = True
    print("✅ Telegram image handler imported successfully")
except ImportError as e:
    telegram_image_available = False
    print(f"❌ Failed to import telegram image handler: {e}")
    enhanced_telegram_router = None

# Resilient model management
try:
    from src.managers.echo_integration import router as resilient_router
    resilient_available = True
    print("✅ Resilient model router imported successfully")
except ImportError as e:
    resilient_available = False
    print(f"❌ Failed to import resilient router: {e}")
    resilient_router = None

# Conversation memory management
try:
    from src.api.legacy.conversation_memory_routes import router as memory_router
    memory_available = True
    print("✅ Conversation memory router imported successfully")
except ImportError as e:
    memory_available = False
    print(f"❌ Failed to import conversation memory router: {e}")
    memory_router = None

# Anime semantic search
try:
    from src.api.legacy.anime_search import router as anime_search_router
    anime_search_available = True
    print("✅ Anime semantic search router imported successfully")
except ImportError as e:
    anime_search_available = False

# Media search endpoints
try:
    from src.api.legacy.media_search import router as media_search_router
    media_search_available = True
    print("✅ Media search router imported successfully")
except ImportError as e:
    media_search_available = False
    print(f"❌ Failed to import media search router: {e}")
    media_search_router = None
    print(f"❌ Failed to import anime search router: {e}")
    anime_search_router = None

# Anime character integration
try:
    from src.api.legacy.anime_integration import router as anime_integration_router
    anime_integration_available = True
    print("✅ Anime character integration router imported successfully")
except ImportError as e:
    anime_integration_available = False
    print(f"❌ Failed to import anime integration router: {e}")
    anime_integration_router = None

# Semantic integration for intelligent creative orchestration
try:
    from src.api.legacy.semantic_integration_routes import router as semantic_integration_router
    semantic_integration_available = True
    print("✅ Semantic integration router imported successfully")
except ImportError as e:
    semantic_integration_available = False
    print(f"❌ Failed to import semantic integration router: {e}")
    semantic_integration_router = None

def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    # Load environment variables
    load_dotenv()

    # Create FastAPI app
    app = FastAPI(
        title="Echo Brain",
        description="Advanced AI orchestrator with modular architecture",
        version="2.0.0"
    )

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Add middleware for user context and permissions
    try:
        from src.middleware.user_context_middleware import UserContextMiddleware, PermissionMiddleware
        app.add_middleware(UserContextMiddleware)
        app.add_middleware(PermissionMiddleware)
        logging.info("✅ User context and permission middleware added")
    except ImportError as e:
        logging.warning(f"⚠️ Could not load user context middleware: {e}")

    # Include routers
    print(f"🔍 Including main_router with routes: {[r.path for r in main_router.routes if hasattr(r, 'path')]}")
    app.include_router(main_router, prefix="", tags=["main"])
    app.include_router(feedback_router, prefix="", tags=["feedback"])
    app.include_router(learning_pipeline_router, prefix="", tags=["learning-pipeline"])
    if verified_execution_available and verified_execution_router:
        app.include_router(verified_execution_router, prefix="", tags=["verified-execution"])
        print("✅ Verified execution routes added to app")
    app.include_router(system_metrics_router, prefix="", tags=["metrics"])
    app.include_router(takeout_router, prefix="", tags=["takeout"])
    print("✅ Takeout processing routes added to app")
    app.include_router(enhanced_system_metrics_router, prefix="", tags=["enhanced-metrics"])
    app.include_router(gpu_monitor_router, prefix="", tags=["gpu-monitoring"])
    app.include_router(neural_metrics_router, prefix="", tags=["neural"])
    app.include_router(learning_router, prefix="", tags=["learning"])
    app.include_router(autonomous_router, prefix="", tags=["autonomous"])
    app.include_router(coordination_router, prefix="", tags=["coordination"])
    app.include_router(integration_router, prefix="", tags=["testing"])
    app.include_router(task_router, prefix="", tags=["tasks"])
    app.include_router(photo_router, prefix="", tags=["vision"])
    app.include_router(improvement_router, prefix="/api/echo", tags=["improvement"])
    app.include_router(delegation_router, prefix="/api/echo", tags=["delegation"])

    # Training status router
    try:
        from src.api.training_status import router as training_router
        app.include_router(training_router, prefix="", tags=["training"])
        print("✅ Training status routes added to app")
    except ImportError as e:
        print(f"❌ Failed to import training status router: {e}")

    # External routers
    app.include_router(agent_dev_router, prefix="", tags=["agents"])
    app.include_router(veteran_router, prefix="", tags=["veteran"])
    app.include_router(general_telegram_router, prefix="", tags=["telegram"])
    app.include_router(telegram_router, prefix="", tags=["telegram"])

    # Enhanced Telegram Executor
    if telegram_executor_available and telegram_executor_router:
        app.include_router(telegram_executor_router, prefix="", tags=["telegram-executor"])
        print("✅ Telegram executor routes added to app")

    # Enhanced Telegram Image Handler
    if telegram_image_available and enhanced_telegram_router:
        app.include_router(enhanced_telegram_router, prefix="", tags=["telegram-images"])
        print("✅ Telegram image handler routes added to app")

    # Resilient model management
    if resilient_available and resilient_router:
        app.include_router(resilient_router, prefix="", tags=["resilient-models"])
        print("✅ Resilient model routes added to app")

    # Conversation memory management
    if memory_available and memory_router:
        app.include_router(memory_router, prefix="", tags=["conversation-memory"])
        print("✅ Conversation memory routes added to app")

    # Anime semantic search
    if anime_search_available and anime_search_router:
        app.include_router(anime_search_router, prefix="", tags=["anime-search"])
        print("✅ Anime semantic search routes added to app")

    # Media search endpoints
    if media_search_available and media_search_router:
        app.include_router(media_search_router, prefix="/api/echo", tags=["media-search"])
        print("✅ Media search routes added to app at /api/echo/search/*")

    # Anime character integration
    if anime_integration_available and anime_integration_router:
        app.include_router(anime_integration_router, prefix="", tags=["anime-integration"])
        print("✅ Anime character integration routes added to app")

    # Semantic integration for intelligent creative orchestration
    if semantic_integration_available and semantic_integration_router:
        app.include_router(semantic_integration_router, prefix="", tags=["semantic-integration"])
        print("✅ Semantic integration routes added to app")

    # Static files
    static_dir = "/opt/tower-echo-brain/static"
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    return app