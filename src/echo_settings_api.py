# 🔴 DEPRECATED: Use unified_router.py instead
# This file is being phased out in favor of single source of truth
# Import from: from src.routing.unified_router import unified_router

#!/usr/bin/env python3
"""
Add settings API endpoint to Echo Brain for frontend control
Just like Claude has temperature, model selection, etc.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

router = APIRouter(prefix="/api/echo")

# Global settings storage (in production, use database)
echo_settings = {
    "mode": "auto",
    "model": None,
    "intelligence_level": 3,
    "speed_vs_quality": 50,
    "temperature": 0.7,
    "max_tokens": 2000,
    "top_p": 0.95,
    "options": {
        "use_context": True,
        "search_kb": True,
        "allow_fallback": True,
        "show_thinking": False,
        "stream_response": False
    }
}

class EchoSettings(BaseModel):
    mode: str  # "auto" or "manual"
    model: Optional[str] = None
    intelligence_level: int = 3  # 1-5 scale
    speed_vs_quality: int = 50  # 0-100 scale
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 0.95
    options: Dict[str, bool] = {}

@router.post("/settings")
async def update_settings(settings: EchoSettings):
    """Update Echo Brain settings from frontend controls"""
    global echo_settings

    echo_settings = settings.dict()

    # Apply settings to model selection logic
    if settings.mode == "manual" and settings.model:
        # Force specific model
        echo_settings["force_model"] = settings.model
    else:
        # Use intelligence level for auto selection
        model_map = {
            1: "tinyllama:latest",  # Quick
            2: "phi3:mini",  # Basic
            3: "llama3.2:3b",  # Standard
            4: "mistral:7b",  # Advanced
            5: "llama3.1:70b"  # Expert
        }
        echo_settings["preferred_model"] = model_map.get(settings.intelligence_level, "llama3.2:3b")

    return {"status": "success", "settings": echo_settings}

@router.get("/settings")
async def get_settings():
    """Get current Echo Brain settings"""
    return echo_settings

def select_model_with_settings(query: str, settings: dict = None) -> str:
    """
    Select model based on frontend settings, not keywords
    This is how professional AI systems work - UI controls, not magic words
    """
    if not settings:
        settings = echo_settings

    # Manual mode - use forced model
    if settings.get("mode") == "manual" and settings.get("force_model"):
        return settings["force_model"]

    # Auto mode - use intelligence level and query analysis
    intelligence_level = settings.get("intelligence_level", 3)
    speed_vs_quality = settings.get("speed_vs_quality", 50)

    # Analyze query complexity
    query_lower = query.lower()
    is_code = any(word in query_lower for word in ["code", "function", "implement", "debug", "fix"])
    is_simple = len(query.split()) < 5
    is_complex = len(query.split()) > 30 or "explain" in query_lower or "analyze" in query_lower

    # Model selection matrix (like Claude's model selector)
    if is_code:
        if speed_vs_quality > 70:  # Prefer quality
            return "qwen2.5-coder:32b"
        else:
            return "deepseek-coder-v2:16b"

    if is_simple and intelligence_level <= 2:
        return "tinyllama:latest"

    if is_complex or intelligence_level >= 4:
        if speed_vs_quality > 80:  # Maximum quality
            return "llama3.1:70b"
        elif intelligence_level >= 4:
            return "mixtral:8x7b"
        else:
            return "mistral:7b"

    # Default based on intelligence level
    default_models = {
        1: "tinyllama:latest",
        2: "phi3:mini",
        3: "llama3.2:3b",
        4: "mistral:7b",
        5: "llama3.1:70b"
    }

    return default_models.get(intelligence_level, "llama3.2:3b")

# Add this to the main Echo Brain service
def integrate_settings_api(app):
    """Integrate settings API into Echo Brain"""
    app.include_router(router)

    # Patch the chat endpoint to use settings
    @app.post("/api/echo/chat")
    async def chat_with_settings(query: str, user_id: str):
        """Chat endpoint that respects frontend settings"""

        # Get model based on settings, not keywords
        model = select_model_with_settings(query)

        # Get other settings
        temperature = echo_settings.get("temperature", 0.7)
        max_tokens = echo_settings.get("max_tokens", 2000)

        # Process with selected model and settings
        response = await process_with_model(
            query=query,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            settings=echo_settings
        )

        return response