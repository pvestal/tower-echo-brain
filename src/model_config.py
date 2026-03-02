"""
Model configuration — maps role names to Ollama model identifiers.

Used by echo_main_router and other components that need model names by role.
"""

import os

# Default model for all roles (override per-role via env vars)
_DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "mistral:7b")

_MODEL_MAP = {
    "general": os.getenv("OLLAMA_MODEL_GENERAL", _DEFAULT_MODEL),
    "reasoning": os.getenv("OLLAMA_MODEL_REASONING", _DEFAULT_MODEL),
    "analysis": os.getenv("OLLAMA_MODEL_ANALYSIS", _DEFAULT_MODEL),
    "extraction": os.getenv("OLLAMA_MODEL_EXTRACTION", os.getenv("EXTRACTION_MODEL", "gemma3:12b")),
    "coding": os.getenv("OLLAMA_MODEL_CODING", "deepseek-coder-v2:16b"),
    "embedding": os.getenv("OLLAMA_MODEL_EMBEDDING", "nomic-embed-text"),
}


def get_model(role: str = "general") -> str:
    """Return the Ollama model name for a given role."""
    return _MODEL_MAP.get(role, _DEFAULT_MODEL)
