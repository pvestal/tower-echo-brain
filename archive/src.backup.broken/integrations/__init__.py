"""
Echo Brain Integrations
Provides connections to external services
"""

from src.integrations.ollama_client import get_ollama_client
from src.integrations.comfyui_client import get_comfyui_client
from src.integrations.email_client import get_email_client
from src.integrations.telegram_client import get_telegram_client

__all__ = [
    'get_ollama_client',
    'get_comfyui_client',
    'get_email_client',
    'get_telegram_client'
]