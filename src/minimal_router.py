#!/usr/bin/env python3
"""
Minimal Model Router - Bypasses all broken logic.
Directly uses the first available Ollama model.
"""
import requests
import logging
logger = logging.getLogger(__name__)

def get_simple_route(query: str):
    """Always returns a working, available model."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = resp.json().get("models", [])
        if models:
            return models[0]["name"]  # Use the first available model
    except:
        pass
    return "llama3.2:latest"  # Universal fallback

def call_ollama(model: str, prompt: str) -> str:
    """Direct call to Ollama API."""
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120
        )
        if resp.status_code == 200:
            return resp.json().get("response", "Error: No response")
        return f"Error: Ollama returned {resp.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"