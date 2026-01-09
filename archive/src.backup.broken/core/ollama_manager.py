#!/usr/bin/env python3
"""Fix Ollama timeout issues in Echo Brain."""

import asyncio
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class OllamaManager:
    """Manage Ollama operations with proper timeout handling."""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.base_url = "http://localhost:11434"

    async def list_models(self) -> Dict:
        """List available models with timeout."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.json()
        except asyncio.TimeoutError:
            logger.error(f"Model listing timed out after {self.timeout} seconds")
            return {"models": [], "error": "timeout"}
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return {"models": [], "error": str(e)}

    async def pull_model(self, model_name: str) -> Dict:
        """Pull a model with progress tracking."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=None) as client:
                # Use streaming for long operations
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/pull",
                    json={"name": model_name}
                ) as response:
                    progress = []
                    async for line in response.aiter_lines():
                        if line:
                            progress.append(line)
                            # Could emit progress updates here
                    return {"status": "success", "model": model_name}
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return {"status": "error", "error": str(e)}

    async def generate(self, model: str, prompt: str, timeout: int = 60) -> Dict:
        """Generate response with configurable timeout."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False
                    }
                )
                return response.json()
        except asyncio.TimeoutError:
            logger.error(f"Generation timed out after {timeout} seconds")
            return {"error": "timeout", "response": ""}
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {"error": str(e), "response": ""}

# Global instance
ollama_manager = OllamaManager()
