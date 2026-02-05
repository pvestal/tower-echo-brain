"""
Async LLM service for Ollama
Replaces blocking requests.post() calls
"""
import os
import aiohttp
import asyncio
from typing import Optional, Dict, Any, AsyncGenerator
from dataclasses import dataclass
from functools import lru_cache
import json

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

@dataclass
class LLMResponse:
    content: str
    model: str
    total_duration_ms: float
    tokens_per_second: float

class LLMService:
    def __init__(self):
        self.base_url = OLLAMA_URL
        self.timeout = aiohttp.ClientTimeout(total=120)

    async def generate(
        self,
        prompt: str,
        model: str = "mistral:7b",
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> LLMResponse:
        """Generate a completion (non-streaming)"""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        if system:
            payload["system"] = system

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.post(f"{self.base_url}/api/generate", json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"Ollama error {resp.status}: {text}")

                data = await resp.json()

                return LLMResponse(
                    content=data.get("response", ""),
                    model=data.get("model", model),
                    total_duration_ms=data.get("total_duration", 0) / 1_000_000,
                    tokens_per_second=data.get("eval_count", 0) / (data.get("eval_duration", 1) / 1_000_000_000)
                )

    async def generate_stream(
        self,
        prompt: str,
        model: str = "mistral:7b",
        system: Optional[str] = None,
        temperature: float = 0.7
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming completion"""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature
            }
        }

        if system:
            payload["system"] = system

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.post(f"{self.base_url}/api/generate", json=payload) as resp:
                async for line in resp.content:
                    if line:
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]

    async def chat(
        self,
        messages: list,
        # model: str = "mistral:7b",
        model: str = "mistral:7b",
        temperature: float = 0.7
    ) -> LLMResponse:
        """Chat completion with message history"""
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.post(f"{self.base_url}/api/chat", json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"Ollama error {resp.status}: {text}")

                data = await resp.json()

                return LLMResponse(
                    content=data.get("message", {}).get("content", ""),
                    model=data.get("model", model),
                    total_duration_ms=data.get("total_duration", 0) / 1_000_000,
                    tokens_per_second=data.get("eval_count", 0) / (data.get("eval_duration", 1) / 1_000_000_000)
                )

@lru_cache()
def get_llm_service() -> LLMService:
    return LLMService()