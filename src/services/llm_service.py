"""Async LLM service for Ollama with GPU Arbiter awareness.

ALL generative models run on CPU (num_gpu=0) to avoid VRAM contention
on the shared AMD RX 9070 XT. Only nomic-embed-text uses GPU (350MB, pinned).

Ollama loads models to the AMD GPU by default, which starves ComfyUI-ROCm
of VRAM for video generation. Even "light" 7B models consume 5-10GB on GPU.
CPU inference on 96GB RAM is fast enough for chat/reasoning tasks.

Heavy models (gemma3:12b vision) still require arbiter claim + GPU access
when vision capabilities are needed (image analysis).
"""

import logging
import os
import aiohttp
from typing import Optional, AsyncGenerator
from dataclasses import dataclass
from functools import lru_cache
import json

logger = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Models that need GPU (vision capabilities) — arbiter check required
HEAVY_MODELS = {"gemma3:12b", "deepseek-coder-v2:16b"}

# Models that should ALWAYS run on CPU to preserve AMD VRAM for ComfyUI-ROCm
# All generative models except when explicitly using vision features
CPU_ONLY_MODELS = {
    "mistral:7b", "deepseek-r1:8b", "qwen2.5-coder:7b",
    "gemma3:12b", "deepseek-coder-v2:16b",
}


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

    async def _check_arbiter(self, model: str) -> bool:
        """Check GPU Arbiter before heavy model calls. Returns True if OK to proceed."""
        if model not in HEAVY_MODELS:
            return True
        try:
            from services.gpu_arbiter_client import arbiter
            return await arbiter.can_use_heavy_model()
        except Exception:
            return True  # Fail open

    async def generate(
        self,
        prompt: str,
        model: str = "mistral:7b",
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        skip_arbiter: bool = False,
        use_gpu: bool = False,
    ) -> LLMResponse:
        """Generate a completion. Forces CPU by default to preserve AMD VRAM.

        Args:
            use_gpu: If True, allows GPU inference (for vision tasks). Must pass
                     arbiter check for heavy models.
        """
        if not skip_arbiter and not await self._check_arbiter(model):
            raise RuntimeError(
                f"GPU busy — {model} deferred by arbiter (ComfyUI-ROCm or vision batch active)"
            )

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            "keep_alive": "5m",
        }
        # Force CPU inference unless explicitly using GPU (vision tasks)
        if not use_gpu and model in CPU_ONLY_MODELS:
            payload["options"]["num_gpu"] = 0
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
                    tokens_per_second=data.get("eval_count", 0) / max(data.get("eval_duration", 1) / 1_000_000_000, 0.001),
                )

    async def generate_stream(
        self,
        prompt: str,
        model: str = "mistral:7b",
        system: Optional[str] = None,
        temperature: float = 0.7,
        use_gpu: bool = False,
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming completion. Forces CPU by default."""
        if not await self._check_arbiter(model):
            raise RuntimeError(
                f"GPU busy — {model} deferred by arbiter"
            )

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": temperature},
            "keep_alive": "5m",
        }
        # Force CPU inference unless explicitly using GPU
        if not use_gpu and model in CPU_ONLY_MODELS:
            payload["options"]["num_gpu"] = 0
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
        model: str = "mistral:7b",
        temperature: float = 0.7,
        skip_arbiter: bool = False,
        use_gpu: bool = False,
    ) -> LLMResponse:
        """Chat completion with message history. Forces CPU by default."""
        if not skip_arbiter and not await self._check_arbiter(model):
            raise RuntimeError(
                f"GPU busy — {model} deferred by arbiter"
            )

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
            "keep_alive": "5m",
        }
        # Force CPU inference unless explicitly using GPU (vision tasks)
        if not use_gpu and model in CPU_ONLY_MODELS:
            payload["options"]["num_gpu"] = 0

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
                    tokens_per_second=data.get("eval_count", 0) / max(data.get("eval_duration", 1) / 1_000_000_000, 0.001),
                )


@lru_cache()
def get_llm_service() -> LLMService:
    return LLMService()
