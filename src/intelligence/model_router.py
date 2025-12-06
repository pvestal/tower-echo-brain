#!/usr/bin/env python3
"""
Model Router for Echo Brain - Routes tasks to appropriate Tower LLMs.
Finally gives Echo the ability to use specialized local models instead of
defaulting to a single endpoint.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, List
import httpx
import asyncio
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class TaskCategory(Enum):
    """Categories of tasks that need different model capabilities."""
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    CODE_REFACTOR = "code_refactor"
    CODE_ANALYSIS = "code_analysis"
    DEBUGGING = "debugging"
    ARCHITECTURE = "architecture"
    REASONING = "reasoning"
    CREATIVE = "creative"
    CONVERSATION = "conversation"
    SUMMARIZATION = "summarization"
    SYSTEM_COMMANDS = "system_commands"
    DATA_ANALYSIS = "data_analysis"

@dataclass
class ModelCapability:
    """Detailed capability profile for a model."""
    name: str
    model_id: str
    base_url: str
    api_type: str  # "ollama", "vllm", "openai", "anthropic"
    context_window: int
    parameters: int  # Size in billions
    specialties: List[TaskCategory]
    performance_scores: Dict[TaskCategory, float]  # 0-1 score per category
    latency_ms: int  # Average response time
    gpu_memory_gb: float  # VRAM required
    is_available: bool = True
    last_health_check: Optional[float] = None

class TowerModelRegistry:
    """
    Registry of all available models on Tower infrastructure.
    Knows what's running, what's good at what, and routes accordingly.
    """

    def __init__(self):
        self.models = self._initialize_tower_models()
        self.health_check_interval = 60  # seconds
        self.last_health_check = {}

    def _initialize_tower_models(self) -> Dict[str, ModelCapability]:
        """Initialize with Tower's known model configuration."""
        return {
            "qwen-coder-7b": ModelCapability(
                name="Qwen 2.5 Coder 7B",
                model_id="qwen2.5-coder:7b",
                base_url="http://localhost:11434",
                api_type="ollama",
                context_window=32768,
                parameters=7,
                specialties=[
                    TaskCategory.CODE_GENERATION,
                    TaskCategory.CODE_REFACTOR,
                    TaskCategory.CODE_REVIEW,
                    TaskCategory.DEBUGGING
                ],
                performance_scores={
                    TaskCategory.CODE_GENERATION: 0.95,
                    TaskCategory.CODE_REFACTOR: 0.92,
                    TaskCategory.CODE_REVIEW: 0.88,
                    TaskCategory.DEBUGGING: 0.90,
                    TaskCategory.REASONING: 0.75,
                    TaskCategory.CONVERSATION: 0.70
                },
                latency_ms=1500,
                gpu_memory_gb=4.5
            ),
            "deepseek-coder": ModelCapability(
                name="DeepSeek Coder",
                model_id="deepseek-coder:latest",
                base_url="http://localhost:11434",
                api_type="ollama",
                context_window=16384,
                parameters=6.7,
                specialties=[
                    TaskCategory.CODE_GENERATION,
                    TaskCategory.CODE_ANALYSIS
                ],
                performance_scores={
                    TaskCategory.CODE_GENERATION: 0.85,
                    TaskCategory.CODE_ANALYSIS: 0.82,
                    TaskCategory.DEBUGGING: 0.78,
                    TaskCategory.REASONING: 0.70
                },
                latency_ms=800,
                gpu_memory_gb=8.0
            ),
            "llama3.2": ModelCapability(
                name="Llama 3.2",
                model_id="llama3.2:latest",
                base_url="http://localhost:11434",
                api_type="ollama",
                context_window=131072,
                parameters=3,
                specialties=[
                    TaskCategory.CONVERSATION,
                    TaskCategory.SUMMARIZATION
                ],
                performance_scores={
                    TaskCategory.CONVERSATION: 0.85,
                    TaskCategory.SUMMARIZATION: 0.80,
                    TaskCategory.REASONING: 0.75,
                    TaskCategory.CREATIVE: 0.70
                },
                latency_ms=500,
                gpu_memory_gb=4.0
            ),
            "mixtral": ModelCapability(
                name="Mixtral 8x7B",
                model_id="mixtral:8x7b",
                base_url="http://localhost:11434",
                api_type="ollama",
                context_window=32768,
                parameters=47,
                specialties=[
                    TaskCategory.REASONING,
                    TaskCategory.ARCHITECTURE,
                    TaskCategory.DATA_ANALYSIS
                ],
                performance_scores={
                    TaskCategory.REASONING: 0.90,
                    TaskCategory.ARCHITECTURE: 0.85,
                    TaskCategory.DATA_ANALYSIS: 0.82,
                    TaskCategory.CODE_REVIEW: 0.75
                },
                latency_ms=3000,
                gpu_memory_gb=26.0
            ),
            "tinyllama": ModelCapability(
                name="TinyLlama",
                model_id="tinyllama:latest",
                base_url="http://localhost:11434",
                api_type="ollama",
                context_window=2048,
                parameters=1.1,
                specialties=[
                    TaskCategory.SYSTEM_COMMANDS
                ],
                performance_scores={
                    TaskCategory.SYSTEM_COMMANDS: 0.70,
                    TaskCategory.CONVERSATION: 0.60
                },
                latency_ms=100,
                gpu_memory_gb=1.0
            )
        }

    async def check_model_health(self, model_id: str) -> bool:
        """Check if a model is actually responding."""
        model = self.models.get(model_id)
        if not model:
            return False

        try:
            # Use longer timeout for larger models
            timeout = 30.0 if "7b" in model_id or "32b" in model_id else 10.0
            async with httpx.AsyncClient(timeout=timeout) as client:
                # Ollama health check
                if model.api_type == "ollama":
                    response = await client.post(
                        f"{model.base_url}/api/generate",
                        json={
                            "model": model.model_id,
                            "prompt": "Hi",
                            "stream": False,
                            "options": {"num_predict": 1}  # Minimal response
                        }
                    )
                    model.is_available = response.status_code == 200
                    import time
                    model.last_health_check = time.time()
                    return model.is_available
        except Exception as e:
            logger.warning(f"Health check failed for {model_id}: {e}")
            model.is_available = False
            return False

    def select_best_model(
        self,
        task_category: TaskCategory,
        context_size_needed: int = 1000,
        max_latency_ms: int = 5000
    ) -> Optional[ModelCapability]:
        """
        Select the best available model for a task.

        Selection criteria:
        1. Model must be available
        2. Model must fit context size
        3. Model must meet latency requirement
        4. Prefer models with higher performance scores for the task
        5. Prefer local models over API models
        """
        candidates = []

        for model in self.models.values():
            # Skip unavailable models
            if not model.is_available:
                continue

            # Skip if context too large
            if context_size_needed > model.context_window:
                continue

            # Skip if too slow
            if model.latency_ms > max_latency_ms:
                continue

            # Get performance score for this task
            score = model.performance_scores.get(task_category, 0.0)

            # Boost score if it's a specialty
            if task_category in model.specialties:
                score *= 1.2

            candidates.append((model, score))

        if not candidates:
            logger.warning(f"No suitable model found for {task_category}")
            return None

        # Sort by score (highest first)
        candidates.sort(key=lambda x: x[1], reverse=True)

        selected = candidates[0][0]
        logger.info(f"Selected {selected.name} for {task_category.value} (score: {candidates[0][1]:.2f})")

        return selected

    def get_model_for_code_task(self) -> Optional[ModelCapability]:
        """Quick helper to get the best code model."""
        return self.select_best_model(TaskCategory.CODE_GENERATION)

    def get_model_for_reasoning(self) -> Optional[ModelCapability]:
        """Quick helper to get the best reasoning model."""
        return self.select_best_model(TaskCategory.REASONING)

    def get_fastest_model(self) -> Optional[ModelCapability]:
        """Get the fastest responding model for simple tasks."""
        available = [m for m in self.models.values() if m.is_available]
        if not available:
            return None
        return min(available, key=lambda m: m.latency_ms)


class UnifiedModelClient:
    """
    Unified interface to call any model regardless of its API type.
    This is what Echo Brain will use to actually execute LLM calls.
    """

    def __init__(self, registry: TowerModelRegistry):
        self.registry = registry
        self.timeout = 120.0
        self.retry_attempts = 2

    async def complete(
        self,
        prompt: str,
        task_category: TaskCategory = TaskCategory.CONVERSATION,
        system: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.7,
        model_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get completion from the best model for the task.

        Returns both the response and metadata about what model was used.
        """
        # Select model
        if model_override:
            model = self.registry.models.get(model_override)
        else:
            model = self.registry.select_best_model(
                task_category,
                context_size_needed=len(prompt) // 4  # Rough token estimate
            )

        if not model:
            raise ValueError(f"No suitable model available for {task_category.value}")

        # Route to appropriate handler
        start_time = asyncio.get_event_loop().time()

        try:
            if model.api_type == "ollama":
                response = await self._call_ollama(
                    model, prompt, system, max_tokens, temperature
                )
            else:
                raise NotImplementedError(f"API type {model.api_type} not implemented")

            latency = int((asyncio.get_event_loop().time() - start_time) * 1000)

            return {
                "response": response,
                "model_used": model.name,
                "model_id": model.model_id,
                "task_category": task_category.value,
                "latency_ms": latency,
                "context_used": len(prompt) // 4,
                "success": True
            }

        except Exception as e:
            logger.error(f"Model call failed: {e}")
            return {
                "response": None,
                "error": str(e),
                "model_attempted": model.name,
                "task_category": task_category.value,
                "success": False
            }

    async def _call_ollama(
        self,
        model: ModelCapability,
        prompt: str,
        system: str,
        max_tokens: int,
        temperature: float
    ) -> str:
        """Call Ollama API with proper error handling."""
        messages = []
        if system:
            full_prompt = f"{system}\n\n{prompt}"
        else:
            full_prompt = prompt

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{model.base_url}/api/generate",
                json={
                    "model": model.model_id,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                }
            )
            response.raise_for_status()
            return response.json()["response"]

    async def code_generation(self, prompt: str, language: str = "python") -> str:
        """Specialized method for code generation tasks."""
        system = f"You are an expert {language} developer. Generate clean, efficient, well-commented code."

        result = await self.complete(
            prompt=prompt,
            task_category=TaskCategory.CODE_GENERATION,
            system=system,
            temperature=0.3,  # Lower temperature for more deterministic code
            max_tokens=4096
        )

        return result["response"] if result["success"] else None

    async def code_review(self, code: str) -> str:
        """Specialized method for code review tasks."""
        prompt = f"Review this code for bugs, security issues, and improvements:\n\n{code}"

        result = await self.complete(
            prompt=prompt,
            task_category=TaskCategory.CODE_REVIEW,
            system="You are a senior software engineer conducting a thorough code review.",
            temperature=0.5,
            max_tokens=2048
        )

        return result["response"] if result["success"] else None


async def test_model_router():
    """Test the model router with Tower's services."""

    # Initialize registry
    registry = TowerModelRegistry()
    client = UnifiedModelClient(registry)

    # Test 1: Check model health with proper timeout
    print("Testing model availability (this may take a moment)...")
    for model_id in ["qwen-coder-7b", "deepseek-coder", "tinyllama"]:
        available = await registry.check_model_health(model_id)
        print(f"  {model_id}: {'✅ Available' if available else '❌ Unavailable'}")

    # Test 2: Code generation (should route to Qwen-Coder)
    print("\nTesting code generation routing...")
    result = await client.complete(
        prompt="Write a Python function to calculate fibonacci",
        task_category=TaskCategory.CODE_GENERATION
    )

    if result["success"]:
        print(f"  Routed to: {result['model_used']}")
        print(f"  Latency: {result['latency_ms']}ms")
        print(f"  Response preview: {result['response'][:100]}...")

    # Test 3: Simple task (should route to TinyLlama)
    print("\nTesting fast model selection...")
    fast_model = registry.get_fastest_model()
    if fast_model:
        print(f"  Fastest model: {fast_model.name} ({fast_model.latency_ms}ms)")

    print("\n✅ Model router test complete")


if __name__ == "__main__":
    asyncio.run(test_model_router())