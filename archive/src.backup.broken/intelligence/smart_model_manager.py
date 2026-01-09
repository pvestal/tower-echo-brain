#!/usr/bin/env python3
"""
Smart Model Manager for Echo Brain - Production-ready model lifecycle management.
Handles the reality that a loaded inferior model beats an unloaded superior one.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from enum import Enum
import asyncio
import httpx
import json
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelState(Enum):
    UNLOADED = "unloaded"      # Not in memory
    LOADING = "loading"         # Currently loading
    LOADED = "loaded"           # Ready for inference
    BUSY = "busy"               # Processing a request
    ERROR = "error"             # Failed health check

class TaskUrgency(Enum):
    INTERACTIVE = "interactive"  # User waiting, need response in <5s
    BACKGROUND = "background"    # Can wait 30+ seconds
    BATCH = "batch"              # Minutes acceptable
    CRITICAL = "critical"        # Emergency task, use anything available

@dataclass
class ModelProfile:
    """Complete profile of a model's characteristics."""
    name: str
    model_id: str
    vram_mb: int                         # Memory requirement when loaded
    load_time_seconds: float             # Cold start time
    avg_inference_ms: float              # Average response time when loaded
    max_context_tokens: int              # Context window size
    capabilities: Dict[str, float]       # Task type -> quality score (0-1)
    preferred_gpu: int = 0               # Which GPU to load on (0=AMD, 1=NVIDIA for Tower)

@dataclass
class ModelStatus:
    """Real-time status of a model."""
    model_id: str
    state: ModelState
    last_used: Optional[datetime] = None
    current_vram_mb: int = 0
    requests_processed: int = 0
    avg_latency_ms: float = 0
    error_message: Optional[str] = None

@dataclass
class GPUStatus:
    """Current GPU state for Tower's dual-GPU setup."""
    device_id: int
    name: str
    total_vram_mb: int
    used_vram_mb: int
    models_loaded: List[str]
    temperature_c: Optional[float] = None

    @property
    def available_vram_mb(self) -> int:
        # Keep 1GB buffer for system
        return max(0, self.total_vram_mb - self.used_vram_mb - 1024)

    @property
    def can_fit_model(self) -> bool:
        """Can this GPU fit another model?"""
        return self.available_vram_mb > 2048  # Need at least 2GB free

class TowerModelManager:
    """
    Production-ready model manager for Tower's infrastructure.

    Key principles:
    1. A loaded 7B model beats an unloaded 32B model for interactive tasks
    2. Memory is precious - evict LRU models when needed
    3. Preload predictively based on usage patterns
    4. Track actual performance, not theoretical
    """

    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.http = httpx.AsyncClient(timeout=120.0)
        self.model_profiles = self._init_tower_models()
        self.model_status: Dict[str, ModelStatus] = {}
        self.gpu_status = self._init_gpu_status()
        self._performance_history: Dict[str, List[float]] = {}
        self._last_check = datetime.now()

    def _init_tower_models(self) -> Dict[str, ModelProfile]:
        """Initialize model profiles based on Tower's actual capabilities."""
        return {
            # Models that actually fit on Tower's GPUs
            "qwen2.5-coder:7b": ModelProfile(
                name="Qwen 2.5 Coder 7B",
                model_id="qwen2.5-coder:7b",
                vram_mb=4500,
                load_time_seconds=12.0,
                avg_inference_ms=1500,
                max_context_tokens=32768,
                capabilities={
                    "code_generation": 0.88,
                    "code_review": 0.85,
                    "code_refactor": 0.83,
                    "debugging": 0.82,
                    "general": 0.70
                },
                preferred_gpu=1  # RTX 3060
            ),
            "deepseek-coder:latest": ModelProfile(
                name="DeepSeek Coder",
                model_id="deepseek-coder:latest",
                vram_mb=8500,
                load_time_seconds=18.0,
                avg_inference_ms=2200,
                max_context_tokens=16384,
                capabilities={
                    "code_generation": 0.85,
                    "code_analysis": 0.82,
                    "debugging": 0.78,
                    "general": 0.65
                },
                preferred_gpu=0  # AMD GPU
            ),
            "llama3.2:3b": ModelProfile(
                name="Llama 3.2 3B",
                model_id="llama3.2:3b",
                vram_mb=2500,
                load_time_seconds=5.0,
                avg_inference_ms=800,
                max_context_tokens=8192,
                capabilities={
                    "general": 0.75,
                    "conversation": 0.78,
                    "summarization": 0.72,
                    "quick_response": 0.85
                },
                preferred_gpu=1  # Fits on either
            ),
            "tinyllama:latest": ModelProfile(
                name="TinyLlama",
                model_id="tinyllama:latest",
                vram_mb=1000,
                load_time_seconds=2.0,
                avg_inference_ms=200,
                max_context_tokens=2048,
                capabilities={
                    "quick_response": 0.90,
                    "status_check": 0.70,
                    "simple_tasks": 0.65,
                    "general": 0.55
                },
                preferred_gpu=1  # Keep on RTX for fast response
            ),
            "mistral:7b-instruct": ModelProfile(
                name="Mistral 7B Instruct",
                model_id="mistral:7b-instruct",
                vram_mb=4800,
                load_time_seconds=10.0,
                avg_inference_ms=1400,
                max_context_tokens=8192,
                capabilities={
                    "reasoning": 0.82,
                    "general": 0.78,
                    "conversation": 0.80,
                    "analysis": 0.75
                },
                preferred_gpu=1  # RTX 3060
            ),
            "llava:7b": ModelProfile(
                name="Llava 7B Vision",
                model_id="llava:7b",
                vram_mb=5000,
                load_time_seconds=12.0,
                avg_inference_ms=2000,
                max_context_tokens=4096,
                capabilities={
                    "vision": 0.85,
                    "image_analysis": 0.88,
                    "general": 0.65
                },
                preferred_gpu=0  # AMD for compute
            )
        }

    def _init_gpu_status(self) -> Dict[int, GPUStatus]:
        """Initialize GPU status for Tower's dual-GPU setup."""
        return {
            0: GPUStatus(
                device_id=0,
                name="AMD RX 9070 XT",
                total_vram_mb=16384,  # 16GB
                used_vram_mb=0,
                models_loaded=[]
            ),
            1: GPUStatus(
                device_id=1,
                name="NVIDIA RTX 3060",
                total_vram_mb=12288,  # 12GB
                used_vram_mb=0,
                models_loaded=[]
            )
        }

    async def refresh_model_status(self) -> None:
        """Update status of all models from Ollama."""
        try:
            # Get currently loaded models
            response = await self.http.get(f"{self.ollama_url}/api/ps")
            if response.status_code == 200:
                data = response.json()
                loaded_models = {m["name"]: m.get("size", 0) for m in data.get("models", [])}

                # Reset GPU usage
                for gpu in self.gpu_status.values():
                    gpu.models_loaded = []
                    gpu.used_vram_mb = 0

                # Update model status
                for model_id, profile in self.model_profiles.items():
                    if model_id in loaded_models:
                        status = ModelStatus(
                            model_id=model_id,
                            state=ModelState.LOADED,
                            last_used=datetime.now(),
                            current_vram_mb=profile.vram_mb
                        )

                        # Update GPU usage
                        gpu = self.gpu_status[profile.preferred_gpu]
                        gpu.models_loaded.append(model_id)
                        gpu.used_vram_mb += profile.vram_mb
                    else:
                        status = ModelStatus(
                            model_id=model_id,
                            state=ModelState.UNLOADED,
                            current_vram_mb=0
                        )

                    self.model_status[model_id] = status

                self._last_check = datetime.now()

        except Exception as e:
            logger.error(f"Failed to refresh model status: {e}")

    async def select_model(
        self,
        task_type: str,
        urgency: TaskUrgency,
        context_size: int = 1000,
        min_quality: float = 0.6
    ) -> Tuple[str, str, float]:
        """
        Select the best available model considering all factors.

        Returns:
            Tuple of (model_id, selection_reason, expected_wait_seconds)
        """
        # Refresh status if stale
        if datetime.now() - self._last_check > timedelta(seconds=30):
            await self.refresh_model_status()

        candidates = []

        for model_id, profile in self.model_profiles.items():
            # Skip if can't handle context size
            if context_size > profile.max_context_tokens:
                continue

            # Get quality score for task
            quality = profile.capabilities.get(task_type, 0.5)
            if quality < min_quality:
                continue

            # Check if loaded
            status = self.model_status.get(model_id, ModelStatus(model_id, ModelState.UNLOADED))
            is_loaded = status.state == ModelState.LOADED

            # Calculate effective score
            if urgency == TaskUrgency.CRITICAL:
                # For critical tasks, heavily favor loaded models
                time_penalty = 0 if is_loaded else 10.0
            elif urgency == TaskUrgency.INTERACTIVE:
                # For interactive, <5s is acceptable
                load_time = 0 if is_loaded else profile.load_time_seconds
                time_penalty = load_time / 5.0 if load_time > 5 else load_time / 10.0
            elif urgency == TaskUrgency.BACKGROUND:
                # For background, up to 30s is fine
                time_penalty = (0 if is_loaded else profile.load_time_seconds) / 30.0
            else:  # BATCH
                # For batch, time doesn't matter much
                time_penalty = (0 if is_loaded else profile.load_time_seconds) / 120.0

            # Factor in inference speed
            speed_score = 1.0 - (profile.avg_inference_ms / 5000.0)  # Normalize to 5s max

            # Composite score
            effective_score = (
                quality * 0.5 +           # Quality most important
                (1.0 if is_loaded else 0.2) * 0.3 +  # Availability
                speed_score * 0.2 -        # Speed
                time_penalty * 0.3         # Penalty for load time
            )

            expected_wait = 0 if is_loaded else profile.load_time_seconds
            expected_wait += profile.avg_inference_ms / 1000.0

            candidates.append({
                "model_id": model_id,
                "name": profile.name,
                "score": effective_score,
                "quality": quality,
                "is_loaded": is_loaded,
                "expected_wait": expected_wait,
                "vram_needed": profile.vram_mb if not is_loaded else 0
            })

        if not candidates:
            raise ValueError(f"No suitable model for task '{task_type}' with context {context_size}")

        # Sort by score
        candidates.sort(key=lambda x: x["score"], reverse=True)
        best = candidates[0]

        # Generate human-readable reason
        if best["is_loaded"]:
            reason = f"{best['name']} (loaded, quality={best['quality']:.2f}, ~{best['expected_wait']:.1f}s)"
        else:
            reason = f"{best['name']} (needs loading ~{best['expected_wait']:.1f}s, quality={best['quality']:.2f})"

        # Add urgency context
        if urgency == TaskUrgency.CRITICAL and not best["is_loaded"]:
            reason += " [CRITICAL: Loading delay unavoidable]"
        elif urgency == TaskUrgency.INTERACTIVE and best["expected_wait"] > 5:
            reason += " [Warning: May exceed interactive threshold]"

        return best["model_id"], reason, best["expected_wait"]

    async def can_load_model(self, model_id: str) -> Tuple[bool, str]:
        """
        Check if a model can be loaded given current memory constraints.

        Returns:
            Tuple of (can_load, reason)
        """
        profile = self.model_profiles.get(model_id)
        if not profile:
            return False, f"Unknown model: {model_id}"

        # Check preferred GPU
        gpu = self.gpu_status[profile.preferred_gpu]

        if gpu.available_vram_mb >= profile.vram_mb:
            return True, f"Can load on {gpu.name} ({gpu.available_vram_mb}MB available)"

        # Check other GPU
        other_gpu_id = 1 - profile.preferred_gpu
        other_gpu = self.gpu_status[other_gpu_id]

        if other_gpu.available_vram_mb >= profile.vram_mb:
            return True, f"Can load on {other_gpu.name} ({other_gpu.available_vram_mb}MB available)"

        # Would need to evict
        return False, f"Insufficient VRAM (need {profile.vram_mb}MB, have {max(gpu.available_vram_mb, other_gpu.available_vram_mb)}MB)"

    async def preload_model(self, model_id: str) -> bool:
        """
        Preload a model into memory.

        Returns:
            True if successfully loaded or already loaded.
        """
        # Check if already loaded
        await self.refresh_model_status()
        status = self.model_status.get(model_id)

        if status and status.state == ModelState.LOADED:
            logger.info(f"Model {model_id} already loaded")
            return True

        # Check if we can load it
        can_load, reason = await self.can_load_model(model_id)
        if not can_load:
            logger.warning(f"Cannot preload {model_id}: {reason}")
            return False

        try:
            # Trigger load with minimal generation
            logger.info(f"Preloading {model_id}...")
            response = await self.http.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": model_id,
                    "prompt": "hi",
                    "stream": False,
                    "options": {"num_predict": 1}
                },
                timeout=60.0
            )

            if response.status_code == 200:
                logger.info(f"Successfully preloaded {model_id}")
                await self.refresh_model_status()
                return True

        except Exception as e:
            logger.error(f"Failed to preload {model_id}: {e}")

        return False

    def get_status_summary(self) -> Dict[str, any]:
        """Get summary of current model and GPU status."""
        loaded = [m for m, s in self.model_status.items() if s.state == ModelState.LOADED]

        return {
            "loaded_models": loaded,
            "total_models": len(self.model_profiles),
            "gpu_status": {
                0: {
                    "name": self.gpu_status[0].name,
                    "used": f"{self.gpu_status[0].used_vram_mb}MB",
                    "available": f"{self.gpu_status[0].available_vram_mb}MB",
                    "models": self.gpu_status[0].models_loaded
                },
                1: {
                    "name": self.gpu_status[1].name,
                    "used": f"{self.gpu_status[1].used_vram_mb}MB",
                    "available": f"{self.gpu_status[1].available_vram_mb}MB",
                    "models": self.gpu_status[1].models_loaded
                }
            },
            "last_refresh": self._last_check.isoformat()
        }

    async def suggest_preloads(self, hour: int) -> List[str]:
        """
        Suggest models to preload based on time of day.

        This is where you'd add usage pattern learning.
        """
        if 9 <= hour <= 17:
            # Work hours: coding focus
            return ["qwen2.5-coder:7b", "deepseek-coder:latest"]
        elif 18 <= hour <= 22:
            # Evening: general tasks
            return ["mistral:7b-instruct", "llama3.2:3b"]
        else:
            # Night: minimal, fast response
            return ["tinyllama:latest"]


async def test_smart_manager():
    """Test the smart model manager."""
    manager = TowerModelManager()

    # Refresh current status
    await manager.refresh_model_status()

    print("Current Status:")
    status = manager.get_status_summary()
    print(json.dumps(status, indent=2))

    # Test selection for different scenarios
    scenarios = [
        ("code_generation", TaskUrgency.INTERACTIVE, "User asks for code help"),
        ("code_generation", TaskUrgency.BACKGROUND, "Background refactoring"),
        ("quick_response", TaskUrgency.CRITICAL, "System emergency"),
        ("reasoning", TaskUrgency.BATCH, "Batch analysis job")
    ]

    print("\n\nModel Selection Tests:")
    print("-" * 60)

    for task_type, urgency, description in scenarios:
        model_id, reason, wait_time = await manager.select_model(task_type, urgency)
        print(f"\n{description}:")
        print(f"  Task: {task_type}, Urgency: {urgency.value}")
        print(f"  Selected: {model_id}")
        print(f"  Reason: {reason}")
        print(f"  Expected wait: {wait_time:.1f}s")

    print("\n✅ Smart model manager test complete")


if __name__ == "__main__":
    asyncio.run(test_smart_manager())