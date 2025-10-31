#!/usr/bin/env python3
"""
Model Preloader for Echo Brain
Preloads frequently used models to reduce cold start latency
"""

import asyncio
import logging
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class ModelPreloader:
    """Preloads and manages Ollama models for optimal performance"""

    def __init__(self):
        self.ollama_api = "http://localhost:11434"

        # Priority models that should be kept warm
        self.priority_models = [
            "tinyllama:latest",      # Quick responses
            "llama3.2:3b",          # Standard queries
            "mistral:7b-instruct",  # Professional queries
            "qwen2.5-coder:32b",    # Coding and expert queries
            "deepseek-coder:latest" # Currently loaded baseline
        ]

        # Track model usage for intelligent caching
        self.usage_stats = defaultdict(lambda: {
            "requests": 0,
            "last_used": None,
            "avg_response_time": 0.0,
            "total_time": 0.0
        })

        # Recent requests for pattern analysis
        self.recent_requests = deque(maxlen=1000)

        # Currently loaded models
        self.loaded_models: Set[str] = set()

    async def check_ollama_health(self) -> bool:
        """Check if Ollama service is healthy"""
        try:
            response = requests.get(f"{self.ollama_api}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False

    async def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded models"""
        try:
            response = requests.get(f"{self.ollama_api}/api/ps", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = [model.get("name", "") for model in data.get("models", [])]
                self.loaded_models = set(models)
                return models
            return []
        except Exception as e:
            logger.error(f"Failed to get loaded models: {e}")
            return []

    async def preload_model(self, model_name: str) -> bool:
        """Preload a specific model"""
        try:
            logger.info(f"Preloading model: {model_name}")

            # Generate a dummy request to load the model
            response = requests.post(
                f"{self.ollama_api}/api/generate",
                json={
                    "model": model_name,
                    "prompt": "Hello",
                    "stream": False,
                    "options": {
                        "num_predict": 1  # Minimal response to just load model
                    }
                },
                timeout=30
            )

            if response.status_code == 200:
                self.loaded_models.add(model_name)
                logger.info(f"✅ Model preloaded: {model_name}")
                return True
            else:
                logger.error(f"Failed to preload {model_name}: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Error preloading {model_name}: {e}")
            return False

    async def preload_priority_models(self) -> Dict[str, bool]:
        """Preload all priority models"""
        results = {}

        if not await self.check_ollama_health():
            logger.error("Ollama not healthy, skipping preload")
            return results

        # Get currently loaded models
        await self.get_loaded_models()

        for model in self.priority_models:
            if model not in self.loaded_models:
                results[model] = await self.preload_model(model)
                # Small delay between preloads to avoid overwhelming
                await asyncio.sleep(2)
            else:
                results[model] = True
                logger.info(f"Model already loaded: {model}")

        return results

    def record_model_usage(self, model_name: str, response_time: float):
        """Record model usage statistics"""
        stats = self.usage_stats[model_name]
        stats["requests"] += 1
        stats["last_used"] = datetime.now()

        # Update average response time
        stats["total_time"] += response_time
        stats["avg_response_time"] = stats["total_time"] / stats["requests"]

        # Record in recent requests
        self.recent_requests.append({
            "model": model_name,
            "timestamp": datetime.now(),
            "response_time": response_time
        })

    def get_usage_patterns(self) -> Dict:
        """Analyze usage patterns for intelligent caching"""
        now = datetime.now()

        # Analyze recent usage (last hour)
        recent_hour = [
            req for req in self.recent_requests
            if (now - req["timestamp"]).total_seconds() < 3600
        ]

        model_frequency = defaultdict(int)
        for req in recent_hour:
            model_frequency[req["model"]] += 1

        # Identify trending models
        trending = sorted(model_frequency.items(), key=lambda x: x[1], reverse=True)

        return {
            "total_requests_hour": len(recent_hour),
            "unique_models_hour": len(model_frequency),
            "trending_models": trending[:5],
            "loaded_models": list(self.loaded_models),
            "priority_models": self.priority_models,
            "usage_stats": dict(self.usage_stats)
        }

    async def intelligent_cache_management(self) -> Dict:
        """Manage model cache based on usage patterns"""
        patterns = self.get_usage_patterns()
        actions_taken = []

        # Preload trending models that aren't loaded
        for model, freq in patterns["trending_models"]:
            if model not in self.loaded_models and freq >= 3:  # 3+ uses in last hour
                success = await self.preload_model(model)
                if success:
                    actions_taken.append(f"preloaded_trending_{model}")

        # Ensure priority models are loaded
        for model in self.priority_models:
            if model not in self.loaded_models:
                success = await self.preload_model(model)
                if success:
                    actions_taken.append(f"preloaded_priority_{model}")

        return {
            "actions_taken": actions_taken,
            "patterns": patterns,
            "timestamp": datetime.now().isoformat()
        }

    async def continuous_optimization(self, check_interval: int = 1800):
        """Continuous model preloading and optimization"""
        logger.info(f"Starting model preloader (check every {check_interval//60} minutes)")

        # Initial preload
        await self.preload_priority_models()

        while True:
            try:
                await asyncio.sleep(check_interval)

                if await self.check_ollama_health():
                    result = await self.intelligent_cache_management()
                    if result["actions_taken"]:
                        logger.info(f"Model cache optimized: {result['actions_taken']}")
                else:
                    logger.warning("Ollama not healthy, skipping cache management")

            except Exception as e:
                logger.error(f"Model preloader error: {e}")
                await asyncio.sleep(300)  # Shorter retry on errors

    async def get_preloader_stats(self) -> Dict:
        """Get preloader statistics"""
        await self.get_loaded_models()
        patterns = self.get_usage_patterns()

        # Calculate average response times
        avg_times = {}
        for model, stats in self.usage_stats.items():
            if stats["requests"] > 0:
                avg_times[model] = stats["avg_response_time"]

        return {
            "loaded_models_count": len(self.loaded_models),
            "loaded_models": list(self.loaded_models),
            "priority_models": self.priority_models,
            "total_requests_tracked": sum(stats["requests"] for stats in self.usage_stats.values()),
            "average_response_times": avg_times,
            "usage_patterns": patterns,
            "ollama_healthy": await self.check_ollama_health(),
            "timestamp": datetime.now().isoformat()
        }

# Global model preloader instance
model_preloader = ModelPreloader()