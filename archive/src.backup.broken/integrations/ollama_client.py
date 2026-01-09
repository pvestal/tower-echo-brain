#!/usr/bin/env python3
"""
Ollama Integration for Echo Brain
Provides LLM reasoning capabilities using local Ollama models
"""

import aiohttp
import asyncio
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class OllamaModel:
    """Ollama model information"""
    name: str
    size: str
    parameters: str
    description: str

class OllamaClient:
    """Client for interacting with Ollama API"""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.available_models: List[OllamaModel] = []

    async def initialize(self):
        """Initialize and check Ollama connection"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        self.available_models = [
                            OllamaModel(
                                name=model["name"],
                                size=model.get("size", "unknown"),
                                parameters=model.get("details", {}).get("parameters", "unknown"),
                                description=model.get("description", "")
                            )
                            for model in data.get("models", [])
                        ]
                        logger.info(f"✅ Connected to Ollama with {len(self.available_models)} models available")
                        return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to Ollama: {e}")
            return False

    async def generate(self, prompt: str, model: str = "llama3.1:8b",
                      temperature: float = 0.7, max_tokens: int = 2048) -> Optional[str]:
        """Generate text using Ollama model"""
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "temperature": temperature,
                "options": {
                    "num_predict": max_tokens
                },
                "stream": False
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("response", "")
                    else:
                        logger.error(f"Ollama generate failed: {response.status}")
                        return None

        except asyncio.TimeoutError:
            logger.error("Ollama generation timed out")
            return None
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            return None

    async def chat(self, messages: List[Dict[str, str]], model: str = "llama3.1:8b",
                   temperature: float = 0.7, max_tokens: int = 2048) -> Optional[str]:
        """Chat with Ollama model using conversation history"""
        try:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "options": {
                    "num_predict": max_tokens
                },
                "stream": False
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("message", {}).get("content", "")
                    else:
                        logger.error(f"Ollama chat failed: {response.status}")
                        return None

        except asyncio.TimeoutError:
            logger.error("Ollama chat timed out")
            return None
        except Exception as e:
            logger.error(f"Ollama chat error: {e}")
            return None

    async def embeddings(self, text: str, model: str = "nomic-embed-text:latest") -> Optional[List[float]]:
        """Generate embeddings using Ollama model"""
        try:
            payload = {
                "model": model,
                "prompt": text
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/embeddings",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("embedding", [])
                    else:
                        logger.error(f"Ollama embeddings failed: {response.status}")
                        return None

        except Exception as e:
            logger.error(f"Ollama embeddings error: {e}")
            return None

    async def analyze_code(self, code: str, language: str = "python") -> Optional[Dict[str, Any]]:
        """Analyze code using specialized coding model"""
        prompt = f"""Analyze the following {language} code and provide:
1. Quality score (0-10)
2. List of issues found
3. Suggestions for improvement
4. Security concerns if any

Code:
```{language}
{code}
```

Provide response in JSON format."""

        # Use coding-specific model if available
        model = "qwen2.5-coder:7b" if any(m.name.startswith("qwen2.5-coder") for m in self.available_models) else "deepseek-coder:latest"

        response = await self.generate(prompt, model=model, temperature=0.3)

        if response:
            try:
                # Try to parse JSON from response
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            except:
                pass

        return None

    async def reason_about_task(self, task: str, context: str = "") -> Optional[str]:
        """Use LLM to reason about a task"""
        prompt = f"""You are Echo Brain, an autonomous AI assistant.

Task: {task}

Context: {context}

Please provide:
1. Your understanding of the task
2. Step-by-step approach to complete it
3. Any potential issues or considerations
4. Expected outcome

Be concise and actionable."""

        # Use best available model for reasoning
        model = "llama3.1:8b" if any(m.name.startswith("llama3.1") for m in self.available_models) else "mistral:7b-instruct"

        return await self.generate(prompt, model=model, temperature=0.7)

    def get_best_model(self, task_type: str) -> str:
        """Select best model for task type"""
        model_preferences = {
            "code": ["qwen2.5-coder:7b", "deepseek-coder-v2:16b", "deepseek-coder:latest"],
            "reasoning": ["llama3.1:8b", "gemma2:9b", "mistral:7b-instruct"],
            "creative": ["llama3.2:3b", "tinyllama:latest"],
            "embeddings": ["nomic-embed-text:latest"],
            "vision": ["llava:7b", "llava:13b", "minicpm-v:8b"]
        }

        preferences = model_preferences.get(task_type, ["llama3.1:8b"])

        for pref in preferences:
            if any(m.name == pref for m in self.available_models):
                return pref

        # Fallback to first available model
        return self.available_models[0].name if self.available_models else "llama3.1:8b"

# Singleton instance
ollama_client = OllamaClient()

async def get_ollama_client() -> OllamaClient:
    """Get initialized Ollama client"""
    if not ollama_client.available_models:
        await ollama_client.initialize()
    return ollama_client