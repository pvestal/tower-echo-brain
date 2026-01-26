"""
Context-Aware Agent Base Class
All agents inherit from this to get automatic context retrieval
"""

import logging
from typing import Dict, Any, Optional
import httpx
import json
import asyncio

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.context import get_optimized_omniscient_context

logger = logging.getLogger(__name__)


class ContextAwareAgent:
    """Base class that all agents inherit from"""

    def __init__(self, model_name: str, system_prompt: str):
        self.context_provider = get_optimized_omniscient_context()
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.ollama_url = "http://localhost:11434"
        self.history = []  # Keep track of recent tasks

    async def process(self, query: str, include_context: bool = True,
                     context_override: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process a query with optional context injection

        Args:
            query: The user's query
            include_context: Whether to fetch and include context
            context_override: Optional pre-fetched context to use instead

        Returns:
            Dict containing response and metadata
        """
        try:
            # Get context if needed
            context = {}
            if context_override:
                context = context_override
            elif include_context:
                context = await self.context_provider.get_context(query)

            # Build prompt with context
            full_prompt = self.build_prompt_with_context(query, context)

            # Call the model
            response = await self.call_model(full_prompt)

            # Store in history
            self.history.append({
                "query": query,
                "response": response,
                "context_used": bool(context),
                "model": self.model_name
            })

            # Keep only last 50 entries
            if len(self.history) > 50:
                self.history = self.history[-50:]

            return {
                "task": query,
                "response": response,
                "model": self.model_name,
                "context_used": context if include_context else None
            }

        except Exception as e:
            logger.error(f"Agent processing failed: {e}")
            return {
                "task": query,
                "response": f"Error: {str(e)}",
                "model": self.model_name,
                "error": str(e)
            }

    def build_prompt_with_context(self, query: str, context: Dict[str, Any]) -> str:
        """Inject context into the prompt"""
        prompt_parts = []

        # Add system prompt
        if self.system_prompt:
            prompt_parts.append(self.system_prompt)

        # Add context if available
        if context:
            context_prompt = self._format_context(context)
            if context_prompt:
                prompt_parts.append("\n## Relevant Context\n" + context_prompt)

        # Add the query
        prompt_parts.append(f"\n## Task\n{query}")

        return "\n".join(prompt_parts)

    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context for inclusion in prompt"""
        formatted_parts = []

        # Format memories
        memories = context.get("memories", [])
        if memories:
            formatted_parts.append("### Relevant Memories")
            for i, memory in enumerate(memories[:5], 1):
                score = memory.get("score", 0)
                text = memory.get("text", "")[:200]  # Truncate long memories
                formatted_parts.append(f"{i}. [Score: {score:.2f}] {text}")

        # Format facts
        facts = context.get("facts", [])
        if facts:
            formatted_parts.append("\n### Related Facts")
            for fact in facts[:10]:
                subject = fact.get("subject", "")
                predicate = fact.get("predicate", "")
                obj = fact.get("object", "")
                formatted_parts.append(f"- {subject} {predicate} {obj}")

        # Format recent conversations
        recent = context.get("recent_conversations", [])
        if recent:
            formatted_parts.append("\n### Recent Conversations")
            for conv in recent[:3]:
                summary = conv.get("summary", "")[:150]
                formatted_parts.append(f"- {summary}")

        return "\n".join(formatted_parts) if formatted_parts else ""

    async def call_model(self, prompt: str) -> str:
        """Call Ollama model with the prompt"""
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                # Prepare the request based on model type
                request_data = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9
                    }
                }

                # Special handling for DeepSeek reasoning model
                if "deepseek-r1" in self.model_name.lower():
                    request_data["options"]["temperature"] = 0.1  # Lower temp for reasoning

                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json=request_data
                )

                if response.status_code == 200:
                    data = response.json()
                    return data.get("response", "No response generated")
                else:
                    logger.error(f"Ollama call failed: {response.status_code}")
                    return f"Model call failed: HTTP {response.status_code}"

        except asyncio.TimeoutError:
            logger.error(f"Ollama call timed out for model {self.model_name}")
            return "Model call timed out (120s limit)"
        except Exception as e:
            logger.error(f"Failed to call model {self.model_name}: {e}")
            return f"Model call failed: {str(e)}"

    async def get_embedding(self, text: str) -> Optional[list]:
        """Get embedding for text using the context provider"""
        return await self.context_provider.generate_embedding(text)

    async def store_learned_fact(self, subject: str, predicate: str,
                                 obj: str, confidence: float = 0.8) -> bool:
        """Store a fact learned during processing"""
        return await self.context_provider.store_fact(
            subject, predicate, obj, confidence,
            source=f"{self.model_name}_agent"
        )