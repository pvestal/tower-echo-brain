#!/usr/bin/env python3
"""
DeepSeek API Adapter for Echo Brain
Secure integration with DeepSeek's reasoning and chat models
"""

import os
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
import time

logger = logging.getLogger(__name__)

class DeepSeekModel(Enum):
    """Available DeepSeek models"""
    REASONER = "deepseek-reasoner"  # Complex reasoning, math, coding
    CHAT = "deepseek-chat"  # General conversation, content generation

@dataclass
class DeepSeekResponse:
    """Structured response from DeepSeek API"""
    content: str
    model_used: str
    reasoning_steps: Optional[str] = None  # For reasoner model
    processing_time: float = 0.0
    token_usage: Optional[Dict[str, int]] = None

class DeepSeekAdapter:
    """
    Secure adapter for DeepSeek API integration
    Uses OpenAI-compatible format with environment variables for credentials
    """

    def __init__(self, model_type: DeepSeekModel = DeepSeekModel.REASONER):
        """
        Initialize DeepSeek adapter with secure configuration

        Args:
            model_type: Which DeepSeek model to use
        """
        # SECURITY: API key from Tower vault (preferred) or environment variable
        api_key = self._get_api_key()
        if not api_key:
            logger.warning("DeepSeek API key not found in vault or environment variables")
            self.client = None
        else:
            try:
                from openai import OpenAI
                self.client = OpenAI(
                    api_key=api_key,
                    base_url="https://api.deepseek.com",
                )
                logger.info(f"✅ DeepSeek adapter initialized with {model_type.value}")
            except ImportError:
                logger.error("OpenAI package not installed. Run: pip install openai")
                self.client = None

        self.model_type = model_type
        self.model_name = model_type.value

    def _get_api_key(self) -> Optional[str]:
        """
        Retrieve DeepSeek API key from Tower vault or environment variables

        Returns:
            API key string or None if not found
        """
        # Try Tower vault first (preferred)
        try:
            import hvac
            vault_client = hvac.Client(url='http://127.0.0.1:8200')

            # Try to get token from environment or use root token
            vault_token = os.getenv('VAULT_TOKEN', 'hvs.yI6XmqUjZABfcJTxJHzTaruB')
            vault_client.token = vault_token

            if vault_client.is_authenticated():
                secret = vault_client.secrets.kv.v2.read_secret_version(path='tower/deepseek')
                api_key = secret['data']['data']['api_key']
                logger.info("✅ DeepSeek API key retrieved from Tower vault")
                return api_key
            else:
                logger.warning("Vault authentication failed")
        except Exception as e:
            logger.warning(f"Failed to read from vault: {e}")

        # Fallback to environment variable
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if api_key:
            logger.info("✅ DeepSeek API key retrieved from environment variable")
            return api_key

        return None

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> DeepSeekResponse:
        """
        Generate response using DeepSeek API

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters for the API

        Returns:
            DeepSeekResponse with content and metadata
        """
        if not self.client:
            logger.error("DeepSeek client not initialized. Check API key.")
            return DeepSeekResponse(
                content="DeepSeek API not configured. Please set DEEPSEEK_API_KEY.",
                model_used=self.model_name,
                processing_time=0.0
            )

        start_time = time.time()

        try:
            # SECURITY: Use structured client calls, never os.system()
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
                **kwargs
            )

            processing_time = time.time() - start_time

            # Extract reasoning if present (for reasoner model)
            content = response.choices[0].message.content
            reasoning_steps = None

            if self.model_type == DeepSeekModel.REASONER and "<think>" in content:
                # Extract thinking process for transparency
                import re
                think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
                if think_match:
                    reasoning_steps = think_match.group(1).strip()
                    # Remove thinking tags from final content
                    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

            return DeepSeekResponse(
                content=content,
                model_used=self.model_name,
                reasoning_steps=reasoning_steps,
                processing_time=processing_time,
                token_usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                }
            )

        except Exception as e:
            logger.error(f"DeepSeek API error: {str(e)}")
            return DeepSeekResponse(
                content=f"Error calling DeepSeek API: {str(e)}",
                model_used=self.model_name,
                processing_time=time.time() - start_time
            )

    def route_by_complexity(self, query: str, complexity_score: float) -> DeepSeekModel:
        """
        Determine which DeepSeek model to use based on query complexity

        Args:
            query: The user's query
            complexity_score: Complexity score from Echo Brain's analyzer (0-100)

        Returns:
            Appropriate DeepSeekModel for the task
        """
        # Complex tasks requiring step-by-step reasoning
        reasoning_keywords = [
            'calculate', 'solve', 'prove', 'derive', 'implement',
            'algorithm', 'optimize', 'debug', 'analyze code',
            'multi-step', 'planning', 'workflow', 'architecture'
        ]

        query_lower = query.lower()

        # Use reasoner for complex or technical queries
        if complexity_score > 70:
            return DeepSeekModel.REASONER
        elif any(keyword in query_lower for keyword in reasoning_keywords):
            return DeepSeekModel.REASONER
        else:
            # Use chat for general conversation
            return DeepSeekModel.CHAT

    def format_for_rag(self, query: str, context: List[str]) -> List[Dict[str, str]]:
        """
        Format query with RAG context for DeepSeek

        Args:
            query: User's question
            context: Retrieved context from vector memory

        Returns:
            Formatted messages for API call
        """
        system_prompt = (
            "You are a helpful AI assistant with access to a knowledge base. "
            "Analyze the provided context carefully and answer based on the relevant information. "
            "If the context doesn't contain enough information, say so clearly."
        )

        context_str = "\n\n".join(context[:5])  # Limit to top 5 contexts

        user_prompt = f"""Context from knowledge base:
{context_str}

Question: {query}

Please provide a comprehensive answer based on the context provided."""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    def format_for_agent_planning(self, task: str, constraints: Optional[List[str]] = None) -> List[Dict[str, str]]:
        """
        Format task for agent workflow planning using reasoner model

        Args:
            task: The task to plan
            constraints: Optional constraints or requirements

        Returns:
            Formatted messages for planning
        """
        system_prompt = (
            "You are an expert AI planner. Break down complex tasks into clear, "
            "actionable steps. Consider dependencies, resources, and potential obstacles. "
            "Provide your reasoning process before the final plan."
        )

        constraint_str = ""
        if constraints:
            constraint_str = "\n\nConstraints:\n" + "\n".join(f"- {c}" for c in constraints)

        user_prompt = f"""Task: {task}{constraint_str}

Please provide:
1. A step-by-step reasoning process
2. A detailed action plan
3. Potential challenges and mitigation strategies"""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]


# Factory function for easy instantiation
def create_deepseek_adapter(model_type: str = "reasoner") -> DeepSeekAdapter:
    """
    Factory function to create DeepSeek adapter

    Args:
        model_type: "reasoner" or "chat"

    Returns:
        Configured DeepSeekAdapter instance
    """
    if model_type == "chat":
        return DeepSeekAdapter(DeepSeekModel.CHAT)
    else:
        return DeepSeekAdapter(DeepSeekModel.REASONER)