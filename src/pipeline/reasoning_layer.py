"""
LAYER 2: Reasoning & Inference

This layer answers: "Given this context, what's the best answer?"

Selects the appropriate model, constructs an enriched prompt,
runs inference, and returns structured results.
"""
import time
import logging
import httpx
import json

from .models import ContextPackage, ReasoningResult, QueryIntent

logger = logging.getLogger("echo.pipeline.reasoning")

OLLAMA_URL = "http://localhost:11434"

# Model selection - USING ACTUAL AVAILABLE MODELS (RTX 3060 12GB)
# Available: mistral:7b, deepseek-r1:8b, llama3.1:8b, gemma2:9b, qwen2.5-coder:7b
MODEL_MAP = {
    QueryIntent.CODING: "qwen2.5-coder:7b",      # Specialized for code
    QueryIntent.REASONING: "deepseek-r1:8b",      # Good at step-by-step reasoning
    QueryIntent.PERSONAL: "llama3.1:8b",          # Good general model
    QueryIntent.FACTUAL: "mistral:7b",            # Fast and accurate
    QueryIntent.CREATIVE: "gemma2:9b",            # Creative tasks
    QueryIntent.CONVERSATIONAL: "llama3.1:8b",    # General conversation
}

# System prompts per intent
SYSTEM_PROMPTS = {
    QueryIntent.CODING: (
        "You are a senior software engineer. Answer coding questions precisely. "
        "Include working code examples. If context provides relevant code snippets, "
        "reference and build on them. Be direct, no fluff."
    ),
    QueryIntent.REASONING: (
        "You are an analytical reasoning engine. Break down complex questions step by step. "
        "Consider multiple perspectives. Cite the provided context when relevant. "
        "State your confidence level and what you're uncertain about."
    ),
    QueryIntent.PERSONAL: (
        "You are Echo Brain, Patrick's personal AI assistant. You know Patrick from "
        "the context provided - his projects, preferences, equipment, and history. "
        "Answer personal questions using ONLY the provided context. If the context "
        "doesn't contain the answer, say so honestly. Do not guess or hallucinate."
    ),
    QueryIntent.FACTUAL: (
        "You are a knowledge retrieval system. Answer factual questions using the provided "
        "context. Be precise and concise. If the context doesn't support an answer, say "
        "'I don't have that information in my current knowledge base.'"
    ),
    QueryIntent.CREATIVE: (
        "You are a creative narration engine for anime production. Generate vivid scenes, "
        "character descriptions, and ComfyUI-compatible prompts when relevant. "
        "Use the provided context for character consistency and project details."
    ),
    QueryIntent.CONVERSATIONAL: (
        "You are Echo Brain, Patrick's AI assistant running on his Tower server. "
        "Be helpful, direct, and conversational. Use provided context to personalize "
        "your responses. Don't be generic."
    ),
}


class ReasoningLayer:
    """
    Model selection + prompt construction + inference.

    Usage:
        layer = ReasoningLayer()
        result = await layer.reason(context_package)
    """

    def __init__(self):
        self.http_client: httpx.AsyncClient = None

    async def initialize(self):
        self.http_client = httpx.AsyncClient(timeout=120.0)  # Reasoning can be slow
        # Verify models are available
        try:
            resp = await self.http_client.get(f"{OLLAMA_URL}/api/tags")
            available = [m["name"] for m in resp.json().get("models", [])]
            for intent, model in MODEL_MAP.items():
                if model not in available and not any(model in a for a in available):
                    logger.warning(f"Model {model} for {intent.value} not found in Ollama!")
                else:
                    logger.info(f"Model verified: {model} -> {intent.value}")
        except Exception as e:
            logger.error(f"Failed to verify Ollama models: {e}")

    async def shutdown(self):
        if self.http_client:
            await self.http_client.aclose()

    async def reason(self, context: ContextPackage) -> ReasoningResult:
        """
        Main entry point. Takes context package, returns reasoned answer.
        """
        start = time.time()

        # Step 1: Select model
        model = self._select_model(context.intent)
        logger.info(f"Selected model: {model} for intent: {context.intent.value}")

        # Step 2: Construct prompt
        system_prompt = SYSTEM_PROMPTS.get(context.intent, SYSTEM_PROMPTS[QueryIntent.CONVERSATIONAL])
        user_prompt = self._build_user_prompt(context)

        # Step 3: Run inference
        try:
            resp = await self.http_client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": model,
                    "system": system_prompt,
                    "prompt": user_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7 if context.intent == QueryIntent.CREATIVE else 0.3,
                        "num_predict": 2048,
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()

            answer = data.get("response", "").strip()
            tokens_in = data.get("prompt_eval_count", 0)
            tokens_out = data.get("eval_count", 0)

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            answer = f"Inference error: {str(e)}"
            tokens_in = 0
            tokens_out = 0

        latency = int((time.time() - start) * 1000)

        # Step 4: Calculate confidence
        confidence = self._calculate_confidence(context, answer)

        logger.info(
            f"Reasoning complete: {tokens_out} tokens, {latency}ms, confidence: {confidence:.2f}"
        )

        return ReasoningResult(
            answer=answer,
            confidence=confidence,
            model_used=model,
            thinking_steps=[],  # Populated if deepseek-r1 exposes chain of thought
            sources_used=context.total_sources_found,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            inference_latency_ms=latency,
        )

    def _select_model(self, intent: QueryIntent) -> str:
        """Select model from the single source of truth map."""
        return MODEL_MAP.get(intent, "llama3.1:8b")

    def _build_user_prompt(self, context: ContextPackage) -> str:
        """Construct the full user prompt with context injection."""
        parts = []

        if context.assembled_context:
            parts.append(context.assembled_context)
            parts.append("")  # Blank line separator

        parts.append(f"QUESTION: {context.query}")

        if context.total_sources_found == 0:
            parts.append(
                "\nNOTE: No relevant context was found in the knowledge base. "
                "Answer based on your general knowledge, and clearly state that "
                "this answer is not grounded in Echo Brain's stored knowledge."
            )

        return "\n".join(parts)

    def _calculate_confidence(self, context: ContextPackage, answer: str) -> float:
        """
        Estimate confidence based on:
        - How many context sources were found
        - Average relevance scores
        - Whether the answer acknowledges uncertainty
        """
        if not answer or "error" in answer.lower():
            return 0.0

        # Base confidence from context availability
        if context.total_sources_found == 0:
            base = 0.3  # No grounding, pure model knowledge
        elif context.total_sources_found < 3:
            base = 0.5
        else:
            base = 0.7

        # Boost from high relevance scores
        if context.sources:
            avg_score = sum(s.relevance_score for s in context.sources) / len(context.sources)
            base = min(1.0, base + avg_score * 0.2)

        # Penalty for uncertainty language
        uncertainty_phrases = ["i don't know", "i'm not sure", "unclear", "i don't have"]
        if any(phrase in answer.lower() for phrase in uncertainty_phrases):
            base *= 0.7

        return round(base, 2)