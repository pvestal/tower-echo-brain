"""
LAYER 2: Reasoning & Inference

This layer answers: "Given this context, what's the best answer?"

Selects the appropriate model, constructs an enriched prompt,
runs inference, and returns structured results.
"""
import re
import time
import logging
import httpx
from typing import List, Tuple

from .models import ContextPackage, ReasoningResult, QueryIntent

logger = logging.getLogger("echo.pipeline.reasoning")

OLLAMA_URL = "http://localhost:11434"

# Model selection - USING ACTUAL AVAILABLE MODELS (AMD RX 9070 XT 16GB via Vulkan)
# Available: mistral:7b, deepseek-r1:8b, llama3.2:3b, gemma2:9b, qwen2.5-coder:7b, gemma3:12b
MODEL_MAP = {
    QueryIntent.CODING: "qwen2.5-coder:7b",      # Specialized for code
    QueryIntent.REASONING: "deepseek-r1:8b",      # Good at step-by-step reasoning
    QueryIntent.PERSONAL: "gemma3:12b",           # Strong general model
    QueryIntent.FACTUAL: "mistral:7b",            # Fast and accurate
    QueryIntent.CREATIVE: "gemma2:9b",            # Creative tasks
    QueryIntent.CONVERSATIONAL: "gemma3:12b",     # General conversation
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

    # Fallback chain: if the preferred model isn't loaded, try these in order
    FALLBACK_CHAIN = ["mistral:7b", "gemma3:12b", "gemma2:9b"]

    def __init__(self):
        self.http_client: httpx.AsyncClient = None
        self._available_models: set = set()
        self._last_model_check: float = 0

    async def initialize(self):
        self.http_client = httpx.AsyncClient(timeout=120.0)  # Reasoning can be slow
        await self._refresh_available_models()

    async def _refresh_available_models(self):
        """Check which models Ollama currently has loaded/available."""
        try:
            resp = await self.http_client.get(f"{OLLAMA_URL}/api/tags")
            models = resp.json().get("models", [])
            self._available_models = set()
            for m in models:
                name = m["name"]
                self._available_models.add(name)
                # Also add without :latest suffix for matching
                if name.endswith(":latest"):
                    self._available_models.add(name.replace(":latest", ""))

            self._last_model_check = time.time()

            for intent, model in MODEL_MAP.items():
                if self._is_model_available(model):
                    logger.info(f"Model verified: {model} -> {intent.value}")
                else:
                    logger.warning(f"Model {model} for {intent.value} NOT available!")
        except Exception as e:
            logger.error(f"Failed to verify Ollama models: {e}")

    def _is_model_available(self, model: str) -> bool:
        """Check if a model is in the available set (handles :latest variants)."""
        return (model in self._available_models or
                f"{model}:latest" in self._available_models or
                any(model in m for m in self._available_models))

    async def shutdown(self):
        if self.http_client:
            await self.http_client.aclose()

    async def reason(self, context: ContextPackage) -> ReasoningResult:
        """
        Main entry point. Takes context package, returns reasoned answer.
        """
        start = time.time()

        # Refresh model list if stale (every 5 minutes)
        if time.time() - self._last_model_check > 300:
            await self._refresh_available_models()

        # Step 1: Select model with fallback
        model = await self._select_model_with_fallback(context.intent)
        logger.info(f"Selected model: {model} for intent: {context.intent.value}")

        # Step 2: Construct prompt
        system_prompt = SYSTEM_PROMPTS.get(context.intent, SYSTEM_PROMPTS[QueryIntent.CONVERSATIONAL])
        user_prompt = self._build_user_prompt(context)

        # Step 3: Run inference
        answer, tokens_in, tokens_out, thinking_steps = await self._run_inference(
            model, system_prompt, user_prompt, context.intent
        )

        latency = int((time.time() - start) * 1000)

        # Step 4: Calculate confidence
        confidence = self._calculate_confidence(context, answer)

        if thinking_steps:
            logger.info(f"Extracted {len(thinking_steps)} thinking steps from {model}")

        logger.info(
            f"Reasoning complete: {tokens_out} tokens, {latency}ms, confidence: {confidence:.2f}"
        )

        return ReasoningResult(
            answer=answer,
            confidence=confidence,
            model_used=model,
            thinking_steps=thinking_steps,
            sources_used=context.total_sources_found,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            inference_latency_ms=latency,
        )

    async def _run_inference(
        self, model: str, system_prompt: str, user_prompt: str, intent: QueryIntent
    ) -> Tuple[str, int, int, List[str]]:
        """Run LLM inference and extract chain-of-thought if present."""
        try:
            resp = await self.http_client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": model,
                    "system": system_prompt,
                    "prompt": user_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7 if intent == QueryIntent.CREATIVE else 0.3,
                        "num_predict": 2048,
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()

            raw_response = data.get("response", "").strip()
            tokens_in = data.get("prompt_eval_count", 0)
            tokens_out = data.get("eval_count", 0)

            # Extract chain-of-thought from <think> blocks (deepseek-r1)
            answer, thinking_steps = self._extract_thinking(raw_response)

            return answer, tokens_in, tokens_out, thinking_steps

        except Exception as e:
            logger.error(f"Inference failed with {model}: {e}")
            return f"Inference error: {str(e)}", 0, 0, []

    def _extract_thinking(self, raw_response: str) -> Tuple[str, List[str]]:
        """Extract <think>...</think> blocks from model output (deepseek-r1).
        Returns (clean_answer, thinking_steps)."""
        thinking_steps = []

        # Match all <think>...</think> blocks
        think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
        matches = think_pattern.findall(raw_response)

        if matches:
            for block in matches:
                # Split into individual reasoning steps
                steps = [s.strip() for s in block.strip().split('\n') if s.strip()]
                thinking_steps.extend(steps)

            # Remove <think> blocks from the final answer
            answer = think_pattern.sub('', raw_response).strip()
        else:
            answer = raw_response

        return answer, thinking_steps

    # Map QueryIntent enum values to agent registry intent strings
    _INTENT_TO_REGISTRY = {
        QueryIntent.CODING: "code_query",
        QueryIntent.REASONING: "self_introspection",
        QueryIntent.PERSONAL: "general_knowledge",
        QueryIntent.FACTUAL: "general_knowledge",
        QueryIntent.CREATIVE: "anime_production",
        QueryIntent.CONVERSATIONAL: "general_knowledge",
    }

    async def _select_model_with_fallback(self, intent: QueryIntent) -> str:
        """Select model with fallback chain if preferred model isn't available.
        Tries the agent registry first, then falls back to MODULE_MAP."""
        # Try agent registry first
        try:
            from src.core.agent_registry import get_agent_registry
            registry = get_agent_registry()
            registry_intent = self._INTENT_TO_REGISTRY.get(intent, "general_knowledge")
            agent = registry.select(registry_intent)
            model = await registry.resolve_model(agent)
            logger.info(f"Registry selected model {model} for {intent.value} via agent '{agent.name}'")
            return model
        except Exception as e:
            logger.warning(f"Agent registry unavailable, using MODEL_MAP fallback: {e}")

        # Original fallback logic
        preferred = MODEL_MAP.get(intent, "llama3.1:8b")

        if self._is_model_available(preferred):
            return preferred

        # Try fallback chain
        for fallback in self.FALLBACK_CHAIN:
            if self._is_model_available(fallback):
                logger.warning(
                    f"Model {preferred} unavailable for {intent.value}, "
                    f"falling back to {fallback}"
                )
                return fallback

        # Last resort: return preferred and let Ollama handle the error
        logger.error(f"No models available! Trying {preferred} anyway.")
        return preferred

    # Confidence gate: if best retrieval score is below this, warn the LLM
    RETRIEVAL_CONFIDENCE_THRESHOLD = 0.35

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
        elif context.sources:
            # Confidence gate: check if best retrieval score is too low
            best_score = max(s.relevance_score for s in context.sources)
            if best_score < self.RETRIEVAL_CONFIDENCE_THRESHOLD:
                parts.append(
                    "\nIMPORTANT: The retrieved context has low relevance scores "
                    f"(best: {best_score:.2f}). If the context does not clearly "
                    "answer the question, say 'I don't have specific information "
                    "about that in my knowledge base' rather than guessing."
                )

            # Conflict detection: check for contradictory facts in sources
            self._check_for_conflicts(context, parts)

        return "\n".join(parts)

    def _check_for_conflicts(self, context: ContextPackage, parts: list) -> None:
        """Check for contradictory information in context sources and warn the LLM."""
        facts = [s for s in context.sources if s.source_type == "facts_table"]
        if len(facts) < 2:
            return

        # Group by subject (from metadata)
        groups = {}
        for f in facts:
            subj = f.metadata.get("subject", "").lower()
            pred = f.metadata.get("predicate", "").lower()
            if subj and pred:
                key = f"{subj}|{pred}"
                groups.setdefault(key, []).append(f)

        for key, group in groups.items():
            objects = set(f.metadata.get("object", "") for f in group if f.metadata.get("object"))
            if len(objects) > 1:
                subj, pred = key.split("|", 1)
                parts.append(
                    f"\nWARNING: Conflicting information found for '{subj} {pred}': "
                    f"{', '.join(repr(o) for o in objects)}. "
                    "Prefer the fact with higher confidence or more recent timestamp."
                )

    def _calculate_confidence(self, context: ContextPackage, answer: str) -> float:
        """
        Estimate confidence based on:
        - How many context sources were found
        - Best and average relevance scores
        - Whether the answer acknowledges uncertainty
        - Whether retrieval scores passed the confidence gate
        """
        if not answer or "error" in answer.lower():
            return 0.0

        # Base confidence from context availability
        if context.total_sources_found == 0:
            base = 0.2  # No grounding — very low confidence
        elif context.total_sources_found < 3:
            base = 0.4
        else:
            base = 0.6

        # Adjust based on relevance scores
        if context.sources:
            best_score = max(s.relevance_score for s in context.sources)
            avg_score = sum(s.relevance_score for s in context.sources) / len(context.sources)

            # Strong retrieval boost
            if best_score >= 0.7:
                base += 0.25
            elif best_score >= self.RETRIEVAL_CONFIDENCE_THRESHOLD:
                base += 0.15
            else:
                # Below confidence gate — penalize
                base -= 0.1

            # Average score contribution
            base += avg_score * 0.1

        # Penalty for uncertainty language (model self-assessed low confidence)
        uncertainty_phrases = ["i don't know", "i'm not sure", "unclear", "i don't have"]
        if any(phrase in answer.lower() for phrase in uncertainty_phrases):
            base *= 0.6

        return round(min(1.0, max(0.0, base)), 2)