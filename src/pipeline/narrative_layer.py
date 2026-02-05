"""
LAYER 3: Narrative Formation

This layer answers: "How should this response be presented to the user?"

Takes the reasoning output and formats it appropriately for the query type.
Adds source citations, adjusts style, performs basic quality checking.
"""
import logging
from .models import ContextPackage, ReasoningResult, NarrativeResponse, QueryIntent

logger = logging.getLogger("echo.pipeline.narrative")


class NarrativeLayer:
    """
    Response formatting and quality assurance.

    Usage:
        layer = NarrativeLayer()
        response = layer.format(context_package, reasoning_result)
    """

    def format(
        self, context: ContextPackage, reasoning: ReasoningResult
    ) -> NarrativeResponse:
        """Format reasoning result into a user-facing response."""

        # Step 1: Apply style based on intent
        styled = self._apply_style(context.intent, reasoning.answer)

        # Step 2: Add source citations if grounded
        citations = self._build_citations(context)

        # Step 3: Quality check
        styled = self._quality_check(context.query, styled, context.intent)

        # Step 4: Build metadata
        metadata = {
            "intent": context.intent.value,
            "context_sources": context.total_sources_found,
            "reasoning_model": reasoning.model_used,
            "context_retrieval_ms": context.retrieval_latency_ms,
            "reasoning_ms": reasoning.inference_latency_ms,
            "total_latency_ms": context.retrieval_latency_ms + reasoning.inference_latency_ms,
            "tokens_in": reasoning.tokens_in,
            "tokens_out": reasoning.tokens_out,
            "confidence": reasoning.confidence,
        }

        logger.info(
            f"Narrative formatted: style={context.intent.value}, "
            f"confidence={reasoning.confidence:.2f}, "
            f"total_latency={metadata['total_latency_ms']}ms"
        )

        return NarrativeResponse(
            response=styled,
            style=context.intent.value,
            sources_cited=citations,
            confidence=reasoning.confidence,
            pipeline_metadata=metadata,
        )

    def _apply_style(self, intent: QueryIntent, answer: str) -> str:
        """Minimal style adjustments. Don't over-process the LLM output."""
        # The LLM was already instructed on style via system prompt.
        # This layer does light post-processing only.

        if intent == QueryIntent.CODING:
            # Ensure code blocks are properly formatted
            return answer

        if intent == QueryIntent.PERSONAL:
            # Strip any robotic prefixes the model might add
            prefixes_to_strip = [
                "Based on the provided context, ",
                "According to the context, ",
                "From the information available, ",
            ]
            for prefix in prefixes_to_strip:
                if answer.startswith(prefix):
                    answer = answer[len(prefix):]
                    answer = answer[0].upper() + answer[1:]  # Re-capitalize
                    break
            return answer

        return answer

    def _build_citations(self, context: ContextPackage) -> list:
        """Build source citation list."""
        citations = []
        for s in context.sources[:5]:  # Max 5 citations
            if s.relevance_score > 0.3:
                citations.append(f"{s.source_type}/{s.collection} (score: {s.relevance_score:.2f})")
        return citations

    def _quality_check(self, query: str, response: str, intent: QueryIntent) -> str:
        """
        Basic quality verification.
        Does NOT use an LLM - just sanity checks.
        """
        # Check: response isn't empty
        if not response or len(response.strip()) < 10:
            return (
                "I wasn't able to generate a meaningful response for this query. "
                "This might indicate a model inference issue."
            )

        # Check: response isn't just echoing the query
        if query.lower().strip() in response.lower() and len(response) < len(query) * 2:
            logger.warning("Response appears to just echo the query")

        # Check: response isn't contaminated with anime content for non-creative queries
        if intent not in (QueryIntent.CREATIVE,):
            contamination_signals = ["goblin slayer", "tokyo debt", "cyber goblin"]
            for signal in contamination_signals:
                if signal in response.lower() and signal not in query.lower():
                    logger.error(f"CONTAMINATION DETECTED: '{signal}' in {intent.value} response")
                    # Strip it but log the issue loudly
                    response = response  # Don't silently modify - the fix needs to happen upstream

        return response