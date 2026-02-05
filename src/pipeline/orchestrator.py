"""
ORCHESTRATOR: Wires Context → Reasoning → Narrative into a single call.

This is the ONLY entry point for processing queries through the pipeline.
No query should bypass this. If it does, it's a bug.
"""
import time
import logging

from .models import PipelineResult
from .context_layer import ContextLayer
from .reasoning_layer import ReasoningLayer
from .narrative_layer import NarrativeLayer

logger = logging.getLogger("echo.pipeline")


class EchoBrainPipeline:
    """
    The complete query processing pipeline.

    Usage:
        pipeline = EchoBrainPipeline()
        await pipeline.initialize()
        result = await pipeline.process("What truck does Patrick drive?")
    """

    def __init__(self):
        self.context = ContextLayer()
        self.reasoning = ReasoningLayer()
        self.narrative = NarrativeLayer()

    async def initialize(self):
        """Initialize all layers. Call once at startup."""
        await self.context.initialize()
        await self.reasoning.initialize()
        logger.info("EchoBrainPipeline fully initialized - all 3 layers ready")

    async def shutdown(self):
        """Clean shutdown of all layers."""
        await self.context.shutdown()
        await self.reasoning.shutdown()

    async def process(self, query: str, debug: bool = False) -> PipelineResult:
        """
        Process a query through all three layers.

        Args:
            query: The user's question
            debug: If True, include full debug info in response

        Returns:
            PipelineResult with the final response and metadata
        """
        start = time.time()
        logger.info(f"Pipeline processing: {query[:100]}")

        # LAYER 1: Context Retrieval
        context_package = await self.context.retrieve(query)
        logger.info(
            f"L1 Context: intent={context_package.intent.value}, "
            f"sources={context_package.total_sources_found}"
        )

        # LAYER 2: Reasoning
        reasoning_result = await self.reasoning.reason(context_package)
        logger.info(
            f"L2 Reasoning: model={reasoning_result.model_used}, "
            f"confidence={reasoning_result.confidence:.2f}, "
            f"tokens={reasoning_result.tokens_out}"
        )

        # LAYER 3: Narrative
        narrative = self.narrative.format(context_package, reasoning_result)
        logger.info(
            f"L3 Narrative: style={narrative.style}, "
            f"citations={len(narrative.sources_cited)}"
        )

        total_latency = int((time.time() - start) * 1000)

        # Build debug info if requested
        debug_info = {}
        if debug:
            debug_info = {
                "context": {
                    "intent": context_package.intent.value,
                    "sources_searched": context_package.total_sources_searched,
                    "sources_found": context_package.total_sources_found,
                    "retrieval_ms": context_package.retrieval_latency_ms,
                    "context_preview": context_package.assembled_context[:500] if context_package.assembled_context else "",
                    "source_details": [
                        {
                            "type": s.source_type,
                            "collection": s.collection,
                            "score": s.relevance_score,
                            "text_preview": s.text[:200],
                        }
                        for s in context_package.sources[:5]
                    ],
                },
                "reasoning": {
                    "model": reasoning_result.model_used,
                    "confidence": reasoning_result.confidence,
                    "tokens_in": reasoning_result.tokens_in,
                    "tokens_out": reasoning_result.tokens_out,
                    "inference_ms": reasoning_result.inference_latency_ms,
                },
                "narrative": {
                    "style": narrative.style,
                    "citations": narrative.sources_cited,
                },
            }

        result = PipelineResult(
            response=narrative.response,
            intent=context_package.intent,
            confidence=reasoning_result.confidence,
            context_sources_found=context_package.total_sources_found,
            reasoning_model=reasoning_result.model_used,
            total_latency_ms=total_latency,
            debug=debug_info,
        )

        logger.info(
            f"Pipeline complete: {total_latency}ms total, "
            f"confidence={result.confidence:.2f}"
        )

        return result