"""
Pydantic models defining exact data contracts between pipeline layers.
Every field is typed. No Any. No Optional without default.
If a layer produces it, the model defines it.
"""
from pydantic import BaseModel, Field
from enum import Enum
from typing import List
from datetime import datetime


class QueryIntent(str, Enum):
    CODING = "coding"
    REASONING = "reasoning"
    PERSONAL = "personal"       # About Patrick, his projects, preferences
    FACTUAL = "factual"         # Lookup-style questions
    CREATIVE = "creative"       # Narration, storytelling, anime
    CONVERSATIONAL = "conversational"  # General chat


class ContextSource(BaseModel):
    """A single piece of retrieved context with provenance."""
    text: str
    source_type: str            # "qdrant_vectors", "postgresql_fts", "facts_table"
    collection: str = ""        # Which Qdrant collection or PG table
    relevance_score: float = 0.0
    metadata: dict = Field(default_factory=dict)


class ContextPackage(BaseModel):
    """Output of Layer 1. Everything the reasoning layer needs."""
    query: str
    intent: QueryIntent
    sources: List[ContextSource] = Field(default_factory=list)
    assembled_context: str = ""  # Formatted context string for prompt injection
    total_sources_searched: int = 0
    total_sources_found: int = 0
    context_token_estimate: int = 0
    retrieval_latency_ms: int = 0


class ReasoningResult(BaseModel):
    """Output of Layer 2. The raw analytical answer."""
    answer: str
    confidence: float = 0.0      # 0.0-1.0, based on context availability
    model_used: str = ""
    thinking_steps: List[str] = Field(default_factory=list)
    sources_used: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    inference_latency_ms: int = 0


class NarrativeResponse(BaseModel):
    """Output of Layer 3. What the user actually sees."""
    response: str
    style: str = ""
    sources_cited: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    pipeline_metadata: dict = Field(default_factory=dict)


class PipelineResult(BaseModel):
    """Complete pipeline output with full observability."""
    response: str                 # The final user-facing response
    intent: QueryIntent
    confidence: float
    context_sources_found: int
    reasoning_model: str
    total_latency_ms: int
    # Debug info - exposed via API flag
    debug: dict = Field(default_factory=dict)