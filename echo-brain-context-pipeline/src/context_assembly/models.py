"""
Context Assembly Pipeline - Data Models

These models define the core data structures used throughout the pipeline.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


class Domain(str, Enum):
    """Query domains for context isolation."""
    TECHNICAL = "technical"
    ANIME = "anime"
    PERSONAL = "personal"
    GENERAL = "general"


class FactType(str, Enum):
    """Types of extracted facts."""
    ENTITY = "entity"           # A thing that exists (person, service, tool)
    RELATIONSHIP = "relationship"  # How things relate to each other
    EVENT = "event"             # Something that happened
    PREFERENCE = "preference"   # User preferences/settings
    TECHNICAL = "technical"     # Technical facts (configs, versions, etc.)
    TEMPORAL = "temporal"       # Time-bound facts


class SourceType(str, Enum):
    """Types of ingested sources."""
    DOCUMENT = "document"
    CONVERSATION = "conversation"
    CODE = "code"
    EXTERNAL = "external"


# ============================================================================
# Ingestion Models
# ============================================================================

class IngestionRecord(BaseModel):
    """Tracks what has been ingested and its processing status."""
    id: UUID
    source_type: SourceType
    source_path: str
    source_hash: str  # SHA256 for deduplication
    
    # Ingestion status
    vector_id: Optional[UUID] = None
    vectorized_at: Optional[datetime] = None
    fact_extracted: bool = False
    fact_extracted_at: Optional[datetime] = None
    facts_count: int = 0
    
    # Metadata
    domain: Optional[Domain] = None
    chunk_count: Optional[int] = None
    token_count: Optional[int] = None
    
    # Audit
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Fact(BaseModel):
    """An extracted fact from source content."""
    id: UUID
    source_id: UUID
    
    # The fact itself
    fact_text: str
    fact_type: FactType
    confidence: float = 1.0
    
    # Domain isolation
    domain: Domain
    
    # Structured extraction (Subject-Predicate-Object triple)
    subject: Optional[str] = None      # Who/what the fact is about
    predicate: Optional[str] = None    # The relationship/action
    object: Optional[str] = None       # The target/value
    
    # Temporal bounds
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None  # NULL = still valid
    
    # Embedding for semantic search
    embedding: Optional[list[float]] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# Query Classification Models
# ============================================================================

class ClassificationResult(BaseModel):
    """Result of classifying a query into a domain."""
    query: str
    domain: Domain
    confidence: float
    reasoning: Optional[str] = None
    
    # Keywords that triggered the classification
    matched_keywords: list[str] = Field(default_factory=list)


# ============================================================================
# Retrieval Models
# ============================================================================

class VectorResult(BaseModel):
    """A result from Qdrant vector search."""
    id: UUID
    content: str
    score: float  # Similarity score
    domain: Domain
    source_type: SourceType
    source_path: str
    created_at: datetime
    
    # Metadata
    chunk_index: Optional[int] = None
    total_chunks: Optional[int] = None


class FactResult(BaseModel):
    """A retrieved fact."""
    id: UUID
    fact_text: str
    fact_type: FactType
    domain: Domain
    confidence: float
    relevance_score: float  # How relevant to the query
    
    # SPO triple if available
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object: Optional[str] = None


class ConversationTurn(BaseModel):
    """A single turn in a conversation."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime
    conversation_id: UUID


class CodeContext(BaseModel):
    """Retrieved code context."""
    file_path: str
    content: str
    language: str
    relevance_score: float
    
    # Code-specific metadata
    function_names: list[str] = Field(default_factory=list)
    class_names: list[str] = Field(default_factory=list)
    imports: list[str] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    """Combined results from all retrieval sources."""
    vectors: list[VectorResult] = Field(default_factory=list)
    facts: list[FactResult] = Field(default_factory=list)
    conversation_history: list[ConversationTurn] = Field(default_factory=list)
    code_context: list[CodeContext] = Field(default_factory=list)
    
    # Metadata
    retrieval_time_ms: int = 0
    total_results: int = 0
    
    def __post_init__(self):
        self.total_results = (
            len(self.vectors) + 
            len(self.facts) + 
            len(self.conversation_history) + 
            len(self.code_context)
        )


# ============================================================================
# Context Assembly Models
# ============================================================================

class TokenBudget(BaseModel):
    """Token allocation for context assembly."""
    total: int = 8192
    system_prompt: int = 500
    facts: int = 1500
    conversation: int = 2000
    code_context: int = 2000
    query_buffer: int = 2192
    
    @property
    def available_for_content(self) -> int:
        return self.total - self.system_prompt - self.query_buffer


class AssembledContext(BaseModel):
    """The final assembled context ready for LLM inference."""
    # Core components
    system_prompt: str
    facts: list[FactResult]
    conversation_history: list[ConversationTurn]
    code_context: list[CodeContext]
    
    # Metadata
    domain: Domain
    classification_confidence: float
    token_count: int
    assembly_time_ms: int
    
    # For debugging/logging
    query: str
    vector_ids_used: list[UUID] = Field(default_factory=list)
    fact_ids_used: list[UUID] = Field(default_factory=list)
    
    def to_prompt_components(self) -> dict:
        """Convert to components ready for LLM."""
        return {
            "system": self.system_prompt,
            "context": self._format_context(),
            "metadata": {
                "domain": self.domain.value,
                "confidence": self.classification_confidence,
                "tokens": self.token_count
            }
        }
    
    def _format_context(self) -> str:
        """Format all context into a single string."""
        parts = []
        
        if self.facts:
            facts_text = "\n".join(f"- {f.fact_text}" for f in self.facts)
            parts.append(f"## Relevant Facts\n{facts_text}")
        
        if self.code_context:
            code_text = "\n\n".join(
                f"### {c.file_path}\n```{c.language}\n{c.content}\n```"
                for c in self.code_context
            )
            parts.append(f"## Code Context\n{code_text}")
        
        if self.conversation_history:
            conv_text = "\n".join(
                f"{t.role}: {t.content}" 
                for t in self.conversation_history
            )
            parts.append(f"## Recent Conversation\n{conv_text}")
        
        return "\n\n".join(parts)


# ============================================================================
# Logging & Metrics Models
# ============================================================================

class ContextAssemblyLog(BaseModel):
    """Log entry for context assembly operations."""
    id: UUID
    
    # Query info
    query_text: str
    classified_domain: Domain
    
    # What was retrieved
    vectors_retrieved: int
    facts_retrieved: int
    conversation_turns: int
    
    # Assembly metrics
    total_tokens: int
    assembly_time_ms: int
    
    # For debugging
    retrieved_vector_ids: list[UUID]
    retrieved_fact_ids: list[UUID]
    
    created_at: datetime = Field(default_factory=datetime.utcnow)


class CoverageReport(BaseModel):
    """Report on ingestion and fact extraction coverage."""
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Overall stats
    total_vectors: int
    vectors_with_facts: int
    overall_coverage_pct: float
    
    # By domain
    domain_coverage: dict[str, dict] = Field(default_factory=dict)
    # Structure: {"technical": {"total": 100, "with_facts": 50, "pct": 50.0}, ...}
    
    # By source type
    source_type_coverage: dict[str, dict] = Field(default_factory=dict)
    # Structure: {"document": {"total": 435, "with_facts": 100, "pct": 23.0}, ...}
    
    # Gaps
    sources_missing_facts: int
    oldest_unprocessed: Optional[datetime] = None


class IngestionGap(BaseModel):
    """Represents a gap in ingestion coverage."""
    source_path: str
    source_type: SourceType
    reason: str  # "not_tracked", "not_vectorized", "facts_not_extracted"
    discovered_at: datetime = Field(default_factory=datetime.utcnow)
