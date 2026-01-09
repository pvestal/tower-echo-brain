#!/usr/bin/env python3
"""
Enhanced Memory Models with Temporal Confidence Management
Extends existing Echo Brain memory system with confidence decay capabilities
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import asyncio
import logging

# Import existing memory and epistemic models
from src.echo_vector_memory import VectorMemory, EchoWithMemory
from src.models.epistemic_models import (
    EpistemicStatus,
    EpistemicKnowledgeType,
    ConfidenceCalibration
)
from src.services.confidence_decay import (
    MemoryWithDecay,
    MemoryCategory,
    DecayConfiguration,
    DecayAlgorithm,
    ConfidenceDecayService
)

logger = logging.getLogger(__name__)

class MemoryType(Enum):
    """Enhanced memory type classification"""
    CONVERSATION = "conversation"       # Chat interactions
    SYSTEM_STATUS = "system_status"     # Service health, configurations
    USER_PREFERENCE = "user_preference" # Personal settings, preferences
    FACTUAL_KNOWLEDGE = "factual"       # General knowledge
    PROCEDURAL = "procedural"           # How-to information
    EPISODIC = "episodic"              # Specific events, contexts
    SEMANTIC = "semantic"               # Abstract concepts, relationships

@dataclass
class MemoryEntry:
    """Enhanced memory entry with decay-aware confidence management"""
    # Core memory data
    id: Optional[str] = None
    content: str = ""
    embedding: Optional[List[float]] = None

    # Enhanced classification
    memory_type: MemoryType = MemoryType.CONVERSATION
    category: MemoryCategory = MemoryCategory.FACTUAL
    importance: float = 0.5  # 0.0 to 1.0

    # Temporal confidence management
    original_confidence: float = 0.5
    current_confidence: float = 0.5
    decay_algorithm: DecayAlgorithm = DecayAlgorithm.IMPORTANCE_BASED

    # Memory metadata
    source: str = "conversation"
    source_reliability: float = 0.5
    tags: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

    # Temporal tracking
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    last_verified: datetime = field(default_factory=datetime.utcnow)
    last_decay_calculation: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    verification_count: int = 0

    # Decay management
    decay_paused: bool = False
    custom_decay_rate: Optional[float] = None
    verification_history: List[Dict[str, Any]] = field(default_factory=list)

    # Cross-references
    epistemic_status: Optional[EpistemicStatus] = None
    qdrant_id: Optional[str] = None
    interaction_id: Optional[int] = None

    def get_current_confidence(self, config: DecayConfiguration) -> float:
        """Calculate current confidence considering decay"""
        if self.decay_paused:
            return self.current_confidence

        time_elapsed = datetime.utcnow() - self.last_decay_calculation
        days_elapsed = time_elapsed.total_seconds() / (24 * 3600)

        if days_elapsed <= 0:
            return self.current_confidence

        # Create temporary MemoryWithDecay for calculation
        temp_memory = MemoryWithDecay(
            epistemic_status=self.epistemic_status,
            category=self.category,
            importance_score=self.importance,
            current_confidence=self.current_confidence,
            decay_algorithm=self.decay_algorithm,
            custom_decay_rate=self.custom_decay_rate,
            last_decay_calculation=self.last_decay_calculation,
            decay_paused=self.decay_paused
        )

        return temp_memory.get_current_confidence(config)

    def should_reverify(self, config: DecayConfiguration) -> bool:
        """Check if memory needs reverification"""
        current_conf = self.get_current_confidence(config)
        return current_conf <= config.reverification_threshold

    def calculate_relevance_score(self, query: str, base_similarity: float) -> float:
        """Calculate relevance considering confidence decay and access patterns"""
        # Base relevance from vector similarity
        relevance = base_similarity

        # Confidence multiplier (low confidence reduces relevance)
        confidence_multiplier = max(0.1, self.current_confidence)
        relevance *= confidence_multiplier

        # Recency bonus (recently accessed memories are more relevant)
        time_since_access = datetime.utcnow() - self.last_accessed
        days_since_access = time_since_access.total_seconds() / (24 * 3600)
        recency_bonus = max(0.5, 1.0 - (days_since_access * 0.1))
        relevance *= recency_bonus

        # Access frequency bonus
        frequency_bonus = min(2.0, 1.0 + (self.access_count * 0.05))
        relevance *= frequency_bonus

        # Importance scaling
        importance_multiplier = 0.5 + (self.importance * 0.5)
        relevance *= importance_multiplier

        return min(1.0, relevance)

    def mark_accessed(self) -> None:
        """Mark memory as accessed for relevance calculation"""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1

    def mark_verified(self, source: str, method: str = "", confidence_boost: float = 0.2) -> None:
        """Mark memory as verified and boost confidence"""
        old_confidence = self.current_confidence
        self.current_confidence = min(1.0, self.current_confidence + confidence_boost)
        self.last_verified = datetime.utcnow()
        self.verification_count += 1

        # Record verification
        verification_record = {
            "timestamp": self.last_verified.isoformat(),
            "source": source,
            "method": method,
            "confidence_before": old_confidence,
            "confidence_after": self.current_confidence
        }
        self.verification_history.append(verification_record)

        # Update epistemic status if available
        if self.epistemic_status:
            self.epistemic_status.mark_verified(f"[{method}] {source}")

    def update_decay_calculation(self, config: DecayConfiguration) -> float:
        """Update current confidence and return new value"""
        self.current_confidence = self.get_current_confidence(config)
        self.last_decay_calculation = datetime.utcnow()
        return self.current_confidence

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "category": self.category.value,
            "importance": self.importance,
            "original_confidence": self.original_confidence,
            "current_confidence": self.current_confidence,
            "decay_algorithm": self.decay_algorithm.value,
            "source": self.source,
            "source_reliability": self.source_reliability,
            "tags": self.tags,
            "context": self.context,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "last_verified": self.last_verified.isoformat(),
            "last_decay_calculation": self.last_decay_calculation.isoformat(),
            "access_count": self.access_count,
            "verification_count": self.verification_count,
            "decay_paused": self.decay_paused,
            "custom_decay_rate": self.custom_decay_rate,
            "verification_history": self.verification_history,
            "qdrant_id": self.qdrant_id,
            "interaction_id": self.interaction_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """Create from dictionary"""
        memory = cls()
        memory.id = data.get("id")
        memory.content = data.get("content", "")
        memory.memory_type = MemoryType(data.get("memory_type", "conversation"))
        memory.category = MemoryCategory(data.get("category", "FACTUAL"))
        memory.importance = data.get("importance", 0.5)
        memory.original_confidence = data.get("original_confidence", 0.5)
        memory.current_confidence = data.get("current_confidence", 0.5)
        memory.decay_algorithm = DecayAlgorithm(data.get("decay_algorithm", "importance_based"))
        memory.source = data.get("source", "conversation")
        memory.source_reliability = data.get("source_reliability", 0.5)
        memory.tags = data.get("tags", [])
        memory.context = data.get("context", {})

        # Parse timestamps
        if data.get("created_at"):
            memory.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("last_accessed"):
            memory.last_accessed = datetime.fromisoformat(data["last_accessed"])
        if data.get("last_verified"):
            memory.last_verified = datetime.fromisoformat(data["last_verified"])
        if data.get("last_decay_calculation"):
            memory.last_decay_calculation = datetime.fromisoformat(data["last_decay_calculation"])

        memory.access_count = data.get("access_count", 0)
        memory.verification_count = data.get("verification_count", 0)
        memory.decay_paused = data.get("decay_paused", False)
        memory.custom_decay_rate = data.get("custom_decay_rate")
        memory.verification_history = data.get("verification_history", [])
        memory.qdrant_id = data.get("qdrant_id")
        memory.interaction_id = data.get("interaction_id")

        return memory

class TemporalVectorMemory(VectorMemory):
    """Enhanced vector memory system with temporal confidence management"""

    def __init__(self, decay_service: Optional[ConfidenceDecayService] = None):
        super().__init__()
        self.decay_service = decay_service
        self.config = DecayConfiguration() if not decay_service else decay_service.config
        self.memory_entries: Dict[str, MemoryEntry] = {}

    async def remember(self, text: str, metadata: Optional[Dict] = None,
                      memory_type: MemoryType = MemoryType.CONVERSATION,
                      importance: float = 0.5) -> bool:
        """
        Enhanced memory storage with classification and confidence tracking

        Args:
            text: The text to remember
            metadata: Optional metadata
            memory_type: Type classification for the memory
            importance: Importance score for decay calculation
        """
        # Generate embedding using parent method
        embedding = await self._generate_embedding(text)
        if not embedding:
            return False

        # Create enhanced memory entry
        memory_entry = MemoryEntry(
            content=text,
            embedding=embedding,
            memory_type=memory_type,
            category=self._infer_category(text, metadata),
            importance=importance,
            source=metadata.get("source", "conversation") if metadata else "conversation",
            tags=metadata.get("tags", []) if metadata else [],
            context=metadata or {}
        )

        # Classify importance and adjust confidence
        memory_entry.original_confidence = self._calculate_initial_confidence(text, metadata)
        memory_entry.current_confidence = memory_entry.original_confidence

        # Generate unique ID
        memory_id = abs(hash(text + str(datetime.utcnow()))) % (2**31)
        memory_entry.id = str(memory_id)
        memory_entry.qdrant_id = str(memory_id)

        # Store in Qdrant using parent method
        success = await super().remember(text, metadata)

        if success:
            # Store enhanced memory entry
            self.memory_entries[memory_entry.id] = memory_entry

            # Create epistemic status if decay service available
            if self.decay_service:
                epistemic_status = self._create_epistemic_status(memory_entry)
                memory_entry.epistemic_status = epistemic_status

            logger.info(f"Stored enhanced memory: {text[:50]}... (confidence: {memory_entry.current_confidence:.2f})")
            return True

        return False

    async def recall(self, query: str, limit: int = 5,
                    min_confidence: float = 0.3) -> List[Dict]:
        """
        Enhanced recall with confidence filtering and temporal relevance

        Args:
            query: Search query
            limit: Maximum results
            min_confidence: Minimum confidence threshold
        """
        # Get base results from parent
        base_results = await super().recall(query, limit * 2)  # Get more for filtering

        enhanced_results = []
        for result in base_results:
            memory_id = str(result.get("id", ""))

            # Get enhanced memory entry
            memory_entry = self.memory_entries.get(memory_id)
            if not memory_entry:
                # Create basic entry for legacy memories
                memory_entry = self._create_legacy_memory_entry(result)
                self.memory_entries[memory_id] = memory_entry

            # Update confidence with decay calculation
            current_confidence = memory_entry.get_current_confidence(self.config)

            # Filter by minimum confidence
            if current_confidence < min_confidence:
                continue

            # Calculate enhanced relevance score
            base_similarity = result.get("score", 0)
            relevance_score = memory_entry.calculate_relevance_score(query, base_similarity)

            # Mark as accessed
            memory_entry.mark_accessed()

            # Enhance result with temporal data
            enhanced_result = result.copy()
            enhanced_result.update({
                "current_confidence": current_confidence,
                "relevance_score": relevance_score,
                "memory_type": memory_entry.memory_type.value,
                "importance": memory_entry.importance,
                "access_count": memory_entry.access_count,
                "days_since_verification": (datetime.utcnow() - memory_entry.last_verified).total_seconds() / (24 * 3600),
                "needs_verification": memory_entry.should_reverify(self.config),
                "verification_count": memory_entry.verification_count
            })

            enhanced_results.append(enhanced_result)

        # Sort by relevance score
        enhanced_results.sort(key=lambda x: x["relevance_score"], reverse=True)

        return enhanced_results[:limit]

    async def verify_memory(self, memory_id: str, verification_source: str,
                           method: str = "") -> bool:
        """Verify a specific memory and boost its confidence"""
        memory_entry = self.memory_entries.get(memory_id)
        if not memory_entry:
            return False

        memory_entry.mark_verified(verification_source, method)

        # Update via decay service if available
        if self.decay_service and memory_entry.epistemic_status:
            await self.decay_service.verify_memory(
                memory_entry.epistemic_status.id,
                verification_source,
                method
            )

        logger.info(f"Verified memory {memory_id}: confidence now {memory_entry.current_confidence:.2f}")
        return True

    async def get_memories_needing_verification(self, limit: int = 20) -> List[Dict]:
        """Get memories that need verification due to low confidence"""
        verification_candidates = []

        for memory_entry in self.memory_entries.values():
            if memory_entry.should_reverify(self.config):
                verification_candidates.append({
                    "id": memory_entry.id,
                    "content": memory_entry.content[:200] + "...",
                    "current_confidence": memory_entry.get_current_confidence(self.config),
                    "days_since_verification": (datetime.utcnow() - memory_entry.last_verified).total_seconds() / (24 * 3600),
                    "verification_count": memory_entry.verification_count,
                    "importance": memory_entry.importance,
                    "memory_type": memory_entry.memory_type.value
                })

        # Sort by lowest confidence first
        verification_candidates.sort(key=lambda x: x["current_confidence"])

        return verification_candidates[:limit]

    async def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory system statistics"""
        if not self.memory_entries:
            return {"error": "No memory entries available"}

        # Calculate confidence statistics
        confidences = [m.get_current_confidence(self.config) for m in self.memory_entries.values()]
        avg_confidence = sum(confidences) / len(confidences)
        low_confidence_count = sum(1 for c in confidences if c < self.config.reverification_threshold)

        # Memory type distribution
        type_distribution = {}
        for memory in self.memory_entries.values():
            mem_type = memory.memory_type.value
            type_distribution[mem_type] = type_distribution.get(mem_type, 0) + 1

        # Category distribution
        category_distribution = {}
        for memory in self.memory_entries.values():
            category = memory.category.value
            category_distribution[category] = category_distribution.get(category, 0) + 1

        # Access patterns
        total_accesses = sum(m.access_count for m in self.memory_entries.values())
        avg_access_count = total_accesses / len(self.memory_entries)

        return {
            "total_memories": len(self.memory_entries),
            "average_confidence": avg_confidence,
            "low_confidence_count": low_confidence_count,
            "verification_needed_percentage": (low_confidence_count / len(self.memory_entries)) * 100,
            "memory_type_distribution": type_distribution,
            "category_distribution": category_distribution,
            "total_accesses": total_accesses,
            "average_access_count": avg_access_count,
            "config": {
                "min_confidence": self.config.minimum_confidence,
                "reverification_threshold": self.config.reverification_threshold
            }
        }

    def _infer_category(self, text: str, metadata: Optional[Dict]) -> MemoryCategory:
        """Infer memory category from content and metadata"""
        text_lower = text.lower()
        tags = metadata.get("tags", []) if metadata else []

        # Check metadata tags first
        if any(tag in tags for tag in ["system", "config", "critical"]):
            return MemoryCategory.CRITICAL
        if any(tag in tags for tag in ["personal", "preference"]):
            return MemoryCategory.PERSONAL

        # Content-based inference
        if any(word in text_lower for word in ["system", "error", "status", "service"]):
            return MemoryCategory.OBSERVED
        if any(word in text_lower for word in ["think", "assume", "might", "probably"]):
            return MemoryCategory.ASSUMED
        if any(word in text_lower for word in ["because", "therefore", "implies"]):
            return MemoryCategory.INFERRED

        return MemoryCategory.FACTUAL

    def _calculate_initial_confidence(self, text: str, metadata: Optional[Dict]) -> float:
        """Calculate initial confidence based on content and source"""
        base_confidence = 0.5

        # Source reliability
        source = metadata.get("source", "") if metadata else ""
        if "system" in source.lower() or "log" in source.lower():
            base_confidence += 0.2
        elif "user" in source.lower():
            base_confidence += 0.1

        # Content indicators
        text_lower = text.lower()
        uncertainty_words = ["might", "maybe", "possibly", "unsure", "unclear"]
        certainty_words = ["definitely", "certainly", "confirmed", "verified"]

        if any(word in text_lower for word in uncertainty_words):
            base_confidence -= 0.2
        if any(word in text_lower for word in certainty_words):
            base_confidence += 0.2

        return max(0.1, min(1.0, base_confidence))

    def _create_epistemic_status(self, memory_entry: MemoryEntry) -> EpistemicStatus:
        """Create epistemic status for integration with decay service"""
        knowledge_type = self._map_memory_type_to_epistemic(memory_entry.memory_type)

        return EpistemicStatus(
            qdrant_memory_id=memory_entry.qdrant_id,
            knowledge_type=knowledge_type,
            confidence=memory_entry.current_confidence,
            source=memory_entry.source,
            source_reliability=memory_entry.source_reliability,
            tags=memory_entry.tags
        )

    def _map_memory_type_to_epistemic(self, memory_type: MemoryType) -> EpistemicKnowledgeType:
        """Map memory type to epistemic knowledge type"""
        mapping = {
            MemoryType.SYSTEM_STATUS: EpistemicKnowledgeType.OBSERVED,
            MemoryType.FACTUAL_KNOWLEDGE: EpistemicKnowledgeType.REMEMBERED,
            MemoryType.PROCEDURAL: EpistemicKnowledgeType.REMEMBERED,
            MemoryType.EPISODIC: EpistemicKnowledgeType.OBSERVED,
            MemoryType.SEMANTIC: EpistemicKnowledgeType.INFERRED,
            MemoryType.CONVERSATION: EpistemicKnowledgeType.UNCERTAIN,
            MemoryType.USER_PREFERENCE: EpistemicKnowledgeType.OBSERVED
        }
        return mapping.get(memory_type, EpistemicKnowledgeType.UNCERTAIN)

    def _create_legacy_memory_entry(self, result: Dict) -> MemoryEntry:
        """Create memory entry for legacy memories without enhanced metadata"""
        content = result.get("text", "")
        memory_entry = MemoryEntry(
            id=str(result.get("id", "")),
            content=content,
            source=result.get("source", "legacy"),
            current_confidence=0.6,  # Default for legacy memories
            original_confidence=0.6,
            importance=0.4,  # Lower importance for legacy
            tags=["legacy"]
        )
        return memory_entry

class TemporalEchoWithMemory(EchoWithMemory):
    """Enhanced Echo with temporal memory management"""

    def __init__(self, temporal_memory: TemporalVectorMemory):
        # Initialize with base VectorMemory interface
        super().__init__(temporal_memory)
        self.temporal_memory = temporal_memory

    async def process_with_temporal_memory(self, user_message: str,
                                         base_response_func,
                                         min_confidence: float = 0.4) -> str:
        """
        Process message with temporal memory context and confidence filtering
        """
        # Get relevant context with confidence filtering
        memories = await self.temporal_memory.recall(
            user_message,
            limit=3,
            min_confidence=min_confidence
        )

        # Build context with confidence indicators
        context = ""
        if memories:
            context = "Relevant memories (with confidence levels):\n"
            for memory in memories:
                confidence = memory.get("current_confidence", 0)
                needs_verification = memory.get("needs_verification", False)
                verification_indicator = " [NEEDS VERIFICATION]" if needs_verification else ""

                context += f"- (confidence: {confidence:.2f}{verification_indicator}) {memory.get('text', '')[:150]}...\n"

        # Enhance prompt with temporal context
        enhanced_prompt = user_message
        if context:
            enhanced_prompt = f"{context}\n\nUser query: {user_message}"

        # Get response
        response = await base_response_func(enhanced_prompt)

        # Learn from exchange with enhanced metadata
        await self.temporal_memory.remember(
            f"User: {user_message}\nEcho: {response}",
            metadata={
                "type": "conversation",
                "user_message": user_message,
                "echo_response": response,
                "learned_at": datetime.utcnow().isoformat(),
                "source": "conversation",
                "tags": ["conversation", "learning"]
            },
            memory_type=MemoryType.CONVERSATION,
            importance=0.6
        )

        return response

    async def get_memory_health_report(self) -> Dict[str, Any]:
        """Get comprehensive memory health report"""
        stats = await self.temporal_memory.get_memory_statistics()
        verification_needed = await self.temporal_memory.get_memories_needing_verification(limit=10)

        return {
            "overall_statistics": stats,
            "verification_needed_sample": verification_needed,
            "health_indicators": {
                "memory_staleness": len(verification_needed) / stats.get("total_memories", 1),
                "average_confidence": stats.get("average_confidence", 0),
                "system_health": "good" if stats.get("average_confidence", 0) > 0.6 else "needs_attention"
            }
        }

# Factory functions for easy integration
async def create_temporal_vector_memory(decay_service: Optional[ConfidenceDecayService] = None) -> TemporalVectorMemory:
    """Create temporal vector memory system"""
    return TemporalVectorMemory(decay_service)

async def create_temporal_echo_with_memory(decay_service: Optional[ConfidenceDecayService] = None) -> TemporalEchoWithMemory:
    """Create enhanced Echo with temporal memory"""
    temporal_memory = await create_temporal_vector_memory(decay_service)
    return TemporalEchoWithMemory(temporal_memory)

if __name__ == "__main__":
    # Example usage
    async def test_temporal_memory():
        # Create temporal memory system
        temporal_memory = await create_temporal_vector_memory()

        # Store different types of memories
        await temporal_memory.remember(
            "Patrick prefers concise technical answers",
            metadata={
                "source": "user_interaction",
                "tags": ["preference", "personal"]
            },
            memory_type=MemoryType.USER_PREFERENCE,
            importance=0.8
        )

        await temporal_memory.remember(
            "Tower dashboard is running on port 8080",
            metadata={
                "source": "system_status",
                "tags": ["system", "configuration"]
            },
            memory_type=MemoryType.SYSTEM_STATUS,
            importance=0.9
        )

        # Test recall with confidence filtering
        memories = await temporal_memory.recall("What does Patrick prefer?", min_confidence=0.3)
        print(f"Found {len(memories)} memories about Patrick's preferences")

        # Get statistics
        stats = await temporal_memory.get_memory_statistics()
        print(f"Memory Statistics: {json.dumps(stats, indent=2, default=str)}")

        # Check for memories needing verification
        verification_needed = await temporal_memory.get_memories_needing_verification()
        print(f"Memories needing verification: {len(verification_needed)}")

    # Run test
    asyncio.run(test_temporal_memory())