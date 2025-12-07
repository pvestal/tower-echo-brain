#!/usr/bin/env python3
"""
Conversation Memory Manager - Multi-turn context with persistence and semantic search.

Solves the core problem: "What services are broken?" → "Fix it"
The system should remember "it" refers to the broken services.

Features:
- Persistent conversation context across sessions
- Entity extraction and resolution (services, files, errors)
- Semantic search for relevant past context
- Integration with verified execution for context-aware actions
- Automatic entity decay to prevent stale references
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from pathlib import Path
from collections import defaultdict
from enum import Enum

# Database and LLM integration
from ..db.database import database
# Direct Ollama API integration for LLM calls

logger = logging.getLogger(__name__)


class EntityType(Enum):
    SERVICE = "service"
    FILE = "file"
    ERROR = "error"
    PORT = "port"
    SYSTEM = "system"
    PERSON = "person"
    TASK = "task"
    GENERIC = "generic"


@dataclass
class Entity:
    """An entity mentioned in conversation with metadata."""
    name: str
    entity_type: EntityType
    value: Any
    confidence: float = 1.0
    first_mentioned: datetime = field(default_factory=datetime.now)
    last_mentioned: datetime = field(default_factory=datetime.now)
    mention_count: int = 1
    context: str = ""  # Context where it was mentioned
    aliases: Set[str] = field(default_factory=set)

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "entity_type": self.entity_type.value,
            "value": str(self.value),
            "confidence": self.confidence,
            "first_mentioned": self.first_mentioned.isoformat(),
            "last_mentioned": self.last_mentioned.isoformat(),
            "mention_count": self.mention_count,
            "context": self.context,
            "aliases": list(self.aliases)
        }

    @classmethod
    def from_dict(cls, data: Dict):
        """Create Entity from dictionary."""
        return cls(
            name=data["name"],
            entity_type=EntityType(data["entity_type"]),
            value=data["value"],
            confidence=data["confidence"],
            first_mentioned=datetime.fromisoformat(data["first_mentioned"]),
            last_mentioned=datetime.fromisoformat(data["last_mentioned"]),
            mention_count=data["mention_count"],
            context=data["context"],
            aliases=set(data["aliases"])
        )


@dataclass
class ConversationTurn:
    """A single turn in a conversation with extracted entities."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime
    conversation_id: str
    entities: List[Entity] = field(default_factory=list)
    intent: Optional[str] = None
    execution_results: List[str] = field(default_factory=list)  # IDs of executions triggered


@dataclass
class ConversationSession:
    """A complete conversation session with context."""
    conversation_id: str
    started_at: datetime
    last_activity: datetime
    turns: List[ConversationTurn] = field(default_factory=list)
    active_entities: Dict[str, Entity] = field(default_factory=dict)
    session_summary: str = ""


class EntityExtractor:
    """Extracts entities from text using patterns and LLM assistance."""

    def __init__(self):
        # Predefined patterns for common Tower entities
        self.patterns = {
            EntityType.SERVICE: [
                r"(tower-echo-brain|echo-brain|echo brain)",
                r"(tower-anime-production|anime-production|anime production)",
                r"(tower-auth|auth service|authentication)",
                r"(tower-kb|knowledge base|kb)",
                r"(tower-dashboard|dashboard)",
                r"(comfyui|comfy ui)",
                r"(tower-apple-music|apple music)",
                r"(nginx|postgresql|redis)",
                r"(ollama|vault)",
            ],
            EntityType.FILE: [
                r"([a-zA-Z_][a-zA-Z0-9_]*\.py)",
                r"([a-zA-Z_][a-zA-Z0-9_]*\.js)",
                r"([a-zA-Z_][a-zA-Z0-9_]*\.md)",
                r"([a-zA-Z_][a-zA-Z0-9_]*\.json)",
                r"([a-zA-Z_][a-zA-Z0-9_]*\.yaml|[a-zA-Z_][a-zA-Z0-9_]*\.yml)",
            ],
            EntityType.PORT: [
                r"port (\d+)",
                r":(\d+)",
                r"(\d+)/tcp",
                r"localhost:(\d+)",
            ],
            EntityType.ERROR: [
                r"(connection refused|connection timeout)",
                r"(404|500|503|502) error",
                r"(failed to start|service failed)",
                r"(out of memory|disk full)",
            ]
        }

    async def extract_entities(self, text: str, conversation_id: str) -> List[Entity]:
        """Extract entities from text using patterns and LLM."""
        entities = []

        # Pattern-based extraction
        entities.extend(self._extract_with_patterns(text))

        # LLM-enhanced extraction for complex entities
        if len(text) > 50:  # Only for substantial text
            llm_entities = await self._extract_with_llm(text, conversation_id)
            entities.extend(llm_entities)

        # Deduplicate and merge
        return self._merge_entities(entities, text)

    def _extract_with_patterns(self, text: str) -> List[Entity]:
        """Extract entities using predefined patterns."""
        entities = []
        text_lower = text.lower()

        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    entity_name = match.group(1) if match.groups() else match.group(0)
                    entity = Entity(
                        name=entity_name,
                        entity_type=entity_type,
                        value=entity_name,
                        confidence=0.9,
                        context=text[:100]
                    )
                    entities.append(entity)

        return entities

    async def _extract_with_llm(self, text: str, conversation_id: str) -> List[Entity]:
        """Use LLM to extract complex entities and relationships."""
        try:
            # Use direct Ollama API for LLM entity extraction
            import httpx

            extraction_prompt = f"""Extract important entities from this text that would be relevant for follow-up questions:

Text: "{text}"

Focus on:
- Services/systems mentioned
- Files or paths mentioned
- Error conditions or problems
- Specific tasks or actions
- Technical components

Return as JSON list with format:
[{{"name": "entity_name", "type": "service|file|error|task|system", "value": "actual_value"}}]

Only include entities that could be referenced later (like with "it", "that", etc.)."""

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "llama3.2:3b",
                        "prompt": extraction_prompt,
                        "stream": False
                    }
                )

                if response.status_code == 200:
                    result = response.json()
                    return self._parse_llm_entities(result.get("response", ""), text)

        except Exception as e:
            logger.warning(f"LLM entity extraction failed: {e}")

        return []

    def _parse_llm_entities(self, llm_response: str, original_text: str) -> List[Entity]:
        """Parse LLM response into Entity objects."""
        entities = []
        try:
            # Extract JSON from LLM response
            json_match = re.search(r'\[.*\]', llm_response, re.DOTALL)
            if json_match:
                entity_data = json.loads(json_match.group())

                for item in entity_data:
                    if isinstance(item, dict) and all(k in item for k in ["name", "type", "value"]):
                        entity_type = EntityType(item["type"]) if item["type"] in [e.value for e in EntityType] else EntityType.GENERIC
                        entity = Entity(
                            name=item["name"],
                            entity_type=entity_type,
                            value=item["value"],
                            confidence=0.8,
                            context=original_text[:100]
                        )
                        entities.append(entity)

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.debug(f"Failed to parse LLM entities: {e}")

        return entities

    def _merge_entities(self, entities: List[Entity], context: str) -> List[Entity]:
        """Merge duplicate entities and add aliases."""
        merged = {}

        for entity in entities:
            key = entity.name.lower()
            if key in merged:
                # Merge with existing
                existing = merged[key]
                existing.confidence = max(existing.confidence, entity.confidence)
                existing.aliases.add(entity.name)
            else:
                merged[key] = entity

        return list(merged.values())


class ConversationMemoryManager:
    """
    Manages conversation context memory with persistence and semantic search.

    Core functionality:
    - Tracks entities across conversation turns
    - Resolves pronouns to recently mentioned entities
    - Persists context to database for cross-session memory
    - Provides context-enhanced prompts for LLM calls
    """

    def __init__(self, storage_path: Path = None):
        self.storage_path = storage_path or Path("/opt/tower-echo-brain/data/conversation_memory.json")
        self.entity_extractor = EntityExtractor()

        # Runtime state
        self.active_sessions: Dict[str, ConversationSession] = {}
        self.global_entities: Dict[str, Entity] = {}

        # Configuration
        self.max_turns_per_session = 50
        self.entity_decay_hours = 24
        self.session_timeout_hours = 4

    async def initialize(self):
        """Initialize the conversation memory system."""
        await self._load_persistent_state()
        logger.info("✅ Conversation memory manager initialized")

    async def add_turn(
        self,
        conversation_id: str,
        role: str,
        content: str,
        intent: str = None,
        execution_results: List[str] = None
    ) -> ConversationTurn:
        """
        Add a new conversation turn and extract entities.

        Args:
            conversation_id: Unique conversation identifier
            role: "user" or "assistant"
            content: The message content
            intent: Classified intent (optional)
            execution_results: List of execution result IDs (optional)

        Returns:
            ConversationTurn with extracted entities
        """
        # Ensure session exists
        if conversation_id not in self.active_sessions:
            self.active_sessions[conversation_id] = ConversationSession(
                conversation_id=conversation_id,
                started_at=datetime.now(),
                last_activity=datetime.now()
            )

        session = self.active_sessions[conversation_id]

        # Extract entities from content
        entities = await self.entity_extractor.extract_entities(content, conversation_id)

        # Create turn
        turn = ConversationTurn(
            role=role,
            content=content,
            timestamp=datetime.now(),
            conversation_id=conversation_id,
            entities=entities,
            intent=intent,
            execution_results=execution_results or []
        )

        # Add to session
        session.turns.append(turn)
        session.last_activity = datetime.now()

        # Update active entities
        now = datetime.now()
        for entity in entities:
            entity.last_mentioned = now

            if entity.name in session.active_entities:
                # Update existing entity
                existing = session.active_entities[entity.name]
                existing.last_mentioned = now
                existing.mention_count += 1
                existing.confidence = max(existing.confidence, entity.confidence)
                existing.aliases.update(entity.aliases)
            else:
                # Add new entity
                session.active_entities[entity.name] = entity
                self.global_entities[entity.name] = entity

        # Cleanup old data
        await self._cleanup_session(session)
        await self._persist_state()

        logger.info(f"📝 Added turn to conversation {conversation_id}: {len(entities)} entities extracted")
        return turn

    async def resolve_reference(self, text: str, conversation_id: str) -> Tuple[str, List[str]]:
        """
        Resolve pronouns and references in text to actual entities.

        Args:
            text: Input text potentially containing pronouns
            conversation_id: Current conversation context

        Returns:
            Tuple of (enhanced_text, list_of_resolved_entities)
        """
        if conversation_id not in self.active_sessions:
            return text, []

        session = self.active_sessions[conversation_id]
        resolved_entities = []
        enhanced_text = text

        # Common pronouns and reference patterns
        reference_patterns = {
            r'\bit\b': self._resolve_singular_entity,
            r'\bthat\b': self._resolve_singular_entity,
            r'\bthis\b': self._resolve_singular_entity,
            r'\bthem\b': self._resolve_plural_entities,
            r'\bthose\b': self._resolve_plural_entities,
            r'\bthese\b': self._resolve_plural_entities,
            r'\bthe service\b': lambda s: self._resolve_by_type(s, EntityType.SERVICE),
            r'\bthe file\b': lambda s: self._resolve_by_type(s, EntityType.FILE),
            r'\bthe error\b': lambda s: self._resolve_by_type(s, EntityType.ERROR),
        }

        for pattern, resolver in reference_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                entities = resolver(session)
                if entities:
                    resolved_entities.extend(entities)
                    # Replace first occurrence with resolved entity
                    entity_names = [e.name for e in entities]
                    replacement = f"{', '.join(entity_names)} (referring to {re.search(pattern, text, re.IGNORECASE).group()})"
                    enhanced_text = re.sub(pattern, replacement, enhanced_text, count=1, flags=re.IGNORECASE)

        return enhanced_text, resolved_entities

    def _resolve_singular_entity(self, session: ConversationSession) -> List[Entity]:
        """Resolve 'it', 'that', 'this' to most recently mentioned singular entity."""
        if not session.active_entities:
            return []

        # Get most recently mentioned entity
        most_recent = max(
            session.active_entities.values(),
            key=lambda e: e.last_mentioned
        )
        return [most_recent]

    def _resolve_plural_entities(self, session: ConversationSession) -> List[Entity]:
        """Resolve 'them', 'those', 'these' to recently mentioned entities."""
        # Return up to 3 most recent entities
        entities = sorted(
            session.active_entities.values(),
            key=lambda e: e.last_mentioned,
            reverse=True
        )
        return entities[:3]

    def _resolve_by_type(self, session: ConversationSession, entity_type: EntityType) -> List[Entity]:
        """Resolve references like 'the service' to most recent entity of that type."""
        typed_entities = [
            e for e in session.active_entities.values()
            if e.entity_type == entity_type
        ]

        if not typed_entities:
            return []

        # Return most recent of this type
        most_recent = max(typed_entities, key=lambda e: e.last_mentioned)
        return [most_recent]

    async def get_context_prompt(self, conversation_id: str, max_turns: int = 5) -> str:
        """
        Generate context prompt for LLM including recent conversation and active entities.

        Args:
            conversation_id: Current conversation
            max_turns: Maximum recent turns to include

        Returns:
            Context string for LLM prompt
        """
        if conversation_id not in self.active_sessions:
            return ""

        session = self.active_sessions[conversation_id]

        # Get recent turns
        recent_turns = session.turns[-max_turns:]

        # Build context
        context_parts = []

        # Active entities
        if session.active_entities:
            context_parts.append("## Recent Context")
            context_parts.append("Currently relevant entities:")
            for name, entity in session.active_entities.items():
                age_minutes = (datetime.now() - entity.last_mentioned).total_seconds() / 60
                context_parts.append(
                    f"- {entity.name} ({entity.entity_type.value}): {entity.value} "
                    f"(mentioned {entity.mention_count}x, last {age_minutes:.0f}m ago)"
                )

        # Recent conversation
        if recent_turns:
            context_parts.append("\n## Recent Conversation")
            for turn in recent_turns[-3:]:  # Last 3 turns
                role_emoji = "👤" if turn.role == "user" else "🤖"
                context_parts.append(f"{role_emoji} {turn.role}: {turn.content}")

        return "\n".join(context_parts)

    async def find_relevant_context(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        max_results: int = 5
    ) -> List[str]:
        """
        Find relevant context from conversation history.

        Uses semantic similarity to identify relevant past conversations.
        """
        relevant_context = []
        context_candidates = []

        if conversation_id and conversation_id in self.active_sessions:
            session = self.active_sessions[conversation_id]

            # Search recent turns in current session
            for turn in reversed(session.turns[-20:]):  # Last 20 turns for better context
                similarity = self._calculate_similarity(query, turn.content)
                if similarity > 0.4:  # Lower threshold for same session
                    context_candidates.append({
                        "text": f"Previous turn ({turn.role}): {turn.content[:300]}...",
                        "similarity": similarity,
                        "timestamp": turn.timestamp,
                        "type": "current_session"
                    })

        # Search across all sessions for similar queries and responses
        for session_id, session in self.active_sessions.items():
            if conversation_id and session_id == conversation_id:
                continue  # Skip current session, already processed

            for turn in session.turns:
                similarity = self._calculate_similarity(query, turn.content)
                if similarity > 0.5:  # Higher threshold for cross-session
                    context_candidates.append({
                        "text": f"Similar {turn.role} ({turn.timestamp.strftime('%Y-%m-%d %H:%M')}): {turn.content[:300]}...",
                        "similarity": similarity,
                        "timestamp": turn.timestamp,
                        "type": "cross_session"
                    })

        # Search global entities for direct matches
        query_lower = query.lower()
        for entity in self.global_entities.values():
            if (entity.name.lower() in query_lower or
                any(alias.lower() in query_lower for alias in entity.aliases)):
                context_candidates.append({
                    "text": f"Known entity: {entity.name} ({entity.entity_type.value}) = {entity.value}",
                    "similarity": 0.9,  # High relevance for direct entity matches
                    "timestamp": entity.last_mentioned,
                    "type": "entity_match"
                })

        # Sort by similarity score and recency
        context_candidates.sort(key=lambda x: (x["similarity"], x["timestamp"]), reverse=True)

        # Build relevant context list
        for candidate in context_candidates[:max_results]:
            relevant_context.append(candidate["text"])

        return relevant_context

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts using multiple metrics.

        Returns:
            Float similarity score between 0.0 and 1.0
        """
        if not text1.strip() or not text2.strip():
            return 0.0

        # Normalize texts
        text1_lower = text1.lower().strip()
        text2_lower = text2.lower().strip()

        # Extract words
        words1 = set(re.findall(r'\w+', text1_lower))
        words2 = set(re.findall(r'\w+', text2_lower))

        if not words1 or not words2:
            return 0.0

        # Jaccard similarity (word overlap)
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        jaccard_sim = len(intersection) / len(union) if union else 0.0

        # Substring similarity (for phrases)
        shorter, longer = (text1_lower, text2_lower) if len(text1_lower) < len(text2_lower) else (text2_lower, text1_lower)
        substring_sim = len(shorter) / len(longer) if shorter in longer else 0.0

        # Exact match bonus
        exact_bonus = 0.3 if text1_lower == text2_lower else 0.0

        # Weighted combination
        similarity = (jaccard_sim * 0.6) + (substring_sim * 0.4) + exact_bonus

        return min(similarity, 1.0)  # Cap at 1.0

    def _is_semantically_similar(self, text1: str, text2: str) -> bool:
        """Simple semantic similarity check (backward compatibility)."""
        return self._calculate_similarity(text1, text2) > 0.3

    async def get_session_summary(self, conversation_id: str) -> str:
        """Generate a summary of the conversation session."""
        if conversation_id not in self.active_sessions:
            return "No active session found."

        session = self.active_sessions[conversation_id]

        if not session.turns:
            return "No conversation turns recorded."

        # Use LLM to generate summary
        try:
            import httpx

            # Prepare conversation text
            conversation_text = []
            for turn in session.turns[-10:]:  # Last 10 turns
                conversation_text.append(f"{turn.role}: {turn.content}")

            summary_prompt = f"""Summarize this conversation focusing on:
- Main topics discussed
- Actions taken or requested
- Current status of any issues
- Key entities mentioned

Conversation:
{chr(10).join(conversation_text)}

Provide a concise summary in 2-3 sentences."""

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "llama3.2:3b",
                        "prompt": summary_prompt,
                        "stream": False
                    }
                )

                if response.status_code == 200:
                    result = response.json()
                    summary = result.get("response", "").strip()
                    if summary:
                        session.session_summary = summary
                        await self._persist_state()
                        return summary

        except Exception as e:
            logger.warning(f"Failed to generate session summary: {e}")

        # Fallback summary
        entities = list(session.active_entities.keys())
        return f"Conversation about {', '.join(entities[:3])}" if entities else "General conversation"

    async def _cleanup_session(self, session: ConversationSession):
        """Clean up old turns and expired entities."""
        now = datetime.now()

        # Remove old turns
        if len(session.turns) > self.max_turns_per_session:
            session.turns = session.turns[-self.max_turns_per_session:]

        # Remove expired entities
        expired_entities = []
        for name, entity in session.active_entities.items():
            age = now - entity.last_mentioned
            if age > timedelta(hours=self.entity_decay_hours):
                expired_entities.append(name)

        for name in expired_entities:
            del session.active_entities[name]

        if expired_entities:
            logger.debug(f"Expired {len(expired_entities)} entities from session {session.conversation_id}")

    async def _persist_state(self):
        """Persist conversation state to disk."""
        try:
            # Prepare data for serialization
            data = {
                "sessions": {},
                "global_entities": {},
                "last_updated": datetime.now().isoformat()
            }

            # Serialize sessions
            for conv_id, session in self.active_sessions.items():
                data["sessions"][conv_id] = {
                    "conversation_id": session.conversation_id,
                    "started_at": session.started_at.isoformat(),
                    "last_activity": session.last_activity.isoformat(),
                    "session_summary": session.session_summary,
                    "entities": {name: entity.to_dict() for name, entity in session.active_entities.items()},
                    "turn_count": len(session.turns)
                }

            # Serialize global entities
            for name, entity in self.global_entities.items():
                data["global_entities"][name] = entity.to_dict()

            # Write to file
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Persisted conversation state: {len(self.active_sessions)} sessions")

        except Exception as e:
            logger.warning(f"Failed to persist conversation state: {e}")

    async def _load_persistent_state(self):
        """Load conversation state from disk."""
        try:
            if not self.storage_path.exists():
                return

            with open(self.storage_path, 'r') as f:
                data = json.load(f)

            # Load global entities
            for name, entity_data in data.get("global_entities", {}).items():
                try:
                    entity = Entity.from_dict(entity_data)
                    self.global_entities[name] = entity
                except Exception as e:
                    logger.warning(f"Failed to load entity {name}: {e}")

            # Load recent sessions (keep only last 24 hours)
            now = datetime.now()
            cutoff = now - timedelta(hours=24)

            for conv_id, session_data in data.get("sessions", {}).items():
                try:
                    last_activity = datetime.fromisoformat(session_data["last_activity"])
                    if last_activity > cutoff:
                        # Reconstruct session (without full turns - those are too much data)
                        session = ConversationSession(
                            conversation_id=conv_id,
                            started_at=datetime.fromisoformat(session_data["started_at"]),
                            last_activity=last_activity,
                            session_summary=session_data.get("session_summary", "")
                        )

                        # Load entities
                        for name, entity_data in session_data.get("entities", {}).items():
                            entity = Entity.from_dict(entity_data)
                            session.active_entities[name] = entity

                        self.active_sessions[conv_id] = session

                except Exception as e:
                    logger.warning(f"Failed to load session {conv_id}: {e}")

            logger.info(f"Loaded conversation state: {len(self.global_entities)} entities, {len(self.active_sessions)} active sessions")

        except Exception as e:
            logger.warning(f"Failed to load conversation state: {e}")


# Singleton instance for Echo Brain
_conversation_memory_instance: Optional[ConversationMemoryManager] = None

async def get_conversation_memory_manager() -> ConversationMemoryManager:
    """Get or create the singleton conversation memory manager."""
    global _conversation_memory_instance
    if _conversation_memory_instance is None:
        _conversation_memory_instance = ConversationMemoryManager()
        await _conversation_memory_instance.initialize()
    return _conversation_memory_instance


# Integration helpers for Echo Brain
async def enhance_query_with_memory(query: str, conversation_id: str) -> Tuple[str, str]:
    """
    Enhance a query with conversation memory context.

    Returns:
        Tuple of (enhanced_query, context_prompt)
    """
    memory_manager = await get_conversation_memory_manager()

    # Resolve references in query
    enhanced_query, resolved_entities = await memory_manager.resolve_reference(query, conversation_id)

    # Get context prompt
    context_prompt = await memory_manager.get_context_prompt(conversation_id)

    return enhanced_query, context_prompt


async def record_conversation_turn(
    conversation_id: str,
    role: str,
    content: str,
    intent: str = None,
    execution_results: List[str] = None
) -> ConversationTurn:
    """
    Record a conversation turn for memory tracking.

    This should be called for both user queries and assistant responses.
    """
    memory_manager = await get_conversation_memory_manager()
    return await memory_manager.add_turn(conversation_id, role, content, intent, execution_results)


async def get_conversation_context_for_execution(conversation_id: str) -> Dict[str, Any]:
    """
    Get conversation context relevant for verified execution.

    Returns dict with entities that might be referenced in execution commands.
    """
    memory_manager = await get_conversation_memory_manager()

    if conversation_id not in memory_manager.active_sessions:
        return {}

    session = memory_manager.active_sessions[conversation_id]

    # Extract execution-relevant entities
    execution_context = {
        "services": [],
        "files": [],
        "ports": [],
        "errors": []
    }

    for entity in session.active_entities.values():
        if entity.entity_type == EntityType.SERVICE:
            execution_context["services"].append(entity.name)
        elif entity.entity_type == EntityType.FILE:
            execution_context["files"].append(entity.name)
        elif entity.entity_type == EntityType.PORT:
            execution_context["ports"].append(entity.value)
        elif entity.entity_type == EntityType.ERROR:
            execution_context["errors"].append(entity.value)

    return execution_context