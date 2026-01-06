#!/usr/bin/env python3
"""
Maintains conversation context for multi-turn coherence.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime, timedelta
from collections import deque

@dataclass
class ConversationTurn:
    """Single turn in conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    entities_mentioned: dict[str, Any] = field(default_factory=dict)

@dataclass
class ConversationContext:
    """
    Tracks conversation state for pronoun resolution and context carryover.

    Solves: "What services are broken?" -> "anime_production"
            "Fix it" -> Should know "it" = anime_production
    """

    max_turns: int = 20
    entity_ttl_minutes: int = 30

    def __post_init__(self):
        self.turns: deque[ConversationTurn] = deque(maxlen=self.max_turns)
        self.active_entities: dict[str, Any] = {}
        self.last_mentioned: dict[str, datetime] = {}

    def add_turn(self, role: str, content: str, entities: dict[str, Any] = None) -> None:
        """Record a conversation turn."""
        turn = ConversationTurn(
            role=role,
            content=content,
            entities_mentioned=entities or {}
        )
        self.turns.append(turn)

        # Update active entities
        if entities:
            now = datetime.now()
            for key, value in entities.items():
                self.active_entities[key] = value
                self.last_mentioned[key] = now

        # Expire old entities
        self._expire_entities()

    def _expire_entities(self) -> None:
        """Remove entities not mentioned recently."""
        now = datetime.now()
        expired = [
            key for key, timestamp in self.last_mentioned.items()
            if now - timestamp > timedelta(minutes=self.entity_ttl_minutes)
        ]
        for key in expired:
            del self.active_entities[key]
            del self.last_mentioned[key]

    def resolve_pronoun(self, pronoun: str) -> Optional[Any]:
        """
        Resolve pronouns like 'it', 'that', 'them' to entities.

        Returns most recently mentioned entity of appropriate type.
        """
        pronoun = pronoun.lower()

        # Simple heuristic: "it" refers to most recent singular entity
        if pronoun in ("it", "that", "this"):
            if self.active_entities:
                # Return most recently mentioned
                most_recent = max(
                    self.last_mentioned.items(),
                    key=lambda x: x[1]
                )
                return self.active_entities.get(most_recent[0])

        # "them" refers to collections
        if pronoun in ("them", "those", "these"):
            collections = [
                v for v in self.active_entities.values()
                if isinstance(v, (list, tuple, set))
            ]
            if collections:
                return collections[-1]

        return None

    def get_context_for_llm(self) -> str:
        """Generate context summary for LLM prompt."""
        if not self.active_entities:
            return ""

        lines = ["Recent context:"]
        for key, value in self.active_entities.items():
            lines.append(f"- {key}: {value}")

        return "\n".join(lines)

# Integration with query handler
def enhance_query_with_context(query: str, context: ConversationContext) -> str:
    """
    Expand pronouns in query using conversation context.

    "Fix it" -> "Fix anime_production (the broken service)"
    """
    pronouns = ["it", "that", "this", "them", "those"]

    enhanced = query
    for pronoun in pronouns:
        if f" {pronoun}" in query.lower() or query.lower().startswith(pronoun):
            resolved = context.resolve_pronoun(pronoun)
            if resolved:
                enhanced = query.replace(
                    pronoun,
                    f"{resolved} (referring to {pronoun})"
                )
                break

    return enhanced


def test_conversation_context():
    """Verify pronoun resolution works."""
    ctx = ConversationContext()

    # User asks about broken services
    ctx.add_turn(
        role="user",
        content="What services are broken?"
    )

    # Assistant responds with entity
    ctx.add_turn(
        role="assistant",
        content="anime_production is broken",
        entities={"broken_service": "anime_production"}
    )

    # User says "fix it"
    resolved = ctx.resolve_pronoun("it")
    assert resolved == "anime_production", f"Expected anime_production, got {resolved}"

    enhanced = enhance_query_with_context("Fix it", ctx)
    assert "anime_production" in enhanced, f"Should expand 'it': {enhanced}"

    print("✅ Multi-turn context resolution working")


if __name__ == "__main__":
    test_conversation_context()