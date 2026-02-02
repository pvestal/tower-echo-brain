#!/usr/bin/env python3
"""
Resolves pronouns to entities from conversation context.
"""

import re
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class PronounResolver:
    """
    Resolves pronouns like 'it', 'that', 'them' to entities from context.

    Examples:
        "What services are broken?" → "anime_production"
        "Fix it" → "Fix anime_production"
    """

    # Pronouns that refer to singular entities
    SINGULAR_PRONOUNS = {"it", "that", "this", "the service", "the file", "the issue"}

    # Pronouns that refer to collections
    PLURAL_PRONOUNS = {"them", "those", "these", "they", "the services", "the files"}

    # Patterns that indicate entity reference
    REFERENCE_PATTERNS = [
        r"\b(it|that|this)\b",
        r"\bthe (service|file|issue|error|problem)\b",
        r"\b(fix|restart|check|repair|analyze) it\b",
    ]

    def __init__(self):
        self.compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.REFERENCE_PATTERNS
        ]

    def needs_resolution(self, query: str) -> bool:
        """Check if query contains pronouns that need resolution."""
        query_lower = query.lower()

        # Check for pronoun patterns
        for pattern in self.compiled_patterns:
            if pattern.search(query_lower):
                return True

        # Check for bare pronouns
        words = set(query_lower.split())
        if words & self.SINGULAR_PRONOUNS:
            return True
        if words & self.PLURAL_PRONOUNS:
            return True

        return False

    def resolve(
        self,
        query: str,
        entities: Dict[str, Any]
    ) -> Tuple[str, Optional[str]]:
        """
        Resolve pronouns in query using entity context.

        Args:
            query: User's query with potential pronouns
            entities: Dict of entity_type -> entity_value from context

        Returns:
            Tuple of (resolved_query, resolved_entity_name)
        """
        if not entities:
            logger.debug("No entities available for resolution")
            return query, None

        if not self.needs_resolution(query):
            logger.debug("Query doesn't need pronoun resolution")
            return query, None

        # Find the most relevant entity to substitute
        # Priority: service > file > error > other
        entity_priority = ["service", "broken_service", "file", "error", "issue"]

        resolved_entity = None
        for entity_type in entity_priority:
            if entity_type in entities:
                resolved_entity = entities[entity_type]
                break

        # If no priority match, use most recent entity
        if not resolved_entity and entities:
            resolved_entity = list(entities.values())[0]

        if not resolved_entity:
            return query, None

        # Build resolved query
        resolved_query = query

        # Add context annotation for the LLM
        context_note = f' (referring to "{resolved_entity}" from our conversation)'

        # Replace common patterns
        replacements = [
            (r"\bfix it\b", f"fix {resolved_entity}"),
            (r"\brestart it\b", f"restart {resolved_entity}"),
            (r"\bcheck it\b", f"check {resolved_entity}"),
            (r"\brepair it\b", f"repair {resolved_entity}"),
            (r"\banalyze it\b", f"analyze {resolved_entity}"),
            (r"\babout it\b", f"about {resolved_entity}"),
            (r"\bis it\b", f"is {resolved_entity}"),
            (r"\bcan you fix it\b", f"can you fix {resolved_entity}"),
        ]

        for pattern, replacement in replacements:
            new_query = re.sub(pattern, replacement, resolved_query, flags=re.IGNORECASE)
            if new_query != resolved_query:
                resolved_query = new_query
                break  # Only apply first matching replacement

        # If no specific replacement matched but resolution needed, append context
        if resolved_query == query and self.needs_resolution(query):
            resolved_query = query + context_note

        logger.info(f"Resolved '{query}' → '{resolved_query}'")
        return resolved_query, resolved_entity