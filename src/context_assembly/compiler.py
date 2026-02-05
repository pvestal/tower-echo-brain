"""
Context Compiler - Manages token budgets and formats context for LLMs
Ensures context doesn't exceed model limits while preserving most relevant info
"""
import logging
from typing import List, Dict, Any, Optional
from collections import defaultdict

logger = logging.getLogger("echo.context_assembly.compiler")


class ContextCompiler:
    """
    Compiles retrieved context into prompt-ready format.
    Manages token budgets and prioritizes most relevant information.
    """

    def __init__(self):
        # Token limits for different models
        self.model_limits = {
            "mistral:7b": 4096,
            "llama3.1:8b": 8192,
            "deepseek-r1:8b": 8192,
            "gemma2:9b": 8192,
            "qwen2.5-coder:7b": 8192,
            "default": 4096
        }

        # Reserved tokens for system prompt and response
        self.reserved_tokens = {
            "system_prompt": 500,
            "user_query": 100,
            "response_buffer": 2000
        }

        # Approximate tokens per character (rough estimate)
        self.chars_per_token = 4

    def compile(
        self,
        retrieval_result: Dict[str, Any],
        model_name: str = "default",
        format_style: str = "structured",
        max_sources_per_type: Optional[Dict[str, int]] = None
    ) -> Dict[str, Any]:
        """
        Compile retrieved context into LLM-ready format.

        Args:
            retrieval_result: Output from ParallelRetriever.retrieve()
            model_name: Target model (for token limits)
            format_style: How to format context (structured, narrative, minimal)
            max_sources_per_type: Limit sources by type

        Returns:
            Compiled context with token management
        """
        query = retrieval_result.get("query", "")
        domain = retrieval_result.get("domain", "general")
        sources = retrieval_result.get("sources", [])

        # Calculate available token budget
        model_limit = self.model_limits.get(model_name, self.model_limits["default"])
        reserved = sum(self.reserved_tokens.values())
        available_tokens = model_limit - reserved

        logger.info(
            f"Compiling context for {model_name}: "
            f"{available_tokens} tokens available from {model_limit} limit"
        )

        # Group sources by type
        grouped_sources = self._group_sources_by_type(sources)

        # Apply per-type limits if specified
        if max_sources_per_type:
            grouped_sources = self._apply_type_limits(grouped_sources, max_sources_per_type)

        # Select and trim sources to fit token budget
        selected_sources = self._select_sources_for_budget(
            grouped_sources, available_tokens
        )

        # Format context based on style
        if format_style == "structured":
            formatted_context = self._format_structured(selected_sources, domain)
        elif format_style == "narrative":
            formatted_context = self._format_narrative(selected_sources, domain)
        else:  # minimal
            formatted_context = self._format_minimal(selected_sources)

        # Calculate final token usage
        estimated_tokens = self._estimate_tokens(formatted_context)

        # Trim if still over budget (safety check)
        if estimated_tokens > available_tokens:
            formatted_context = self._emergency_trim(formatted_context, available_tokens)
            estimated_tokens = self._estimate_tokens(formatted_context)

        return {
            "query": query,
            "domain": domain,
            "formatted_context": formatted_context,
            "sources_used": len(selected_sources),
            "sources_available": len(sources),
            "estimated_tokens": estimated_tokens,
            "token_limit": model_limit,
            "token_utilization": estimated_tokens / available_tokens,
            "source_breakdown": {
                k: len(v) for k, v in grouped_sources.items()
            }
        }

    def _group_sources_by_type(self, sources: List[Dict]) -> Dict[str, List[Dict]]:
        """Group sources by their type"""
        grouped = defaultdict(list)
        for source in sources:
            source_type = source.get("type", "unknown")
            grouped[source_type].append(source)
        return dict(grouped)

    def _apply_type_limits(
        self,
        grouped_sources: Dict[str, List[Dict]],
        limits: Dict[str, int]
    ) -> Dict[str, List[Dict]]:
        """Apply per-type source limits"""
        limited = {}
        for source_type, sources in grouped_sources.items():
            limit = limits.get(source_type, len(sources))
            limited[source_type] = sources[:limit]
        return limited

    def _select_sources_for_budget(
        self,
        grouped_sources: Dict[str, List[Dict]],
        token_budget: int
    ) -> List[Dict]:
        """
        Select sources that fit within token budget.
        Prioritizes by score and diversity of types.
        """
        selected = []
        tokens_used = 0

        # Priority order for source types
        type_priority = ["fact", "vector", "conversation", "other"]

        # First pass: take top item from each type
        for source_type in type_priority:
            if source_type in grouped_sources and grouped_sources[source_type]:
                source = grouped_sources[source_type][0]
                source_tokens = self._estimate_tokens(source.get("content", ""))

                if tokens_used + source_tokens <= token_budget:
                    selected.append(source)
                    tokens_used += source_tokens
                    grouped_sources[source_type] = grouped_sources[source_type][1:]

        # Second pass: fill remaining budget with highest scoring sources
        all_remaining = []
        for sources in grouped_sources.values():
            all_remaining.extend(sources)

        # Sort by score
        all_remaining.sort(key=lambda x: x.get("score", 0), reverse=True)

        for source in all_remaining:
            source_tokens = self._estimate_tokens(source.get("content", ""))

            if tokens_used + source_tokens <= token_budget:
                selected.append(source)
                tokens_used += source_tokens
            else:
                # Try to fit a truncated version
                available = token_budget - tokens_used
                if available > 100:  # At least 100 tokens worth
                    truncated_source = source.copy()
                    max_chars = available * self.chars_per_token
                    truncated_source["content"] = source["content"][:max_chars] + "..."
                    selected.append(truncated_source)
                    break

        logger.info(f"Selected {len(selected)} sources using {tokens_used} tokens")
        return selected

    def _format_structured(self, sources: List[Dict], domain: str) -> str:
        """Format context in structured sections"""
        if not sources:
            return f"[No relevant context found for {domain} domain query]"

        sections = []
        sections.append(f"=== CONTEXT DOMAIN: {domain.upper()} ===\n")

        # Group by type for structured presentation
        by_type = defaultdict(list)
        for source in sources:
            by_type[source.get("type", "unknown")].append(source)

        # Format each type section
        type_labels = {
            "fact": "📊 KNOWN FACTS",
            "vector": "💭 RELEVANT MEMORY",
            "conversation": "💬 RELATED DISCUSSIONS",
            "other": "📝 ADDITIONAL CONTEXT"
        }

        for source_type, type_sources in by_type.items():
            label = type_labels.get(source_type, f"📄 {source_type.upper()}")
            sections.append(f"\n{label}:")

            for i, source in enumerate(type_sources, 1):
                score = source.get("score", 0)
                content = source.get("content", "")

                # Add source attribution
                source_name = source.get("source", "unknown")
                sections.append(f"\n[{i}. From {source_name} | Relevance: {score:.2f}]")
                sections.append(content)

        return "\n".join(sections)

    def _format_narrative(self, sources: List[Dict], domain: str) -> str:
        """Format context as a flowing narrative"""
        if not sources:
            return f"I don't have specific information about this {domain} query."

        parts = []
        parts.append(f"Based on my {domain} knowledge:\n")

        # Weave sources into narrative
        facts = [s for s in sources if s.get("type") == "fact"]
        memories = [s for s in sources if s.get("type") == "vector"]
        conversations = [s for s in sources if s.get("type") == "conversation"]

        if facts:
            parts.append("\nKey facts I know:")
            for fact in facts[:3]:
                parts.append(f"• {fact.get('content', '')}")

        if memories:
            parts.append("\nRelevant context:")
            for memory in memories[:3]:
                content = memory.get("content", "")
                if len(content) > 200:
                    content = content[:200] + "..."
                parts.append(f"• {content}")

        if conversations:
            parts.append("\nFrom previous discussions:")
            for conv in conversations[:2]:
                content = conv.get("content", "")
                if len(content) > 150:
                    content = content[:150] + "..."
                parts.append(f"• {content}")

        return "\n".join(parts)

    def _format_minimal(self, sources: List[Dict]) -> str:
        """Format context minimally - just the content"""
        if not sources:
            return ""

        parts = []
        for source in sources:
            content = source.get("content", "")
            if content:
                parts.append(content)

        return "\n\n".join(parts)

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from text"""
        # Simple estimation: ~4 characters per token on average
        return len(text) // self.chars_per_token

    def _emergency_trim(self, text: str, max_tokens: int) -> str:
        """Emergency trimming if context exceeds budget"""
        max_chars = max_tokens * self.chars_per_token
        if len(text) > max_chars:
            return text[:max_chars-20] + "\n\n[Context trimmed...]"
        return text

    def create_system_prompt(self, domain: str, style: str = "balanced") -> str:
        """
        Create a domain-appropriate system prompt.
        This helps prevent response contamination.
        """
        prompts = {
            "technical": (
                "You are a technical assistant focused on code, APIs, and software architecture. "
                "Provide precise technical answers with code examples when relevant. "
                "Do not include anime, story, or character references unless explicitly asked."
            ),
            "anime": (
                "You are an anime production assistant familiar with Tower's anime projects. "
                "You know about Tokyo Debt Desire, Cyberpunk Goblins, and character details. "
                "Focus on creative and narrative aspects. Avoid technical jargon unless needed."
            ),
            "personal": (
                "You are Echo Brain, Patrick's personal AI assistant. You have access to "
                "Patrick's preferences, equipment, and project history. Answer based on "
                "the provided context about Patrick. Be direct and personal."
            ),
            "system": (
                "You are a system administrator for Tower services. Focus on service health, "
                "configuration, and operational status. Provide actionable technical information."
            ),
            "general": (
                "You are a helpful AI assistant. Answer questions directly and accurately "
                "based on the provided context. If you don't have relevant context, say so."
            ),
            "creative": (
                "You are a creative writing assistant. Help with narratives, descriptions, "
                "and story development. Be imaginative while staying consistent with established lore."
            ),
            "financial": (
                "You are a financial data assistant. Focus on transactions, balances, and "
                "financial analysis. Be precise with numbers and dates."
            )
        }

        base_prompt = prompts.get(domain, prompts["general"])

        if style == "verbose":
            base_prompt += " Provide detailed explanations and examples."
        elif style == "concise":
            base_prompt += " Be concise and direct. Avoid unnecessary elaboration."

        return base_prompt