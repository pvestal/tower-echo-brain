"""
Agent Router for Echo Brain
Routes queries to the most appropriate agent
"""

import logging
import re
from typing import Dict, Any, Optional, Literal
from src.agents.coding_agent import coding_agent
from src.agents.reasoning_agent import reasoning_agent
from src.agents.narration_agent import narration_agent
from src.core.unified_context import get_context_provider

logger = logging.getLogger(__name__)

AgentType = Literal["coding", "reasoning", "narration", "general", "auto"]


class AgentRouter:
    """Routes queries to the most appropriate agent"""

    def __init__(self):
        self.coding_agent = coding_agent
        self.reasoning_agent = reasoning_agent
        self.narration_agent = narration_agent
        self.context_provider = get_context_provider()

        # Classification keywords
        self.coding_keywords = [
            "code", "debug", "fix", "implement", "function", "class",
            "error", "bug", "python", "javascript", "typescript",
            "refactor", "optimize", "algorithm", "script", "program",
            "compile", "syntax", "variable", "method", "api", "endpoint",
            "test", "unittest", "exception", "traceback", "import"
        ]

        self.reasoning_keywords = [
            "analyze", "decide", "compare", "evaluate", "think",
            "pros", "cons", "should", "would", "which", "why",
            "explain", "understand", "reason", "logic", "consider",
            "strategy", "plan", "approach", "solution", "alternative",
            "tradeoff", "assess", "determine", "conclude"
        ]

        self.narration_keywords = [
            "story", "anime", "character", "scene", "dialogue",
            "narrative", "describe", "narrate", "mood", "atmosphere",
            "visual", "cinematic", "dramatic", "emotional", "creative",
            "write", "compose", "craft", "goblin", "cyborg", "tokyo",
            "action", "romance", "adventure", "fantasy"
        ]

    async def route(self, query: str, agent: AgentType = "auto",
                    include_context: bool = True) -> Dict[str, Any]:
        """
        Route a query to the appropriate agent

        Args:
            query: The user's query
            agent: Specific agent to use, or "auto" for automatic selection
            include_context: Whether to include context from knowledge base

        Returns:
            Response from the selected agent
        """
        try:
            # Determine which agent to use
            if agent == "auto":
                selected_agent = self.classify_intent(query)
            else:
                # Validate agent name and default to reasoning for invalid names
                valid_agents = ["coding", "reasoning", "narration", "general"]
                if agent in valid_agents:
                    selected_agent = agent if agent != "general" else "reasoning"
                else:
                    logger.warning(f"Invalid agent name '{agent}', defaulting to reasoning")
                    selected_agent = "reasoning"

            logger.info(f"Routing query to {selected_agent} agent: {query[:50]}...")

            # Process with the selected agent
            if selected_agent == "coding":
                result = await self.coding_agent.process(query, include_context=include_context)
                # Ensure model field exists
                if "model" not in result:
                    result["model"] = self.coding_agent.model_name
            elif selected_agent == "narration":
                result = await self.narration_agent.process(query, include_context=include_context)
                # Standardize NarrationAgent response fields
                if "scene" in result and "task" not in result:
                    result["task"] = result.get("scene", query)
                if "narration" in result and "response" not in result:
                    result["response"] = result.get("narration", "")
                # Ensure model field exists
                if "model" not in result:
                    result["model"] = self.narration_agent.model_name
            else:  # reasoning or general
                result = await self.reasoning_agent.process(query, include_context=include_context)
                # Ensure model field exists
                if "model" not in result:
                    result["model"] = self.reasoning_agent.model_name

            # Ensure required fields exist for API response
            if "task" not in result:
                result["task"] = query
            if "response" not in result:
                result["response"] = result.get("narration", result.get("analysis", "No response generated"))

            # Add routing metadata (ensure these are always strings)
            result["agent_used"] = selected_agent
            result["routing_method"] = "manual" if agent != "auto" else "automatic"

            return result

        except Exception as e:
            logger.error(f"Agent routing failed: {e}")

            # Fallback to reasoning agent with context only
            try:
                logger.info("Falling back to reasoning agent with context...")
                context = await self.context_provider.get_context(query) if include_context else {}

                return {
                    "task": query,
                    "response": f"I encountered an error but here's what I found in the knowledge base:\n\n{self._format_fallback_context(context)}",
                    "agent_used": "reasoning",
                    "routing_method": "fallback",
                    "model": "fallback",
                    "error": str(e),
                    "context_used": context
                }
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                return {
                    "task": query,
                    "response": "I'm having trouble processing your request. Please try again.",
                    "agent_used": "none",  # Changed from None to "none" string
                    "routing_method": "failed",
                    "model": "none",  # Add model field
                    "error": str(e)
                }

    def classify_intent(self, query: str) -> str:
        """
        Classify the intent of a query to determine the best agent

        Returns: 'coding' | 'reasoning' | 'narration' | 'general'
        """
        query_lower = query.lower()

        # Count keyword matches for each category
        coding_score = sum(1 for kw in self.coding_keywords if kw in query_lower)
        reasoning_score = sum(1 for kw in self.reasoning_keywords if kw in query_lower)
        narration_score = sum(1 for kw in self.narration_keywords if kw in query_lower)

        # Strong indicators (immediate classification)
        if any(pattern in query_lower for pattern in ["write.*code", "debug.*error", "fix.*bug", "implement.*function"]):
            return "coding"

        if any(pattern in query_lower for pattern in ["anime.*scene", "write.*story", "create.*narration", "describe.*character"]):
            return "narration"

        if any(pattern in query_lower for pattern in ["should i", "pros.*cons", "compare.*options", "analyze.*decision"]):
            return "reasoning"

        # Score-based classification
        scores = {
            "coding": coding_score * 1.2,  # Slight bias toward coding for technical queries
            "reasoning": reasoning_score,
            "narration": narration_score * 1.1  # Slight bias for creative tasks
        }

        # If there's a clear winner
        max_score = max(scores.values())
        if max_score > 0:
            for agent_type, score in scores.items():
                if score == max_score:
                    logger.debug(f"Classified as {agent_type} (score: {score})")
                    return agent_type

        # Default heuristics for edge cases
        if len(query) > 200:  # Long queries often need reasoning
            return "reasoning"

        if "?" in query:  # Questions often need reasoning
            return "reasoning"

        # Default to reasoning for general queries
        return "reasoning"

    async def get_context_only(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """
        Get context without agent processing
        Useful for external consumers who want to use context themselves

        Args:
            query: The search query
            limit: Maximum number of results per source

        Returns:
            Context from all sources
        """
        return await self.context_provider.get_context(query, limit)

    def _format_fallback_context(self, context: Dict) -> str:
        """Format context for fallback response"""
        if not context:
            return "No relevant context found."

        parts = []

        memories = context.get("memories", [])
        if memories:
            parts.append("**Related Information:**")
            for mem in memories[:3]:
                parts.append(f"- {mem.get('text', '')[:200]}")

        facts = context.get("facts", [])
        if facts:
            parts.append("\n**Known Facts:**")
            for fact in facts[:5]:
                parts.append(f"- {fact['subject']} {fact['predicate']} {fact['object']}")

        return "\n".join(parts) if parts else "Limited context available."

    async def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        return {
            "agents": [
                {
                    "name": "CodingAgent",
                    "model": self.coding_agent.model_name,
                    "status": "active",
                    "tasks_processed": len(self.coding_agent.history)
                },
                {
                    "name": "ReasoningAgent",
                    "model": self.reasoning_agent.model_name,
                    "status": "active",
                    "tasks_processed": len(self.reasoning_agent.history)
                },
                {
                    "name": "NarrationAgent",
                    "model": self.narration_agent.model_name,
                    "status": "active",
                    "tasks_processed": len(self.narration_agent.history)
                }
            ],
            "routing_stats": {
                "classification_keywords": {
                    "coding": len(self.coding_keywords),
                    "reasoning": len(self.reasoning_keywords),
                    "narration": len(self.narration_keywords)
                }
            }
        }


# Singleton instance
_agent_router = None

def get_agent_router() -> AgentRouter:
    """Get or create the singleton agent router"""
    global _agent_router
    if _agent_router is None:
        _agent_router = AgentRouter()
    return _agent_router