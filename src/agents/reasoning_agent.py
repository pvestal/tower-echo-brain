"""Reasoning Agent - Complex multi-step reasoning and analysis"""
import logging
from typing import Dict, List, Optional
from .context_aware_base import ContextAwareAgent

logger = logging.getLogger(__name__)

class ReasoningAgent(ContextAwareAgent):
    """Agent for complex reasoning, analysis, and decision-making"""

    SYSTEM_PROMPT = """You are an expert reasoning agent. You excel at:
1. Breaking complex problems into manageable steps
2. Analyzing tradeoffs and alternatives
3. Providing structured, logical analysis
4. Considering edge cases and counterarguments
5. Reaching well-supported conclusions

Format your response with clear sections:
## Analysis
[Break down the problem and key considerations]

## Reasoning
[Step through your logic, numbered if appropriate]

## Conclusion
[Your final answer with confidence assessment]"""

    def __init__(self):
        super().__init__(
            model_name="deepseek-r1:8b",  # R1 is optimized for reasoning
            system_prompt=self.SYSTEM_PROMPT
        )

    async def process(self, task: str, include_context: bool = True, context: Dict = None) -> Dict:
        """Process a reasoning task"""
        context = context or {}

        # Get unified context from Qdrant and PostgreSQL
        unified_context = {}
        if include_context:
            unified_context = await self.context_provider.get_context(task)

        # Combine all context
        combined_context = {
            **unified_context,
            "user_context": context
        }

        # Build chain-of-thought prompt with unified context
        prompt = self.build_prompt_with_context(task, combined_context)

        # Get reasoning response
        logger.info(f"ReasoningAgent processing: {task[:50]}...")
        response = await self.call_model(prompt)
        
        # Extract structured components
        analysis = self._extract_section(response, "Analysis")
        reasoning = self._extract_section(response, "Reasoning")
        conclusion = self._extract_section(response, "Conclusion")
        
        result = {
            "task": task,
            "response": response,
            "analysis": analysis,
            "reasoning": reasoning,
            "conclusion": conclusion,
            "model": self.model_name,
            "context_used": {
                "memories": len(unified_context.get("memories", [])),
                "facts": len(unified_context.get("facts", [])),
                "recent_conversations": len(unified_context.get("recent_conversations", []))
            }
        }

        # Update history (handled by parent class now)
        self.history.append(result)
        if len(self.history) > 50:
            self.history = self.history[-50:]

        return result
    
    def build_prompt_with_context(self, task: str, context: Dict) -> str:
        """Override parent method to handle reasoning-specific context formatting"""
        parts = []

        # Add system prompt
        parts.append(self.system_prompt)

        # Add unified context (memories, facts, recent conversations)
        if context:
            parent_formatted = self._format_context(context)
            if parent_formatted:
                parts.append("\n## Knowledge Base Context")
                parts.append(parent_formatted)

        # Add user-provided context
        user_context = context.get("user_context", {})
        if user_context.get("background"):
            parts.append(f"\nBackground: {user_context['background']}")

        if user_context.get("constraints"):
            parts.append(f"Constraints: {user_context['constraints']}")
        
        if user_context.get("options"):
            parts.append("Options to consider:")
            for opt in user_context["options"]:
                parts.append(f"- {opt}")

        if user_context.get("examples"):
            parts.append("Examples:")
            for ex in user_context["examples"]:
                parts.append(f"- {ex}")
        
        parts.append(f"\nQuestion/Task: {task}")
        parts.append("\nThink through this systematically.")
        
        return "\n".join(parts)
    
    def _extract_section(self, text: str, section: str) -> str:
        """Extract a section from formatted response"""
        markers = [f"## {section}", f"**{section}**", f"{section}:"]
        
        for marker in markers:
            if marker in text:
                start = text.find(marker) + len(marker)
                # Find next section
                next_markers = ["## ", "**", "\n\n\n"]
                end = len(text)
                for nm in next_markers:
                    pos = text.find(nm, start + 1)
                    if pos > start and pos < end:
                        end = pos
                return text[start:end].strip()
        return ""

# Singleton
reasoning_agent = ReasoningAgent()
