"""Reasoning Agent - Complex multi-step reasoning and analysis"""
import logging
from typing import Dict, List, Optional
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class ReasoningAgent(BaseAgent):
    """Agent for complex reasoning, analysis, and decision-making"""
    
    def __init__(self):
        super().__init__(
            name="ReasoningAgent",
            model="deepseek-r1:8b"  # R1 is optimized for reasoning
        )
        self.system_prompt = """You are an expert reasoning agent. You excel at:
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

    async def process(self, task: str, context: Dict = None) -> Dict:
        """Process a reasoning task"""
        context = context or {}
        
        # Build chain-of-thought prompt
        prompt = self._build_prompt(task, context)
        
        # Get reasoning response
        logger.info(f"ReasoningAgent processing: {task[:50]}...")
        response = await self.call_model(prompt, self.system_prompt)
        
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
            "model": self.model
        }
        
        self.add_to_history(task, {"has_conclusion": bool(conclusion)})
        return result
    
    def _build_prompt(self, task: str, context: Dict) -> str:
        """Build reasoning prompt with context"""
        parts = []
        
        if context.get("background"):
            parts.append(f"Background: {context['background']}")
        
        if context.get("constraints"):
            parts.append(f"Constraints: {context['constraints']}")
        
        if context.get("options"):
            parts.append("Options to consider:")
            for opt in context["options"]:
                parts.append(f"- {opt}")
        
        if context.get("examples"):
            parts.append("Examples:")
            for ex in context["examples"]:
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
