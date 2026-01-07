"""
DeepSeek Reasoning System with <think> tag support.
Enables chain-of-thought reasoning for complex queries.
"""

import re
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)

class DeepSeekReasoner:
    """
    Handles DeepSeek R1 model's reasoning capabilities.
    Processes <think> tags for internal reasoning steps.
    """

    def __init__(self):
        self.model = "deepseek-r1:8b"
        self.think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
        logger.info("🧠 DeepSeek Reasoner initialized")

    def needs_reasoning(self, query: str) -> bool:
        """
        Determine if a query requires deep reasoning.
        """
        reasoning_indicators = [
            'explain how', 'explain why', 'analyze', 'compare',
            'evaluate', 'design', 'architect', 'solve',
            'calculate', 'derive', 'prove', 'optimize',
            'trade-off', 'pros and cons', 'implications'
        ]

        query_lower = query.lower()

        # Check for reasoning indicators
        if any(indicator in query_lower for indicator in reasoning_indicators):
            return True

        # Check for complex multi-part questions
        if query.count('?') > 1 or len(query) > 200:
            return True

        return False

    def format_reasoning_prompt(self, query: str) -> str:
        """
        Format query with reasoning instructions for DeepSeek.
        """
        return f"""<think>
First, I need to understand what is being asked and break it down into logical steps.
Then I'll work through each part systematically to arrive at a comprehensive answer.
</think>

{query}

Please provide a detailed, step-by-step response. Use <think> tags for your internal reasoning process, then provide the final answer clearly."""

    def extract_reasoning_and_answer(self, response: str) -> Dict[str, str]:
        """
        Extract thinking process and final answer from response.
        """
        # Find all thinking sections
        thinking_sections = self.think_pattern.findall(response)

        # Remove thinking sections to get the answer
        answer = self.think_pattern.sub('', response).strip()

        # Clean up the thinking text
        thinking = '\n'.join(thinking_sections).strip() if thinking_sections else ""

        return {
            "thinking": thinking,
            "answer": answer,
            "full_response": response
        }

    async def reason(self, query: str, context: Optional[Dict] = None) -> Dict:
        """
        Execute reasoning process with DeepSeek model.
        """
        try:
            # Check if reasoning is needed
            if not self.needs_reasoning(query):
                logger.info("Query doesn't require deep reasoning, using standard response")
                return {
                    "success": True,
                    "needs_reasoning": False,
                    "query": query
                }

            logger.info(f"🤔 Deep reasoning triggered for: {query[:50]}...")

            # Format the prompt with reasoning instructions
            reasoning_prompt = self.format_reasoning_prompt(query)

            # Here we would normally call the model
            # For now, return structured response
            return {
                "success": True,
                "needs_reasoning": True,
                "model": self.model,
                "prompt": reasoning_prompt,
                "query": query,
                "reasoning_triggered": True
            }

        except Exception as e:
            logger.error(f"Reasoning error: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def process_response(self, response: str) -> Dict:
        """
        Process a response that may contain thinking tags.
        """
        result = self.extract_reasoning_and_answer(response)

        # Log if reasoning was detected
        if result["thinking"]:
            logger.info(f"🧠 Extracted {len(result['thinking'])} chars of reasoning")

        return {
            "has_reasoning": bool(result["thinking"]),
            "thinking": result["thinking"],
            "answer": result["answer"],
            "thinking_steps": len(result["thinking"].split('\n')) if result["thinking"] else 0
        }

# Global instance
reasoner = DeepSeekReasoner()

def should_use_reasoning(query: str) -> bool:
    """Quick check if query needs reasoning."""
    return reasoner.needs_reasoning(query)