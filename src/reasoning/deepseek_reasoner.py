"""
DeepSeek Reasoning System - ACTUAL IMPLEMENTATION
Calls deepseek-r1:8b and parses <think> tags from responses.
"""
import re
import logging
import httpx
import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"

@dataclass
class ReasoningResult:
    """Structured reasoning output"""
    thinking_steps: List[str]
    final_answer: str
    model_used: str
    raw_response: str
    success: bool = True
    error: Optional[str] = None

def should_use_reasoning(query: str) -> bool:
    """Detect if query needs deep reasoning"""
    query_lower = query.lower()

    # Explicit reasoning requests
    reasoning_triggers = [
        'think through', 'analyze', 'compare', 'evaluate',
        'pros and cons', 'trade-off', 'should i', 'which is better',
        'why would', 'how would', 'step by step', 'reasoning',
        'architecture', 'design decision', 'strategy', 'explain how',
        'explain why', 'implications', 'consider'
    ]

    # Complex queries (long or multi-part)
    is_complex = len(query) > 150 or query.count('?') > 1

    return any(trigger in query_lower for trigger in reasoning_triggers) or is_complex

def parse_think_tags(response: str) -> Tuple[List[str], str]:
    """
    Extract thinking steps from response.
    Handles both <think> tags AND numbered/bulleted reasoning.
    """
    thinking_steps = []
    final_answer = response

    # Method 1: Extract <think> tags
    think_pattern = r'<think>(.*?)</think>'
    think_matches = re.findall(think_pattern, response, re.DOTALL)

    if think_matches:
        for match in think_matches:
            steps = [s.strip() for s in match.split('\n') if s.strip()]
            thinking_steps.extend(steps)
        final_answer = response.split('</think>')[-1].strip()
    else:
        # Method 2: Extract numbered steps (1. 2. 3. or Step 1: Step 2:)
        step_patterns = [
            r'(?:^|\n)\s*(\d+)\.\s*(.+?)(?=\n\s*\d+\.|\n\n|$)',  # "1. First..."
            r'(?:^|\n)\s*Step\s*(\d+)[:\.]?\s*(.+?)(?=Step\s*\d+|\n\n|$)',  # "Step 1: First..."
            r'(?:^|\n)\s*[-•]\s*(.+?)(?=\n\s*[-•]|\n\n|$)',  # "- First..." or "• First..."
        ]

        for pattern in step_patterns:
            matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        step_text = match[-1].strip()  # Get the text part
                    else:
                        step_text = match.strip()
                    if step_text and len(step_text) > 10:
                        thinking_steps.append(step_text)
                break

        # Method 3: If still no steps, split on sentence boundaries for long responses
        if not thinking_steps and len(response) > 200:
            sentences = re.split(r'(?<=[.!?])\s+', response)
            # Take first few sentences as "thinking", rest as answer
            if len(sentences) > 3:
                thinking_steps = sentences[:len(sentences)//2]
                final_answer = ' '.join(sentences[len(sentences)//2:])

    # Clean up steps
    thinking_steps = [s[:200] for s in thinking_steps if s and len(s) > 5]

    return thinking_steps, final_answer

async def execute_reasoning(query: str, model: str = "deepseek-r1:8b") -> ReasoningResult:
    """
    Execute reasoning query and parse structured response.
    """
    # Build reasoning-optimized prompt
    prompt = f"""You are a careful analytical thinker. Break down this problem step by step.

Format your response as:
Step 1: [first consideration]
Step 2: [next consideration]
...
Final Answer: [your conclusion]

Question: {query}"""

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            logger.info(f"🤔 Calling {model} for reasoning on: {query[:50]}...")

            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 2048,
                        "top_k": 40,
                        "top_p": 0.9,
                    }
                }
            )
            response.raise_for_status()
            result = response.json()
            raw_response = result.get("response", "")

            if not raw_response:
                logger.warning("Empty response from model")
                return ReasoningResult(
                    thinking_steps=["No response received from model"],
                    final_answer="Unable to generate response",
                    model_used=model,
                    raw_response="",
                    success=False,
                    error="Empty response"
                )

            # Parse thinking and answer
            thinking_steps, final_answer = parse_think_tags(raw_response)

            logger.info(f"🧠 DeepSeek reasoning completed: {len(thinking_steps)} thinking steps extracted")

            # If no thinking steps were extracted, but we have a response,
            # create a single step from the response
            if not thinking_steps and raw_response:
                thinking_steps = ["Direct reasoning applied"]

            return ReasoningResult(
                thinking_steps=thinking_steps,
                final_answer=final_answer or raw_response,
                model_used=model,
                raw_response=raw_response,
                success=True
            )

    except httpx.TimeoutException:
        logger.error("DeepSeek reasoning timed out")
        return ReasoningResult(
            thinking_steps=["Reasoning request timed out after 120 seconds"],
            final_answer="Unable to complete reasoning - request timed out. The query may be too complex or the model may be overloaded.",
            model_used=model,
            raw_response="",
            success=False,
            error="Timeout"
        )
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error in DeepSeek reasoning: {e}")
        return ReasoningResult(
            thinking_steps=[f"HTTP Error {e.response.status_code}"],
            final_answer=f"Reasoning failed due to HTTP error: {e.response.status_code}",
            model_used=model,
            raw_response="",
            success=False,
            error=str(e)
        )
    except Exception as e:
        logger.error(f"DeepSeek reasoning failed: {e}")
        return ReasoningResult(
            thinking_steps=[f"Error: {str(e)}"],
            final_answer=f"Reasoning failed: {str(e)}",
            model_used=model,
            raw_response="",
            success=False,
            error=str(e)
        )

def execute_reasoning_sync(query: str, model: str = "deepseek-r1:8b") -> ReasoningResult:
    """Synchronous wrapper for execute_reasoning"""
    try:
        # Get or create event loop
        try:
            loop = asyncio.get_running_loop()
            # We're already in an async context, can't use asyncio.run
            return asyncio.create_task(execute_reasoning(query, model))
        except RuntimeError:
            # No running loop, safe to use asyncio.run
            return asyncio.run(execute_reasoning(query, model))
    except Exception as e:
        logger.error(f"Failed to execute reasoning sync: {e}")
        return ReasoningResult(
            thinking_steps=[f"Sync execution error: {str(e)}"],
            final_answer="Unable to execute reasoning",
            model_used=model,
            raw_response="",
            success=False,
            error=str(e)
        )

# Legacy compatibility
class DeepSeekReasoner:
    """Legacy class for backwards compatibility"""

    def __init__(self):
        self.model = "deepseek-r1:8b"
        logger.info("🧠 DeepSeek Reasoner initialized (legacy mode)")

    def needs_reasoning(self, query: str) -> bool:
        return should_use_reasoning(query)

    async def reason(self, query: str, context: Optional[Dict] = None) -> Dict:
        """Execute reasoning and return result"""
        if not should_use_reasoning(query):
            return {
                "success": True,
                "needs_reasoning": False,
                "query": query
            }

        result = await execute_reasoning(query, self.model)

        return {
            "success": result.success,
            "needs_reasoning": True,
            "model": result.model_used,
            "response": result.final_answer,
            "thinking_steps": result.thinking_steps,
            "raw_response": result.raw_response,
            "error": result.error
        }

# Global instance for legacy compatibility
reasoner = DeepSeekReasoner()