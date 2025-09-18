#!/usr/bin/env python3
"""
Core intelligence routing system for Echo Brain with ML decision engine
"""

import asyncio
import aiohttp
import logging
import time
from .echo_system_prompt import get_echo_system_prompt
from typing import Dict, List, Optional
from echo_brain_thoughts import echo_brain
from model_decision_engine import get_decision_engine
from src.db.database import database

logger = logging.getLogger(__name__)

class EchoIntelligenceRouter:
    """Core intelligence routing system for Echo Brain with ML decision engine"""

    def __init__(self):
        self.ollama_url = "http://localhost:11434/api/generate"

        # Initialize decision engine for intelligent model selection
        self.decision_engine = get_decision_engine(database.db_config)

        # Legacy hierarchy for backward compatibility
        self.model_hierarchy = {
            "quick": "tinyllama:latest",        # 1B parameters
            "standard": "llama3.2:3b",          # 3B parameters
            "professional": "mistral:7b",       # 7B parameters
            "expert": "qwen2.5-coder:32b",     # 32B parameters
            "genius": "llama3.1:70b",          # 70B parameters
        }
        self.specialized_models = {
            "coding": "deepseek-coder:latest",  # Updated to use new lightweight version
            "creative": "mixtral:8x7b",
            "analysis": "codellama:70b"
        }
        self.escalation_history = []
        self.decision_history = []  # Track decisions for learning

    def analyze_complexity(self, query: str, context: Dict) -> str:
        """Analyze query complexity to determine optimal model"""
        complexity_score = 0.0

        # Basic query analysis
        complexity_score += len(query.split()) * 0.3
        complexity_score += query.count('?') * 3
        complexity_score += query.count('.') * 1

        # Technical complexity indicators
        technical_terms = [
            'database', 'architecture', 'algorithm', 'implementation',
            'refactor', 'optimization', 'integration', 'system'
        ]
        complexity_score += sum(8 for term in technical_terms if term.lower() in query.lower())

        # Programming language detection
        code_terms = ['python', 'javascript', 'sql', 'function', 'class', 'async']
        if any(term in query.lower() for term in code_terms):
            complexity_score += 15

        # Context complexity
        if context.get('previous_failures', 0) > 0:
            complexity_score += 20  # Escalate if previous attempts failed

        if context.get('user_expertise') == 'expert':
            complexity_score += 10

        # Route based on score (adjusted for better escalation)
        if complexity_score < 8:
            return "quick"
        elif complexity_score < 25:
            return "standard"
        elif complexity_score < 40:
            return "professional"
        elif complexity_score < 60:
            return "expert"
        else:
            return "genius"

    def detect_specialization(self, query: str) -> Optional[str]:
        """Detect if query requires specialized model"""
        query_lower = query.lower()

        if any(term in query_lower for term in ['code', 'program', 'function', 'debug']):
            return "coding"
        elif any(term in query_lower for term in ['creative', 'story', 'write', 'imagine']):
            return "creative"
        elif any(term in query_lower for term in ['analyze', 'data', 'research', 'study']):
            return "analysis"
        return None

    def _calculate_complexity_score(self, query: str, context: Dict) -> float:
        """Calculate numerical complexity score for brain visualization"""
        score = 0.0

        # Basic query analysis
        score += len(query.split()) * 0.3
        score += query.count('?') * 3
        score += query.count('.') * 1

        # Technical complexity indicators (increased scoring)
        technical_terms = [
            'database', 'architecture', 'algorithm', 'implementation',
            'refactor', 'optimization', 'integration', 'system', 'distributed',
            'microservice', 'scalable', 'performance', 'design', 'patterns'
        ]
        score += sum(12 for term in technical_terms if term.lower() in query.lower())

        # Programming language detection (increased weight)
        code_terms = ['python', 'javascript', 'sql', 'function', 'class', 'async']
        if any(term in query.lower() for term in code_terms):
            score += 20

        # Context complexity
        if context.get('previous_failures', 0) > 0:
            score += 20

        return min(score, 100.0)

    async def query_model(self, model: str, prompt: str, max_tokens: int = 2048, validate_code: bool = True) -> Dict:
        """Query specific Ollama model with automatic validation and reloading for code generation"""
        try:
            # Construct full prompt with Echo system prompt
            system_prompt = get_echo_system_prompt()
            user_prefix = "\n\nUser: "
            echo_suffix = "\n\nEcho:"
            full_prompt = system_prompt + user_prefix + prompt + echo_suffix

            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": 0.7
                    }
                }

                start_time = asyncio.get_event_loop().time()
                async with session.post(self.ollama_url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        processing_time = asyncio.get_event_loop().time() - start_time
                        response_text = result.get("response", "")

                        # Validate and reload if needed for code generation tasks
                        if validate_code and self.decision_engine:
                            final_response, final_model, validation = await self.decision_engine.validate_and_reload_if_needed(
                                response_text,
                                prompt,
                                model,
                                "code" if self._is_code_prompt(prompt) else "general"
                            )

                            if validation.is_gibberish or validation.requires_reload:
                                logger.info(f"🔄 Reloaded model from {model} to {final_model} due to quality issues")

                            return {
                                "success": True,
                                "response": final_response,
                                "processing_time": processing_time,
                                "model": final_model,
                                "original_model": model if final_model != model else None,
                                "validation_score": validation.quality_score if validation else None,
                                "was_reloaded": final_model != model
                            }

                        return {
                            "success": True,
                            "response": response_text,
                            "processing_time": processing_time,
                            "model": model
                        }
                    else:
                        return {"success": False, "error": f"HTTP {response.status}"}

        except Exception as e:
            logger.error(f"Model query failed for {model}: {e}")
            return {"success": False, "error": str(e)}

    def _is_code_prompt(self, prompt: str) -> bool:
        """Detect if the prompt is asking for code generation"""
        code_indicators = [
            "write code", "generate code", "create a function", "implement",
            "write a script", "create a class", "code for", "program",
            "python", "javascript", "sql", "bash", "typescript", "fix this",
            "debug", "error in", "syntax"
        ]
        prompt_lower = prompt.lower()
        return any(indicator in prompt_lower for indicator in code_indicators)

    async def progressive_escalation(self, query: str, context: Dict) -> Dict:
        """Implement progressive model escalation with ML-driven decision engine"""
        escalation_path = []
        start_time = time.time()

        # 🧠 START THINKING - Begin neural visualization
        thought_id = await echo_brain.start_thinking("query_processing", query)

        # 🧠 PROCESS INPUT
        await echo_brain.think_about_input(thought_id, query)

        # Use ML decision engine for intelligent model selection
        decision = await self.decision_engine.decide_model(query, context)

        # 🧠 ANALYZE COMPLEXITY
        await echo_brain.analyze_complexity(thought_id, decision["complexity_score"] / 100.0)

        model = decision["model"]
        escalation_path.append(f"{decision['tier']}:{model}")

        # 🧠 DECISION: ML-DRIVEN MODEL SELECTION
        await echo_brain.make_decision(
            thought_id,
            "ml_selection",
            f"Selected {model} based on complexity {decision['complexity_score']:.1f}"
        )

        # Check if we need to use API
        if decision.get("use_api"):
            # Use DeepSeek API for extreme complexity
            result = await self._query_deepseek_api(query)
            escalation_path.append("api_fallback")
        else:
            # 🧠 GENERATE RESPONSE
            await echo_brain.generate_response(thought_id, f"{decision['tier']} response")

            # Attempt query with selected model
            result = await self.query_model(model, query)

        if result["success"]:
            # 🧠 SUCCESS - Complete thinking
            await echo_brain.finish_thinking(thought_id)

            # Record performance for learning
            elapsed = time.time() - start_time
            token_count = len(result.get("response", "").split())

            return {
                **result,
                "intelligence_level": decision["tier"],
                "escalation_path": escalation_path,
                "thought_id": thought_id,
                "brain_activity": echo_brain.get_brain_state(),
                "complexity_score": decision["complexity_score"],
                "decision_reason": decision["reason"]
            }
        else:
            # 🧠 ESCALATION NEEDED
            await echo_brain.emotional_response(thought_id, "concern", "Initial model failed")

            # Try API fallback for failed queries
            if decision["tier"] != "cloud":
                logger.info(f"Escalating to DeepSeek API after {model} failure")
                escalation_path.append("api_fallback")

                # 🧠 ESCALATION DECISION
                await echo_brain.make_decision(thought_id, "api_escalation", "Escalating to cloud API")

                fallback_result = await self._query_deepseek_api(query)
                await echo_brain.finish_thinking(thought_id)

                return {
                    **fallback_result,
                    "intelligence_level": "cloud",
                    "escalation_path": escalation_path,
                    "thought_id": thought_id,
                    "brain_activity": echo_brain.get_brain_state()
                }
            else:
                await echo_brain.finish_thinking(thought_id)
                return {
                    **result,
                    "thought_id": thought_id,
                    "brain_activity": echo_brain.get_brain_state()
                }

    async def _query_deepseek_api(self, query: str) -> Dict:
        """Query DeepSeek API for extreme complexity"""
        try:
            # Use existing DeepSeek service on port 8306
            async with aiohttp.ClientSession() as session:
                payload = {
                    "prompt": query,
                    "model": "deepseek-coder",
                    "max_tokens": 2048
                }

                async with session.post(
                    "http://localhost:8306/api/deepseek/chat",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "success": True,
                            "response": result.get("response", ""),
                            "processing_time": result.get("processing_time", 0),
                            "model": "deepseek-api"
                        }
                    else:
                        return {"success": False, "error": f"API returned {response.status}"}

        except Exception as e:
            logger.error(f"DeepSeek API query failed: {e}")
            return {"success": False, "error": str(e)}

# Global intelligence router instance
intelligence_router = EchoIntelligenceRouter()