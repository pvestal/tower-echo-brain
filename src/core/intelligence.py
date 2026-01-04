#!/usr/bin/env python3
"""
Core intelligence routing system for Echo Brain with ML decision engine
"""

import asyncio
import aiohttp
import logging
import time
from src.core.echo_system_prompt import get_echo_system_prompt
from src.core.complexity_analyzer import ComplexityAnalyzer
from typing import Dict, List, Optional
from src.core.echo.echo_brain_thoughts import echo_brain
from src.misc.model_decision_engine import get_decision_engine
from src.db.database import database
from src.core.resilient_context import get_resilient_omniscient_context

logger = logging.getLogger(__name__)

class EchoIntelligenceRouter:
    """Core intelligence routing system for Echo Brain with ML decision engine"""

    def __init__(self):
        self.ollama_url = "http://localhost:11434/api/generate"

        # Initialize decision engine for intelligent model selection
        self.decision_engine = get_decision_engine(database.db_config)

        # ENHANCED hierarchy with DeepSeek integration
        self.model_hierarchy = {
            "quick": "llama3.1:8b",                # Fast local model
            "standard": "deepseek-r1:8b",          # DeepSeek for general queries
            "professional": "deepseek-r1:8b",      # DeepSeek for professional queries
            "expert": "deepseek-coder-v2:16b",     # DeepSeek for complex reasoning
            "genius": "deepseek-coder-v2:16b",     # DeepSeek heavyweight model
        }
        self.specialized_models = {
            "coding": "deepseek-coder-v2:16b",      # DeepSeek excels at coding
            "creative": "deepseek-r1:8b",           # DeepSeek for creative tasks
            "analysis": "deepseek-coder-v2:16b",    # DeepSeek for deep analysis
            "reasoning": "deepseek-coder-v2:16b",   # Complex reasoning tasks
            "math": "deepseek-coder-v2:16b"         # Mathematical problem solving
        }
        self.escalation_history = []
        self.decision_history = []  # Track decisions for learning

        # Import fixed model selector
        try:
            from src.core.model_selector_fix import model_selector
            self.model_selector = model_selector
        except ImportError:
            self.model_selector = None

    def analyze_complexity(self, query: str, context: Dict) -> str:
        """
        Analyze query complexity to determine optimal model tier
        REFACTORED: Now delegates to ComplexityAnalyzer (Oct 22, 2025)
        """
        result = ComplexityAnalyzer.analyze(query, context)
        return result.tier  # Returns: tiny, small, medium, large, cloud

    
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
        """
        Calculate numerical complexity score for brain visualization
        REFACTORED: Now delegates to ComplexityAnalyzer (Oct 22, 2025)
        """
        result = ComplexityAnalyzer.analyze(query, context)
        return result.score  # Returns: 0-100

    
    async def query_model(self, model: str, prompt: str, context: Dict = None, max_tokens: int = 2048, validate_code: bool = True) -> Dict:
        """Query specific Ollama model with conversation context and automatic validation"""
        try:
            # MEMORY AUGMENTATION - Add stored memories to prompt
            try:
                from src.middleware.memory_augmentation_middleware import augment_with_memories
                import uuid
                original_prompt = prompt
                request_id = str(uuid.uuid4())
                prompt = augment_with_memories(prompt, request_id)
                if prompt != original_prompt:
                    logger.info(f"📚 Query augmented with stored memories (request_id: {request_id})")
            except Exception as e:
                logger.debug(f"Memory augmentation skipped: {e}")

            # Construct full prompt with Echo system prompt and conversation history
            system_prompt = get_echo_system_prompt()

            # Add conversation history if available
            conversation_context = ""
            if context and context.get("conversation_history"):
                logger.info(f"🧠 CONTEXT: Found {len(context['conversation_history'])} previous messages")
                conversation_context = "\n\nPREVIOUS CONVERSATION:\n"
                for i, msg in enumerate(context["conversation_history"][-5:]):  # Last 5 messages for context
                    logger.info(f"🧠 MESSAGE {i}: {msg.get('query_text', '')[:50]}...")
                    conversation_context += f"User: {msg.get('query_text', '')}\nEcho: {msg.get('response', '')}\n"
                conversation_context += "\nCURRENT CONVERSATION:"
            else:
                logger.warning("🧠 NO CONTEXT: conversation_history not found in context")

            # 🛡️ RESILIENT OMNISCIENT CONTEXT INTEGRATION
            omniscient_context = ""
            try:
                context_manager = get_resilient_omniscient_context()
                if not context_manager.connection:
                    await context_manager.connect()

                conversation_id = context.get("conversation_id") if context else None
                omniscient_summary = await context_manager.build_context_summary_resilient(prompt, conversation_id)
                omniscient_context = f"\n\n{omniscient_summary}\n"
                logger.info(f"🛡️ RESILIENT OMNISCIENT CONTEXT: Error-resistant context integration")
            except Exception as e:
                logger.warning(f"🛡️ RESILIENT OMNISCIENT CONTEXT: Gracefully handled context failure - {e}")
                # Context system will provide fallback results automatically
                omniscient_context = "\n\n📋 BASIC CONTEXT: System working with fallback data\n"

            user_prefix = "\n\nUser: "
            echo_suffix = "\n\nEcho:"
            full_prompt = system_prompt + conversation_context + omniscient_context + user_prefix + prompt + echo_suffix

            # Log basic prompt info
            logger.info(f"🔍 Prompt length: {len(full_prompt)} chars, context: {len(omniscient_context)} chars")

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

                # Handle DeepSeek models via DeepSeek API
                if model.startswith("deepseek"):
                    return await self._query_deepseek_via_adapter(model, prompt, max_tokens)

                async with session.post(self.ollama_url, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as response:
                    if response.status == 200:
                        result = await response.json()
                        processing_time = asyncio.get_event_loop().time() - start_time
                        response_text = result.get("response", "")

                        # DISABLED: Validation system was corrupting responses
                        # if validate_code and self.decision_engine:
                        #     final_response, final_model, validation = await self.decision_engine.validate_and_reload_if_needed(
                        #         response_text,
                        #         prompt,
                        #         model,
                        #         "code" if self._is_code_prompt(prompt) else "general"
                        #     )

                        #     if validation.is_gibberish or validation.requires_reload:
                        #         logger.info(f"🔄 Reloaded model from {model} to {final_model} due to quality issues")

                        #     return {
                        #         "success": True,
                        #         "response": final_response,
                        #         "processing_time": processing_time,
                        #         "model": final_model,
                        #         "original_model": model if final_model != model else None,
                        #         "validation_score": validation.quality_score if validation else None,
                        #         "was_reloaded": final_model != model
                        #     }

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

        # Check if we need to use larger model
        if decision.get("use_api"):
            # Use larger local model for extreme complexity
            result = await self._query_ollama_model("deepseek-coder-v2:16b", query)
            escalation_path.append("large_model")
        else:
            # 🧠 GENERATE RESPONSE
            await echo_brain.generate_response(thought_id, f"{decision['tier']} response")

            # Attempt query with selected model and conversation context
            result = await self.query_model(model, query, context)

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

            # Try local model escalation for failed queries
            if decision["tier"] != "cloud":
                logger.info(f"Escalating to larger local model after {model} failure")
                escalation_path.append("model_escalation")

                # 🧠 ESCALATION DECISION
                await echo_brain.make_decision(thought_id, "model_escalation", "Escalating to larger local model")

                fallback_result = await self._query_ollama_model("deepseek-coder-v2:16b", query)
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

    async def _query_deepseek_via_adapter(self, model: str, prompt: str, max_tokens: int) -> Dict:
        """Query DeepSeek using the new adapter with vault integration"""
        try:
            from src.models.deepseek_adapter import create_deepseek_adapter

            start_time = asyncio.get_event_loop().time()

            # Determine adapter type based on model
            adapter_type = "reasoner" if "reasoner" in model else "chat"
            adapter = create_deepseek_adapter(adapter_type)

            if not adapter.client:
                return {"success": False, "error": "DeepSeek adapter initialization failed"}

            # Prepare messages
            messages = [{"role": "user", "content": prompt}]

            # Call DeepSeek
            response = adapter.generate(
                messages=messages,
                temperature=0.7,
                max_tokens=max_tokens
            )

            processing_time = asyncio.get_event_loop().time() - start_time

            return {
                "success": True,
                "response": response.content,
                "processing_time": processing_time,
                "model": response.model_used,
                "reasoning_steps": response.reasoning_steps,
                "token_usage": response.token_usage
            }

        except Exception as e:
            logger.error(f"DeepSeek adapter query failed: {e}")
            return {"success": False, "error": str(e)}

    async def _query_deepseek_api(self, query: str) -> Dict:
        """Legacy DeepSeek API method - replaced by adapter"""
        return await self._query_deepseek_via_adapter("deepseek-reasoner", query, 2048)

# Global intelligence router instance
intelligence_router = EchoIntelligenceRouter()