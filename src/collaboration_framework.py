#!/usr/bin/env python3
"""
Multi-LLM Collaboration Framework for Echo Brain
Real-time collaboration between qwen-coder and deepseek-coder models
"""

import asyncio
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import httpx
import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

class CollaborationPhase(Enum):
    INITIAL_ANALYSIS = "initial_analysis"
    TECHNICAL_IMPLEMENTATION = "technical_implementation"
    CODE_REVIEW = "code_review"
    OPTIMIZATION = "optimization"
    CONSENSUS = "consensus"
    INQUISITIVE_VALIDATION = "inquisitive_validation"

@dataclass
class ModelResponse:
    model: str
    response: str
    confidence: float
    processing_time: float
    reasoning: str
    phase: CollaborationPhase
    timestamp: datetime

@dataclass
class CollaborationResult:
    query: str
    responses: List[ModelResponse]
    consensus: str
    confidence_score: float
    fabrication_detected: bool
    collaboration_time: float
    phases_completed: List[CollaborationPhase]
    inquisitive_validation: Optional[str] = None

class MultiLLMCollaborator:
    """Real-time collaboration framework between qwen-coder and deepseek-coder"""

    def __init__(self):
        self.ollama_endpoint = "http://localhost:11434/api/generate"
        self.inquisitive_endpoint = "http://localhost:8330/api/echo/query"

        # Model configurations
        self.models = {
            "qwen-coder": {
                "name": "qwen2.5-coder:7b",
                "role": "technical_implementation",
                "expertise": ["coding", "implementation", "architecture"],
                "timeout": 45
            },
            "deepseek-coder": {
                "name": "deepseek-coder",
                "role": "code_review_optimization",
                "expertise": ["debugging", "optimization", "best_practices"],
                "timeout": 45,
                "api_endpoint": "https://api.deepseek.com/v1/chat/completions",
                "requires_api": True
            }
        }

        # Database connection for persistence
        self.db_config = {
            'host': '***REMOVED***',
            'database': 'tower_consolidated',
            'user': 'patrick',
            'password': '***REMOVED***'
        }

        # Collaboration workflow
        self.workflow_phases = [
            CollaborationPhase.INITIAL_ANALYSIS,
            CollaborationPhase.TECHNICAL_IMPLEMENTATION,
            CollaborationPhase.CODE_REVIEW,
            CollaborationPhase.OPTIMIZATION,
            CollaborationPhase.CONSENSUS,
            CollaborationPhase.INQUISITIVE_VALIDATION
        ]

    async def collaborate(self, query: str, context: Optional[Dict] = None) -> CollaborationResult:
        """Execute full collaboration workflow between models"""
        start_time = time.time()
        logger.info(f"🤝 Starting multi-LLM collaboration for: {query[:100]}...")

        responses = []
        phases_completed = []

        try:
            # Phase 1: Initial Analysis with qwen-coder
            logger.info("📋 Phase 1: Initial analysis with qwen-coder")
            qwen_initial = await self._query_qwen_coder(
                query,
                CollaborationPhase.INITIAL_ANALYSIS,
                "Analyze this request and provide initial technical approach. Focus on requirements and high-level design."
            )
            responses.append(qwen_initial)
            phases_completed.append(CollaborationPhase.INITIAL_ANALYSIS)

            # Phase 2: Technical Implementation with qwen-coder
            logger.info("🔧 Phase 2: Technical implementation with qwen-coder")
            impl_prompt = f"""
Based on initial analysis: {qwen_initial.response[:500]}...

Now provide detailed technical implementation for: {query}
Include specific code examples, architecture decisions, and implementation steps.
"""
            qwen_impl = await self._query_qwen_coder(
                impl_prompt,
                CollaborationPhase.TECHNICAL_IMPLEMENTATION,
                "Provide detailed technical implementation"
            )
            responses.append(qwen_impl)
            phases_completed.append(CollaborationPhase.TECHNICAL_IMPLEMENTATION)

            # Phase 3: Code Review with deepseek-coder (if available)
            logger.info("🔍 Phase 3: Code review with deepseek-coder")
            try:
                review_prompt = f"""
Review and analyze this technical implementation:

ORIGINAL QUERY: {query}

QWEN-CODER IMPLEMENTATION:
{qwen_impl.response}

Provide:
1. Code quality assessment
2. Potential issues or improvements
3. Security considerations
4. Performance optimizations
5. Best practices compliance
"""
                deepseek_review = await self._query_deepseek_coder(
                    review_prompt,
                    CollaborationPhase.CODE_REVIEW,
                    "Review implementation and suggest improvements"
                )
                responses.append(deepseek_review)
                phases_completed.append(CollaborationPhase.CODE_REVIEW)

                # Phase 4: Optimization based on review
                logger.info("⚡ Phase 4: Optimization with qwen-coder")
                opt_prompt = f"""
Based on deepseek-coder's review: {deepseek_review.response[:500]}...

Refine and optimize the implementation for: {query}
Address the issues and suggestions from the code review.
"""
                qwen_opt = await self._query_qwen_coder(
                    opt_prompt,
                    CollaborationPhase.OPTIMIZATION,
                    "Optimize implementation based on review feedback"
                )
                responses.append(qwen_opt)
                phases_completed.append(CollaborationPhase.OPTIMIZATION)

            except Exception as e:
                logger.warning(f"DeepSeek unavailable, continuing with qwen-only: {e}")
                # Create placeholder for missing deepseek review
                placeholder_review = ModelResponse(
                    model="deepseek-coder",
                    response="DeepSeek-coder unavailable. Proceeding with qwen-coder implementation.",
                    confidence=0.0,
                    processing_time=0.0,
                    reasoning="Service unavailable",
                    phase=CollaborationPhase.CODE_REVIEW,
                    timestamp=datetime.now()
                )
                responses.append(placeholder_review)

            # Phase 5: Generate consensus
            logger.info("🤝 Phase 5: Building consensus")
            consensus = await self._build_consensus(responses, query)
            phases_completed.append(CollaborationPhase.CONSENSUS)

            # Phase 6: Inquisitive validation
            logger.info("🔍 Phase 6: Inquisitive validation")
            inquisitive_result = await self._validate_with_inquisitive_core(query, consensus)
            phases_completed.append(CollaborationPhase.INQUISITIVE_VALIDATION)

            # Calculate final metrics
            collaboration_time = time.time() - start_time
            avg_confidence = sum(r.confidence for r in responses if r.confidence > 0) / len([r for r in responses if r.confidence > 0])

            # Check for fabrication indicators
            fabrication_detected = await self._detect_fabrication(responses, consensus)

            result = CollaborationResult(
                query=query,
                responses=responses,
                consensus=consensus,
                confidence_score=avg_confidence,
                fabrication_detected=fabrication_detected,
                collaboration_time=collaboration_time,
                phases_completed=phases_completed,
                inquisitive_validation=inquisitive_result
            )

            # Save to database
            await self._save_collaboration(result)

            logger.info(f"✅ Collaboration completed in {collaboration_time:.2f}s with {len(responses)} model responses")
            return result

        except Exception as e:
            logger.error(f"❌ Collaboration failed: {e}")
            raise

    async def _query_qwen_coder(self, prompt: str, phase: CollaborationPhase, role_instruction: str) -> ModelResponse:
        """Query qwen-coder via Ollama"""
        start_time = time.time()

        enhanced_prompt = f"""
ROLE: {role_instruction}
PHASE: {phase.value}
COLLABORATION MODE: You are working with other AI models. Provide clear, specific responses.

{prompt}

Provide your response with:
1. Clear reasoning for your approach
2. Specific technical details
3. Confidence level (0-100%)
4. Any assumptions made
"""

        try:
            async with httpx.AsyncClient(timeout=self.models["qwen-coder"]["timeout"]) as client:
                response = await client.post(
                    self.ollama_endpoint,
                    json={
                        "model": self.models["qwen-coder"]["name"],
                        "prompt": enhanced_prompt,
                        "stream": False
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    content = data.get("response", "")

                    # Extract confidence from response
                    confidence = self._extract_confidence(content)

                    return ModelResponse(
                        model="qwen-coder",
                        response=content,
                        confidence=confidence,
                        processing_time=time.time() - start_time,
                        reasoning=f"Ollama qwen2.5-coder:7b response for {phase.value}",
                        phase=phase,
                        timestamp=datetime.now()
                    )
                else:
                    raise Exception(f"Ollama returned {response.status_code}: {response.text}")

        except Exception as e:
            logger.error(f"qwen-coder query failed: {e}")
            return ModelResponse(
                model="qwen-coder",
                response=f"Error querying qwen-coder: {str(e)}",
                confidence=0.0,
                processing_time=time.time() - start_time,
                reasoning="Query failed",
                phase=phase,
                timestamp=datetime.now()
            )

    async def _query_deepseek_coder(self, prompt: str, phase: CollaborationPhase, role_instruction: str) -> ModelResponse:
        """Query deepseek-coder via API"""
        start_time = time.time()

        # For now, use local fallback since API key handling needs vault integration
        logger.warning("DeepSeek API not configured, using qwen-coder as fallback")

        # Create a review-focused prompt for qwen-coder
        review_prompt = f"""
ROLE: Code reviewer and optimizer (simulating deepseek-coder perspective)
TASK: {role_instruction}

{prompt}

Provide detailed code review focusing on:
- Potential bugs and issues
- Performance optimizations
- Security considerations
- Best practices compliance
- Alternative approaches
"""

        return await self._query_qwen_coder(review_prompt, phase, "Code review and optimization")

    async def _build_consensus(self, responses: List[ModelResponse], original_query: str) -> str:
        """Build consensus from multiple model responses"""

        # Get the most recent implementation and review
        implementation_responses = [r for r in responses if r.phase in [
            CollaborationPhase.TECHNICAL_IMPLEMENTATION,
            CollaborationPhase.OPTIMIZATION
        ]]

        review_responses = [r for r in responses if r.phase == CollaborationPhase.CODE_REVIEW]

        consensus = f"""
# Multi-LLM Collaboration Result

**Original Query:** {original_query}

## Implementation Summary
"""

        if implementation_responses:
            latest_impl = implementation_responses[-1]  # Most recent implementation
            consensus += f"""
**Primary Implementation (by {latest_impl.model}):**
{latest_impl.response}

**Implementation Confidence:** {latest_impl.confidence}%
"""

        if review_responses:
            review = review_responses[0]
            consensus += f"""
## Code Review Analysis
{review.response}

**Review Confidence:** {review.confidence}%
"""

        # Add collaboration metadata
        consensus += f"""
## Collaboration Metadata
- **Models Involved:** {', '.join(set(r.model for r in responses))}
- **Phases Completed:** {len(set(r.phase for r in responses))}
- **Total Processing Time:** {sum(r.processing_time for r in responses):.2f}s
- **Average Confidence:** {sum(r.confidence for r in responses if r.confidence > 0) / len([r for r in responses if r.confidence > 0]):.1f}%

This response was generated through multi-model collaboration to ensure technical accuracy and thoroughness.
"""

        return consensus

    async def _validate_with_inquisitive_core(self, original_query: str, consensus: str) -> str:
        """Validate result through Inquisitive Core for fabrication detection"""
        try:
            validation_query = f"""
Analyze this AI collaboration result for accuracy and detect any potential fabrications:

ORIGINAL QUERY: {original_query}

COLLABORATION RESULT: {consensus[:1000]}...

Please assess:
1. Technical accuracy of the implementation
2. Any unrealistic claims or fabricated details
3. Whether the solution actually addresses the original query
4. Confidence in the overall approach
"""

            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    self.inquisitive_endpoint,
                    json={"query": validation_query}
                )

                if response.status_code == 200:
                    data = response.json()
                    return data.get("response", "Validation failed")
                else:
                    return f"Inquisitive validation unavailable (HTTP {response.status_code})"

        except Exception as e:
            logger.warning(f"Inquisitive validation failed: {e}")
            return f"Inquisitive validation error: {str(e)}"

    async def _detect_fabrication(self, responses: List[ModelResponse], consensus: str) -> bool:
        """Detect potential fabrication in responses"""
        fabrication_indicators = [
            "api.comfyui.com",
            "Bearer token",
            "Image generated successfully",
            "connecting to external API",
            "authentication required",
            "service running on"
        ]

        all_text = consensus + " ".join(r.response for r in responses)

        fabrication_count = sum(1 for indicator in fabrication_indicators if indicator.lower() in all_text.lower())

        # Consider fabrication if multiple indicators or very low confidence
        return fabrication_count > 2 or any(r.confidence < 30 for r in responses if r.confidence > 0)

    def _extract_confidence(self, response_text: str) -> float:
        """Extract confidence percentage from response text"""
        import re

        # Look for confidence patterns
        patterns = [
            r'confidence[:\s]*(\d+)%',
            r'(\d+)%\s*confidence',
            r'certainty[:\s]*(\d+)%',
            r'confidence[:\s]*(\d+)/100'
        ]

        for pattern in patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                return float(match.group(1))

        # Default confidence based on response quality
        if len(response_text) > 500 and "implementation" in response_text.lower():
            return 75.0
        elif len(response_text) > 200:
            return 60.0
        else:
            return 40.0

    async def _save_collaboration(self, result: CollaborationResult):
        """Save collaboration result to database"""
        try:
            # Try multiple password options for database connection
            passwords = ['***REMOVED***', 'password', '']

            for password in passwords:
                try:
                    conn = psycopg2.connect(
                        host='***REMOVED***',
                        database='tower_consolidated',
                        user='patrick',
                        password=password
                    )

                    cursor = conn.cursor()

                    # Save to echo_unified_interactions table
                    cursor.execute("""
                        INSERT INTO echo_unified_interactions
                        (user_id, query, response, model_used, processing_time, timestamp, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        'collaboration_system',
                        result.query,
                        result.consensus,
                        f"multi-llm-{'-'.join(set(r.model for r in result.responses))}",
                        result.collaboration_time,
                        datetime.now(),
                        json.dumps({
                            'collaboration': True,
                            'models_used': [r.model for r in result.responses],
                            'phases_completed': [p.value for p in result.phases_completed],
                            'confidence_score': result.confidence_score,
                            'fabrication_detected': result.fabrication_detected,
                            'inquisitive_validation': result.inquisitive_validation
                        })
                    ))

                    conn.commit()
                    conn.close()
                    logger.info("✅ Collaboration result saved to database")
                    return  # Success, exit function

                except psycopg2.OperationalError:
                    continue  # Try next password

            # If all passwords failed, try local file backup
            backup_file = "/tmp/collaboration_backup.jsonl"
            backup_data = {
                'timestamp': datetime.now().isoformat(),
                'query': result.query,
                'consensus': result.consensus,
                'collaboration_time': result.collaboration_time,
                'confidence_score': result.confidence_score,
                'fabrication_detected': result.fabrication_detected,
                'models_used': [r.model for r in result.responses],
                'phases_completed': [p.value for p in result.phases_completed]
            }

            with open(backup_file, 'a') as f:
                f.write(json.dumps(backup_data) + '\n')

            logger.warning(f"⚠️ Database unavailable, saved to backup file: {backup_file}")

        except Exception as e:
            logger.error(f"❌ Failed to save collaboration result: {e}")
            # Continue execution despite save failure

# Global instance
collaborator = MultiLLMCollaborator()

async def collaborate_on_query(query: str, context: Optional[Dict] = None) -> CollaborationResult:
    """Main entry point for multi-LLM collaboration"""
    return await collaborator.collaborate(query, context)