#!/usr/bin/env python3
"""
Semantic Memory Client for Echo Brain
Provides intelligent creative orchestration with semantic capabilities
"""

import asyncio
import aiohttp
import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import traceback

logger = logging.getLogger(__name__)

class SemanticMemoryClient:
    """
    Client for interacting with the semantic memory system (port 8332)
    Provides character consistency, style preferences, and creative DNA learning
    """

    def __init__(self, base_url: str = "http://localhost:8332"):
        self.base_url = base_url
        self.api_prefix = "/api/echo/anime/semantic"
        self.session = None

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def _get_session(self):
        """Get or create session"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session

    async def health_check(self) -> Dict[str, Any]:
        """Check semantic memory service health"""
        try:
            session = await self._get_session()
            url = f"{self.base_url}{self.api_prefix}/health"

            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {"status": "healthy", "service": "semantic_memory", **data}
                else:
                    return {"status": "unhealthy", "error": f"HTTP {response.status}"}

        except Exception as e:
            logger.error(f"Semantic memory health check failed: {e}")
            return {"status": "error", "error": str(e)}

    async def search_similar_content(
        self,
        query: str,
        content_type: Optional[str] = None,
        max_results: int = 10,
        similarity_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Search for semantically similar content
        Used for finding similar past generations, character references, etc.
        """
        try:
            session = await self._get_session()
            url = f"{self.base_url}{self.api_prefix}/search"

            payload = {
                "query": query,
                "content_type": content_type,
                "max_results": max_results,
                "similarity_threshold": similarity_threshold
            }

            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    return {
                        "error": f"HTTP {response.status}",
                        "detail": error_text,
                        "query": query,
                        "results": [],
                        "total_results": 0
                    }

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return {"error": str(e), "query": query, "results": [], "total_results": 0}

    async def search_characters(
        self,
        description: str,
        max_results: int = 5,
        similarity_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Search for similar characters for consistency checking
        """
        try:
            session = await self._get_session()
            url = f"{self.base_url}{self.api_prefix}/character/search"

            payload = {
                "description": description,
                "max_results": max_results,
                "similarity_threshold": similarity_threshold
            }

            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    return {
                        "error": f"HTTP {response.status}",
                        "detail": error_text,
                        "description": description,
                        "results": [],
                        "total_results": 0
                    }

        except Exception as e:
            logger.error(f"Character search failed: {e}")
            return {"error": str(e), "description": description, "results": [], "total_results": 0}

    async def store_character_memory(
        self,
        character_name: str,
        character_id: str,
        visual_description: str,
        personality_traits: Dict[str, Any] = {},
        appearance_details: Dict[str, Any] = {}
    ) -> Dict[str, Any]:
        """
        Store character information for consistency tracking
        """
        try:
            session = await self._get_session()
            url = f"{self.base_url}{self.api_prefix}/character/store"

            payload = {
                "character_name": character_name,
                "character_id": character_id,
                "visual_description": visual_description,
                "personality_traits": personality_traits,
                "appearance_details": appearance_details
            }

            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    return {"error": f"HTTP {response.status}", "detail": error_text}

        except Exception as e:
            logger.error(f"Character memory storage failed: {e}")
            return {"error": str(e)}

    async def store_text_embedding(
        self,
        text: str,
        content_type: str = "prompt",
        content_id: Optional[str] = None,
        metadata: Dict[str, Any] = {}
    ) -> Dict[str, Any]:
        """
        Store text embedding for future semantic search
        """
        try:
            session = await self._get_session()
            url = f"{self.base_url}{self.api_prefix}/embed/text"

            payload = {
                "text": text,
                "content_type": content_type,
                "content_id": content_id or f"echo_{int(datetime.now().timestamp())}",
                "store": True,
                "metadata": metadata
            }

            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    return {"error": f"HTTP {response.status}", "detail": error_text}

        except Exception as e:
            logger.error(f"Text embedding storage failed: {e}")
            return {"error": str(e)}


class SemanticEnhancedOrchestrator:
    """
    Enhanced Echo Brain orchestrator with semantic memory integration
    """

    def __init__(self):
        self.semantic_client = SemanticMemoryClient()
        self.logger = logging.getLogger(f"{__name__}.SemanticEnhancedOrchestrator")

    async def enhance_prompt_with_semantic_memory(
        self,
        original_prompt: str,
        generation_type: str = "anime_character"
    ) -> Dict[str, Any]:
        """
        Enhance prompt generation using semantic memory

        1. Search for similar past prompts
        2. Extract successful patterns
        3. Add character consistency recommendations
        4. Include style preferences from successful generations
        """
        try:
            async with self.semantic_client as client:
                # Search for similar prompts/content
                similar_content = await client.search_similar_content(
                    query=original_prompt,
                    content_type="prompt",
                    max_results=5,
                    similarity_threshold=0.6
                )

                # Extract character references for consistency checking
                character_matches = await self._extract_character_references(
                    original_prompt, client
                )

                # Build enhancement recommendations
                enhancement = {
                    "original_prompt": original_prompt,
                    "similar_prompts": similar_content.get("results", []),
                    "character_consistency": character_matches,
                    "style_recommendations": self._extract_style_patterns(similar_content),
                    "semantic_enhancement": self._build_semantic_enhancement(
                        original_prompt, similar_content
                    ),
                    "generation_type": generation_type,
                    "timestamp": datetime.now().isoformat()
                }

                # Generate enhanced prompt
                enhanced_prompt = self._generate_enhanced_prompt(enhancement)
                enhancement["enhanced_prompt"] = enhanced_prompt

                return enhancement

        except Exception as e:
            self.logger.error(f"Prompt enhancement failed: {e}")
            self.logger.error(traceback.format_exc())
            return {
                "original_prompt": original_prompt,
                "enhanced_prompt": original_prompt,  # Fallback to original
                "error": str(e),
                "similar_prompts": [],
                "character_consistency": {},
                "style_recommendations": {}
            }

    async def check_character_consistency(
        self,
        character_description: str,
        character_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Check character consistency against stored character memories
        """
        try:
            async with self.semantic_client as client:
                # Search for similar characters
                similar_chars = await client.search_characters(
                    description=character_description,
                    max_results=3,
                    similarity_threshold=0.7
                )

                consistency_report = {
                    "character_description": character_description,
                    "character_name": character_name,
                    "similar_characters": similar_chars.get("results", []),
                    "consistency_score": self._calculate_consistency_score(similar_chars),
                    "recommendations": self._generate_consistency_recommendations(similar_chars),
                    "timestamp": datetime.now().isoformat()
                }

                return consistency_report

        except Exception as e:
            self.logger.error(f"Character consistency check failed: {e}")
            return {
                "character_description": character_description,
                "error": str(e),
                "consistency_score": 0.5,  # Neutral score on error
                "recommendations": ["Unable to check consistency due to error"]
            }

    async def update_creative_dna(
        self,
        generation_result: Dict[str, Any],
        quality_scores: Dict[str, float],
        user_feedback: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update creative DNA based on generation results and feedback
        """
        try:
            async with self.semantic_client as client:
                # Determine if this was successful based on quality scores
                avg_quality = sum(quality_scores.values()) / len(quality_scores) if quality_scores else 0
                success_threshold = 7.0  # Quality score threshold for "successful" generation

                # Store the generation data for future learning
                metadata = {
                    "quality_scores": quality_scores,
                    "average_quality": avg_quality,
                    "successful": avg_quality >= success_threshold,
                    "user_feedback": user_feedback,
                    "generation_params": generation_result.get("generation_params", {}),
                    "job_id": generation_result.get("job_id"),
                    "timestamp": datetime.now().isoformat()
                }

                # Store the prompt and results for semantic search
                storage_result = await client.store_text_embedding(
                    text=generation_result.get("prompt", ""),
                    content_type="generation_result",
                    content_id=f"gen_{generation_result.get('job_id', 'unknown')}",
                    metadata=metadata
                )

                # Store character information if present
                if "character_name" in generation_result:
                    await self._store_character_from_generation(generation_result, client)

                dna_update = {
                    "generation_id": generation_result.get("job_id"),
                    "quality_scores": quality_scores,
                    "learning_stored": storage_result.get("stored", False),
                    "dna_evolution": self._analyze_dna_evolution(metadata),
                    "timestamp": datetime.now().isoformat()
                }

                return dna_update

        except Exception as e:
            self.logger.error(f"Creative DNA update failed: {e}")
            return {"error": str(e), "generation_id": generation_result.get("job_id")}

    async def get_semantic_quality_validation(
        self,
        generated_content: Dict[str, Any],
        original_prompt: str
    ) -> Dict[str, Any]:
        """
        Validate generation quality using semantic analysis
        """
        try:
            async with self.semantic_client as client:
                # Search for similar successful generations
                similar_success = await client.search_similar_content(
                    query=original_prompt,
                    content_type="generation_result",
                    similarity_threshold=0.6,
                    max_results=5
                )

                validation = {
                    "original_prompt": original_prompt,
                    "similar_successful_generations": similar_success.get("results", []),
                    "semantic_consistency": self._validate_semantic_consistency(
                        generated_content, similar_success
                    ),
                    "style_adherence": self._validate_style_adherence(
                        generated_content, similar_success
                    ),
                    "quality_prediction": self._predict_quality_from_semantic(similar_success),
                    "timestamp": datetime.now().isoformat()
                }

                return validation

        except Exception as e:
            self.logger.error(f"Semantic quality validation failed: {e}")
            return {"error": str(e), "original_prompt": original_prompt}

    # Helper methods

    async def _extract_character_references(
        self, prompt: str, client: SemanticMemoryClient
    ) -> Dict[str, Any]:
        """Extract and validate character references from prompt"""
        # Simple character name extraction (could be enhanced with NLP)
        import re
        potential_names = re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', prompt)

        character_info = {}
        for name in potential_names:
            char_search = await client.search_characters(
                description=f"character named {name}",
                max_results=1,
                similarity_threshold=0.8
            )
            if char_search.get("results"):
                character_info[name] = char_search["results"][0]

        return character_info

    def _extract_style_patterns(self, similar_content: Dict[str, Any]) -> Dict[str, Any]:
        """Extract successful style patterns from similar content"""
        patterns = {
            "common_keywords": [],
            "successful_styles": [],
            "quality_indicators": []
        }

        results = similar_content.get("results", [])
        for result in results:
            metadata = result.get("metadata", {})
            if metadata.get("successful", False):
                # Extract patterns from successful generations
                patterns["successful_styles"].append({
                    "content": result.get("content_text", ""),
                    "quality": metadata.get("average_quality", 0),
                    "params": metadata.get("generation_params", {})
                })

        return patterns

    def _build_semantic_enhancement(
        self, original_prompt: str, similar_content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build semantic enhancement recommendations"""
        enhancement = {
            "semantic_keywords": [],
            "style_suggestions": [],
            "quality_improvements": []
        }

        # Analyze successful similar content
        successful_results = [
            r for r in similar_content.get("results", [])
            if r.get("metadata", {}).get("successful", False)
        ]

        if successful_results:
            # Extract common successful elements
            enhancement["style_suggestions"] = [
                "Based on similar successful generations",
                f"Found {len(successful_results)} successful similar prompts"
            ]

            # Average quality of similar successful content
            avg_quality = sum(
                r.get("metadata", {}).get("average_quality", 0)
                for r in successful_results
            ) / len(successful_results)

            enhancement["expected_quality"] = avg_quality

        return enhancement

    def _generate_enhanced_prompt(self, enhancement: Dict[str, Any]) -> str:
        """Generate enhanced prompt from semantic analysis"""
        original = enhancement["original_prompt"]

        # Start with original prompt
        enhanced = original

        # Add style recommendations from successful similar generations
        style_recs = enhancement.get("style_recommendations", {})
        successful_styles = style_recs.get("successful_styles", [])

        if successful_styles:
            # Find the highest quality similar generation
            best_style = max(successful_styles, key=lambda x: x.get("quality", 0))
            quality = best_style.get("quality", 0)

            if quality > 7.0:  # Only use high-quality references
                enhanced += f" (style reference: high quality anime generation, score {quality:.1f})"

        # Add character consistency notes
        char_consistency = enhancement.get("character_consistency", {})
        if char_consistency:
            enhanced += " [maintain character consistency]"

        return enhanced

    def _calculate_consistency_score(self, similar_chars: Dict[str, Any]) -> float:
        """Calculate character consistency score"""
        results = similar_chars.get("results", [])
        if not results:
            return 0.5  # Neutral score if no data

        # Use similarity scores as consistency indicator
        scores = [r.get("similarity_score", 0) for r in results]
        return max(scores) if scores else 0.5

    def _generate_consistency_recommendations(
        self, similar_chars: Dict[str, Any]
    ) -> List[str]:
        """Generate character consistency recommendations"""
        recommendations = []
        results = similar_chars.get("results", [])

        if not results:
            recommendations.append("No similar characters found - this is a new character")
        else:
            best_match = max(results, key=lambda x: x.get("similarity_score", 0))
            score = best_match.get("similarity_score", 0)

            if score > 0.8:
                recommendations.append(
                    f"High similarity to existing character: {best_match.get('character_name')}"
                )
                recommendations.append("Maintain consistent visual traits")
            elif score > 0.6:
                recommendations.append(
                    f"Moderate similarity to: {best_match.get('character_name')}"
                )
                recommendations.append("Consider differentiating features")
            else:
                recommendations.append("Unique character - establish new consistent style")

        return recommendations

    async def _store_character_from_generation(
        self, generation_result: Dict[str, Any], client: SemanticMemoryClient
    ):
        """Store character information extracted from generation result"""
        try:
            character_name = generation_result.get("character_name", "")
            prompt = generation_result.get("prompt", "")

            if character_name and prompt:
                await client.store_character_memory(
                    character_name=character_name,
                    character_id=f"gen_{generation_result.get('job_id')}_{character_name.lower().replace(' ', '_')}",
                    visual_description=prompt,
                    personality_traits=generation_result.get("personality_traits", {}),
                    appearance_details=generation_result.get("appearance_details", {})
                )
        except Exception as e:
            self.logger.error(f"Character storage from generation failed: {e}")

    def _analyze_dna_evolution(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how the creative DNA should evolve based on this generation"""
        return {
            "quality_trend": "improving" if metadata.get("successful", False) else "needs_work",
            "learning_points": [
                f"Generation quality: {metadata.get('average_quality', 0):.1f}",
                f"Successful: {metadata.get('successful', False)}"
            ]
        }

    def _validate_semantic_consistency(
        self, generated_content: Dict[str, Any], similar_success: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate semantic consistency with successful generations"""
        return {
            "consistency_score": 0.8,  # Placeholder - would implement actual validation
            "similar_successes": len(similar_success.get("results", []))
        }

    def _validate_style_adherence(
        self, generated_content: Dict[str, Any], similar_success: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate style adherence to successful patterns"""
        return {
            "style_score": 0.75,  # Placeholder - would implement actual validation
            "style_patterns_matched": len(similar_success.get("results", []))
        }

    def _predict_quality_from_semantic(self, similar_success: Dict[str, Any]) -> Dict[str, Any]:
        """Predict quality based on semantic analysis of similar content"""
        results = similar_success.get("results", [])
        if not results:
            return {"predicted_quality": 5.0, "confidence": 0.5}

        # Calculate average quality of similar content
        qualities = [
            r.get("metadata", {}).get("average_quality", 5.0)
            for r in results
        ]

        avg_quality = sum(qualities) / len(qualities)
        confidence = min(len(results) / 5.0, 1.0)  # Higher confidence with more data

        return {
            "predicted_quality": avg_quality,
            "confidence": confidence,
            "based_on_samples": len(results)
        }