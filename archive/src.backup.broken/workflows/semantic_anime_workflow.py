#!/usr/bin/env python3
"""
Semantic-Enhanced Anime Generation Workflow
Integrates Echo Brain AI orchestration with semantic memory for intelligent creative generation
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import traceback

from src.services.semantic_memory_client import SemanticEnhancedOrchestrator

logger = logging.getLogger(__name__)

class SemanticAnimeWorkflow:
    """
    Enhanced anime generation workflow with semantic memory integration

    This workflow provides:
    1. Intelligent prompt enhancement using semantic memory
    2. Character consistency checking across generations
    3. Style preference learning and application
    4. Quality prediction and validation
    5. Creative DNA evolution based on results
    """

    def __init__(
        self,
        anime_api_url: str = "http://localhost:8331",
        echo_api_url: str = "http://localhost:8309"
    ):
        self.anime_api_url = anime_api_url
        self.echo_api_url = echo_api_url
        self.semantic_orchestrator = SemanticEnhancedOrchestrator()
        self.session = None

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def generate_with_semantic_intelligence(
        self,
        user_prompt: str,
        generation_params: Optional[Dict[str, Any]] = None,
        character_name: Optional[str] = None,
        project_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Complete semantic-enhanced anime generation workflow

        Steps:
        1. Analyze user intent and extract character information
        2. Search semantic memory for similar content and characters
        3. Enhance prompt with semantic intelligence
        4. Check character consistency
        5. Predict quality and suggest optimizations
        6. Execute generation with enhanced parameters
        7. Validate results against semantic patterns
        8. Update creative DNA with learning
        """
        workflow_start = datetime.now()
        workflow_id = f"semantic_workflow_{int(workflow_start.timestamp())}"

        try:
            logger.info(f"Starting semantic anime workflow: {workflow_id}")

            # Step 1: Semantic prompt enhancement
            enhancement_result = await self._enhance_prompt_semantically(
                user_prompt, character_name
            )

            # Step 2: Character consistency validation
            character_analysis = await self._analyze_character_consistency(
                enhancement_result.get("enhanced_prompt", user_prompt),
                character_name
            )

            # Step 3: Quality prediction and parameter optimization
            optimization_result = await self._optimize_generation_parameters(
                enhancement_result, generation_params
            )

            # Step 4: Execute anime generation
            generation_result = await self._execute_anime_generation(
                enhancement_result["enhanced_prompt"],
                optimization_result["optimized_params"],
                project_id
            )

            # Step 5: Semantic quality validation
            quality_validation = await self._validate_generation_semantically(
                generation_result,
                user_prompt,
                enhancement_result["enhanced_prompt"]
            )

            # Step 6: Update creative DNA with learning
            learning_update = await self._update_creative_learning(
                generation_result,
                quality_validation,
                user_prompt,
                character_name
            )

            workflow_time = (datetime.now() - workflow_start).total_seconds()

            # Compile complete workflow result
            complete_result = {
                "workflow_id": workflow_id,
                "user_prompt": user_prompt,
                "enhanced_prompt": enhancement_result["enhanced_prompt"],
                "character_name": character_name,
                "semantic_enhancement": enhancement_result,
                "character_analysis": character_analysis,
                "parameter_optimization": optimization_result,
                "generation_result": generation_result,
                "quality_validation": quality_validation,
                "creative_learning": learning_update,
                "workflow_time_seconds": workflow_time,
                "timestamp": datetime.now().isoformat(),
                "success": generation_result.get("success", False)
            }

            logger.info(f"Completed semantic workflow {workflow_id} in {workflow_time:.2f}s")
            return complete_result

        except Exception as e:
            logger.error(f"Semantic anime workflow failed: {e}")
            logger.error(traceback.format_exc())
            return {
                "workflow_id": workflow_id,
                "error": str(e),
                "user_prompt": user_prompt,
                "success": False,
                "timestamp": datetime.now().isoformat()
            }

    async def _enhance_prompt_semantically(
        self,
        user_prompt: str,
        character_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Enhance prompt using semantic memory intelligence"""
        try:
            enhancement = await self.semantic_orchestrator.enhance_prompt_with_semantic_memory(
                original_prompt=user_prompt,
                generation_type="anime_character"
            )

            logger.info(f"Prompt enhanced: {len(user_prompt)} -> {len(enhancement['enhanced_prompt'])} chars")
            return enhancement

        except Exception as e:
            logger.error(f"Prompt enhancement failed: {e}")
            return {
                "original_prompt": user_prompt,
                "enhanced_prompt": user_prompt,  # Fallback to original
                "error": str(e),
                "similar_prompts": [],
                "style_recommendations": {}
            }

    async def _analyze_character_consistency(
        self,
        enhanced_prompt: str,
        character_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze character consistency using semantic memory"""
        try:
            if not character_name:
                # Try to extract character name from prompt
                character_name = self._extract_character_name_from_prompt(enhanced_prompt)

            if character_name:
                consistency = await self.semantic_orchestrator.check_character_consistency(
                    character_description=enhanced_prompt,
                    character_name=character_name
                )
                logger.info(f"Character consistency score: {consistency.get('consistency_score', 0):.2f}")
                return consistency
            else:
                return {
                    "character_name": None,
                    "consistency_score": 1.0,  # New character, no consistency issues
                    "recommendations": ["New character - establishing baseline"],
                    "similar_characters": []
                }

        except Exception as e:
            logger.error(f"Character consistency analysis failed: {e}")
            return {
                "error": str(e),
                "consistency_score": 0.5,
                "recommendations": ["Unable to analyze consistency"]
            }

    async def _optimize_generation_parameters(
        self,
        enhancement_result: Dict[str, Any],
        base_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Optimize generation parameters based on semantic intelligence"""
        try:
            # Start with default parameters
            default_params = {
                "steps": 28,
                "cfg_scale": 7.5,
                "sampler": "dpmpp_2m",
                "scheduler": "karras",
                "resolution": "768x768",
                "frames": 120,
                "fps": 24
            }

            # Apply base parameters if provided
            if base_params:
                default_params.update(base_params)

            optimized_params = default_params.copy()

            # Apply optimizations based on semantic analysis
            style_recs = enhancement_result.get("style_recommendations", {})
            successful_styles = style_recs.get("successful_styles", [])

            if successful_styles:
                # Find the highest quality successful style
                best_style = max(
                    successful_styles,
                    key=lambda x: x.get("quality", 0)
                )

                best_params = best_style.get("params", {})
                if best_params and best_style.get("quality", 0) > 7.0:
                    # Apply successful parameters
                    for key, value in best_params.items():
                        if key in optimized_params:
                            optimized_params[key] = value
                    logger.info(f"Applied parameters from successful generation (quality: {best_style.get('quality'):.1f})")

            # Character consistency optimizations
            character_analysis = enhancement_result.get("character_consistency", {})
            consistency_score = character_analysis.get("consistency_score", 0.5)

            if consistency_score > 0.8:
                # High consistency character - use proven settings
                optimized_params["cfg_scale"] = min(optimized_params["cfg_scale"], 7.0)
                optimized_params["steps"] = max(optimized_params["steps"], 30)
                logger.info("Applied consistency optimization for established character")

            return {
                "original_params": default_params,
                "optimized_params": optimized_params,
                "optimization_reason": self._build_optimization_reason(
                    enhancement_result, successful_styles, consistency_score
                )
            }

        except Exception as e:
            logger.error(f"Parameter optimization failed: {e}")
            return {
                "error": str(e),
                "original_params": base_params or {},
                "optimized_params": base_params or {}
            }

    async def _execute_anime_generation(
        self,
        enhanced_prompt: str,
        optimized_params: Dict[str, Any],
        project_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Execute anime generation using optimized parameters"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            # Prepare generation request
            generation_request = {
                "prompt": enhanced_prompt,
                "project_id": project_id,
                **optimized_params
            }

            # Submit generation job
            async with self.session.post(
                f"{self.anime_api_url}/api/anime/jobs",
                json=generation_request
            ) as response:
                if response.status == 200:
                    job_data = await response.json()
                    job_id = job_data.get("job_id")

                    if job_id:
                        # Monitor job progress
                        final_result = await self._monitor_generation_job(job_id)
                        return final_result
                    else:
                        return {"error": "No job ID returned", "success": False}
                else:
                    error_text = await response.text()
                    return {"error": f"HTTP {response.status}: {error_text}", "success": False}

        except Exception as e:
            logger.error(f"Anime generation execution failed: {e}")
            return {"error": str(e), "success": False}

    async def _monitor_generation_job(self, job_id: int) -> Dict[str, Any]:
        """Monitor anime generation job until completion"""
        try:
            max_wait_time = 600  # 10 minutes max
            poll_interval = 5  # Check every 5 seconds
            start_time = datetime.now()

            logger.info(f"Monitoring generation job {job_id}")

            while True:
                # Check job status
                async with self.session.get(
                    f"{self.anime_api_url}/api/anime/jobs/{job_id}/status"
                ) as response:
                    if response.status == 200:
                        status_data = await response.json()
                        job_status = status_data.get("status")

                        if job_status == "completed":
                            logger.info(f"Generation job {job_id} completed successfully")
                            return {
                                "job_id": job_id,
                                "success": True,
                                "status": "completed",
                                "result": status_data,
                                "generation_time": (datetime.now() - start_time).total_seconds()
                            }
                        elif job_status == "failed":
                            logger.error(f"Generation job {job_id} failed")
                            return {
                                "job_id": job_id,
                                "success": False,
                                "status": "failed",
                                "error": status_data.get("error", "Unknown error")
                            }
                        elif job_status in ["pending", "running"]:
                            # Job still running, check for timeout
                            elapsed = (datetime.now() - start_time).total_seconds()
                            if elapsed > max_wait_time:
                                logger.warning(f"Job {job_id} timed out after {elapsed}s")
                                return {
                                    "job_id": job_id,
                                    "success": False,
                                    "status": "timeout",
                                    "error": "Generation timed out"
                                }

                            # Wait and continue monitoring
                            await asyncio.sleep(poll_interval)
                            continue
                        else:
                            logger.warning(f"Unknown job status: {job_status}")
                            return {
                                "job_id": job_id,
                                "success": False,
                                "status": job_status,
                                "error": f"Unknown status: {job_status}"
                            }
                    else:
                        error_text = await response.text()
                        logger.error(f"Status check failed: HTTP {response.status}: {error_text}")
                        return {
                            "job_id": job_id,
                            "success": False,
                            "error": f"Status check failed: {error_text}"
                        }

        except Exception as e:
            logger.error(f"Job monitoring failed: {e}")
            return {
                "job_id": job_id,
                "success": False,
                "error": f"Monitoring failed: {str(e)}"
            }

    async def _validate_generation_semantically(
        self,
        generation_result: Dict[str, Any],
        original_prompt: str,
        enhanced_prompt: str
    ) -> Dict[str, Any]:
        """Validate generation quality using semantic analysis"""
        try:
            if not generation_result.get("success", False):
                return {
                    "validation_score": 0.0,
                    "error": "Generation failed, cannot validate",
                    "original_prompt": original_prompt
                }

            validation = await self.semantic_orchestrator.get_semantic_quality_validation(
                generated_content=generation_result,
                original_prompt=enhanced_prompt
            )

            # Extract validation metrics
            semantic_score = validation.get("semantic_consistency", {}).get("consistency_score", 0.5)
            style_score = validation.get("style_adherence", {}).get("style_score", 0.5)
            quality_prediction = validation.get("quality_prediction", {}).get("predicted_quality", 5.0)

            # Calculate overall validation score
            overall_score = (semantic_score + style_score + (quality_prediction / 10.0)) / 3.0

            validation_result = {
                "overall_validation_score": round(overall_score, 2),
                "semantic_consistency_score": semantic_score,
                "style_adherence_score": style_score,
                "predicted_quality": quality_prediction,
                "validation_details": validation,
                "original_prompt": original_prompt,
                "enhanced_prompt": enhanced_prompt
            }

            logger.info(f"Semantic validation score: {overall_score:.2f}")
            return validation_result

        except Exception as e:
            logger.error(f"Semantic validation failed: {e}")
            return {
                "error": str(e),
                "overall_validation_score": 0.5,
                "original_prompt": original_prompt
            }

    async def _update_creative_learning(
        self,
        generation_result: Dict[str, Any],
        quality_validation: Dict[str, Any],
        original_prompt: str,
        character_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Update creative DNA with learning from this generation"""
        try:
            # Create quality scores from validation
            quality_scores = {
                "semantic_consistency": quality_validation.get("semantic_consistency_score", 0.5),
                "style_adherence": quality_validation.get("style_adherence_score", 0.5),
                "predicted_quality": quality_validation.get("predicted_quality", 5.0),
                "overall_validation": quality_validation.get("overall_validation_score", 0.5)
            }

            # Prepare generation result for learning
            learning_data = {
                "job_id": generation_result.get("job_id"),
                "prompt": original_prompt,
                "character_name": character_name,
                "generation_params": generation_result.get("generation_params", {}),
                "success": generation_result.get("success", False)
            }

            # Update creative DNA
            learning_update = await self.semantic_orchestrator.update_creative_dna(
                generation_result=learning_data,
                quality_scores=quality_scores,
                user_feedback=None  # Could be added later
            )

            logger.info(f"Updated creative DNA for job {generation_result.get('job_id')}")
            return learning_update

        except Exception as e:
            logger.error(f"Creative learning update failed: {e}")
            return {"error": str(e)}

    def _extract_character_name_from_prompt(self, prompt: str) -> Optional[str]:
        """Extract character name from prompt using simple heuristics"""
        import re

        # Look for patterns like "Character Name" or character-like names
        name_patterns = [
            r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b',  # First Last
            r'\b([A-Z][a-z]+)\s+(?:character|girl|boy|person)',  # Name + descriptor
            r'(?:character named|called)\s+([A-Z][a-z]+)',  # "character named X"
        ]

        for pattern in name_patterns:
            matches = re.findall(pattern, prompt, re.IGNORECASE)
            if matches:
                return matches[0] if isinstance(matches[0], str) else matches[0][0]

        return None

    def _build_optimization_reason(
        self,
        enhancement_result: Dict[str, Any],
        successful_styles: List[Dict[str, Any]],
        consistency_score: float
    ) -> str:
        """Build human-readable optimization reasoning"""
        reasons = []

        if successful_styles:
            best_quality = max(s.get("quality", 0) for s in successful_styles)
            reasons.append(f"Applied parameters from similar generation (quality: {best_quality:.1f})")

        if consistency_score > 0.8:
            reasons.append(f"Character consistency optimization (score: {consistency_score:.2f})")
        elif consistency_score < 0.4:
            reasons.append("New character - using exploratory parameters")

        similar_count = len(enhancement_result.get("similar_prompts", []))
        if similar_count > 0:
            reasons.append(f"Based on {similar_count} similar successful prompts")

        return "; ".join(reasons) if reasons else "Default optimization applied"