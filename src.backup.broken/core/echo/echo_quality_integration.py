#!/usr/bin/env python3
"""
Echo Brain Quality Integration System
Connects Echo Brain with quality assessment pipeline and multimedia orchestration
"""

import os
import asyncio
import json
import logging
from typing import Dict, Optional, List, Tuple
from pathlib import Path
import aiohttp
import numpy as np
from dataclasses import dataclass
from enum import Enum

# Quality Levels
class QualityLevel(Enum):
    PREMIUM = 0.95
    PRODUCTION = 0.85
    ACCEPTABLE = 0.70
    DRAFT = 0.50
    FAILED = 0.0

@dataclass
class QualityResult:
    """Quality assessment result"""
    overall_score: float
    level: QualityLevel
    technical_score: float
    semantic_score: float
    style_score: float
    issues: List[str]
    suggestions: List[str]
    retry_params: Optional[Dict] = None

class EchoQualityIntegration:
    """Integrates Echo Brain with quality assessment and multimedia pipeline"""

    def __init__(self):
        self.echo_url = "http://localhost:8309"
        self.comfyui_url = "http://localhost:8188"
        self.quality_url = "http://localhost:8328"
        self.voice_url = "http://localhost:8312"
        self.music_url = "http://localhost:8315"
        self.llava_url = "http://localhost:11434"

        self.logger = logging.getLogger(__name__)
        self.retry_count = 3
        self.quality_threshold = QualityLevel.PRODUCTION.value

    async def generate_with_quality(self, prompt: str, character: str = "",
                                   setting: str = "", media_type: str = "image") -> Dict:
        """Generate content with quality verification and retry logic"""

        best_result = None
        best_score = 0.0

        for attempt in range(self.retry_count):
            self.logger.info(f"Generation attempt {attempt + 1}/{self.retry_count}")

            # Adjust parameters based on previous attempts
            workflow_params = self._adjust_workflow_params(prompt, attempt, best_result)

            # Generate content
            if media_type == "image":
                result = await self._generate_image(prompt, character, setting, workflow_params)
            elif media_type == "video":
                result = await self._generate_video(prompt, character, setting, workflow_params)
            elif media_type == "soundmovie":
                result = await self._generate_soundmovie(prompt, character, setting, workflow_params)
            else:
                raise ValueError(f"Unknown media type: {media_type}")

            if not result:
                continue

            # Assess quality
            quality = await self._assess_quality(result, prompt, media_type)

            # Track best result
            if quality.overall_score > best_score:
                best_score = quality.overall_score
                best_result = {
                    "content": result,
                    "quality": quality,
                    "attempt": attempt + 1
                }

            # Check if quality meets threshold
            if quality.overall_score >= self.quality_threshold:
                self.logger.info(f"Quality threshold met: {quality.overall_score:.2f}")
                return best_result

            # Log quality issues for learning
            await self._log_quality_feedback(prompt, result, quality)

        # Return best result even if below threshold
        if best_result:
            self.logger.warning(f"Quality below threshold. Best score: {best_score:.2f}")

        return best_result

    async def _generate_image(self, prompt: str, character: str,
                            setting: str, params: Dict) -> Optional[str]:
        """Generate image through ComfyUI with enhanced workflow"""

        workflow = self._create_enhanced_workflow(prompt, character, setting, params)

        async with aiohttp.ClientSession() as session:
            # Submit to ComfyUI
            async with session.post(f"{self.comfyui_url}/api/prompt",
                                  json=workflow) as resp:
                if resp.status != 200:
                    self.logger.error(f"ComfyUI error: {resp.status}")
                    return None

                result = await resp.json()
                prompt_id = result.get("prompt_id")

            # Wait for completion with optimized polling
            image_path = await self._wait_for_completion(prompt_id)

        return image_path

    async def _generate_video(self, prompt: str, character: str,
                            setting: str, params: Dict) -> Optional[str]:
        """Generate video with synchronized audio"""

        # Generate frames in parallel
        frames = await self._generate_frames_batch(prompt, character, setting, params)

        if not frames or len(frames) < params.get("min_frames", 24):
            self.logger.error("Insufficient frames generated")
            return None

        # Generate audio track
        audio = await self._generate_audio_track(prompt, character, len(frames))

        # Compile video with audio sync
        video_path = await self._compile_video(frames, audio, params)

        return video_path

    async def _generate_soundmovie(self, prompt: str, character: str,
                                  setting: str, params: Dict) -> Optional[str]:
        """Generate complete soundmovie with voice, music, and visuals"""

        # Parse script for scenes and dialogue
        scenes = self._parse_script(prompt)

        # Generate all components in parallel
        tasks = []
        for scene in scenes:
            tasks.append(self._generate_scene_components(scene, character, setting, params))

        components = await asyncio.gather(*tasks)

        # Assemble soundmovie with synchronization
        soundmovie_path = await self._assemble_soundmovie(components, params)

        return soundmovie_path

    async def _assess_quality(self, content_path: str, prompt: str,
                             media_type: str) -> QualityResult:
        """Comprehensive quality assessment using multiple methods"""

        # Technical quality assessment
        technical = await self._assess_technical_quality(content_path, media_type)

        # Semantic quality using LLaVA
        semantic = await self._assess_semantic_quality(content_path, prompt)

        # Style consistency check
        style = await self._assess_style_quality(content_path, media_type)

        # Calculate overall score
        overall = (technical * 0.3 + semantic * 0.4 + style * 0.3)

        # Determine quality level
        level = self._get_quality_level(overall)

        # Identify issues and suggestions
        issues = self._identify_issues(technical, semantic, style)
        suggestions = self._generate_suggestions(issues)
        retry_params = self._generate_retry_params(issues) if level.value < self.quality_threshold else None

        return QualityResult(
            overall_score=overall,
            level=level,
            technical_score=technical,
            semantic_score=semantic,
            style_score=style,
            issues=issues,
            suggestions=suggestions,
            retry_params=retry_params
        )

    async def _assess_semantic_quality(self, image_path: str, prompt: str) -> float:
        """Use LLaVA to assess semantic quality"""

        vision_prompt = f'''Analyze this anime image against the original prompt: "{prompt}"
        Rate from 0.0 to 1.0:
        1. Character accuracy (matches description)
        2. Scene composition (logical placement)
        3. Art style consistency (anime aesthetic)
        4. Overall prompt fidelity
        Provide a single overall score between 0.0 and 1.0'''

        async with aiohttp.ClientSession() as session:
            # Call LLaVA through Ollama
            payload = {
                "model": "llava:7b",
                "prompt": vision_prompt,
                "images": [image_path],
                "stream": False
            }

            async with session.post(f"{self.llava_url}/api/generate",
                                  json=payload) as resp:
                if resp.status != 200:
                    self.logger.error(f"LLaVA error: {resp.status}")
                    return 0.5  # Default score on error

                result = await resp.json()
                response = result.get("response", "")

                # Extract score from response
                try:
                    # Parse score from LLaVA response
                    import re
                    scores = re.findall(r"0\.\d+|1\.0", response)
                    if scores:
                        return float(scores[-1])  # Use last score as overall
                except:
                    pass

        return 0.5  # Default if parsing fails

    async def _assess_technical_quality(self, content_path: str, media_type: str) -> float:
        """Assess technical quality metrics"""

        async with aiohttp.ClientSession() as session:
            # Call quality assessment service
            payload = {
                "file_path": content_path,
                "media_type": media_type
            }

            async with session.post(f"{self.quality_url}/api/assess",
                                  json=payload) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return result.get("technical_score", 0.5)

        return 0.5  # Default score

    async def _assess_style_quality(self, content_path: str, media_type: str) -> float:
        """Assess anime style consistency"""

        if media_type in ["image", "video"]:
            # Check for anime-specific features
            features = await self._extract_anime_features(content_path)

            score = 0.0
            score += 0.25 if features.get("has_anime_eyes", False) else 0
            score += 0.25 if features.get("has_anime_proportions", False) else 0
            score += 0.25 if features.get("has_cel_shading", False) else 0
            score += 0.25 if features.get("has_vibrant_colors", False) else 0

            return score

        return 0.75  # Default for non-visual media

    def _adjust_workflow_params(self, prompt: str, attempt: int,
                               previous_result: Optional[Dict]) -> Dict:
        """Adjust generation parameters based on attempt and previous results"""

        params = {
            "steps": 20 + (attempt * 5),  # Increase steps each attempt
            "cfg_scale": 7.0 + (attempt * 0.5),  # Increase guidance
            "sampler_name": "euler_a" if attempt == 0 else "dpm_2m",
            "denoise": 1.0 - (attempt * 0.05),  # Decrease denoise
            "seed": -1  # Random seed each attempt
        }

        # Adjust based on previous quality issues
        if previous_result and previous_result.get("quality"):
            quality = previous_result["quality"]

            if "blur" in quality.issues:
                params["steps"] += 10
                params["cfg_scale"] += 1.0

            if "low_detail" in quality.issues:
                params["steps"] += 5
                prompt += ", highly detailed, intricate"

            if "wrong_style" in quality.issues:
                prompt += ", anime style, manga aesthetic"

        return params

    def _create_enhanced_workflow(self, prompt: str, character: str,
                                 setting: str, params: Dict) -> Dict:
        """Create ComfyUI workflow with quality enhancements"""

        # Base workflow structure
        workflow = {
            "prompt": {
                "3": {
                    "inputs": {
                        "seed": params.get("seed", -1),
                        "steps": params.get("steps", 30),
                        "cfg": params.get("cfg_scale", 8.0),
                        "sampler_name": params.get("sampler_name", "euler_a"),
                        "scheduler": "normal",
                        "denoise": params.get("denoise", 1.0),
                        "model": ["4", 0],
                        "positive": ["6", 0],
                        "negative": ["7", 0],
                        "latent_image": ["5", 0]
                    },
                    "class_type": "KSampler"
                }
            }
        }

        # Enhanced positive prompt
        enhanced_prompt = f"{prompt}, {character}, {setting}, "
        enhanced_prompt += "masterpiece, best quality, highly detailed, "
        enhanced_prompt += "anime style, sharp focus, intricate details"

        # Enhanced negative prompt
        negative_prompt = "worst quality, low quality, blurry, "
        negative_prompt += "bad anatomy, bad hands, missing fingers, "
        negative_prompt += "jpeg artifacts, watermark, username"

        workflow["prompt"]["6"] = {
            "inputs": {"text": enhanced_prompt},
            "class_type": "CLIPTextEncode"
        }

        workflow["prompt"]["7"] = {
            "inputs": {"text": negative_prompt},
            "class_type": "CLIPTextEncode"
        }

        return workflow

    async def _wait_for_completion(self, prompt_id: str, timeout: int = 60) -> Optional[str]:
        """Wait for ComfyUI completion with optimized polling"""

        start_time = asyncio.get_event_loop().time()

        async with aiohttp.ClientSession() as session:
            while (asyncio.get_event_loop().time() - start_time) < timeout:
                # Check history for completion
                async with session.get(f"{self.comfyui_url}/api/history/{prompt_id}") as resp:
                    if resp.status == 200:
                        history = await resp.json()

                        if prompt_id in history:
                            outputs = history[prompt_id].get("outputs", {})
                            for node_outputs in outputs.values():
                                if "images" in node_outputs:
                                    for img_info in node_outputs["images"]:
                                        # Fixed path
                                        image_path = f"/home/{os.getenv("TOWER_USER", "patrick")}/Projects/ComfyUI-Working/output/{img_info['filename']}"
                                        if Path(image_path).exists():
                                            return image_path

                # Optimized polling interval
                await asyncio.sleep(0.5)

        return None

    def _get_quality_level(self, score: float) -> QualityLevel:
        """Determine quality level from score"""

        if score >= QualityLevel.PREMIUM.value:
            return QualityLevel.PREMIUM
        elif score >= QualityLevel.PRODUCTION.value:
            return QualityLevel.PRODUCTION
        elif score >= QualityLevel.ACCEPTABLE.value:
            return QualityLevel.ACCEPTABLE
        elif score >= QualityLevel.DRAFT.value:
            return QualityLevel.DRAFT
        else:
            return QualityLevel.FAILED

    def _identify_issues(self, technical: float, semantic: float, style: float) -> List[str]:
        """Identify quality issues from scores"""

        issues = []

        if technical < 0.7:
            issues.append("low_technical_quality")
            if technical < 0.5:
                issues.append("blur")
                issues.append("artifacts")

        if semantic < 0.7:
            issues.append("poor_prompt_adherence")
            if semantic < 0.5:
                issues.append("wrong_content")

        if style < 0.7:
            issues.append("wrong_style")
            if style < 0.5:
                issues.append("not_anime")

        return issues

    def _generate_suggestions(self, issues: List[str]) -> List[str]:
        """Generate improvement suggestions based on issues"""

        suggestions = []

        if "blur" in issues:
            suggestions.append("Increase sampling steps to 30+")
            suggestions.append("Increase CFG scale to 9.0")

        if "low_detail" in issues:
            suggestions.append("Add 'highly detailed, intricate' to prompt")
            suggestions.append("Use higher resolution latent")

        if "wrong_style" in issues:
            suggestions.append("Emphasize 'anime style, manga' in prompt")
            suggestions.append("Consider using anime-specific model")

        if "poor_prompt_adherence" in issues:
            suggestions.append("Simplify prompt to key elements")
            suggestions.append("Increase guidance scale")

        return suggestions

    def _generate_retry_params(self, issues: List[str]) -> Dict:
        """Generate retry parameters based on issues"""

        params = {}

        if "blur" in issues:
            params["steps"] = 35
            params["cfg_scale"] = 9.0

        if "wrong_style" in issues:
            params["model"] = "animagine_xl_3.1"
            params["style_lora"] = 0.8

        return params

    async def _log_quality_feedback(self, prompt: str, result: str, quality: QualityResult):
        """Log quality feedback for learning"""

        feedback = {
            "prompt": prompt,
            "result": result,
            "quality_score": quality.overall_score,
            "issues": quality.issues,
            "suggestions": quality.suggestions,
            "timestamp": asyncio.get_event_loop().time()
        }

        # Save to knowledge base
        async with aiohttp.ClientSession() as session:
            await session.post(
                "https://192.168.50.135/api/kb/articles",
                json={
                    "title": f"Quality Feedback: {quality.level.name}",
                    "content": json.dumps(feedback, indent=2),
                    "category": "quality_learning",
                    "tags": ["quality", "feedback", "learning"]
                },
                ssl=False
            )

# Integration with Echo Brain
class EchoBrainEnhanced:
    """Enhanced Echo Brain with quality integration"""

    def __init__(self):
        self.quality_system = EchoQualityIntegration()
        self.logger = logging.getLogger(__name__)

    async def generate_anime_content(self, request: Dict) -> Dict:
        """Generate anime content with quality assurance"""

        prompt = request.get("prompt", "")
        character = request.get("character", "")
        setting = request.get("setting", "")
        media_type = request.get("media_type", "image")

        # Generate with quality verification
        result = await self.quality_system.generate_with_quality(
            prompt, character, setting, media_type
        )

        if result:
            return {
                "success": True,
                "content": result["content"],
                "quality_score": result["quality"].overall_score,
                "quality_level": result["quality"].level.name,
                "attempts": result["attempt"],
                "issues": result["quality"].issues,
                "suggestions": result["quality"].suggestions
            }
        else:
            return {
                "success": False,
                "error": "Failed to generate content meeting quality standards"
            }

if __name__ == "__main__":
    # Test the quality integration
    async def test():
        echo = EchoBrainEnhanced()
        result = await echo.generate_anime_content({
            "prompt": "A cyberpunk samurai in neon Tokyo",
            "character": "Kai",
            "setting": "nighttime cityscape",
            "media_type": "image"
        })
        print(json.dumps(result, indent=2))

    asyncio.run(test())