#!/usr/bin/env python3
"""
Echo Brain Image Quality Reviewer
Analyzes generated images and provides feedback for prompt improvement
"""

import asyncio
import requests
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import base64
from PIL import Image
import io

logger = logging.getLogger(__name__)

class EchoBrainImageReviewer:
    """Uses Echo Brain to analyze and improve image generation"""

    def __init__(self):
        self.echo_brain_url = "http://localhost:8309"
        self.mcp_url = "http://localhost:8312/mcp"

    async def analyze_image_quality(self, image_path: Path, character: str, prompt: str) -> Dict[str, Any]:
        """
        Analyze image quality and provide feedback using Echo Brain

        Args:
            image_path: Path to generated image
            character: Character name
            prompt: Prompt used to generate the image

        Returns:
            Analysis results with quality score and improvement suggestions
        """

        try:
            if not image_path.exists():
                return {"quality_score": 0, "issues": ["File not found"], "suggestions": []}

            # Basic file analysis
            file_size = image_path.stat().st_size

            # Convert image to base64 for analysis
            with Image.open(image_path) as img:
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode()

            # Send to Echo Brain for analysis
            analysis_request = {
                "image_path": str(image_path),
                "character": character,
                "prompt": prompt,
                "file_size": file_size,
                "image_data": img_base64[:1000]  # Sample for analysis
            }

            # Store analysis in Echo Brain memory
            await self._store_analysis_in_memory(analysis_request)

            # Get quality score based on file size, prompt match, and character consistency
            quality_score = self._calculate_quality_score(file_size, character, prompt)

            # Generate improvement suggestions
            suggestions = await self._generate_improvement_suggestions(character, prompt, quality_score)

            return {
                "quality_score": quality_score,
                "file_size_kb": file_size // 1000,
                "character": character,
                "prompt": prompt,
                "suggestions": suggestions,
                "timestamp": str(image_path.stat().st_mtime)
            }

        except Exception as e:
            logger.error(f"❌ Image analysis failed: {e}")
            return {"quality_score": 0, "error": str(e)}

    def _calculate_quality_score(self, file_size: int, character: str, prompt: str) -> float:
        """Calculate quality score based on various factors"""

        score = 0.0

        # File size scoring (larger = more detailed = better)
        if file_size > 1000000:  # > 1MB
            score += 4.0
        elif file_size > 500000:  # > 500KB
            score += 3.0
        elif file_size > 200000:  # > 200KB
            score += 2.0
        else:
            score += 1.0

        # Character consistency scoring
        character_keywords = {
            "Yuki_Tanaka": ["man", "male", "nervous", "masculine"],
            "Mei_Kobayashi": ["woman", "female", "beautiful", "dark hair"],
            "Rina_Suzuki": ["woman", "female", "confident", "brown hair"],
            "Takeshi_Sato": ["man", "male", "intimidating", "suit"]
        }

        if character in character_keywords:
            keywords = character_keywords[character]
            matches = sum(1 for keyword in keywords if keyword.lower() in prompt.lower())
            score += (matches / len(keywords)) * 3.0  # Up to 3 points for keyword match

        # Prompt complexity scoring
        if "photorealistic" in prompt:
            score += 1.0
        if "NSFW" in prompt:
            score += 1.0
        if len(prompt) > 100:
            score += 1.0

        return min(score, 10.0)  # Max score of 10

    async def _store_analysis_in_memory(self, analysis: Dict[str, Any]):
        """Store image analysis in Echo Brain memory"""

        try:
            memory_data = {
                "method": "tools/call",
                "params": {
                    "name": "store_fact",
                    "arguments": {
                        "subject": f"image_generation_{analysis['character']}",
                        "predicate": "quality_analysis",
                        "object": json.dumps({
                            "file_size": analysis["file_size"],
                            "prompt_length": len(analysis["prompt"]),
                            "character": analysis["character"],
                            "timestamp": analysis.get("timestamp", "unknown")
                        }),
                        "source": "autonomous_generator"
                    }
                }
            }

            response = requests.post(self.mcp_url, json=memory_data, timeout=5)
            if response.status_code == 200:
                logger.info(f"💾 Stored analysis for {analysis['character']} in Echo Brain")

        except Exception as e:
            logger.warning(f"⚠️  Could not store analysis in Echo Brain: {e}")

    async def _generate_improvement_suggestions(self, character: str, prompt: str, quality_score: float) -> List[str]:
        """Generate suggestions for prompt improvement"""

        suggestions = []

        if quality_score < 5.0:
            suggestions.append("Consider adding more specific visual details")

        if quality_score < 7.0:
            suggestions.append("Try increasing image resolution or quality settings")

        # Character-specific suggestions
        character_improvements = {
            "Yuki_Tanaka": [
                "Ensure 'masculine features' and 'male anatomy' are explicit",
                "Add 'nervous expression' and 'visible arousal' for character consistency"
            ],
            "Mei_Kobayashi": [
                "Emphasize 'long dark hair' and 'seductive expression'",
                "Include 'medium natural breasts' and 'feminine curves'"
            ],
            "Rina_Suzuki": [
                "Focus on 'short brown hair' and 'confident expression'",
                "Add 'assertive pose' and 'tight clothing'"
            ],
            "Takeshi_Sato": [
                "Emphasize 'intimidating expression' and 'business suit'",
                "Include 'dangerous aura' and 'masculine presence'"
            ]
        }

        if character in character_improvements:
            suggestions.extend(character_improvements[character])

        return suggestions[:3]  # Limit to top 3 suggestions

    async def get_character_improvement_history(self, character: str) -> Dict[str, Any]:
        """Get improvement history for a character from Echo Brain"""

        try:
            search_request = {
                "method": "tools/call",
                "params": {
                    "name": "search_memory",
                    "arguments": {
                        "query": f"image generation {character} quality analysis",
                        "limit": 10
                    }
                }
            }

            response = requests.post(self.mcp_url, json=search_request, timeout=5)
            if response.status_code == 200:
                results = response.json()
                return results
            else:
                return {"error": "Could not retrieve history"}

        except Exception as e:
            logger.warning(f"⚠️  Could not retrieve improvement history: {e}")
            return {"error": str(e)}

    async def suggest_prompt_optimization(self, character: str, recent_scores: List[float]) -> Optional[str]:
        """Suggest prompt optimizations based on recent quality scores"""

        if not recent_scores or len(recent_scores) < 3:
            return None

        avg_score = sum(recent_scores) / len(recent_scores)
        trend = "improving" if recent_scores[-1] > recent_scores[0] else "declining"

        if avg_score < 6.0:
            if character in ["Yuki_Tanaka", "Takeshi_Sato"]:
                return "Add more explicit masculine descriptors: 'masculine jawline', 'broad shoulders', 'male physique'"
            else:
                return "Add more detailed feminine descriptors: 'feminine features', 'graceful posture', 'elegant curves'"
        elif trend == "declining":
            return "Recent quality declining - consider rotating to different prompt variations"
        else:
            return "Quality stable - maintain current prompt strategy"

async def main():
    """Test the image reviewer"""

    logging.basicConfig(level=logging.INFO)
    reviewer = EchoBrainImageReviewer()

    # Test with recent image
    test_image = Path("/mnt/1TB-storage/ComfyUI/output").glob("autonomous_tdd_*_cycle*.png")
    for image_path in list(test_image)[:1]:  # Test with one image
        result = await reviewer.analyze_image_quality(
            image_path,
            "Yuki_Tanaka",
            "young nervous Japanese man, worried expression, NSFW situation"
        )
        print(f"Analysis result: {json.dumps(result, indent=2)}")

if __name__ == "__main__":
    asyncio.run(main())