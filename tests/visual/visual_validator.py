"""
Visual Validation System for Echo Brain Anime Generation
Uses LLaVA model to validate character consistency, style compliance, and quality.
"""

import asyncio
import base64
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import hashlib

import cv2
import numpy as np
from PIL import Image
import requests
import aiohttp
import psycopg2.pool


@dataclass
class ValidationResult:
    """Result of visual validation"""
    image_path: str
    character_name: str
    consistency_score: float  # 0-10
    style_score: float       # 0-10
    quality_score: float     # 0-10
    emotion_match: bool
    validation_time: float
    error: Optional[str] = None
    llava_response: Optional[str] = None


@dataclass
class CharacterReference:
    """Reference data for character validation"""
    character_name: str
    reference_images: List[str]
    expected_features: Dict[str, str]
    emotional_range: List[str]
    style_notes: str


class LLaVAClient:
    """Client for interacting with LLaVA model"""

    def __init__(self,
                 model_endpoint: str = "http://***REMOVED***:11434",
                 model_name: str = "llava:latest",
                 timeout: int = 30):
        self.endpoint = model_endpoint
        self.model_name = model_name
        self.timeout = timeout
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 for LLaVA"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            raise ValueError(f"Failed to encode image {image_path}: {e}")

    async def analyze_image(self, image_path: str, prompt: str) -> Dict:
        """Send image and prompt to LLaVA for analysis"""
        if not self.session:
            raise RuntimeError("LLaVAClient must be used as async context manager")

        image_b64 = self.encode_image(image_path)

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False
        }

        try:
            async with self.session.post(f"{self.endpoint}/api/generate", json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"LLaVA API error {response.status}: {error_text}")

                result = await response.json()
                return result

        except aiohttp.ClientError as e:
            raise RuntimeError(f"Connection error to LLaVA: {e}")


class CharacterDatabase:
    """Database operations for character references and validation results"""

    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
        self.connection_pool = None

    def initialize_pool(self):
        """Initialize database connection pool"""
        try:
            self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                1, 10,
                host=self.db_config["host"],
                database=self.db_config["database"],
                user=self.db_config["user"],
                password=self.db_config["password"]
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize database pool: {e}")

    def get_character_references(self, character_name: str) -> Optional[CharacterReference]:
        """Get character reference data from database"""
        if not self.connection_pool:
            self.initialize_pool()

        conn = None
        try:
            conn = self.connection_pool.getconn()
            cursor = conn.cursor()

            # Get character data
            cursor.execute("""
                SELECT name, reference_images, expected_features, emotional_range, style_notes
                FROM character_references
                WHERE LOWER(name) = LOWER(%s)
            """, (character_name,))

            result = cursor.fetchone()
            if not result:
                return None

            return CharacterReference(
                character_name=result[0],
                reference_images=json.loads(result[1]) if result[1] else [],
                expected_features=json.loads(result[2]) if result[2] else {},
                emotional_range=json.loads(result[3]) if result[3] else [],
                style_notes=result[4] or ""
            )

        except Exception as e:
            logging.error(f"Database error getting character {character_name}: {e}")
            return None
        finally:
            if conn:
                self.connection_pool.putconn(conn)

    def save_validation_result(self, result: ValidationResult):
        """Save validation result to database"""
        if not self.connection_pool:
            self.initialize_pool()

        conn = None
        try:
            conn = self.connection_pool.getconn()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO validation_results
                (image_path, character_name, consistency_score, style_score,
                 quality_score, emotion_match, validation_time, error_message,
                 llava_response, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """, (
                result.image_path, result.character_name, result.consistency_score,
                result.style_score, result.quality_score, result.emotion_match,
                result.validation_time, result.error, result.llava_response
            ))

            conn.commit()

        except Exception as e:
            logging.error(f"Failed to save validation result: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                self.connection_pool.putconn(conn)


class VisualValidator:
    """Main visual validation system"""

    # Validation prompts for different aspects
    VALIDATION_PROMPTS = {
        "consistency": """
            Look at this anime character image. Compare it to the character description: {description}.
            Rate the visual consistency on a scale of 1-10, where:
            - 10: Perfect match to description, all features correct
            - 8-9: Very good match, minor differences
            - 6-7: Good match, some notable differences
            - 4-5: Partial match, several differences
            - 1-3: Poor match, major differences

            Respond with ONLY a number from 1-10, nothing else.
        """,

        "style": """
            Analyze this image for anime art style quality. Rate from 1-10:
            - 10: Perfect anime style (clean lines, vibrant colors, proper proportions)
            - 8-9: High quality anime style with minor imperfections
            - 6-7: Good anime style with some issues
            - 4-5: Acceptable anime style but noticeable problems
            - 1-3: Poor anime style or not anime-like

            Respond with ONLY a number from 1-10, nothing else.
        """,

        "quality": """
            Rate the technical image quality from 1-10:
            - 10: Perfect clarity, no artifacts, excellent composition
            - 8-9: High quality with minimal flaws
            - 6-7: Good quality with some minor issues
            - 4-5: Acceptable quality but noticeable problems
            - 1-3: Poor quality with major issues

            Look for: blur, artifacts, distortions, incomplete features.
            Respond with ONLY a number from 1-10, nothing else.
        """,

        "emotion": """
            What emotion is this character expressing? Choose from:
            happy, sad, angry, surprised, neutral, excited, concerned, determined, confused

            Respond with ONLY the emotion word, nothing else.
        """
    }

    def __init__(self,
                 llava_endpoint: str = "http://***REMOVED***:11434",
                 db_config: Optional[Dict] = None):
        self.llava_client = LLaVAClient(llava_endpoint)
        self.db_client = CharacterDatabase(db_config) if db_config else None
        self.logger = logging.getLogger(__name__)

        # Performance thresholds
        self.thresholds = {
            "consistency_min": 8.5,
            "style_min": 9.0,
            "quality_min": 7.0,
            "max_validation_time": 30.0
        }

    def _extract_score(self, response_text: str) -> float:
        """Extract numeric score from LLaVA response"""
        try:
            # Look for numbers in the response
            import re
            numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', response_text)
            if numbers:
                score = float(numbers[0])
                return min(max(score, 1.0), 10.0)  # Clamp to 1-10
            return 5.0  # Default middle score
        except:
            return 5.0

    def _extract_emotion(self, response_text: str) -> str:
        """Extract emotion from LLaVA response"""
        emotions = ["happy", "sad", "angry", "surprised", "neutral",
                   "excited", "concerned", "determined", "confused"]

        response_lower = response_text.lower()
        for emotion in emotions:
            if emotion in response_lower:
                return emotion
        return "neutral"

    async def validate_image(self,
                           image_path: str,
                           character_name: str,
                           expected_emotion: str = "neutral",
                           reference_description: str = "") -> ValidationResult:
        """Validate a single image"""
        start_time = time.time()

        try:
            # Verify image exists
            if not os.path.exists(image_path):
                return ValidationResult(
                    image_path=image_path,
                    character_name=character_name,
                    consistency_score=0.0,
                    style_score=0.0,
                    quality_score=0.0,
                    emotion_match=False,
                    validation_time=0.0,
                    error=f"Image file not found: {image_path}"
                )

            # Get character reference if available
            character_ref = None
            if self.db_client:
                character_ref = self.db_client.get_character_references(character_name)

            # Use reference description if available
            if character_ref and character_ref.expected_features:
                description = json.dumps(character_ref.expected_features)
            else:
                description = reference_description or f"Character named {character_name}"

            async with self.llava_client as client:
                # Test consistency
                consistency_prompt = self.VALIDATION_PROMPTS["consistency"].format(
                    description=description
                )
                consistency_result = await client.analyze_image(image_path, consistency_prompt)
                consistency_score = self._extract_score(consistency_result.get("response", "5"))

                # Test style
                style_result = await client.analyze_image(
                    image_path, self.VALIDATION_PROMPTS["style"]
                )
                style_score = self._extract_score(style_result.get("response", "5"))

                # Test quality
                quality_result = await client.analyze_image(
                    image_path, self.VALIDATION_PROMPTS["quality"]
                )
                quality_score = self._extract_score(quality_result.get("response", "5"))

                # Test emotion
                emotion_result = await client.analyze_image(
                    image_path, self.VALIDATION_PROMPTS["emotion"]
                )
                detected_emotion = self._extract_emotion(emotion_result.get("response", "neutral"))
                emotion_match = detected_emotion.lower() == expected_emotion.lower()

                validation_time = time.time() - start_time

                # Create result
                result = ValidationResult(
                    image_path=image_path,
                    character_name=character_name,
                    consistency_score=consistency_score,
                    style_score=style_score,
                    quality_score=quality_score,
                    emotion_match=emotion_match,
                    validation_time=validation_time,
                    llava_response=json.dumps({
                        "consistency": consistency_result.get("response"),
                        "style": style_result.get("response"),
                        "quality": quality_result.get("response"),
                        "emotion": emotion_result.get("response")
                    })
                )

                # Save to database
                if self.db_client:
                    self.db_client.save_validation_result(result)

                return result

        except Exception as e:
            validation_time = time.time() - start_time
            self.logger.error(f"Validation failed for {image_path}: {e}")

            return ValidationResult(
                image_path=image_path,
                character_name=character_name,
                consistency_score=0.0,
                style_score=0.0,
                quality_score=0.0,
                emotion_match=False,
                validation_time=validation_time,
                error=str(e)
            )

    async def validate_batch(self,
                           image_paths: List[str],
                           character_name: str,
                           expected_emotions: Optional[List[str]] = None,
                           reference_description: str = "") -> List[ValidationResult]:
        """Validate multiple images in batch"""
        if expected_emotions is None:
            expected_emotions = ["neutral"] * len(image_paths)

        # Ensure lists are same length
        if len(expected_emotions) != len(image_paths):
            expected_emotions = expected_emotions[:len(image_paths)] + \
                             ["neutral"] * (len(image_paths) - len(expected_emotions))

        tasks = []
        for image_path, emotion in zip(image_paths, expected_emotions):
            task = self.validate_image(image_path, character_name, emotion, reference_description)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(ValidationResult(
                    image_path=image_paths[i],
                    character_name=character_name,
                    consistency_score=0.0,
                    style_score=0.0,
                    quality_score=0.0,
                    emotion_match=False,
                    validation_time=0.0,
                    error=str(result)
                ))
            else:
                final_results.append(result)

        return final_results

    def meets_thresholds(self, result: ValidationResult) -> bool:
        """Check if validation result meets quality thresholds"""
        return (
            result.consistency_score >= self.thresholds["consistency_min"] and
            result.style_score >= self.thresholds["style_min"] and
            result.quality_score >= self.thresholds["quality_min"] and
            result.validation_time <= self.thresholds["max_validation_time"] and
            result.error is None
        )

    def generate_report(self, results: List[ValidationResult]) -> Dict:
        """Generate summary report from validation results"""
        if not results:
            return {"error": "No results to analyze"}

        passed_results = [r for r in results if self.meets_thresholds(r)]
        failed_results = [r for r in results if not self.meets_thresholds(r)]

        report = {
            "total_images": len(results),
            "passed": len(passed_results),
            "failed": len(failed_results),
            "pass_rate": len(passed_results) / len(results) * 100,
            "average_scores": {
                "consistency": sum(r.consistency_score for r in results) / len(results),
                "style": sum(r.style_score for r in results) / len(results),
                "quality": sum(r.quality_score for r in results) / len(results)
            },
            "average_validation_time": sum(r.validation_time for r in results) / len(results),
            "emotion_accuracy": sum(r.emotion_match for r in results) / len(results) * 100,
            "thresholds": self.thresholds,
            "failed_images": [r.image_path for r in failed_results]
        }

        return report


# CLI interface for standalone testing
if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Visual validation for anime generation")
    parser.add_argument("images", nargs="+", help="Image file paths to validate")
    parser.add_argument("--character", required=True, help="Character name")
    parser.add_argument("--emotion", default="neutral", help="Expected emotion")
    parser.add_argument("--description", default="", help="Character description")
    parser.add_argument("--llava-endpoint", default="http://***REMOVED***:11434",
                       help="LLaVA API endpoint")
    parser.add_argument("--output", help="Output JSON file for results")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Create validator
    validator = VisualValidator(args.llava_endpoint)

    async def main():
        results = await validator.validate_batch(
            args.images,
            args.character,
            [args.emotion] * len(args.images),
            args.description
        )

        report = validator.generate_report(results)

        if args.output:
            with open(args.output, 'w') as f:
                json.dump({
                    "results": [vars(r) for r in results],
                    "report": report
                }, f, indent=2)
            print(f"Results saved to {args.output}")
        else:
            print(json.dumps(report, indent=2))

        # Exit with non-zero if validation failed
        if report["pass_rate"] < 100:
            sys.exit(1)

    asyncio.run(main())