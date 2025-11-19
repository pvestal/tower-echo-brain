"""
Character Generation Accuracy Tests
Tests the complete character generation pipeline from request to final output validation.
"""

import asyncio
import json
import os
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pytest
import httpx
import psycopg2
from PIL import Image

from ..visual.visual_validator import VisualValidator, ValidationResult


class CharacterGenerationTest:
    """Test character generation accuracy and consistency"""

    def __init__(self,
                 anime_api_endpoint: str = "http://192.168.50.135:8328",
                 echo_api_endpoint: str = "http://192.168.50.135:8309",
                 comfyui_endpoint: str = "http://192.168.50.135:8188",
                 test_timeout: int = 300):  # 5 minutes max per test
        self.anime_api = anime_api_endpoint
        self.echo_api = echo_api_endpoint
        self.comfyui_api = comfyui_endpoint
        self.timeout = test_timeout
        self.validator = VisualValidator()

        # Test characters with known attributes
        self.test_characters = {
            "Ryuu": {
                "description": "Young male warrior with spiky black hair, determined expression, wearing blue tunic",
                "expected_features": {
                    "hair_color": "black",
                    "hair_style": "spiky",
                    "gender": "male",
                    "age_range": "young adult",
                    "clothing": "blue tunic"
                },
                "test_emotions": ["determined", "angry", "neutral"],
                "test_scenarios": [
                    "standing in forest clearing",
                    "holding sword in battle stance",
                    "looking thoughtful by campfire"
                ]
            },
            "Yuki": {
                "description": "Young female mage with long silver hair, gentle blue eyes, wearing white robes",
                "expected_features": {
                    "hair_color": "silver",
                    "hair_style": "long",
                    "eye_color": "blue",
                    "gender": "female",
                    "clothing": "white robes"
                },
                "test_emotions": ["gentle", "concerned", "happy"],
                "test_scenarios": [
                    "casting spell with glowing hands",
                    "reading ancient tome in library",
                    "smiling while tending to flowers"
                ]
            },
            "Kai": {
                "description": "Cyberpunk hacker with neon blue hair, cybernetic eye implant, dark jacket",
                "expected_features": {
                    "hair_color": "neon blue",
                    "augmentation": "cybernetic eye",
                    "clothing": "dark jacket",
                    "style": "cyberpunk"
                },
                "test_emotions": ["focused", "suspicious", "excited"],
                "test_scenarios": [
                    "typing on holographic keyboard",
                    "standing in neon-lit alley",
                    "examining data streams"
                ]
            }
        }

    async def test_character_consistency(self, character_name: str, num_variations: int = 5) -> Dict:
        """Test that character appears consistently across multiple generations"""
        if character_name not in self.test_characters:
            raise ValueError(f"Unknown test character: {character_name}")

        character_data = self.test_characters[character_name]
        results = {
            "character_name": character_name,
            "test_started": datetime.now().isoformat(),
            "generations": [],
            "validation_results": [],
            "consistency_score": 0.0,
            "errors": []
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Generate multiple variations of the same character
                for i in range(num_variations):
                    generation_start = time.time()

                    # Create generation request
                    prompt = f"{character_data['description']}, anime style, high quality"
                    generation_request = {
                        "character_name": character_name,
                        "prompt": prompt,
                        "style": "anime",
                        "quality": "high",
                        "seed": i * 12345,  # Different seed for variation
                        "project_id": f"test_consistency_{character_name}_{uuid.uuid4().hex[:8]}"
                    }

                    try:
                        # Submit generation request
                        response = await client.post(
                            f"{self.anime_api}/api/anime/generate",
                            json=generation_request
                        )

                        if response.status_code != 200:
                            error_msg = f"Generation request failed: {response.status_code} - {response.text}"
                            results["errors"].append(error_msg)
                            continue

                        generation_data = response.json()
                        job_id = generation_data.get("job_id")

                        if not job_id:
                            results["errors"].append("No job_id returned from generation request")
                            continue

                        # Wait for completion and get result
                        image_path = await self._wait_for_completion(client, job_id)

                        if image_path:
                            generation_time = time.time() - generation_start

                            generation_result = {
                                "iteration": i + 1,
                                "job_id": job_id,
                                "image_path": image_path,
                                "generation_time": generation_time,
                                "prompt": prompt,
                                "seed": generation_request["seed"]
                            }

                            results["generations"].append(generation_result)

                            # Validate the generated image
                            validation_result = await self.validator.validate_image(
                                image_path,
                                character_name,
                                expected_emotion="neutral",
                                reference_description=character_data["description"]
                            )

                            results["validation_results"].append(validation_result)

                        else:
                            results["errors"].append(f"Generation {i+1} failed to produce image")

                    except Exception as e:
                        results["errors"].append(f"Generation {i+1} error: {str(e)}")

                # Calculate overall consistency score
                if results["validation_results"]:
                    consistency_scores = [r.consistency_score for r in results["validation_results"]]
                    results["consistency_score"] = sum(consistency_scores) / len(consistency_scores)

                results["test_completed"] = datetime.now().isoformat()
                return results

        except Exception as e:
            results["errors"].append(f"Test framework error: {str(e)}")
            results["test_completed"] = datetime.now().isoformat()
            return results

    async def test_emotion_accuracy(self, character_name: str) -> Dict:
        """Test that character can express different emotions accurately"""
        if character_name not in self.test_characters:
            raise ValueError(f"Unknown test character: {character_name}")

        character_data = self.test_characters[character_name]
        emotions_to_test = character_data["test_emotions"]

        results = {
            "character_name": character_name,
            "test_started": datetime.now().isoformat(),
            "emotion_tests": [],
            "emotion_accuracy": 0.0,
            "errors": []
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                for emotion in emotions_to_test:
                    generation_start = time.time()

                    # Create emotion-specific prompt
                    emotion_prompt = f"{character_data['description']}, {emotion} expression, anime style"
                    generation_request = {
                        "character_name": character_name,
                        "prompt": emotion_prompt,
                        "style": "anime",
                        "emotion": emotion,
                        "project_id": f"test_emotion_{character_name}_{emotion}_{uuid.uuid4().hex[:8]}"
                    }

                    try:
                        # Submit generation request
                        response = await client.post(
                            f"{self.anime_api}/api/anime/generate",
                            json=generation_request
                        )

                        if response.status_code != 200:
                            error_msg = f"Emotion test {emotion} failed: {response.status_code}"
                            results["errors"].append(error_msg)
                            continue

                        generation_data = response.json()
                        job_id = generation_data.get("job_id")

                        # Wait for completion
                        image_path = await self._wait_for_completion(client, job_id)

                        if image_path:
                            generation_time = time.time() - generation_start

                            # Validate emotion expression
                            validation_result = await self.validator.validate_image(
                                image_path,
                                character_name,
                                expected_emotion=emotion,
                                reference_description=character_data["description"]
                            )

                            emotion_test = {
                                "emotion": emotion,
                                "job_id": job_id,
                                "image_path": image_path,
                                "generation_time": generation_time,
                                "validation_result": validation_result,
                                "emotion_match": validation_result.emotion_match
                            }

                            results["emotion_tests"].append(emotion_test)

                        else:
                            results["errors"].append(f"Emotion test {emotion} failed to produce image")

                    except Exception as e:
                        results["errors"].append(f"Emotion test {emotion} error: {str(e)}")

                # Calculate emotion accuracy
                if results["emotion_tests"]:
                    matches = sum(1 for test in results["emotion_tests"] if test["emotion_match"])
                    results["emotion_accuracy"] = (matches / len(results["emotion_tests"])) * 100

                results["test_completed"] = datetime.now().isoformat()
                return results

        except Exception as e:
            results["errors"].append(f"Emotion test framework error: {str(e)}")
            results["test_completed"] = datetime.now().isoformat()
            return results

    async def test_scenario_adaptability(self, character_name: str) -> Dict:
        """Test character consistency across different scenarios"""
        if character_name not in self.test_characters:
            raise ValueError(f"Unknown test character: {character_name}")

        character_data = self.test_characters[character_name]
        scenarios = character_data["test_scenarios"]

        results = {
            "character_name": character_name,
            "test_started": datetime.now().isoformat(),
            "scenario_tests": [],
            "average_consistency": 0.0,
            "errors": []
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                for scenario in scenarios:
                    generation_start = time.time()

                    # Create scenario-specific prompt
                    scenario_prompt = f"{character_data['description']}, {scenario}, anime style"
                    generation_request = {
                        "character_name": character_name,
                        "prompt": scenario_prompt,
                        "style": "anime",
                        "scenario": scenario,
                        "project_id": f"test_scenario_{character_name}_{uuid.uuid4().hex[:8]}"
                    }

                    try:
                        # Submit generation request
                        response = await client.post(
                            f"{self.anime_api}/api/anime/generate",
                            json=generation_request
                        )

                        if response.status_code != 200:
                            error_msg = f"Scenario test failed: {response.status_code}"
                            results["errors"].append(error_msg)
                            continue

                        generation_data = response.json()
                        job_id = generation_data.get("job_id")

                        # Wait for completion
                        image_path = await self._wait_for_completion(client, job_id)

                        if image_path:
                            generation_time = time.time() - generation_start

                            # Validate character consistency in scenario
                            validation_result = await self.validator.validate_image(
                                image_path,
                                character_name,
                                expected_emotion="neutral",
                                reference_description=character_data["description"]
                            )

                            scenario_test = {
                                "scenario": scenario,
                                "job_id": job_id,
                                "image_path": image_path,
                                "generation_time": generation_time,
                                "validation_result": validation_result,
                                "consistency_score": validation_result.consistency_score
                            }

                            results["scenario_tests"].append(scenario_test)

                        else:
                            results["errors"].append(f"Scenario test failed to produce image: {scenario}")

                    except Exception as e:
                        results["errors"].append(f"Scenario test error: {str(e)}")

                # Calculate average consistency across scenarios
                if results["scenario_tests"]:
                    consistency_scores = [test["consistency_score"] for test in results["scenario_tests"]]
                    results["average_consistency"] = sum(consistency_scores) / len(consistency_scores)

                results["test_completed"] = datetime.now().isoformat()
                return results

        except Exception as e:
            results["errors"].append(f"Scenario test framework error: {str(e)}")
            results["test_completed"] = datetime.now().isoformat()
            return results

    async def _wait_for_completion(self, client: httpx.AsyncClient, job_id: str,
                                 max_wait: int = 600) -> Optional[str]:
        """Wait for generation job to complete and return image path"""
        start_time = time.time()

        while time.time() - start_time < max_wait:
            try:
                # Check job status
                response = await client.get(f"{self.anime_api}/api/anime/status/{job_id}")

                if response.status_code != 200:
                    return None

                status_data = response.json()
                status = status_data.get("status", "unknown")

                if status == "completed":
                    # Get the generated image path
                    output_path = status_data.get("output_path")
                    if output_path and os.path.exists(output_path):
                        return output_path

                elif status == "failed" or status == "error":
                    return None

                # Wait before checking again
                await asyncio.sleep(5)

            except Exception:
                # Continue trying
                await asyncio.sleep(5)

        return None

    def meets_production_standards(self, test_results: Dict) -> bool:
        """Check if test results meet production quality standards"""
        if test_results.get("errors"):
            return False

        # Check consistency score
        consistency_score = test_results.get("consistency_score", 0)
        if consistency_score < 8.5:  # Below threshold
            return False

        # Check emotion accuracy if present
        emotion_accuracy = test_results.get("emotion_accuracy", 100)
        if emotion_accuracy < 85:  # Below 85% accuracy
            return False

        # Check generation times
        generations = test_results.get("generations", [])
        for gen in generations:
            if gen.get("generation_time", 0) > 60:  # Over 1 minute
                return False

        return True


# Pytest test cases
class TestCharacterGeneration:
    """Pytest test class for character generation"""

    @pytest.fixture
    def character_test(self):
        return CharacterGenerationTest()

    @pytest.mark.asyncio
    async def test_ryuu_consistency(self, character_test):
        """Test Ryuu character consistency across generations"""
        results = await character_test.test_character_consistency("Ryuu", num_variations=3)

        # Assertions
        assert not results["errors"], f"Errors occurred: {results['errors']}"
        assert len(results["generations"]) >= 2, "Should generate at least 2 variations"
        assert results["consistency_score"] >= 7.0, f"Consistency too low: {results['consistency_score']}"

        # Check generation times are reasonable
        for gen in results["generations"]:
            assert gen["generation_time"] < 120, f"Generation too slow: {gen['generation_time']}s"

    @pytest.mark.asyncio
    async def test_yuki_emotions(self, character_test):
        """Test Yuki character emotion expression accuracy"""
        results = await character_test.test_emotion_accuracy("Yuki")

        # Assertions
        assert not results["errors"], f"Errors occurred: {results['errors']}"
        assert len(results["emotion_tests"]) >= 2, "Should test multiple emotions"
        assert results["emotion_accuracy"] >= 70, f"Emotion accuracy too low: {results['emotion_accuracy']}%"

    @pytest.mark.asyncio
    async def test_kai_scenarios(self, character_test):
        """Test Kai character across different scenarios"""
        results = await character_test.test_scenario_adaptability("Kai")

        # Assertions
        assert not results["errors"], f"Errors occurred: {results['errors']}"
        assert len(results["scenario_tests"]) >= 2, "Should test multiple scenarios"
        assert results["average_consistency"] >= 7.0, f"Scenario consistency too low: {results['average_consistency']}"

    @pytest.mark.asyncio
    async def test_production_standards(self, character_test):
        """Test that all characters meet production quality standards"""
        characters_to_test = ["Ryuu", "Yuki"]  # Limited for CI/CD
        all_passed = True
        failed_characters = []

        for character_name in characters_to_test:
            results = await character_test.test_character_consistency(character_name, num_variations=2)

            if not character_test.meets_production_standards(results):
                all_passed = False
                failed_characters.append(character_name)

        assert all_passed, f"Characters failed production standards: {failed_characters}"

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_comprehensive_validation(self, character_test):
        """Comprehensive test of all characters and scenarios (slow test)"""
        all_results = {}

        for character_name in character_test.test_characters.keys():
            # Test consistency
            consistency_results = await character_test.test_character_consistency(character_name, num_variations=3)
            # Test emotions
            emotion_results = await character_test.test_emotion_accuracy(character_name)
            # Test scenarios
            scenario_results = await character_test.test_scenario_adaptability(character_name)

            all_results[character_name] = {
                "consistency": consistency_results,
                "emotions": emotion_results,
                "scenarios": scenario_results
            }

        # Save comprehensive results
        results_file = f"/tmp/character_validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

        print(f"Comprehensive validation results saved to: {results_file}")

        # Check overall system health
        total_tests = sum(len(char_results.keys()) for char_results in all_results.values())
        failed_tests = 0

        for character_name, char_results in all_results.items():
            for test_type, test_result in char_results.items():
                if not character_test.meets_production_standards(test_result):
                    failed_tests += 1

        success_rate = ((total_tests - failed_tests) / total_tests) * 100
        assert success_rate >= 80, f"Overall system success rate too low: {success_rate}%"


if __name__ == "__main__":
    # CLI interface for standalone testing
    import argparse

    parser = argparse.ArgumentParser(description="Character generation testing")
    parser.add_argument("--character", required=True, help="Character name to test")
    parser.add_argument("--test-type", choices=["consistency", "emotions", "scenarios", "all"],
                       default="consistency", help="Type of test to run")
    parser.add_argument("--variations", type=int, default=3, help="Number of variations for consistency test")
    parser.add_argument("--output", help="Output file for results")

    args = parser.parse_args()

    async def main():
        test_runner = CharacterGenerationTest()

        if args.test_type == "consistency":
            results = await test_runner.test_character_consistency(args.character, args.variations)
        elif args.test_type == "emotions":
            results = await test_runner.test_emotion_accuracy(args.character)
        elif args.test_type == "scenarios":
            results = await test_runner.test_scenario_adaptability(args.character)
        elif args.test_type == "all":
            consistency_results = await test_runner.test_character_consistency(args.character, args.variations)
            emotion_results = await test_runner.test_emotion_accuracy(args.character)
            scenario_results = await test_runner.test_scenario_adaptability(args.character)

            results = {
                "consistency": consistency_results,
                "emotions": emotion_results,
                "scenarios": scenario_results
            }

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Results saved to {args.output}")
        else:
            print(json.dumps(results, indent=2, default=str))

    asyncio.run(main())