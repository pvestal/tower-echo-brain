#!/usr/bin/env python3
"""
Enhanced Autonomous Tokyo Debt Desire Image Generator with Echo Brain Integration
Continuously generates photorealistic images with quality feedback and prompt optimization
"""

import asyncio
import requests
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any
from image_quality_reviewer import EchoBrainImageReviewer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedAutonomousImageGenerator:
    """Enhanced autonomous image generator with Echo Brain feedback"""

    def __init__(self):
        self.comfyui_url = "http://localhost:8188"
        self.output_dir = Path("/mnt/1TB-storage/ComfyUI/output")
        self.is_running = False
        self.generation_count = 0

        # Initialize Echo Brain reviewer
        self.reviewer = EchoBrainImageReviewer()

        # Character quality tracking
        self.character_scores = {
            "Mei_Kobayashi": [],
            "Rina_Suzuki": [],
            "Yuki_Tanaka": [],
            "Takeshi_Sato": []
        }

        # Enhanced character prompts with optimization tracking
        self.character_prompts = {
            "Mei_Kobayashi": {
                "base_prompts": [
                    "beautiful Japanese woman, female anatomy, long dark hair, seductive gentle expression, wearing skimpy apron with visible cleavage and bare legs, cooking in Tokyo apartment kitchen, curvy body, medium natural breasts, vagina, feminine curves, photorealistic, natural skin texture, sexy pose, aroused expression",
                    "beautiful curvy Japanese woman, female anatomy, long dark hair, sultry smile, wearing tight low-cut top and short skirt, sitting with legs crossed on couch, visible cleavage, Tokyo apartment, feminine body, photorealistic, detailed skin, alluring pose, slightly aroused",
                    "beautiful Japanese woman, female anatomy, long dark hair, caring but sexy expression, wearing revealing crop top and short shorts while cleaning, bent over pose showing curves from behind, rear view, Tokyo apartment, natural lighting, feminine curves, photorealistic, attractive body, arousal visible"
                ],
                "optimizations": [],
                "current_index": 0
            },
            "Rina_Suzuki": {
                "base_prompts": [
                    "attractive confident Japanese woman, female anatomy, short brown hair, assertive seductive expression, hands on hips emphasizing curves, wearing tight mini dress with deep neckline, medium natural breasts, curvy hips, vagina, feminine curves, Tokyo apartment, photorealistic, sexy stance, aroused",
                    "attractive Japanese woman, female anatomy, short brown hair, seductive confident expression, wearing lingerie or bikini, bedroom setting, curvy body with medium natural breasts, feminine body, provocative pose from behind, rear view, photorealistic, sensual lighting, arousal state",
                    "attractive Japanese woman, female anatomy, short brown hair, arguing but sexy pose, aggressive stance with chest pushed out, wearing revealing tight clothing, medium natural breasts visible, feminine curves, Tokyo apartment living room, photorealistic, alluring, sexually excited"
                ],
                "optimizations": [],
                "current_index": 0
            },
            "Yuki_Tanaka": {
                "base_prompts": [
                    "young nervous Japanese man, male anatomy, average build, worried anxious expression, counting money bills, wearing casual t-shirt, surrounded by two sexy women in revealing clothing, Tokyo apartment, photorealistic, harem protagonist, visible male arousal, penis, masculine body, NSFW situation",
                    "young Japanese man, male anatomy, nervous worried expression, sitting shirtless on couch between two curvy women touching him, casual clothes partially removed, masculine chest, penis visible, caught in compromising situation, visible physical attraction and arousal, photorealistic, awkward but sexually excited, NSFW scene",
                    "young Japanese man, male anatomy, stressed but sexually aroused expression, holding bills while attractive woman in lingerie seductively distracts him from behind, rear view of woman, Tokyo apartment setting, masculine features, penis, photorealistic, conflicted between money and desire, visible male arousal, NSFW content"
                ],
                "optimizations": [],
                "current_index": 0
            },
            "Takeshi_Sato": {
                "base_prompts": [
                    "intimidating middle-aged Japanese man, male anatomy, short black hair, cold menacing expression, expensive dark business suit, standing in dark Tokyo alley, masculine features, photorealistic, dangerous aura",
                    "intimidating Japanese man in business suit, male anatomy, menacing expression, standing at apartment door while sexy women in background, threatening pose, masculine build, urban setting, photorealistic, power dynamic",
                    "intimidating yakuza man, male anatomy, dark suit, cold calculating expression, office setting with city view and attractive secretary, masculine presence, photorealistic detailed features, authority figure"
                ],
                "optimizations": [],
                "current_index": 0
            }
        }

    async def start_enhanced_generation(self):
        """Start the enhanced generation process with Echo Brain feedback"""

        logger.info("🧠 Starting Enhanced Autonomous Tokyo Debt Desire Generator with Echo Brain")
        self.is_running = True

        generation_cycle = 0
        max_images = 1000  # Stop at 1000 images for user feedback
        review_frequency = 5  # Review every 5 generations

        while self.is_running and self.generation_count < max_images:
            try:
                generation_cycle += 1
                logger.info(f"🎯 Starting enhanced generation cycle {generation_cycle}")

                # Generate one image for each character
                for character_name, character_data in self.character_prompts.items():
                    # Get current prompt (base or optimized)
                    current_prompt = self._get_current_prompt(character_name)

                    success = await self._generate_character_image(character_name, current_prompt, generation_cycle)
                    if success:
                        self.generation_count += 1
                        logger.info(f"✅ Generated {character_name} NSFW image #{self.generation_count}")

                        # Echo Brain analysis every few generations
                        if generation_cycle % review_frequency == 0:
                            await self._echo_brain_review(character_name, current_prompt, generation_cycle)

                        # Prompt optimization check
                        if generation_cycle % 10 == 0:  # Every 10 cycles
                            await self._optimize_prompts(character_name)

                    else:
                        logger.warning(f"❌ Failed to generate {character_name}")

                    # Small delay between characters
                    await asyncio.sleep(3)

                logger.info(f"🔄 Enhanced cycle {generation_cycle} complete. Generated {self.generation_count} total images.")

                # Check if we've reached the limit
                if self.generation_count >= max_images:
                    logger.info(f"🎯 Reached {max_images} image limit. Stopping for user feedback...")
                    await self._generate_final_report()
                    break

                # Wait before next cycle
                await asyncio.sleep(30)

            except KeyboardInterrupt:
                logger.info("🛑 Stopping enhanced generation...")
                self.is_running = False
                break
            except Exception as e:
                logger.error(f"❌ Error in enhanced generation cycle: {e}")
                await asyncio.sleep(10)

    def _get_current_prompt(self, character: str) -> str:
        """Get the current best prompt for a character"""

        char_data = self.character_prompts[character]

        # Use optimization if available and performing well
        if char_data["optimizations"]:
            return char_data["optimizations"][-1]  # Latest optimization

        # Otherwise use rotating base prompts
        prompt_index = char_data["current_index"] % len(char_data["base_prompts"])
        char_data["current_index"] += 1
        return char_data["base_prompts"][prompt_index]

    async def _echo_brain_review(self, character: str, prompt: str, cycle: int):
        """Perform Echo Brain quality analysis on generated image"""

        try:
            # Find the generated image
            image_path = self.output_dir / f"autonomous_enhanced_tdd_{character}_cycle{cycle}_00001_.png"

            if not image_path.exists():
                # Try alternative naming
                alt_path = self.output_dir / f"autonomous_enhanced_tdd_{character}_cycle{cycle}_00002_.png"
                if alt_path.exists():
                    image_path = alt_path

            if image_path.exists():
                # Get Echo Brain analysis
                analysis = await self.reviewer.analyze_image_quality(image_path, character, prompt)

                quality_score = analysis.get("quality_score", 0)
                suggestions = analysis.get("suggestions", [])

                # Track quality scores
                self.character_scores[character].append(quality_score)

                logger.info(f"🧠 Echo Brain Analysis - {character}: Quality Score {quality_score:.1f}/10")
                if suggestions:
                    logger.info(f"💡 Suggestions: {', '.join(suggestions[:2])}")

                # Store in character's optimization data
                if quality_score < 7.0 and suggestions:
                    await self._apply_suggestions(character, suggestions)

            else:
                logger.warning(f"⚠️  Could not find image for Echo Brain analysis: {image_path}")

        except Exception as e:
            logger.error(f"❌ Echo Brain review failed for {character}: {e}")

    async def _apply_suggestions(self, character: str, suggestions: List[str]):
        """Apply Echo Brain suggestions to improve prompts"""

        try:
            char_data = self.character_prompts[character]
            base_prompt = char_data["base_prompts"][0]  # Use first prompt as base

            # Apply suggestions to create optimized prompt
            optimized_prompt = base_prompt

            for suggestion in suggestions:
                if "masculine features" in suggestion and character in ["Yuki_Tanaka", "Takeshi_Sato"]:
                    if "masculine jawline" not in optimized_prompt:
                        optimized_prompt += ", masculine jawline, broad shoulders"

                elif "detailed feminine descriptors" in suggestion and character in ["Mei_Kobayashi", "Rina_Suzuki"]:
                    if "graceful posture" not in optimized_prompt:
                        optimized_prompt += ", graceful posture, elegant curves"

                elif "visual details" in suggestion:
                    if "high detail" not in optimized_prompt:
                        optimized_prompt += ", high detail, sharp focus, professional lighting"

            # Store optimization
            char_data["optimizations"].append(optimized_prompt)
            logger.info(f"🔧 Applied optimization to {character} prompts")

        except Exception as e:
            logger.error(f"❌ Failed to apply suggestions for {character}: {e}")

    async def _optimize_prompts(self, character: str):
        """Optimize prompts based on recent quality scores"""

        try:
            recent_scores = self.character_scores[character][-5:]  # Last 5 scores

            if len(recent_scores) >= 3:
                optimization_suggestion = await self.reviewer.suggest_prompt_optimization(character, recent_scores)

                if optimization_suggestion:
                    logger.info(f"🎯 {character} optimization suggestion: {optimization_suggestion}")

                    # Apply optimization suggestion
                    await self._apply_suggestions(character, [optimization_suggestion])

        except Exception as e:
            logger.error(f"❌ Prompt optimization failed for {character}: {e}")

    async def _generate_character_image(self, character: str, prompt: str, cycle: int) -> bool:
        """Generate a single character image"""

        try:
            workflow = self._create_workflow(character, prompt, cycle)

            response = requests.post(f"{self.comfyui_url}/prompt", json={"prompt": workflow["prompt"]}, timeout=15)
            result = response.json()

            if "prompt_id" in result:
                logger.info(f"📷 Queued enhanced {character} generation: {result['prompt_id']}")

                # Wait for generation to complete (increased for queue processing)
                await asyncio.sleep(20)

                # Check if file was created
                expected_file = self.output_dir / f"autonomous_enhanced_tdd_{character}_cycle{cycle}_00001_.png"
                if expected_file.exists():
                    file_size = expected_file.stat().st_size
                    if file_size > 100000:  # At least 100KB
                        return True

                return False
            else:
                logger.error(f"❌ No prompt_id returned for {character}")
                return False

        except Exception as e:
            logger.error(f"❌ Generation failed for {character}: {e}")
            return False

    def _create_workflow(self, character: str, prompt: str, cycle: int) -> dict:
        """Create ComfyUI workflow for character generation"""

        return {
            "prompt": {
                "1": {
                    "inputs": {
                        "ckpt_name": "realisticVision_v51.safetensors"
                    },
                    "class_type": "CheckpointLoaderSimple"
                },
                "2": {
                    "inputs": {
                        "text": f"photograph, {prompt}, ultra realistic, high detail, professional photography, 4k, masterpiece, natural lighting, NSFW, explicit, adult content, sexually suggestive",
                        "clip": ["1", 1]
                    },
                    "class_type": "CLIPTextEncode"
                },
                "3": {
                    "inputs": {
                        "text": "cartoon, anime, illustration, drawing, painting, low quality, blurry, deformed, ugly, distorted",
                        "clip": ["1", 1]
                    },
                    "class_type": "CLIPTextEncode"
                },
                "4": {
                    "inputs": {
                        "width": 768,
                        "height": 1024,
                        "batch_size": 1
                    },
                    "class_type": "EmptyLatentImage"
                },
                "5": {
                    "inputs": {
                        "seed": int(time.time()) % 1000000 + cycle,
                        "steps": 25,
                        "cfg": 6.5,
                        "sampler_name": "dpmpp_2m_sde",
                        "scheduler": "karras",
                        "denoise": 1,
                        "model": ["1", 0],
                        "positive": ["2", 0],
                        "negative": ["3", 0],
                        "latent_image": ["4", 0]
                    },
                    "class_type": "KSampler"
                },
                "6": {
                    "inputs": {
                        "samples": ["5", 0],
                        "vae": ["1", 2]
                    },
                    "class_type": "VAEDecode"
                },
                "7": {
                    "inputs": {
                        "filename_prefix": f"autonomous_enhanced_tdd_{character}_cycle{cycle}",
                        "images": ["6", 0]
                    },
                    "class_type": "SaveImage"
                }
            }
        }

    async def _generate_final_report(self):
        """Generate final quality report using Echo Brain"""

        logger.info("📊 Generating Echo Brain Quality Report...")

        for character, scores in self.character_scores.items():
            if scores:
                avg_score = sum(scores) / len(scores)
                max_score = max(scores)
                min_score = min(scores)

                logger.info(f"📈 {character}: Avg Quality {avg_score:.1f}, Range {min_score:.1f}-{max_score:.1f} ({len(scores)} samples)")

        logger.info("🎯 Echo Brain autonomous generation complete with quality tracking!")

    def get_status(self) -> dict:
        """Get current generation status with Echo Brain metrics"""

        return {
            "is_running": self.is_running,
            "total_generated": self.generation_count,
            "output_directory": str(self.output_dir),
            "echo_brain_enabled": True,
            "character_quality_scores": {
                char: {
                    "count": len(scores),
                    "average": sum(scores) / len(scores) if scores else 0,
                    "latest": scores[-1] if scores else None
                } for char, scores in self.character_scores.items()
            }
        }

async def main():
    """Main function"""

    generator = EnhancedAutonomousImageGenerator()

    logger.info("🚀 Echo Brain Enhanced Tokyo Debt Desire Generator")
    logger.info("🧠 Will continuously generate with quality feedback and optimization")
    logger.info("🔄 Press Ctrl+C to stop")

    try:
        await generator.start_enhanced_generation()
    except KeyboardInterrupt:
        logger.info("🛑 Enhanced generation stopped by user")

    # Show final status
    status = generator.get_status()
    logger.info(f"📊 Final Status: Generated {status['total_generated']} images with Echo Brain feedback")

if __name__ == "__main__":
    asyncio.run(main())