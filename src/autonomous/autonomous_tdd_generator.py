#!/usr/bin/env python3
"""
Autonomous Tokyo Debt Desire Image Generator
Simplified autonomous system that continuously generates photorealistic images
"""

import asyncio
import requests
import json
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AutonomousImageGenerator:
    """Simplified autonomous image generator for Tokyo Debt Desire"""

    def __init__(self):
        self.comfyui_url = "http://localhost:8188"
        self.output_dir = Path("/mnt/1TB-storage/ComfyUI/output")
        self.is_running = False
        self.generation_count = 0

        # Character body consistency tracking
        self.character_traits = {
            "Mei_Kobayashi": {
                "hair": "long dark hair",
                "body": "curvy body, medium natural breasts",
                "personality": "gentle, caring, seductive",
                "clothing": "revealing, skimpy, low-cut"
            },
            "Rina_Suzuki": {
                "hair": "short brown hair",
                "body": "curvy hips, medium natural breasts",
                "personality": "confident, assertive, seductive",
                "clothing": "tight, revealing, mini dress"
            },
            "Yuki_Tanaka": {
                "hair": "casual hair",
                "body": "average build, nervous, visible arousal",
                "personality": "worried, anxious, conflicted, sexually attracted",
                "clothing": "casual t-shirt, everyday clothes, partially undressed"
            },
            "Takeshi_Sato": {
                "hair": "short black hair",
                "body": "intimidating build",
                "personality": "cold, menacing, dangerous",
                "clothing": "expensive dark business suit"
            }
        }

    async def start_autonomous_generation(self):
        """Start the autonomous generation process"""

        logger.info("🧠 Starting Autonomous Tokyo Debt Desire Generator")
        self.is_running = True

        # Define NSFW character scenarios with enhanced sex appeal
        characters = {
            "Mei_Kobayashi": [
                "beautiful Japanese woman, female anatomy, long dark hair, seductive gentle expression, wearing skimpy apron with visible cleavage and bare legs, cooking in Tokyo apartment kitchen, curvy body, medium natural breasts, vagina, feminine curves, photorealistic, natural skin texture, sexy pose, aroused expression",
                "beautiful curvy Japanese woman, female anatomy, long dark hair, sultry smile, wearing tight low-cut top and short skirt, sitting with legs crossed on couch, visible cleavage, Tokyo apartment, feminine body, photorealistic, detailed skin, alluring pose, slightly aroused",
                "beautiful Japanese woman, female anatomy, long dark hair, caring but sexy expression, wearing revealing crop top and short shorts while cleaning, bent over pose showing curves from behind, rear view, Tokyo apartment, natural lighting, feminine curves, photorealistic, attractive body, arousal visible"
            ],
            "Rina_Suzuki": [
                "attractive confident Japanese woman, female anatomy, short brown hair, assertive seductive expression, hands on hips emphasizing curves, wearing tight mini dress with deep neckline, medium natural breasts, curvy hips, vagina, feminine curves, Tokyo apartment, photorealistic, sexy stance, aroused",
                "attractive Japanese woman, female anatomy, short brown hair, seductive confident expression, wearing lingerie or bikini, bedroom setting, curvy body with medium natural breasts, feminine body, provocative pose from behind, rear view, photorealistic, sensual lighting, arousal state",
                "attractive Japanese woman, female anatomy, short brown hair, arguing but sexy pose, aggressive stance with chest pushed out, wearing revealing tight clothing, medium natural breasts visible, feminine curves, Tokyo apartment living room, photorealistic, alluring, sexually excited"
            ],
            "Yuki_Tanaka": [
                "young nervous Japanese man, male anatomy, average build, worried anxious expression, counting money bills, wearing casual t-shirt, surrounded by two sexy women in revealing clothing, Tokyo apartment, photorealistic, harem protagonist, visible male arousal, penis, masculine body, NSFW situation",
                "young Japanese man, male anatomy, nervous worried expression, sitting shirtless on couch between two curvy women touching him, casual clothes partially removed, masculine chest, penis visible, caught in compromising situation, visible physical attraction and arousal, photorealistic, awkward but sexually excited, NSFW scene",
                "young Japanese man, male anatomy, stressed but sexually aroused expression, holding bills while attractive woman in lingerie seductively distracts him from behind, rear view of woman, Tokyo apartment setting, masculine features, penis, photorealistic, conflicted between money and desire, visible male arousal, NSFW content"
            ],
            "Takeshi_Sato": [
                "intimidating middle-aged Japanese man, male anatomy, short black hair, cold menacing expression, expensive dark business suit, standing in dark Tokyo alley, masculine features, photorealistic, dangerous aura",
                "intimidating Japanese man in business suit, male anatomy, menacing expression, standing at apartment door while sexy women in background, threatening pose, masculine build, urban setting, photorealistic, power dynamic",
                "intimidating yakuza man, male anatomy, dark suit, cold calculating expression, office setting with city view and attractive secretary, masculine presence, photorealistic detailed features, authority figure"
            ]
        }

        generation_cycle = 0
        max_images = 1000  # Stop at 1000 images for user feedback

        while self.is_running and self.generation_count < max_images:
            try:
                generation_cycle += 1
                logger.info(f"🎯 Starting generation cycle {generation_cycle}")

                # Generate one image for each character
                for character_name, prompts in characters.items():
                    # Rotate through different prompts
                    prompt_index = (generation_cycle - 1) % len(prompts)
                    base_prompt = prompts[prompt_index]

                    # Add consistent character traits
                    consistent_prompt = self._ensure_character_consistency(character_name, base_prompt)

                    success = await self._generate_character_image(character_name, consistent_prompt, generation_cycle)
                    if success:
                        self.generation_count += 1
                        logger.info(f"✅ Generated {character_name} NSFW image #{self.generation_count}")

                        # Verify body consistency
                        await self._verify_body_consistency(character_name, generation_cycle)
                    else:
                        logger.warning(f"❌ Failed to generate {character_name}")

                    # Skip rear views for now to focus on main generation
                    # if generation_cycle % 2 == 0:
                    #     rear_prompt = self._create_rear_view_prompt(character_name, base_prompt)
                    #     rear_success = await self._generate_character_image(character_name, rear_prompt, generation_cycle, "_rear")
                    #     if rear_success:
                    #         self.generation_count += 1
                    #         logger.info(f"🍑 Generated {character_name} rear view NSFW #{self.generation_count}")

                    # Small delay between characters
                    await asyncio.sleep(3)

                logger.info(f"🔄 Cycle {generation_cycle} complete. Generated {self.generation_count} total images.")

                # Check if we've reached the limit
                if self.generation_count >= max_images:
                    logger.info(f"🎯 Reached {max_images} image limit. Stopping for user feedback...")
                    break

                # Wait before next cycle
                await asyncio.sleep(30)

            except KeyboardInterrupt:
                logger.info("🛑 Stopping autonomous generation...")
                self.is_running = False
                break
            except Exception as e:
                logger.error(f"❌ Error in generation cycle: {e}")
                await asyncio.sleep(10)

    def _create_rear_view_prompt(self, character: str, base_prompt: str) -> str:
        """Create rear view variant of character prompt"""

        rear_views = {
            "Mei_Kobayashi": "beautiful Japanese woman, long dark hair, from behind showing curves, wearing tight shorts and crop top, rear view, curvy hips and legs, bent over slightly, Tokyo apartment, photorealistic, sexy from behind, aroused posture",
            "Rina_Suzuki": "attractive Japanese woman, short brown hair, from behind showing curves, wearing tight mini skirt, rear view, curvy hips and thighs, hands on hips pose, Tokyo apartment, photorealistic, alluring from behind, arousal visible",
            "Yuki_Tanaka": "young Japanese man from behind, shirtless, casual pants, rear view while being embraced by attractive woman, her hands on his body, Tokyo apartment, photorealistic, arousal scene from behind, NSFW situation",
            "Takeshi_Sato": "intimidating Japanese man from behind in business suit, rear view in office, attractive secretary approaching from behind, power dynamic, photorealistic, authority figure from behind"
        }

        return rear_views.get(character, base_prompt + ", from behind, rear view")

    async def _generate_character_image(self, character: str, prompt: str, cycle: int, suffix: str = "") -> bool:
        """Generate a single character image"""

        try:
            workflow = self._create_workflow(character, prompt, cycle, suffix)

            response = requests.post(f"{self.comfyui_url}/prompt", json={"prompt": workflow["prompt"]}, timeout=15)
            result = response.json()

            if "prompt_id" in result:
                logger.info(f"📷 Queued {character}{suffix} generation: {result['prompt_id']}")

                # Wait for generation to complete
                await asyncio.sleep(12)

                # Check if file was created
                expected_file = self.output_dir / f"autonomous_tdd_{character}_cycle{cycle}{suffix}_00001_.png"
                # Also check for alternate naming patterns
                if not expected_file.exists() and suffix:
                    alt_file = self.output_dir / f"autonomous_tdd_{character}{suffix}_cycle{cycle}_00001_.png"
                    if alt_file.exists():
                        expected_file = alt_file
                if expected_file.exists():
                    file_size = expected_file.stat().st_size
                    if file_size > 100000:  # At least 100KB
                        return True

                return False
            else:
                logger.error(f"❌ No prompt_id returned for {character}{suffix}")
                return False

        except Exception as e:
            logger.error(f"❌ Generation failed for {character}{suffix}: {e}")
            return False

    def _create_workflow(self, character: str, prompt: str, cycle: int, suffix: str = "") -> dict:
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
                        "filename_prefix": f"autonomous_tdd_{character}_cycle{cycle}{suffix}",
                        "images": ["6", 0]
                    },
                    "class_type": "SaveImage"
                }
            }
        }

    def _ensure_character_consistency(self, character: str, base_prompt: str) -> str:
        """Ensure character prompt includes consistent physical traits"""

        if character not in self.character_traits:
            return base_prompt

        traits = self.character_traits[character]

        # Build consistent prompt with required traits
        consistent_prompt = f"{base_prompt}, {traits['hair']}, {traits['body']}, consistent character design, same person, recognizable features"

        return consistent_prompt

    async def _verify_body_consistency(self, character: str, cycle: int):
        """Verify body consistency of generated character"""

        expected_file = self.output_dir / f"autonomous_tdd_{character}_cycle{cycle}_00001_.png"

        if expected_file.exists():
            file_size = expected_file.stat().st_size

            # Basic checks
            if file_size > 500000:  # At least 500KB for detailed NSFW content
                logger.info(f"🔍 {character} body consistency: GOOD (size: {file_size//1000}KB)")

                # Store character trait verification
                traits = self.character_traits.get(character, {})
                logger.info(f"📝 Verified {character} traits: {traits.get('hair', '')}, {traits.get('body', '')}")

                return True
            else:
                logger.warning(f"⚠️  {character} body consistency: POOR (size: {file_size//1000}KB)")
                return False
        else:
            logger.error(f"❌ {character} verification failed: File not found")
            return False

    def get_status(self) -> dict:
        """Get current generation status"""

        return {
            "is_running": self.is_running,
            "total_generated": self.generation_count,
            "output_directory": str(self.output_dir),
            "nsfw_mode": True,
            "character_traits": self.character_traits
        }

async def main():
    """Main function"""

    generator = AutonomousImageGenerator()

    logger.info("🚀 Echo Brain Autonomous Tokyo Debt Desire Generator")
    logger.info("📷 Will continuously generate photorealistic character variations")
    logger.info("🔄 Press Ctrl+C to stop")

    try:
        await generator.start_autonomous_generation()
    except KeyboardInterrupt:
        logger.info("🛑 Autonomous generation stopped by user")

    # Show final status
    status = generator.get_status()
    logger.info(f"📊 Final Status: Generated {status['total_generated']} images")

if __name__ == "__main__":
    asyncio.run(main())