#!/usr/bin/env python3
"""
Goblin Slayer Violence Scene Generator
Specialized generator for cyberpunk violence and gore scenes
"""

import json
import requests
import random
from datetime import datetime
from pathlib import Path
import psycopg2
from psycopg2.extras import RealDictCursor

# ComfyUI endpoint
COMFYUI_URL = "http://localhost:8188"

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'anime_production',
    'user': 'patrick',
    'password': 'RP78eIrW7cI2jYvL5akt1yurE',
    'port': 5432
}

class GoblinSlayerGenerator:
    """Generate violent action scenes for Goblin Slayer style content"""

    def __init__(self):
        self.scene_templates = {
            "massacre": {
                "base_prompt": "goblin slayer in cyberpunk armor, brutal combat scene, massacring goblins",
                "loras": ["DreamLTXV.safetensors", "LTX2_SS_Motion_7K.safetensors"],
                "strengths": [0.6, 0.8],
                "violence_tags": ["blood_splatter", "dismemberment", "gore_explosion"]
            },
            "execution": {
                "base_prompt": "goblin slayer performing execution, cyberpunk setting, neon lights",
                "loras": ["DreamLTXV.safetensors"],
                "strengths": [0.7],
                "violence_tags": ["decapitation", "blood_pool", "execution"]
            },
            "torture_interrogation": {
                "base_prompt": "dark interrogation scene, goblin captured, cyberpunk dungeon",
                "loras": [],
                "strengths": [],
                "violence_tags": ["torture", "brutal_beating"]
            },
            "sword_combat": {
                "base_prompt": "intense sword fight, energy blade combat, cyberpunk arena",
                "loras": ["LTX2_SS_Motion_7K.safetensors"],
                "strengths": [0.9],
                "violence_tags": ["sword_slash", "blood_splatter", "hack_and_slash"]
            },
            "gore_aftermath": {
                "base_prompt": "battlefield aftermath, corpses and blood, dark cyberpunk atmosphere",
                "loras": ["DreamLTXV.safetensors"],
                "strengths": [0.5],
                "violence_tags": ["corpse_pile", "blood_pool", "severed_limbs"]
            }
        }

    def get_db_connection(self):
        """Get database connection"""
        return psycopg2.connect(**DB_CONFIG)

    def build_violence_prompt(self, scene_type: str, character_name: str = "Goblin Slayer"):
        """Build prompt with violence elements"""
        template = self.scene_templates.get(scene_type, self.scene_templates["massacre"])

        # Add violence tags to prompt
        violence_elements = ", ".join(template["violence_tags"])
        full_prompt = f"{character_name}, {template['base_prompt']}, {violence_elements}"

        # Add cyberpunk elements
        cyberpunk_additions = [
            "neon blood effects",
            "cybernetic gore",
            "holographic blood spray",
            "mechanical dismemberment",
            "plasma blade cuts"
        ]

        selected_additions = random.sample(cyberpunk_additions, 2)
        full_prompt += ", " + ", ".join(selected_additions)

        return full_prompt, template["loras"], template["strengths"]

    def generate_violence_scene(self, scene_type: str, output_name: str = None):
        """Generate a violent scene"""

        prompt, lora_names, strengths = self.build_violence_prompt(scene_type)

        if not output_name:
            output_name = f"goblin_slayer_{scene_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        print(f"🗡️ Generating {scene_type} scene")
        print(f"   Prompt: {prompt}")
        print(f"   LoRAs: {', '.join(lora_names) if lora_names else 'None'}")

        # Build ComfyUI workflow
        workflow = self.build_workflow(prompt, lora_names, strengths, output_name)

        # Submit to ComfyUI
        response = requests.post(f"{COMFYUI_URL}/prompt", json={"prompt": workflow})

        if response.status_code == 200:
            result = response.json()
            if "prompt_id" in result:
                prompt_id = result["prompt_id"]
                print(f"✅ Scene submitted: {prompt_id}")

                # Save to database
                self.save_to_database(scene_type, prompt, prompt_id, output_name)

                return {
                    "success": True,
                    "prompt_id": prompt_id,
                    "output_name": output_name,
                    "scene_type": scene_type
                }
            else:
                print(f"❌ Error: {result}")
                return {"success": False, "error": result}
        else:
            print(f"❌ Failed: {response.status_code}")
            return {"success": False, "error": response.text}

    def build_workflow(self, prompt: str, lora_names: list, strengths: list, output_name: str):
        """Build ComfyUI workflow for violence scene"""

        workflow = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "ltxv-2b-fp8.safetensors"}
            },
            "2": {
                "class_type": "CLIPLoader",
                "inputs": {"clip_name": "t5xxl_fp16.safetensors", "type": "ltxv"}
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": prompt,
                    "clip": ["2", 0]
                }
            }
        }

        # Add LoRAs if specified
        current_model = ["1", 0]
        current_clip = ["2", 0]

        for idx, (lora_name, strength) in enumerate(zip(lora_names, strengths)):
            node_id = str(10 + idx)
            workflow[node_id] = {
                "class_type": "LoraLoader",
                "inputs": {
                    "lora_name": lora_name,
                    "strength_model": strength,
                    "strength_clip": strength * 0.8,
                    "model": current_model,
                    "clip": current_clip
                }
            }
            current_model = [node_id, 0]
            current_clip = [node_id, 1]

        # Sampler
        workflow["20"] = {
            "class_type": "LTXVBaseSampler",
            "inputs": {
                "model": current_model,
                "cond": ["3", 0],
                "uncond": ["21", 0],
                "fps": 12,  # Higher FPS for action
                "width": 768,
                "height": 512,
                "length": 97,  # Longer for violence scenes
                "seed": random.randint(0, 999999),
                "steps": 35,
                "cfg": 4.0,  # Higher CFG for more adherence
                "denoise": 1.0,
                "sampler_name": "euler"
            }
        }

        # Negative prompt
        workflow["21"] = {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "censored, blurred, pixelated, child-like, cartoon",
                "clip": current_clip
            }
        }

        # VAE Decode - use simple decode without tiling
        # Note: The checkpoint loader returns [model, clip, vae]
        # We need index 2 for VAE
        workflow["22"] = {
            "class_type": "LTXVTiledVAEDecode",
            "inputs": {
                "latents": ["20", 0],
                "vae": ["1", 2],  # VAE is index 2 from checkpoint
                "vertical_tiles": 2,
                "horizontal_tiles": 2,
                "overlap": 0.1,
                "last_frame_fix": True
            }
        }

        # Video output
        workflow["23"] = {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["22", 0],
                "frame_rate": 12,
                "loop_count": 0,
                "filename_prefix": output_name,
                "format": "video/h264-mp4",
                "pingpong": False,
                "save_output": True,
                "videopreview": {"format": "webp"}
            }
        }

        return workflow

    def save_to_database(self, scene_type: str, prompt: str, prompt_id: str, output_name: str):
        """Save violence scene generation to database"""
        conn = self.get_db_connection()
        try:
            cursor = conn.cursor()

            # Create table if not exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS violence_scene_generations (
                    id SERIAL PRIMARY KEY,
                    scene_type VARCHAR(50),
                    prompt TEXT,
                    prompt_id VARCHAR(100),
                    output_name VARCHAR(255),
                    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Insert record
            cursor.execute("""
                INSERT INTO violence_scene_generations
                (scene_type, prompt, prompt_id, output_name)
                VALUES (%s, %s, %s, %s)
            """, (scene_type, prompt, prompt_id, output_name))

            conn.commit()
        finally:
            conn.close()

    def generate_full_sequence(self):
        """Generate a full violence sequence"""
        scenes = ["sword_combat", "massacre", "execution", "gore_aftermath"]

        results = []
        for scene in scenes:
            result = self.generate_violence_scene(scene)
            results.append(result)
            print(f"   Scene {scene}: {'✅' if result['success'] else '❌'}")

        return results


def main():
    """Test violence scene generation"""
    generator = GoblinSlayerGenerator()

    print("🗡️ GOBLIN SLAYER VIOLENCE SCENE GENERATOR")
    print("=" * 50)

    # Test single scene
    print("\n1. Generating massacre scene...")
    result = generator.generate_violence_scene("massacre")

    if result["success"]:
        print(f"✅ Success! Check output: {result['output_name']}")
    else:
        print(f"❌ Failed: {result.get('error')}")

    # Generate full sequence
    print("\n2. Generating full violence sequence...")
    sequence_results = generator.generate_full_sequence()

    success_count = sum(1 for r in sequence_results if r["success"])
    print(f"\n✅ Generated {success_count}/{len(sequence_results)} scenes successfully")


if __name__ == "__main__":
    main()