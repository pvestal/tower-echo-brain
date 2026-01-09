#!/usr/bin/env python3
"""
LoRA Dataset Creator for Patrick's Characters
Prepares training datasets from generated images
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class LoRADatasetCreator:
    """Create LoRA training datasets for Patrick's characters"""

    def __init__(self):
        self.source_dir = Path("/mnt/1TB-storage/ComfyUI/output")
        self.dataset_base = Path("/mnt/1TB-storage/lora_datasets")
        self.dataset_base.mkdir(exist_ok=True)

    def create_character_dataset(self, character_name: str, project: str) -> Dict:
        """Create a LoRA dataset for a specific character"""

        # Find all images for this character
        pattern = f"patrick_{character_name}_*.png"
        images = list(self.source_dir.glob(pattern))

        if not images:
            return {"error": f"No images found for {character_name}"}

        # Create dataset directory
        dataset_dir = self.dataset_base / f"{project}_{character_name}"
        dataset_dir.mkdir(exist_ok=True)

        # Create subdirectories
        img_dir = dataset_dir / f"10_patrick_{character_name}"
        img_dir.mkdir(parents=True, exist_ok=True)

        # Copy images and create captions
        for idx, img_path in enumerate(images[:20]):  # Max 20 for training
            # Copy image
            dest_img = img_dir / f"{idx:04d}.png"
            shutil.copy2(img_path, dest_img)

            # Create caption file
            caption = self._generate_caption(character_name, project, idx)
            caption_file = img_dir / f"{idx:04d}.txt"
            caption_file.write_text(caption)

        # Create training config
        config = {
            "character": character_name,
            "project": project,
            "num_images": len(images),
            "dataset_path": str(dataset_dir),
            "training_params": {
                "resolution": "1024,1024",
                "batch_size": 1,
                "max_train_steps": 2000,
                "learning_rate": "1e-4",
                "network_dim": 32,
                "network_alpha": 16,
                "trigger_word": f"patrick_{character_name}"
            }
        }

        config_file = dataset_dir / "training_config.json"
        config_file.write_text(json.dumps(config, indent=2))

        logger.info(f"✅ Created LoRA dataset for {character_name} at {dataset_dir}")

        return {
            "status": "success",
            "character": character_name,
            "dataset_path": str(dataset_dir),
            "num_images": len(images),
            "ready_for_training": True
        }

    def _generate_caption(self, character: str, project: str, index: int) -> str:
        """Generate training captions with variety"""

        # Patrick's character descriptions
        descriptions = {
            "tokyo_debt_crisis": {
                "riku": "male protagonist with brown hair, casual clothes",
                "yuki": "yakuza daughter with long black hair",
                "sakura": "childhood friend with pink hair"
            },
            "goblin_slayer_neon": {
                "kai_nakamura": "tech specialist with spiky black hair",
                "ryuu": "sniper with silver hair",
                "hiroshi": "infiltrator with short dark hair"
            }
        }

        base_desc = descriptions.get(project, {}).get(character, "anime character")

        # Variety in captions for better training
        variations = [
            f"patrick_{character}, {base_desc}",
            f"a photo of patrick_{character}, {base_desc}",
            f"patrick_{character} character, {base_desc}, masterpiece",
            f"artwork of patrick_{character}, {base_desc}, best quality",
            f"patrick_{character}, {base_desc}, detailed"
        ]

        return variations[index % len(variations)]

    def create_all_datasets(self) -> Dict:
        """Create datasets for all Patrick's characters"""

        results = {}

        # Tokyo Debt Crisis characters
        for char in ["riku", "yuki", "sakura"]:
            result = self.create_character_dataset(char, "tokyo_debt_crisis")
            results[char] = result

        # Goblin Slayer characters
        for char in ["ryuu", "hiroshi", "kai_nakamura"]:
            result = self.create_character_dataset(char, "goblin_slayer_neon")
            results[char] = result

        return results

# Singleton
lora_creator = LoRADatasetCreator()