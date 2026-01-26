#!/usr/bin/env python3
"""
Tower LoRA Training Studio
Integrated training system for anime character LoRAs
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import psycopg2
from psycopg2.extras import RealDictCursor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'anime_production',
    'user': 'patrick',
    'password': 'RP78eIrW7cI2jYvL5akt1yurE',
    'port': 5432
}

# Paths
KOHYA_PATH = Path("/opt/tower-anime-production/training/kohya_real")
MODELS_PATH = Path("/mnt/1TB-storage/models")
LORA_OUTPUT_PATH = MODELS_PATH / "loras"
DATASET_BASE_PATH = Path("/opt/tower-lora-studio/datasets")
CONFIG_PATH = Path("/opt/tower-lora-studio/configs")


@dataclass
class TrainingConfig:
    """Configuration for LoRA training"""
    model_name: str
    base_model: str = "/mnt/1TB-storage/models/checkpoints/realisticVisionV51_v51VAE.safetensors"
    resolution: int = 512
    batch_size: int = 1
    max_train_epochs: int = 10
    learning_rate: float = 1e-4
    network_dim: int = 32
    network_alpha: int = 16
    save_every_n_epochs: int = 2
    caption_extension: str = ".txt"
    seed: int = 42
    mixed_precision: str = "fp16"
    xformers: bool = True
    cache_latents: bool = True
    gradient_checkpointing: bool = False


class LoRATrainingStudio:
    """Main training studio class"""

    def __init__(self):
        self.kohya_path = KOHYA_PATH
        self.ensure_directories()

    def ensure_directories(self):
        """Ensure all required directories exist"""
        for path in [DATASET_BASE_PATH, CONFIG_PATH, LORA_OUTPUT_PATH]:
            path.mkdir(parents=True, exist_ok=True)

    def get_db_connection(self):
        """Get database connection"""
        return psycopg2.connect(**DB_CONFIG)

    async def prepare_dataset(self,
                             dataset_name: str,
                             source_videos: List[str],
                             frame_interval: int = 30) -> Path:
        """Extract frames from videos for training dataset"""
        dataset_path = DATASET_BASE_PATH / dataset_name
        dataset_path.mkdir(parents=True, exist_ok=True)

        images_path = dataset_path / "images"
        images_path.mkdir(exist_ok=True)

        logger.info(f"Preparing dataset: {dataset_name}")

        frame_count = 0
        for video_path in source_videos:
            if not Path(video_path).exists():
                logger.warning(f"Video not found: {video_path}")
                continue

            # Extract frames using ffmpeg
            cmd = [
                "ffmpeg", "-i", video_path,
                "-vf", f"select='not(mod(n\\,{frame_interval}))'",
                "-vsync", "vfr",
                "-q:v", "2",
                f"{images_path}/frame_%04d.jpg"
            ]

            try:
                subprocess.run(cmd, check=True, capture_output=True)
                # Count extracted frames
                new_frames = len(list(images_path.glob("*.jpg"))) - frame_count
                frame_count += new_frames
                logger.info(f"Extracted {new_frames} frames from {Path(video_path).name}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to extract frames: {e}")

        logger.info(f"Total frames extracted: {frame_count}")
        return dataset_path

    async def create_captions(self, dataset_path: Path, trigger_word: str, base_prompt: str):
        """Create caption files for training images"""
        images_path = dataset_path / "images"

        for image_file in images_path.glob("*.jpg"):
            caption_file = image_file.with_suffix(".txt")
            caption = f"{trigger_word}, {base_prompt}"
            caption_file.write_text(caption)

        logger.info(f"Created captions for {len(list(images_path.glob('*.txt')))} images")

    def generate_training_config(self, config: TrainingConfig, dataset_path: Path) -> Path:
        """Generate Kohya training configuration file"""
        config_file = CONFIG_PATH / f"{config.model_name}_config.toml"

        toml_content = f"""
[model_arguments]
pretrained_model_name_or_path = "{config.base_model}"

[dataset_arguments]
train_data_dir = "{dataset_path / 'images'}"
resolution = {config.resolution}
caption_extension = "{config.caption_extension}"
cache_latents = {str(config.cache_latents).lower()}

[training_arguments]
output_dir = "{LORA_OUTPUT_PATH}"
output_name = "{config.model_name}"
max_train_epochs = {config.max_train_epochs}
train_batch_size = {config.batch_size}
learning_rate = {config.learning_rate}
save_every_n_epochs = {config.save_every_n_epochs}
seed = {config.seed}
mixed_precision = "{config.mixed_precision}"
xformers = {str(config.xformers).lower()}
gradient_checkpointing = {str(config.gradient_checkpointing).lower()}

[network_arguments]
network_module = "networks.lora"
network_dim = {config.network_dim}
network_alpha = {config.network_alpha}

[optimizer_arguments]
optimizer_type = "AdamW8bit"
lr_scheduler = "cosine_with_restarts"
lr_scheduler_num_cycles = 3
        """

        config_file.write_text(toml_content)
        logger.info(f"Generated config: {config_file}")
        return config_file

    async def start_training(self,
                           model_name: str,
                           dataset_path: Path,
                           config: Optional[TrainingConfig] = None) -> bool:
        """Start the actual training process"""
        if config is None:
            config = TrainingConfig(model_name=model_name)

        # Generate config file
        config_file = self.generate_training_config(config, dataset_path)

        # Prepare training script
        training_script = self.kohya_path / "train_network.py"

        if not training_script.exists():
            logger.error(f"Training script not found: {training_script}")
            return False

        # Build command
        cmd = [
            sys.executable,
            str(training_script),
            "--config_file", str(config_file),
            "--enable_bucket",
            "--min_bucket_reso", "256",
            "--max_bucket_reso", "1024",
            "--bucket_reso_steps", "64",
        ]

        logger.info(f"Starting training with command: {' '.join(cmd)}")

        # Run training
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.kohya_path)
            )

            # Stream output
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                logger.info(line.decode().strip())

            await process.wait()

            if process.returncode == 0:
                logger.info("Training completed successfully")

                # Update database
                await self.update_database(model_name, dataset_path)
                return True
            else:
                logger.error(f"Training failed with code {process.returncode}")
                return False

        except Exception as e:
            logger.error(f"Training error: {e}")
            return False

    async def update_database(self, model_name: str, dataset_path: Path):
        """Update database with training results"""
        conn = self.get_db_connection()
        try:
            cursor = conn.cursor()

            # Check if this is a character LoRA
            cursor.execute("""
                UPDATE characters
                SET lora_path = %s, updated_at = NOW()
                WHERE LOWER(REPLACE(name, ' ', '_')) = %s
                RETURNING id, name
            """, (
                str(LORA_OUTPUT_PATH / f"{model_name}.safetensors"),
                model_name.replace('_lora', '').lower()
            ))

            result = cursor.fetchone()
            if result:
                logger.info(f"Updated character {result[1]} with LoRA path")

            conn.commit()
        finally:
            conn.close()

    async def quick_test_training(self):
        """Quick test with minimal dataset"""
        logger.info("Starting quick test training")

        # Use existing test videos
        test_videos = [
            "/mnt/1TB-storage/ComfyUI/output/custom_lora_test_00001.mp4"
        ]

        # Prepare minimal dataset
        dataset_path = await self.prepare_dataset(
            "test_cowgirl_position",
            test_videos,
            frame_interval=10  # Extract every 10th frame for quick test
        )

        # Create captions
        await self.create_captions(
            dataset_path,
            trigger_word="cowgirl_position",
            base_prompt="a woman in cowgirl position, intimate scene"
        )

        # Configure for quick test
        config = TrainingConfig(
            model_name="test_cowgirl_lora",
            max_train_epochs=5,  # Quick test
            learning_rate=5e-5,
            network_dim=16,  # Smaller for faster training
            network_alpha=8,
            save_every_n_epochs=1
        )

        # Start training
        success = await self.start_training(
            "test_cowgirl_lora",
            dataset_path,
            config
        )

        return success


async def main():
    """Main entry point"""
    studio = LoRATrainingStudio()

    # Run quick test
    success = await studio.quick_test_training()

    if success:
        logger.info("Test training completed successfully")
        logger.info(f"Check output at: {LORA_OUTPUT_PATH}")
    else:
        logger.error("Test training failed")


if __name__ == "__main__":
    asyncio.run(main())